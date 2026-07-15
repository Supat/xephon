# Plugin architecture plan

**Status (2026-07-16):** Phases 0–3 implemented on `plugin-arch`.
Phase 0 (seams, registry, payloads) → Phase 1 (page/menu/export
slots, DebugSamplePlugin) → Phase 2 (inference carve-out,
EvalFormPlugin) → Phase 3 (persistent plugin storage, import
service, section proposals, template-pack import, payload v1→v2
migration exercise). Notable deltas from the plan as written:
session events are a `PluginHandle` callback rather than an
`AsyncSequence` (deterministic, less test flake); the settings
toggle UI shipped with Phase 1 rather than Phase 0; the inference
carve-out shipped with its consumer in Phase 2 rather than Phase 0.
Phase 3 decision points: the first retrofit (keyword-review sheet)
is DEFERRED until the EvalForm ground-truth eval exists — two
in-flight reshapings of that surface at once isn't worth it; T3
scripting still has no customer and stays deferred. Post-Phase-3
the EvalForm plugin kept evolving on this branch (supplementary
distillation, CSV export, synthetic eval harness, rubric
conformance, road provenance — see docs/evalform_plugin.md /
docs/evalform_pipeline.md). Open: ground-truth eval per the
research doc §6, on-device verification sweep.
**Companion:** `docs/eval_form_autofill_research.md` — the evaluation-form
auto-fill is the first plugin and the forcing function for every extension
point below.

---

## 1. What "plugin" can mean on this platform

iOS forbids loading native code that isn't signed into the app bundle — no
`dlopen` of downloaded dylibs, no JIT. A sideloaded research app is not
exempt. That leaves three honest tiers:

| Tier | What ships | When it loads | Feasible on iPadOS |
|---|---|---|---|
| **T1: compiled plugin modules** | Swift SPM targets in this repo, registered at startup | compile time | yes |
| **T2: data packs** | declarative content (form templates, vocab, render layouts) as JSON documents | runtime, via file import | yes |
| **T3: scripted plugins** | logic in an embedded interpreter (JavaScriptCore, WASM via WasmKit) | runtime | yes, but heavy |

**Plan: T1 + T2. T3 is explicitly deferred** — it buys third-party
extensibility this single-team research app doesn't need yet, at the cost of
a sandboxed API bridge, a second language, and a much larger test surface.
The seam to add it later is the same host API T1 defines.

The practical meaning of "plugin" here is therefore: **first-party features
built against a frozen host API instead of against `RecordingController`'s
internals** — enforced by the module boundary (a plugin target *cannot*
import the app target) — plus **runtime-importable content** for the parts
that genuinely vary per use case (evaluation-form definitions).

## 2. Shape

```
Package.swift
  ├─ XephonPluginKit        NEW — plugin protocol + host service protocols.
  │                          Depends only on Fusion/Export value types.
  ├─ Plugins/
  │   └─ EvalFormPlugin     NEW — first plugin. Depends on XephonPluginKit
  │                          (+ XephonUtilities). Never on the app target.
  └─ (existing Core targets unchanged)

Xephon app target
  ├─ PluginHost/            NEW — the only place that knows both worlds:
  │    PluginRegistry        compile-time list of installed plugins
  │    SessionHostAdapter    implements SessionReading over RecordingController
  │    InferenceHostAdapter  implements InferenceService over SummarizerCoordinator
  │    UIHostAdapters        page/menu/exporter/sheet plumbing
  └─ (existing views gain plugin slots; see §4)
```

Registration is a hardcoded array in `XephonApp` (`PluginRegistry.install
([EvalFormPlugin()])`). No discovery magic; adding a plugin is a one-line
diff plus a `project.yml` target entry.

## 3. The host API (XephonPluginKit)

The protocol surface — kept deliberately small; everything a plugin touches
goes through it:

```swift
public protocol XephonPlugin: Sendable {
    static var id: PluginID { get }          // reverse-DNS-ish, stable
    static var displayName: String { get }   // localized
    static var payloadVersion: Int { get }   // for .xph payload migration
    @MainActor func activate(host: any PluginHost) -> PluginHandle
}
```

`PluginHost` hands out capability services (each its own protocol, so tests
can stub them and future T3 sandboxing can gate them):

- **`SessionReading`** — value-type snapshots of utterances, speaker names,
  sections, keywords, diarization timeline, session title; an
  `AsyncSequence` of session events (`loaded`, `cleared`,
  `utterancesChanged(version:)`). Read-only. No `RecordingController`
  types cross the boundary.
- **`SessionAnnotating`** — the narrow write set plugins may need: propose
  sections, contribute keyword groups (seeded once, user-editable after),
  register undoable actions on the session's `UndoManager` via a host
  wrapper. Every write is undoable and journaled by plugin id.
- **`InferenceService`** — `generate(prompt:schema:budget:) async throws ->
  JSONValue` routed through the user's configured summarizer backend
  (Apple FM / MLX / LM Studio), plus availability queries. Plugins never
  link MLX or talk HTTP themselves. This inherits the app's privacy
  posture for free: a plugin *cannot* reach the network except through
  host services that honor the cloud-consent rules.
- **`PluginStorage`** — (a) per-plugin settings via namespaced defaults;
  (b) per-session payload: `Data` blob stored in the `.xph` bundle keyed by
  plugin id + `payloadVersion` (see §5); (c) per-plugin scratch directory.
- **`ExportRegistering`** — register named exporters (`(SessionSnapshot,
  payload) -> ExportProduct`); the host routes actual file I/O through
  `FilePickerCoordinator` and the single modifier pair at the root.
  Content types a plugin can write are declared statically by the plugin
  (compile-time), because `DataFileDocument.{readable,writable}ContentTypes`
  is a compile-time whitelist.
- **`UIContributing`** — declarative descriptors, not free-floating views:
  - `controlPanePages: [PluginPageDescriptor]` — title + `@MainActor`
    view builder; host appends them to the control-pane `TabView` (tags
    allocated after the built-in pages; the documented rotation/rebuild
    traps in `ControlPaneView` stay the host's problem, not the plugin's).
  - `menuCommands: [PluginMenuDescriptor]` — routed through the existing
    `MenuCommands` bus (plugins never touch `CommandGroup` directly; the
    iPadOS-26 traps live in one place).
  - Sheets/alerts driven by plugin-owned `@Observable` state, presented
    from the plugin's own page — inheriting the single-enum modal
    discipline (`KeywordsCard.Presentation` precedent).

**Rules of the road** (inherited from CLAUDE.md, enforced in review + a
PluginKit lint test where cheap): typed errors per plugin, `os.Logger`
category per plugin id, Swift 6 strict concurrency, `String(localized:)`,
no inline `.fileImporter`/`.fileExporter`/`.alert` stacks, no audio or
transcript bytes leaving the device except through host export/inference
services.

## 4. Host-side seams (what the app must grow)

1. **`SessionHostAdapter`** — the read facade. `RecordingController` already
   exposes almost everything as `@Observable` value types; the adapter's job
   is snapshotting + the event stream, keyed on the existing
   `utterancesVersion` / `sessionToken` so plugins get the same invalidation
   signals the built-in memos use.
2. **`InferenceHostAdapter`** — carve the backend dispatch out of
   `SummarizerCoordinator` into a schema-in/JSON-out call. This is the same
   seam the eval research doc needs for its backend bake-off, so the work
   pays twice. Includes a serialization gate: plugin inference and built-in
   summarize/review runs contend for the same model memory; the host
   queues, plugins just await.
3. **`.xph` payload field** — `SessionDocument` gains
   `pluginPayloads: [String: PluginPayload]?` where `PluginPayload =
   {version: Int, data: Data}`. Nil round-trips (older bundles unchanged);
   **unknown plugin ids are preserved verbatim on load→save**, so opening a
   session on a build without some plugin never destroys its data.
4. **Control-pane page injection** — the `TabView` page list becomes
   built-ins + registry pages. Tag stability across enable/disable matters
   (the page controller rebuild traps are already documented in that file).
5. **Settings** — a Plugins section listing installed plugins with
   enable/disable toggles (disabled = not activated at startup; its `.xph`
   payloads still round-trip opaquely).

## 5. First plugin: EvalFormPlugin

Maps the pipeline from `docs/eval_form_autofill_research.md` onto the API:

| Pipeline stage | Host service used |
|---|---|
| form template (A-1 first) | **T2 data pack**: JSON document — items, scales, step semantics, reference roads, vocabulary seeds, render layout. Imported via host file-import; new templates (cornering sheet, other labs' forms) are data, not code |
| onomatopoeia tagging | `SessionAnnotating.contributeKeywords` (seeds the existing bank; matching/homophone review is the app's, untouched) |
| road segmentation | `SessionReading` rows → `SessionAnnotating.proposeSections` |
| spoken-score capture | plugin-internal (regex/lexicon over snapshot rows) |
| LLM field extraction | `InferenceService.generate(prompt:schema:)` — stated-vs-inferred / null-first / evidence-rows policy from the research doc |
| draft form + review state | `PluginStorage` session payload (versioned) |
| review UI (field ↔ row ↔ audio) | plugin control-pane page + sheet; row playback via a `SessionReading.requestPlayback(row:)` host call |
| export (markdown/CSV) | `ExportRegistering` |

The plugin ships with the A-1 template embedded as its default pack, so the
data-pack mechanism has exactly one consumer and one embedded example from
day one.

## 6. What is deliberately not in scope

- **No dynamic native code loading** — platform-impossible on iOS; not
  attempted.
- **No scripting tier yet** (T3) — revisit only when someone outside the
  repo needs to add logic without building the app.
- **No big-bang retrofit** of existing features (keywords, sections,
  summarizer) into plugins. The boundary is proven by *new* code first.
  After EvalFormPlugin ships, retrofit candidates get assessed one at a
  time — likely first candidate: the keyword-review sheet, which already
  has the shape of a plugin (own model, own sheet, session-scoped state).
- **No plugin marketplace/versioned ABI concerns** — everything compiles
  together; API breaks are compile errors fixed in the same PR.

## 7. Phasing

- **Phase 0 — seams, no visible change.** `XephonPluginKit` target;
  `SessionHostAdapter` + event stream; `InferenceHostAdapter` carve-out;
  `SessionDocument.pluginPayloads` (with opaque-preservation round-trip
  tests); registry + settings toggles. Exit: an empty `HelloPlugin` in
  `Tests/` activates, receives session events, persists a payload through
  Save → Open, and cannot import the app target (build-graph enforced).
- **Phase 1 — UI + export slots.** Page injection, menu descriptors,
  exporter registration, `requestPlayback`. Exit: HelloPlugin shows a page
  with live session data and exports a text file through the root picker.
- **Phase 2 — EvalFormPlugin.** Per §5, template pack + deterministic
  passes + LLM extraction + review UI + exports; ground-truth eval per the
  research doc's §6, numbers to `docs/eval_log.md`.
- **Phase 3 — hardening + second consumer.** Template-pack import UX,
  payload migration exercise (bump `payloadVersion` once, on purpose),
  first retrofit candidate, and a decision point on whether T3 scripting
  has a real customer.

## 8. Risks / open questions

- **Facade drift** — the host adapters must not become a second
  `RecordingController` API; keep `SessionReading` snapshot-shaped and
  resist adding one-off accessors per plugin request.
- **TabView fragility** — page injection touches the most trap-dense view
  in the app (documented rotation/rebuild workarounds); Phase 1 needs
  device testing on exactly those traps.
- **Model contention** — plugin inference vs auto-summarize/review runs;
  the host gate serializes, but UX for "queued behind summarizer" needs a
  design pass in Phase 2.
- **Payload forward-compat** — opaque preservation is specified in Phase 0
  precisely because it's the easiest thing to get silently wrong later.
- **Where prompts live** — plugin code (versioned with logic) vs data pack
  (varies per form). Current call: prompt *templates* in plugin code,
  form-specific vocabulary/field lists interpolated from the pack; revisit
  if packs turn out to need prompt-level variation.
