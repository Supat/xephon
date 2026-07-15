# EvalFormPlugin — how the evaluation plugin works

Component reference for `Plugins/EvalFormPlugin/` as built
(2026-07-15): structure, lifecycle, host integration, state, UI,
persistence, and extension points. The extraction *algorithm*
(candidate selection → deterministic tier → LLM tier → merge) has
its own minute-detail reference in `docs/evalform_pipeline.md`;
this doc covers everything around it — the plugin as a component
of the plugin architecture (`docs/plugin_architecture.md`).

## 1. What it is

The first shipped consumer of the Xephon plugin architecture: a
compiled (tier-T1) plugin that fills a Japanese vehicle
ride-quality evaluation sheet from a session transcript, with the
sheet definition itself carried as a replaceable (tier-T2) data
pack. Registered unconditionally in `xephonInstalledPlugins()`
(Xephon/PluginHost/InstalledPlugins.swift) — it ships in Release
builds, next to the Debug-only sample plugin.

Identity: `PluginID("xephon.evalform")`, display name 官能評価シート
/ "Evaluation Form", `payloadVersion 2`.

## 2. Module boundary

`EvalFormPlugin` is an SPM target (Package.swift → path
`Plugins/EvalFormPlugin`) depending ONLY on:

- `XephonPluginKit` — the frozen host API (its sole window into
  the app),
- `Fusion` — `UtteranceEstimate` value type,
- `XephonUtilities`, `XephonLogging`.

The app target is unreachable from here by construction: the
plugin cannot see `RecordingController`, the pipeline, pickers, or
any view of the main app. Everything it does goes through the
capability services handed to it at activation. Localized strings
live in the target's own catalog
(`Resources/Localizable.xcstrings`, en + ja, accessed via
`bundle: .module`).

## 3. File map

| File | Role |
|---|---|
| `EvalFormPlugin.swift` | `XephonPlugin` conformance: activation, keyword seeding, page + menu contribution |
| `EvalFormModel.swift` | `@MainActor @Observable` state: phase, draft, template lifecycle, run entry, exports, review toggles |
| `EvalFormTemplate.swift` | the data-pack schema (`Codable`) + the embedded A-1 default (items, scales incl. ※ rubric anchors, metadata fields/cues) |
| `EvalFormRunner.swift` | headless fill pipeline (shared by the plugin page and the eval harness) |
| `EvalFormExtractor.swift` | pure extraction machinery: candidates, ±2 context window, spoken-score grammar hookup, prompts/schemas, lenient parsing + truncation repair, merge policy, road-section proposals |
| `SpokenScoreParser.swift` | the deterministic spoken-score / preference grammar |
| `EvalFormDraft.swift` | the result document (payload v2) + version-aware restore/migration |
| `EvalFormCoverage.swift` | the 未検出 (to-fill) computation shared by card + exports |
| `EvalFormCard.swift` | the control-pane page UI |
| `EvalFormScaleViews.swift` | the sheet's two axes drawn as printed (strength / preference) |
| `EvalFormMarkdown.swift`, `EvalFormCSV.swift` | export renderers |
| `EvalFormSynthetic.swift` | known-answer session generator + scorer for the model eval harness |

## 4. Lifecycle

**Install.** The registry activates the plugin at startup when its
settings toggle (Plugins card, summarizer page) is enabled —
default on.

**Activation** (`activate(host:)`), in order:
1. `EvalFormModel(host:)` is created. Its init loads the active
   template — an imported pack from persistent storage
   (`templatePack` key) when present and non-empty, else the
   embedded A-1 — and restores the session draft from the plugin
   payload (version-aware, §9).
2. `model.seedKeywords()` — the ACTIVE template's vocabulary goes
   into the user's keyword bank under a group named after the
   template. Idempotent by the host contract (never duplicates,
   never overwrites); this is what lights up keyword highlighting,
   homophone review, and the keyword timeline strip for the
   evaluation vocabulary even if the user never runs a fill.
3. A `PluginHandle` is returned carrying:
   - the session-event callback → `model.handle(_:)`;
   - one page descriptor (`xephon.evalform.page`, checklist icon)
     whose content closure builds `EvalFormCard(model:)`;
   - one menu command (`xephon.evalform.export`, File menu,
     "評価シートを書き出す…") gated by `isEnabled: { model.canExport }`
     — reading the model's observable state at menu render keeps
     the item live.

**Session events** (all delivered on the MainActor):
- `.sessionLoaded` → re-restore the draft from the (just-replaced)
  payload;
- `.sessionCleared` → drop draft, reset phase;
- `.utterancesChanged` → ignored; staleness is derived by
  comparing the draft's stored `utterancesVersion` against the
  live snapshot at render time, not evented.

**Deactivation** (toggle off) drops the handle: page and menu item
disappear, the model deallocates. Stored payloads and the imported
pack are untouched.

## 5. Host services consumed

| Service | Used for |
|---|---|
| `SessionReading.snapshot()` | every read of utterances / speaker names / version — value snapshots at call time, no live references |
| `InferenceService.generate` | the per-item, supplementary, and metadata LLM calls (schema-in / raw-string-out; backend enforcement is the host's) |
| `InferenceService.withBatch` | wraps the whole fill so MLX backends pay ONE model load/unload cycle per run |
| `InferenceService.availability` | gates the Fill Form button and surfaces the reason string when unavailable |
| `PluginStorage.sessionPayloadData/-Version` | the draft, persisted inside the `.xph` |
| `PluginStorage.persistentData` | the imported template pack (cross-session, defaults-backed) |
| `ExportPresenting.presentExport` | markdown / CSV through the app's single root exporter |
| `ImportPresenting.presentImport` | template-pack JSON import (host owns security scopes; the plugin receives bytes) |
| `SessionAnnotating.contributeKeywords` | vocabulary seeding (§4) |
| `SessionAnnotating.proposeSections` | road-section proposals (§11) |
| `PluginHost.requestPlayback` | evidence-chip taps → row audio, same semantics as a transcript row's play button |

Deliberately NOT used: no network, no ML runtime, no file paths —
none are reachable from the module, which is the point.

## 6. `EvalFormModel` — observable state

- `phase: idle | running(step) | failed(reason)` — drives the
  progress row / error line. `running`'s string is the per-item
  progress ("ヒョコヒョコ (2/6)") forwarded from the runner's
  `onProgress`.
- `draft: EvalFormDraft?` — the current result document.
- `template: EvalFormTemplate` — the active sheet.
- `lastExport`, `lastSectionDetection` — one-line feedback.
- Derived: `draftIsStale` (version stamp vs live),
  `draftMatchesTemplate` (a pack swap orphans a draft — item ids
  belong to the old sheet, so results only render when ids match),
  `canExport` (draft present AND matches), `candidateCounts` (the
  pre-run coverage readout: mention rows per item),
  `usesImportedTemplate`.
- Actions: `run()` (gates → `EvalFormRunner.fill` inside the batch
  → persist), `importTemplatePack()` / `resetTemplateToDefault()`,
  `detectRoadSections()`, `toggleReviewed(_:)` / `isReviewed(_:)`,
  `playRow(_:)`, `exportMarkdown()` / `exportCSV()`.

`run()` catches only cancellation (abort, keep the previous
draft); every other failure is absorbed per-stage by the runner
(degradation table in the pipeline doc).

## 7. Template pack system (tier T2)

`EvalFormTemplate` is plain `Codable` — a pack is a JSON file:

- `id` (stable; drafts record it), `name` (sheet title; also the
  keyword group name);
- `strengthScale` {minimum, maximum, step, `anchors[]`} — anchors
  are the sheet's ※ rubric (magnitude + meaning), used in the
  extraction prompt and the card footnote; optional for older
  packs;
- `preferenceScale` {minimum, maximum};
- `items[]` {id, number, titleJa, titleEn, definition,
  referenceRoads[], vocabulary[]} — `vocabulary` drives candidate
  matching AND keyword seeding; `referenceRoads` derive the
  fallback road lexicon (speeds stripped);
- `metadataFields[]` + optional `metadataCues[]`;
- optional `roadCallouts[]` {road, surfaces[]} — the explicit
  section-detection lexicon: canonical label + every transcript
  form that counts as its callout (partial/alias forms, course
  naming). When present it replaces the derived labels.

**Import flow**: Template menu → Import Template… → root picker
(JSON) → decode + validate (non-empty items; failures surface
`evalform.error.badPack`) → persist bytes to plugin persistent
storage → swap `model.template` → re-seed keywords under the new
pack's name. **Reset** removes the stored pack and returns to the
embedded A-1 (re-seeding again). Because drafts record their
`templateID`, a swap never renders an old draft against the wrong
sheet — the card shows the mismatch note + coverage instead, and
exports are gated off until a re-fill.

Everything sheet-specific is data. Code changes are needed only
for new *behaviors* (a new scale type, a new pass), not new sheets.

## 8. UI composition

The host wraps the page in the app's standard scroll/inset chrome;
the plugin ships one glass card:

1. **Header**: sheet name; Fill Form (disabled while running or
   when inference is unavailable — the reason renders beneath) and
   Export (menu: Markdown / CSV; gated by `canExport`); the
   Template menu (import / reset) and Detect road sections with a
   "+N" feedback count.
2. **Pre-run**: the vocabulary-coverage readout (mention-row count
   per item — what a run would chew on before spending inference
   time). Post-run this is replaced by results.
3. **Results per item**: reviewed-confirmed toggle (green check,
   payload state), title, numeric score badge (stated plain /
   inferred orange + `?` / — when empty), then the sheet's two
   axes as printed — 強い −1…+1 弱い with 0.125 ticks and labeled
   majors, 嫌い 1…9 好き — marker coding matching the badge; the
   comment; evidence chips (adaptive grid, tap = playback,
   collapsed past 6 behind "+N"); orange conflict notes
   (revision trails, llmFailed, 極性要確認).
4. **補足コメント** with its own evidence chips.
5. **未検出（要手動記入）** — the reviewer's to-fill list
   (`EvalFormCoverage`, same data the exports render).
6. **Staleness / template-mismatch banners** when applicable, and
   the **※ rubric footnote** (always visible) from the template's
   anchors.

The card never presents its own pickers/alerts — exports and
imports go through the host (the app's single-modifier
disciplines), and the failure surface is the phase line.

## 9. Persistence

**Session draft** → the plugin's `.xph` payload, stamped with
`payloadVersion` (2). Restore rules (`EvalFormDraft.restore`):
v2 decodes; v1 (before `reviewedItemIDs`) migrates with an empty
review state; a payload stamped by a NEWER plugin build returns
nil — the draft doesn't render, but the bytes remain in the bundle
untouched for the build that wrote them (the host preserves
unknown/undecoded payloads verbatim through load → save). New
optional fields inside v2 are missing-key tolerant.

**Template pack** → plugin persistent storage (cross-session,
survives restarts and session resets; not part of any `.xph`).

**Review toggles** (`reviewedItemIDs`) mutate the draft and
persist immediately — they are sheet state, so they ride Save →
Open with the session.

## 10. Menu integration

One File-menu item under the app's plugin command group: "Export
Evaluation Form…" → `model.exportMarkdown()`. Its `isEnabled`
closure reads `model.canExport`, so the item enables the moment a
matching draft exists and disables on session clear / pack swap —
without the plugin touching `CommandGroup` or the menu bus.

## 11. Road sections

"Detect road sections" is deterministic: the pack's
`roadCallouts` lexicon (canonical label + alias surfaces; falls
back to labels derived from `referenceRoads`) matched
width/case-folded against each row; callout-to-callout segments
proposed via `SessionAnnotating.proposeSections`, which validates
bounds and skips existing titles (re-running never duplicates;
sections are user-owned once created). Repeat visits get numbered
titles; all surfaces of one road count as one road. It assumes
live-callout protocol speech; courses with different naming
(e.g. 5ヘルツ路面) are a pack edit, not a code change.

## 12. Testing surfaces

- `EvalFormPluginTests` (unit): grammar fixtures, context-window
  attribution, merge policy incl. polarity, parsing/repair,
  coverage phrases, CSV escaping, codable round-trips, migration.
- `PluginKitTests.evalFormPluginActivatesWithContributions`:
  activation against the stub host — page/menu counts, keyword
  seeding, menu gating.
- The headless runner + `EvalFormSynthetic` power the
  model-independent eval harness (offline floor in unit tests;
  live LM Studio suite in EvalTests) — see docs/eval_log.md.

## 13. Adding a new sheet (checklist)

1. Author a pack JSON matching §7 (unique `id`; per-item
   `vocabulary` including kana variants; scale + anchors as
   printed; metadata fields + cues; road names as actually spoken
   on the course).
2. Import it via the Template menu — keywords re-seed under the
   pack's name automatically.
3. Run Fill Form; check the coverage readout first (zero-mention
   items signal vocabulary gaps in the pack, not pipeline
   failures).
4. Nothing else — persistence, exports, coverage, scales, and the
   harness all key off the template.

## 14. Known limitations

- Extraction quality of the inferred channel is unvalidated
  against human judgment (ground-truth eval pending).
- Candidate matching is exact-substring; ASR-garbled onomatopoeia
  reach the pipeline only after hand-correction / homophone
  review.
- The polarity check is a direction-word heuristic (blind to
  negation) and flags rather than fixes — by design.
- Manual score entry is out of scope by design; the 未検出 list
  is the hand-off to the paper sheet.
- Scale endpoint labels (強い/弱い/嫌い/好き) are currently fixed
  in the views, not the pack.
