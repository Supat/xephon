# SwiftUI `.fileImporter` collision post-mortem

## TL;DR

**Don't attach two `.fileImporter` modifiers to the same view chain.
Use one `.fileImporter` and switch its `allowedContentTypes` +
result-handler by a mode flag.** Multiple importers stacked on the
same view silently collide: the menu fires, the binding flips, the
document picker plumbing initializes, and nothing presents.

## Symptom

Adding a new "Import Session…" file picker alongside the existing
"Open Audio File…" picker, both as separate `.fileImporter` modifiers
on `ContentView`, produced this exact failure mode:

- Menu item action fires (`AppLog.app.info("Import Session menu item
  pressed")` ✓).
- The token-bridge `onChange` handler runs (`AppLog.app.info("Import
  Session triggered: …")` ✓).
- The presentation binding flips to `true` (`AppLog.app.info(
  "showingImportSession set to true")` ✓).
- LaunchServices logs appear (`Plugin query method called`, the
  document-picker XPC service initializing).
- **No picker UI ever appears.**

The audio picker — attached to the same view chain via a different
binding — kept working. The two importers had distinct
`@State`-backed bindings, distinct `allowedContentTypes`, and distinct
result handlers, but the second one silently lost.

## What didn't work

### 1. Detaching the second importer to a `.background { Color.clear }`

The idea: put the second `.fileImporter` on a sibling view in the
hierarchy rather than the same modifier chain. This worked at the
state-binding level — the binding flipped, the result handler was
wired — but the `UIDocumentPickerViewController` never presented,
presumably because SwiftUI couldn't find a presenting view controller
through the `Color.clear` background.

The LaunchServices log noise (`Plugin query method called`, the
`-54` LSDReadService chatter) is the document picker XPC service
trying to spin up. The picker controller is *requested* but its
presentation doesn't reach a real responder chain.

### 2. Reordering the two `.fileImporter` modifiers

Putting the session importer before the audio one — or vice versa —
made no observable difference. The conflict isn't about z-order; it's
about SwiftUI's internal bookkeeping of presentation-style modifiers
on the same view.

### 3. Two `.fileImporter` modifiers in the same `ViewModifier`

A `ViewModifier` that chains its own modifiers (`.fileExporter` plus
`.fileImporter`) onto `content` is no different from chaining them
directly in the body. The composition site doesn't matter; the
collision is at the modified-view level.

## What worked

**One `.fileImporter`, parameterized by a mode flag set immediately
before presentation:**

```swift
@State private var showingFilePicker: Bool = false
@State private var filePickerMode: FilePickerMode = .audio
enum FilePickerMode { case audio, session }

// In menu observers:
.onChange(of: menuCommands.openAudioFileToken) { _, _ in
    filePickerMode = .audio
    showingFilePicker = true
}
.onChange(of: menuCommands.importSessionToken) { _, _ in
    filePickerMode = .session
    showingFilePicker = true
}

// One modifier in the view chain:
.fileImporter(
    isPresented: $showingFilePicker,
    allowedContentTypes: filePickerAllowedTypes,
    allowsMultipleSelection: false,
    onCompletion: handleFilePickerResult
)
```

`filePickerAllowedTypes` and `handleFilePickerResult` both switch on
`filePickerMode`. SwiftUI captures whatever `allowedContentTypes`
returns at the moment of presentation, so as long as `filePickerMode`
is set *before* `showingFilePicker = true`, the picker presents the
right list and the handler routes the result to the right downstream
flow.

`.fileExporter` doesn't have this problem and can stay where it is.
The collision is specific to `.fileImporter`.

## Why the "two importers" approach feels right but isn't

Each `.fileImporter` looks declarative and independent: separate
binding, separate content types, separate handler. It reads like the
SwiftUI way of doing things. But SwiftUI's presentation system
internally treats `.fileImporter` similarly to `.sheet` — there's a
single "presented modal" slot per view, and the framework picks one
winner when multiple modifiers race for it.

This isn't documented anywhere obvious; it surfaces only when you try
it and find one importer silently inert. The state-binding logs show
everything succeeds; the presentation never happens. That's the
fingerprint of this bug.

## Generalization

The same pattern probably applies to any pair of presentation-style
modifiers attached to the same view:

- Two `.sheet(isPresented:)` modifiers — known to collide; use
  `.sheet(item:)` with an optional enum for multiplexing.
- Two `.fileImporter` — collides as documented here.
- Two `.fileExporter` — likely the same shape, though we haven't hit
  it in practice yet.

The rule: **one presentation-style modifier per view, multiplexed via
state when multiple flows need it.**

## Related code

- `Xephon/ContentView.swift` — single `.fileImporter` plus
  `filePickerMode`, `filePickerAllowedTypes`, `handleFilePickerResult`.
- `Xephon/SessionFileDocument.swift` — `SessionIOModifier` still
  bundles the `.fileExporter` and the error alert; the importer was
  removed and routed through ContentView's shared picker instead.
- `Xephon/XephonApp.swift` — menu commands set their token UUIDs;
  ContentView's `onChange` observers translate token bumps into
  `filePickerMode` + `showingFilePicker` updates.
