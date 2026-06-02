---
name: file-picker
description: Use when adding, editing, or reviewing any file-picker code in Xephon (SwiftUI `.fileImporter` / `.fileExporter`, file Open/Save UI, document import/export, exporting transcripts/sessions/audio, importing audio/sessions). The Xephon rule is that every picker routes through `FilePickerCoordinator` — inline modifiers silently break on iPadOS 26. This skill enforces the routing pattern and the UTType whitelist update.
---

# File picker handling in Xephon

Every file-pickup or file-save in this app routes through **one** shared
coordinator and **one** pair of `.fileImporter` / `.fileExporter`
modifiers at the `ContentView` navigation root. Inline pickers silently
no-op on iPadOS 26 — the binding flips, nothing presents.

## Hard rules

1. **Never** attach `.fileImporter` or `.fileExporter` inline on any
   view. There is exactly one of each in `Xephon/App/ContentView.swift`
   bound to `filePicker.isImporterPresented` / `isExporterPresented`.
2. Every caller goes through
   `FilePickerCoordinator.presentImport(allowedTypes:onResult:)` or
   `presentExport(data:contentType:defaultFilename:onResult:)`.
   Same coordinator instance flows through the view tree as
   `FilePickerCoordinator` (already in scope in `ContentView`,
   `ControlPaneView`, etc.).
3. If you need a new export `UTType`, **add it to both**
   `DataFileDocument.readableContentTypes` and `writableContentTypes`
   in `Xephon/FileIO/FilePickerCoordinator.swift`. Missing types log
   *"Attempting to export a document using a content type … not
   included in its writableContentTypes"* and abort the export.

## Pattern — import

```swift
filePicker.presentImport(allowedTypes: [.audio]) { result in
    switch result {
    case .success(let url):
        // Security scope is the CALLER's responsibility:
        guard url.startAccessingSecurityScopedResource() else { return }
        defer { url.stopAccessingSecurityScopedResource() }
        // … read / decode …
    case .failure(let error):
        // user cancel or framework error — usually safe to ignore
        // user cancel; log framework errors
        AppLog.app.error("import failed: \(String(describing: error), privacy: .public)")
    }
}
```

## Pattern — export

```swift
let data: Data = …                 // already-encoded bytes
filePicker.presentExport(
    data: data,
    contentType: .json,            // must be in DataFileDocument's whitelist
    defaultFilename: "session.json"
) { result in
    switch result {
    case .success(let url):
        AppLog.app.info("exported to \(url.path, privacy: .public)")
    case .failure(let error):
        AppLog.app.error("export failed: \(String(describing: error), privacy: .public)")
    }
}
```

## What to do if you see inline `.fileImporter` / `.fileExporter`

Treat it as a bug. Rewrite the call site to use the coordinator. If
the file landed there because a quick-fix needed a content type the
coordinator doesn't yet whitelist, the right fix is to add the type
to `DataFileDocument`, not to attach a sibling modifier.

## Same singular-modifier discipline elsewhere

The same iPadOS-26 trap applies to crowded `.alert` /
`.confirmationDialog` stacks on a single view chain. If you find
yourself adding the second one to a view, drive both from a single
enum-of-cases (see `KeywordsCard.Presentation` for the pattern) or
extract one into its own `ViewModifier` (see
`SpeakerRenameAlertModifier` in `ContentView.swift`).

## Files to know

- `Xephon/FileIO/FilePickerCoordinator.swift` — the coordinator,
  `ImportRequest` / `ExportRequest` types, and `DataFileDocument`
  (the `UTType` whitelist lives here).
- `Xephon/App/ContentView.swift` — the single pair of
  `.fileImporter` / `.fileExporter` modifiers at the nav root.
