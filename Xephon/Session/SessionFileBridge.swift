import SwiftUI

/// Attaches every SwiftUI surface that the `SessionFileCoordinator`
/// drives: the shared `.fileImporter`, the discard-before-import
/// alert, the JSON-export share sheet, the Save Session
/// `.fileExporter` (via `SessionIOModifier`), and the four
/// menu-command observers (`openAudioFileToken`, `exportJSONToken`,
/// `saveSessionToken`, `importSessionToken`).
///
/// Pulling these out of `ContentView.mainBody` keeps the body
/// readable AND keeps the Swift type-checker comfortably under its
/// `.sheet`/`.alert` chain budget — without an explicit bound the
/// chained modifiers tipped the body's inference cost into "unable
/// to type-check this expression in reasonable time" territory.
struct SessionFileBridge: ViewModifier {
    let recorder: RecordingController
    @Bindable var coord: SessionFileCoordinator
    let menuCommands: MenuCommands

    func body(content: Content) -> some View {
        content
            // Save Session export panel + the shared I/O error
            // alert that both save and load funnel into.
            .modifier(SessionIOModifier(
                showingSaveSession: $coord.showingSaveSession,
                pendingSaveDocument: $coord.pendingSaveDocument,
                sessionIOError: $coord.sessionIOError,
                defaultFilename: coord.defaultSessionFilename
            ))
            .fileImporter(
                isPresented: $coord.showingFilePicker,
                allowedContentTypes: coord.filePickerAllowedTypes,
                allowsMultipleSelection: false
            ) { result in
                coord.handleFilePickerResult(result, recorder: recorder)
            }
            .sheet(item: $coord.shareURL) { url in
                ShareSheet(items: [url])
            }
            .alert(
                String(localized: "record.discardConfirm.title"),
                isPresented: $coord.showingFileDiscardConfirm
            ) {
                Button(String(localized: "record.discardConfirm.confirm"), role: .destructive) {
                    // Discard accepted — start analysis directly.
                    coord.startFromPendingFile(recorder: recorder)
                }
                Button(String(localized: "record.discardConfirm.cancel"), role: .cancel) {
                    coord.cancelPendingFile()
                }
            } message: {
                Text(
                    String(
                        format: String(localized: "record.discardConfirm.message"),
                        recorder.utterances.count
                    )
                )
            }
            // File → Open… (⌘O) command pipe. The menu writes a
            // fresh UUID into `menuCommands.openAudioFileToken`; we
            // observe the change and raise the same `.fileImporter`
            // the on-screen button does.
            .onChange(of: menuCommands.openAudioFileToken) { _, _ in
                coord.presentAudioPicker(recorder: recorder)
            }
            // File → Export to JSON (⌘S) command pipe.
            .onChange(of: menuCommands.exportJSONToken) { _, _ in
                Task { await coord.exportJSON(recorder: recorder) }
            }
            // File → Save Session… (⇧⌘S).
            .onChange(of: menuCommands.saveSessionToken) { _, _ in
                Task { await coord.saveSession(recorder: recorder) }
            }
            // File → Import Session… (⇧⌘O).
            .onChange(of: menuCommands.importSessionToken) { _, _ in
                coord.presentSessionPicker(recorder: recorder)
            }
    }
}
