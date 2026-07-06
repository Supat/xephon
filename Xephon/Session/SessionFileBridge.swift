import SwiftUI

/// Attaches every SwiftUI surface that the `SessionFileCoordinator`
/// drives: the discard-before-import alert, the JSON-export share
/// sheet, the session I/O error alert, and the four menu-command
/// observers (`openAudioFileToken`, `exportJSONToken`,
/// `saveSessionToken`, `importSessionToken`).
///
/// All file picker surfaces (audio open, session import, session
/// save) route through `FilePickerCoordinator` instead of attaching
/// their own `.fileImporter` / `.fileExporter` modifiers — that
/// keeps exactly one importer + one exporter at the navigation
/// root, sidestepping the iPadOS 26 multi-modifier-on-the-same-
/// view-chain collision that silently swallows presentations.
///
/// Pulling these out of `ContentView.mainBody` keeps the body
/// readable AND keeps the Swift type-checker comfortably under its
/// `.sheet`/`.alert` chain budget — without an explicit bound the
/// chained modifiers tipped the body's inference cost into "unable
/// to type-check this expression in reasonable time" territory.
struct SessionFileBridge: ViewModifier {
    let recorder: RecordingController
    @Bindable var coord: SessionFileCoordinator
    let filePicker: FilePickerCoordinator
    let menuCommands: MenuCommands

    func body(content: Content) -> some View {
        content
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
            // Shared error alert for both save and load failures —
            // moved here from the old `SessionIOModifier` when its
            // `.fileExporter` migrated into `FilePickerCoordinator`.
            .alert(
                "Session I/O Error",
                isPresented: Binding(
                    get: { coord.sessionIOError != nil },
                    set: { if !$0 { coord.sessionIOError = nil } }
                ),
                presenting: coord.sessionIOError
            ) { _ in
                Button("OK", role: .cancel) { coord.sessionIOError = nil }
            } message: { msg in
                Text(msg)
            }
            // File → Open… (⌘O) command pipe. The menu writes a
            // fresh UUID into `menuCommands.openAudioFileToken`; we
            // observe the change and ask the centralized file
            // picker to raise the importer.
            .onChange(of: menuCommands.openAudioFileToken) { _, _ in
                coord.presentAudioPicker(recorder: recorder, filePicker: filePicker)
            }
            // File → Export to JSON (⌘S) command pipe.
            .onChange(of: menuCommands.exportJSONToken) { _, _ in
                Task { await coord.exportJSON(recorder: recorder) }
            }
            // File → Save Session… (⇧⌘S).
            .onChange(of: menuCommands.saveSessionToken) { _, _ in
                Task { await coord.saveSession(recorder: recorder, filePicker: filePicker) }
            }
            // File → Import Session… (⇧⌘O).
            .onChange(of: menuCommands.importSessionToken) { _, _ in
                coord.presentSessionPicker(recorder: recorder, filePicker: filePicker)
            }
            // File → Export Session Audio….
            .onChange(of: menuCommands.exportRecordedAudioToken) { _, _ in
                coord.exportSessionAudio(recorder: recorder, filePicker: filePicker)
            }
    }
}
