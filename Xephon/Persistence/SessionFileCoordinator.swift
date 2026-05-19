import Foundation
import SwiftUI
import UniformTypeIdentifiers
import Export
import XephonLogging

/// Owns every piece of session-file state that ContentView used to
/// carry inline: the shared `.fileImporter` mode + presentation flag,
/// the Save Session export panel snapshot, the discard-before-import
/// confirmation, the picked-URL security-scope lifecycle, the JSON
/// export share-sheet URL, and a single error surface for both
/// save and load.
///
/// Mutating UI flags (those bound to SwiftUI `.fileImporter` /
/// `.alert` / `.sheet` modifiers) stay default-observed so the view
/// re-evaluates when they flip. The internal bookkeeping
/// (`pendingFileURL`, `pendingFileScopeAcquired`) is
/// `@ObservationIgnored` — only the discard alert's action consumes
/// them, and we don't want a security-scope flip invalidating
/// unrelated views.
@MainActor
@Observable
final class SessionFileCoordinator {
    enum FilePickerMode { case audio, session }

    // MARK: - File-importer (shared between Open Audio / Import Session)

    /// `.fileImporter` presentation flag. Two `.fileImporter`
    /// modifiers stacked on the same view chain silently collide on
    /// iPadOS 26 — one importer switching its content type by mode
    /// is the reliable shape.
    var showingFilePicker = false
    /// What the next picker presentation should accept and what its
    /// result handler should do with the URL.
    var filePickerMode: FilePickerMode = .audio

    // MARK: - Save Session export panel

    /// True while the session-save panel is up.
    var showingSaveSession = false
    /// Snapshot bundled when the user invokes Save Session.
    /// Captured at command time (synchronously) so the file-exporter
    /// sheet writes a stable copy even if the user keeps interacting
    /// with the app while it's open. Nil = no save in progress.
    var pendingSaveDocument: SessionFileDocument?

    // MARK: - Open Audio: pending URL + scope lifecycle + discard alert

    /// True iff a picked audio URL is awaiting the discard-confirm
    /// dialog because the recorder already has utterances.
    var showingFileDiscardConfirm = false
    /// URL the picker handed us — held across the discard dialog
    /// hop. `@ObservationIgnored` because nothing in the view tree
    /// observes the URL itself; it's consumed by the alert's
    /// Confirm action.
    @ObservationIgnored
    private var pendingFileURL: URL?
    /// True when we successfully called
    /// `startAccessingSecurityScopedResource()` on `pendingFileURL`.
    /// The picker's implicit grant can expire over the multi-dialog
    /// hop to `startFromFile`, so we pin scope as soon as the URL
    /// arrives and release it once the recorder has taken its own
    /// ref (or the user has cancelled).
    @ObservationIgnored
    private var pendingFileScopeAcquired = false

    // MARK: - Shared error surface + JSON export share sheet

    /// Last error from a save/load attempt; surfaces as an alert
    /// inside `SessionIOModifier`.
    var sessionIOError: String?
    /// Drives the JSON-export ShareSheet presentation. Identifiable
    /// via the `URL: @retroactive Identifiable` extension so it
    /// plugs into `.sheet(item:)`.
    var shareURL: URL?

    // MARK: - Read-only derived

    /// Content types the single shared fileImporter advertises,
    /// based on which menu command opened it. `xephonSession` is
    /// registered via `project.yml`'s `UTExportedTypeDeclarations`,
    /// so the picker greys out non-`.xph` files when in session
    /// mode.
    var filePickerAllowedTypes: [UTType] {
        switch filePickerMode {
        case .audio:
            return [.audio, .mp3, .wav, .mpeg4Audio, .aiff]
        case .session:
            return [.xephonSession]
        }
    }

    /// Filename suggestion for the Save Session… panel. ISO-8601-ish
    /// stamp so successive saves don't collide and the user can scan
    /// the file list chronologically. Locale-pinned to `en_US_POSIX`
    /// per Apple's guidance for fixed-format strings — without it
    /// `DateFormatter` would localize digits / separators (e.g.
    /// Arabic-Indic numerals under ar locale) and break the
    /// filename scheme.
    var defaultSessionFilename: String {
        let fmt = DateFormatter()
        fmt.locale = Locale(identifier: "en_US_POSIX")
        fmt.dateFormat = "yyyy-MM-dd_HHmm"
        return "xephon-\(fmt.string(from: Date())).xph"
    }

    // MARK: - Menu-command entry points

    /// File → Open… / on-screen Open button. Pins the picker mode
    /// to `.audio` so a stale mode left over from an earlier Import
    /// Session… invocation can't leak through and make this entry
    /// accept `.xph` files. Gated on busy state.
    func presentAudioPicker(recorder: RecordingController) {
        guard !recorder.isRecording, !recorder.isAnalyzing else { return }
        filePickerMode = .audio
        showingFilePicker = true
    }

    /// File → Import Session… (⇧⌘O). Reuses the single fileImporter
    /// by switching its mode to `.session` before raising it.
    func presentSessionPicker(recorder: RecordingController) {
        guard !recorder.isRecording, !recorder.isAnalyzing else { return }
        filePickerMode = .session
        showingFilePicker = true
    }

    /// File → Save Session… Snapshot the recorder's state into a
    /// `SessionDocument` synchronously, stash it in
    /// `pendingSaveDocument`, and raise the `.fileExporter`. The
    /// exporter dismisses by clearing the pending doc so
    /// re-triggering works.
    func saveSession(recorder: RecordingController) async {
        guard !recorder.utterances.isEmpty,
              !recorder.isRecording,
              !recorder.isAnalyzing else { return }
        do {
            let doc = try await recorder.makeSessionDocument()
            pendingSaveDocument = SessionFileDocument(session: doc)
            showingSaveSession = true
        } catch {
            sessionIOError = String(describing: error)
        }
    }

    /// File → Export to JSON / toolbar export button. Same gating
    /// as the toolbar button so cmd-S during recording / analyzing /
    /// empty-utterances no-ops cleanly. Additionally guards
    /// `shareURL == nil` so repeated invocations while the share
    /// sheet is already up don't write a fresh file + reassign
    /// shareURL — `.sheet(item:)` interprets a new URL as
    /// "dismiss + represent", which under rapid presses appears as
    /// a stacking sheet.
    func exportJSON(recorder: RecordingController) async {
        guard !recorder.utterances.isEmpty,
              !recorder.isRecording,
              !recorder.isAnalyzing,
              shareURL == nil else { return }
        if let url = await recorder.exportJSON() {
            shareURL = url
        }
    }

    /// Toolbar export entry — bypasses the empty-utterances /
    /// busy-state guards `exportJSON` enforces because the toolbar
    /// button itself is already `.disabled(!isIdleWithTranscript)`.
    /// Keeps the share-sheet stack guard so a double-tap doesn't
    /// restack the sheet.
    func exportJSONFromToolbar(recorder: RecordingController) async {
        guard shareURL == nil else { return }
        if let url = await recorder.exportJSON() {
            shareURL = url
        }
    }

    // MARK: - File-picker result + audio scope lifecycle

    /// Single dispatcher for the shared fileImporter. The audio
    /// path pins security scope and hands off to the pacing dialog
    /// (or starts analysis immediately if the transcript is empty);
    /// the session path reads + decodes off MainActor.
    func handleFilePickerResult(
        _ result: Result<[URL], any Error>,
        recorder: RecordingController
    ) {
        switch result {
        case .success(let urls):
            guard let url = urls.first else { return }
            switch filePickerMode {
            case .audio:
                pendingFileURL = url
                pendingFileScopeAcquired = url.startAccessingSecurityScopedResource()
                if !recorder.utterances.isEmpty {
                    showingFileDiscardConfirm = true
                } else {
                    startFromPendingFile(recorder: recorder)
                }
            case .session:
                Task { await loadSessionFromPickedFile(url, recorder: recorder) }
            }
        case .failure(let error):
            AppLog.app.error("file picker: \(String(describing: error), privacy: .public)")
            if filePickerMode == .session {
                sessionIOError = String(describing: error)
            }
        }
    }

    /// Discard-confirmed: hand the URL to the recorder (which
    /// acquires its own scope ref synchronously in `startFromFile`),
    /// then release the picker's ref we've been holding through the
    /// dialog hop.
    func startFromPendingFile(recorder: RecordingController) {
        guard let url = pendingFileURL else { return }
        let acquired = pendingFileScopeAcquired
        pendingFileScopeAcquired = false
        pendingFileURL = nil
        Task {
            await recorder.startFromFile(url)
            if acquired {
                url.stopAccessingSecurityScopedResource()
            }
        }
    }

    /// Cancel path for the discard alert: release the picker's
    /// scope ref we grabbed in the importer callback and clear the
    /// pending URL.
    func cancelPendingFile() {
        if pendingFileScopeAcquired, let url = pendingFileURL {
            url.stopAccessingSecurityScopedResource()
        }
        pendingFileScopeAcquired = false
        pendingFileURL = nil
    }

    /// Read a picked `.xph` URL into the recorder. Security-scoped:
    /// the picker hands us a scoped URL; we hold it just long
    /// enough to read the bytes and let `loadSession` extract any
    /// audio into the app's sandbox.
    private func loadSessionFromPickedFile(
        _ url: URL,
        recorder: RecordingController
    ) async {
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }
        do {
            let data = try Data(contentsOf: url)
            let document = try SessionBundle.decode(data)
            try await recorder.loadSession(document)
        } catch {
            sessionIOError = String(describing: error)
        }
    }
}
