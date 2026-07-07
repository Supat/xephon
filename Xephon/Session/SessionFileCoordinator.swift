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
    /// in `SessionFileBridge`.
    var sessionIOError: String?
    /// Drives the JSON-export ShareSheet presentation. Identifiable
    /// via the `URL: @retroactive Identifiable` extension so it
    /// plugs into `.sheet(item:)`.
    var shareURL: URL?

    // MARK: - Read-only derived

    /// Content types accepted by the audio Open path. Hard-coded
    /// here so the menu / button callers don't have to know what
    /// the picker should accept.
    private static let audioContentTypes: [UTType] = [
        .audio, .mp3, .wav, .mpeg4Audio, .aiff,
    ]

    /// Timestamped fallback filename for the Save Session…
    /// panel — used when the user hasn't set a session title.
    /// ISO-8601-ish stamp so successive saves don't collide and
    /// the user can scan the file list chronologically. Locale-
    /// pinned to `en_US_POSIX` per Apple's guidance for fixed-
    /// format strings — without it `DateFormatter` would
    /// localize digits / separators (e.g. Arabic-Indic numerals
    /// under ar locale) and break the filename scheme.
    var defaultSessionFilename: String {
        "\(defaultSessionFilenameBase).xph"
    }

    /// Extension-less variant of `defaultSessionFilename`, shared
    /// with the recorded-audio export (which appends the recording's
    /// own container extension instead of `.xph`).
    var defaultSessionFilenameBase: String {
        let fmt = DateFormatter()
        fmt.locale = Locale(identifier: "en_US_POSIX")
        fmt.dateFormat = "yyyy-MM-dd_HHmm"
        return "xephon-\(fmt.string(from: Date()))"
    }

    /// Suggested Save Session… filename for the given title.
    /// Sanitized so the file picker accepts it as-is (the user
    /// can still edit it in the dialog before confirming);
    /// otherwise falls back to `defaultSessionFilename` when the
    /// title is empty or sanitizes to empty (whitespace-only).
    ///
    /// Sanitization replaces the two characters macOS / iOS
    /// reject in filenames — `/` and `:` — with `-`. Everything
    /// else (Unicode, spaces, emoji, punctuation) the system
    /// allows in a filename, so we leave it alone.
    func sessionFilename(forTitle title: String) -> String {
        "\(sessionFilenameBase(forTitle: title)).xph"
    }

    /// Extension-less variant of `sessionFilename(forTitle:)` — the
    /// sanitized title, or the timestamped default when the title is
    /// empty / sanitizes to empty.
    func sessionFilenameBase(forTitle title: String) -> String {
        let trimmed = title.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return defaultSessionFilenameBase }
        let sanitized = trimmed
            .replacingOccurrences(of: "/", with: "-")
            .replacingOccurrences(of: ":", with: "-")
        guard !sanitized.isEmpty else { return defaultSessionFilenameBase }
        return sanitized
    }

    /// File → Export Session Audio… / leading toolbar button.
    /// Hands the session audio — a fresh mic recording, a loaded
    /// `.xph` bundle extraction, or a file-opened session source —
    /// to the centralized exporter. `.mappedIfSafe` keeps a long
    /// WAV from being copied wholesale into RAM — the
    /// DataFileDocument write streams it from disk. The scope dance
    /// covers the file-opened case (picker URLs need an active
    /// security scope to read); it no-ops for app-owned
    /// temp/recording files. Gated on `canExportSessionAudio`
    /// (audio exists + idle); the menu item and toolbar button
    /// mirror the same gate, so this guard is defense-in-depth.
    func exportSessionAudio(
        recorder: RecordingController,
        filePicker: FilePickerCoordinator
    ) {
        guard recorder.canExportSessionAudio,
              let url = recorder.sessionAudioFileURL else { return }
        let stillScoped = url.startAccessingSecurityScopedResource()
        defer {
            if stillScoped { url.stopAccessingSecurityScopedResource() }
        }
        do {
            let data = try Data(contentsOf: url, options: .mappedIfSafe)
            let ext = url.pathExtension.lowercased()
            // Derive the UTType from the actual container —
            // imported sources can be mp3/aiff, not just the
            // recorder m4a/wav pair. Falls back to .data
            // (whitelisted) for exotic extensions rather than
            // lying about the type.
            let contentType: UTType = UTType(
                filenameExtension: ext,
                conformingTo: .audio
            ) ?? .data
            let filename = "\(sessionFilenameBase(forTitle: recorder.sessionTitle)).\(ext)"
            filePicker.presentExport(
                data: data,
                contentType: contentType,
                defaultFilename: filename
            ) { [weak self] result in
                switch result {
                case .success(let savedURL):
                    AppLog.app.info("recorded audio exported to \(savedURL.lastPathComponent, privacy: .public)")
                case .failure(let error):
                    self?.sessionIOError = String(describing: error)
                }
            }
        } catch {
            sessionIOError = String(describing: error)
        }
    }

    // MARK: - Menu-command entry points

    /// File → Open… / on-screen Open button. Routes through the
    /// app-level `FilePickerCoordinator` so we don't stack a
    /// second `.fileImporter` modifier on the view chain (which
    /// collides on iPadOS 26 — see FilePickerCoordinator's doc).
    /// Gated on busy state.
    func presentAudioPicker(
        recorder: RecordingController,
        filePicker: FilePickerCoordinator
    ) {
        guard !recorder.isRecording, !recorder.isAnalyzing else { return }
        filePicker.presentImport(allowedTypes: Self.audioContentTypes) { [weak self, weak recorder] result in
            guard let self, let recorder else { return }
            self.handleAudioPickerResult(result, recorder: recorder)
        }
    }

    /// File → Import Session… (⇧⌘O). Same centralized importer
    /// as the audio path, with `.xephonSession` as the allowed
    /// type so non-`.xph` files grey out.
    func presentSessionPicker(
        recorder: RecordingController,
        filePicker: FilePickerCoordinator
    ) {
        guard !recorder.isRecording, !recorder.isAnalyzing else { return }
        filePicker.presentImport(allowedTypes: [.xephonSession]) { [weak self, weak recorder] result in
            guard let self, let recorder else { return }
            self.handleSessionPickerResult(result, recorder: recorder)
        }
    }

    /// File → Save Session… Snapshot the recorder's state into a
    /// `SessionDocument`, encode synchronously, and hand the bytes
    /// to the app-level exporter. The export callback flips the
    /// error alert on failure; success dismisses the picker
    /// without further work.
    func saveSession(
        recorder: RecordingController,
        filePicker: FilePickerCoordinator
    ) async {
        guard !recorder.utterances.isEmpty,
              !recorder.isRecording,
              !recorder.isAnalyzing else { return }
        do {
            let doc = try await recorder.makeSessionDocument()
            let data = try SessionBundle.encode(doc)
            let filename = sessionFilename(forTitle: recorder.sessionTitle)
            filePicker.presentExport(
                data: data,
                contentType: .xephonSession,
                defaultFilename: filename
            ) { [weak self, weak recorder] result in
                switch result {
                case .success(let savedURL):
                    // Track the user's rename in the save dialog.
                    // If they edited the suggested filename before
                    // confirming, the chrome's title field should
                    // follow — otherwise the visible title would
                    // diverge from what's on disk for the rest of
                    // the session and confuse the next save.
                    guard let recorder else { return }
                    let savedBase = savedURL
                        .deletingPathExtension()
                        .lastPathComponent
                    if !savedBase.isEmpty {
                        recorder.sessionTitle = savedBase
                    }
                case .failure(let error):
                    self?.sessionIOError = String(describing: error)
                }
            }
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

    /// Audio-import callback. Pins security scope on the picked
    /// URL (the picker's implicit grant can expire across the
    /// multi-dialog hop to `startFromFile`), then either gates on
    /// the discard-confirm dialog (utterances present) or starts
    /// analysis directly (empty list).
    private func handleAudioPickerResult(
        _ result: Result<URL, any Error>,
        recorder: RecordingController
    ) {
        switch result {
        case .success(let url):
            pendingFileURL = url
            pendingFileScopeAcquired = url.startAccessingSecurityScopedResource()
            if !recorder.utterances.isEmpty {
                showingFileDiscardConfirm = true
            } else {
                startFromPendingFile(recorder: recorder)
            }
        case .failure(let error):
            AppLog.app.error("audio file picker: \(String(describing: error), privacy: .public)")
        }
    }

    /// Session-import callback. Off-MainActor read + decode.
    /// Errors surface through the shared `sessionIOError` alert
    /// (the audio path doesn't surface picker failures the same
    /// way because the user can also cancel that flow benignly).
    private func handleSessionPickerResult(
        _ result: Result<URL, any Error>,
        recorder: RecordingController
    ) {
        switch result {
        case .success(let url):
            Task { await loadSessionFromPickedFile(url, recorder: recorder) }
        case .failure(let error):
            AppLog.app.error("session file picker: \(String(describing: error), privacy: .public)")
            sessionIOError = String(describing: error)
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
            // Detached, not `Task {}` — this method is MainActor-
            // isolated, so an inheriting Task would run the read +
            // decode of the whole bundle (embedded audio included,
            // easily tens of MB) on the main thread and beachball
            // the UI for the duration. The security scope acquired
            // above is process-wide, so the detached read is
            // covered; `SessionDocument` is Sendable.
            let document = try await Task.detached(priority: .userInitiated) {
                try SessionBundle.decode(try Data(contentsOf: url))
            }.value
            try await recorder.loadSession(document)
            // Always override the loaded `sessionTitle` with the
            // .xph file's base name. The user has just picked
            // this specific file out of Files / iCloud / wherever
            // — its on-disk name is the most current intent
            // (renames in Files happen after Save Session,
            // so the bundle's persisted title can lag). The
            // bundle's `sessionTitle` field is still preserved
            // inside `SessionDocument` for downstream consumers
            // that care; only the live chrome TextField reads
            // through `recorder.sessionTitle`.
            recorder.sessionTitle = url.deletingPathExtension().lastPathComponent
        } catch {
            sessionIOError = String(describing: error)
        }
    }
}
