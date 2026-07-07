import Foundation
@preconcurrency import AVFoundation
import Audio
import Fusion
import XephonLogging

/// Owns per-utterance / arbitrary-range playback: the AVAudioPlayer
/// lifecycle, the playback-source URL (+ security-scope balancing),
/// the cross-instance session latch, and the play/stop entry points.
/// Extracted from RecordingController along the SummarizerCoordinator
/// seam (unowned parent + thin forwarders on the controller keep
/// every existing call site source-compatible) — see
/// docs/mvvm_audit.md F1/rec 4.
///
/// Cross-cutting reads (phase, errorMessage) go through the unowned
/// parent; the coordinator never outlives the controller and both
/// share MainActor isolation.
@MainActor
@Observable
final class PlaybackCoordinator {
    private unowned let parent: RecordingController

    /// File URL whose utterances are currently loaded for playback.
    /// Non-nil iff the most recent (or in-progress) session has
    /// playable audio. Mutated through `setPlaybackSourceURL` so the
    /// matching security-scoped access ref is balanced.
    private(set) var playbackSourceURL: URL?
    /// URL we currently hold a `startAccessingSecurityScopedResource`
    /// ref on. Distinct from `playbackSourceURL` because the start
    /// call can fail (e.g. for a non-scoped URL); we only stash here
    /// after a successful start so the matching stop is balanced.
    private var scopedPlaybackURL: URL?
    /// ID of the utterance currently playing back, nil when nothing
    /// is playing. Rows flip their play icon / disable off this.
    private(set) var playingUtteranceID: UUID?
    var playbackPlayer: AVAudioPlayer?
    var playbackStopTask: Task<Void, Never>?
    /// True while a `playRange(start:end:)` preview is in flight —
    /// distinct from row-level playback so the Edit Utterance
    /// sheet's play button toggles without lighting up row controls.
    private(set) var isPreviewPlaying: Bool = false

    init(parent: RecordingController) {
        self.parent = parent
    }

    /// PROCESS-WIDE "some controller is playing" latch. The
    /// per-instance guards (`phase == .idle`, `playbackPlayer == nil`)
    /// that gate `refreshInputs`' category swap and the idle input
    /// poll are invisible across instances: when a second live
    /// RecordingController exists (see 1eefe4f — observed in field
    /// logs, not yet root-caused at the SwiftUI level), the idle
    /// instance's poll saw itself as idle while the foreground
    /// instance was mid-playback, and thrashed the shared session
    /// under the active player — category swaps produced the
    /// periodic ~3-5 s dropouts; after the deactivate-before-swap
    /// hardening it became a hard kill (~2 s in, player silent while
    /// the stop task keeps running). Claimed on every playback start,
    /// released in `stopPlayback` ONLY by the claiming instance — so
    /// a defensive stopPlayback() on some other instance can't
    /// unlatch an active playback. Checked (non-nil) by every
    /// instance before touching the shared session while idle.
    @MainActor static var playbackSessionOwner: ObjectIdentifier?

    /// True while ANY controller instance has playback running.
    @MainActor static var playbackSessionActive: Bool {
        playbackSessionOwner != nil
    }

    /// Assign `playbackSourceURL` while keeping the security-scoped
    /// access ref balanced. File-picker URLs only stay readable while
    /// some part of the app holds a ref via
    /// `startAccessingSecurityScopedResource()`; AudioFileCapture's
    /// ref drops at the end of analysis, so we hold our own ref here
    /// for the duration that the URL is exposed for playback. The
    /// `start` can fail for already-accessible URLs (e.g. in tests),
    /// which is fine — we just don't stash a stop counterpart.
    func setPlaybackSourceURL(_ newURL: URL?) {
        if let scoped = scopedPlaybackURL {
            scoped.stopAccessingSecurityScopedResource()
            scopedPlaybackURL = nil
        }
        playbackSourceURL = newURL
        if let url = newURL {
            let ok = url.startAccessingSecurityScopedResource()
            AppLog.app.info(
                "playback scope start: \(ok ? "ok" : "skipped", privacy: .public) for \(url.lastPathComponent, privacy: .public)"
            )
            if ok { scopedPlaybackURL = url }
        }
    }

    /// Toggle playback of the audio range `[utterance.start, utterance.end]`
    /// from `playbackSourceURL`. No-op when there's no source URL (mic
    /// session) or when analysis is still running — the row gates the
    /// button so this is defense-in-depth. Tapping the row that's
    /// currently playing stops it; tapping a different row stops the
    /// previous playback and starts the new one.
    func togglePlayback(for utterance: UtteranceEstimate) {
        AppLog.app.info(
            "togglePlayback called: utt=\(utterance.id, privacy: .public) src=\(self.playbackSourceURL?.lastPathComponent ?? "nil", privacy: .public) phase=\(String(describing: self.parent.phase), privacy: .public) scoped=\(self.scopedPlaybackURL?.lastPathComponent ?? "nil", privacy: .public)"
        )
        guard let url = playbackSourceURL else {
            AppLog.app.warning("togglePlayback: no playbackSourceURL")
            return
        }
        guard parent.phase == .idle else {
            AppLog.app.warning("togglePlayback: phase not idle: \(String(describing: self.parent.phase), privacy: .public)")
            return
        }
        if playingUtteranceID == utterance.id {
            stopPlayback()
            return
        }
        stopPlayback()
        // Claim the shared session BEFORE activating it so no other
        // controller instance's idle poll deactivates/swaps it out
        // from under the player. Released by `stopPlayback` on every
        // exit path.
        Self.playbackSessionOwner = ObjectIdentifier(self)
        #if os(iOS) || targetEnvironment(macCatalyst)
        do {
            let session = AVAudioSession.sharedInstance()
            AppLog.app.info(
                "playback session BEFORE: category=\(session.category.rawValue, privacy: .public) mode=\(session.mode.rawValue, privacy: .public) outputs=\(session.currentRoute.outputs.map(\.portType.rawValue).joined(separator: ","), privacy: .public)"
            )
            try session.setCategory(.playback, mode: .default, options: [])
            try session.setActive(true)
            AppLog.app.info(
                "playback session AFTER:  category=\(session.category.rawValue, privacy: .public) mode=\(session.mode.rawValue, privacy: .public) outputs=\(session.currentRoute.outputs.map(\.portType.rawValue).joined(separator: ","), privacy: .public)"
            )
        } catch {
            AppLog.app.warning("playback session setup failed: \(String(describing: error), privacy: .public)")
        }
        #endif
        do {
            let player = try AVAudioPlayer(contentsOf: url)
            // Retain the player BEFORE calling play(). If the local
            // reference is the only retain at the moment of play(),
            // the optimizer is free to release it on the next line —
            // not normally an issue, but we've seen the case where
            // state doesn't progress past idle on the second session.
            playbackPlayer = player
            player.prepareToPlay()
            player.currentTime = max(0, utterance.start)
            let didStart = player.play()
            AppLog.app.info(
                "togglePlayback: play() returned \(didStart ? "true" : "false", privacy: .public), duration=\(player.duration, privacy: .public)s, seek=\(player.currentTime, privacy: .public)s"
            )
            guard didStart else {
                AppLog.app.warning("playback failed to start for \(utterance.id, privacy: .public)")
                // Through stopPlayback so the session latch clears
                // and the just-activated session deactivates.
                stopPlayback()
                return
            }
            playingUtteranceID = utterance.id
            AppLog.app.info("togglePlayback: playingUtteranceID set to \(utterance.id, privacy: .public)")
            let duration = max(0, utterance.end - utterance.start)
            playbackStopTask = Task { @MainActor [weak self] in
                try? await Task.sleep(for: .seconds(duration))
                guard !Task.isCancelled else { return }
                self?.stopPlayback()
            }
        } catch {
            AppLog.app.error("playback open failed: \(String(describing: error), privacy: .public)")
            stopPlayback()
        }
    }


    /// Play `[start, end]` from the source file. Used by the Edit
    /// Utterance dialog's preview button — the dialog's spinners
    /// hold arbitrary times that don't correspond to an existing
    /// utterance row, so `togglePlayback(for:)` doesn't apply.
    ///
    /// `owner` ties the playback session to a specific utterance id
    /// when one applies — the review sheet's inline play buttons use
    /// this so each card knows whether it's the active one. Pass
    /// nil for arbitrary-range previews (the edit dialog's preview
    /// button, where the spinners may not correspond to any row).
    func playRange(
        start: TimeInterval,
        end: TimeInterval,
        owner: UUID? = nil
    ) {
        guard let url = playbackSourceURL else { return }
        guard parent.phase == .idle else { return }
        guard end > start else { return }
        stopPlayback()
        // Same cross-instance session claim as togglePlayback.
        Self.playbackSessionOwner = ObjectIdentifier(self)
        #if os(iOS) || targetEnvironment(macCatalyst)
        do {
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.playback, mode: .default, options: [])
            try session.setActive(true)
        } catch {
            AppLog.app.warning(
                "playRange: session setup failed: \(String(describing: error), privacy: .public)"
            )
        }
        #endif
        do {
            let player = try AVAudioPlayer(contentsOf: url)
            playbackPlayer = player
            player.prepareToPlay()
            player.currentTime = max(0, start)
            guard player.play() else {
                stopPlayback()
                return
            }
            isPreviewPlaying = true
            playingUtteranceID = owner
            let duration = max(0, end - start)
            playbackStopTask = Task { @MainActor [weak self] in
                try? await Task.sleep(for: .seconds(duration))
                guard !Task.isCancelled else { return }
                self?.stopPlayback()
            }
        } catch {
            AppLog.app.error(
                "playRange failed: \(String(describing: error), privacy: .public)"
            )
            stopPlayback()
        }
    }


    /// Stop any in-flight playback. Safe to call when nothing is
    /// playing — it just clears the latch.
    func stopPlayback(caller: String = #function) {
        let hadPlayer = playbackPlayer != nil
        if playingUtteranceID != nil || hadPlayer {
            AppLog.app.info("stopPlayback called by \(caller, privacy: .public), playingID=\(self.playingUtteranceID?.uuidString ?? "nil", privacy: .public)")
        }
        playbackStopTask?.cancel()
        playbackStopTask = nil
        playbackPlayer?.stop()
        playbackPlayer = nil
        playingUtteranceID = nil
        isPreviewPlaying = false
        // Release only our own claim: stopPlayback is the universal
        // terminus for every playback path on THIS coordinator
        // (including failed starts), but defensive calls on another
        // instance must not unlatch an active playback elsewhere.
        if Self.playbackSessionOwner == ObjectIdentifier(self) {
            Self.playbackSessionOwner = nil
        }
        #if os(iOS) || targetEnvironment(macCatalyst)
        // Deactivate the session a playback activated. Leaving it
        // active in `.playback` broke the idle input poll's safety
        // premise: `refreshInputs`' category swap is metadata-only
        // ONLY on an inactive session — on the active one it took
        // effect immediately, cycling the USB input up and down
        // (clock renegotiation) every ~2 s between utterances, which
        // bled audible dropouts into the next playback. Gated on
        // `hadPlayer` so the many defensive stopPlayback() calls
        // (sheet dismissals, row taps) don't churn the session when
        // nothing was playing — AND on `parent.phase == .idle`:
        // `prepareSessionForRecording` calls stopPlayback AFTER
        // `capture.start()` has activated the `.record` session
        // (phase is already .recording), and deactivating there
        // would stall the freshly-started capture engine.
        if hadPlayer, parent.phase == .idle {
            try? AVAudioSession.sharedInstance().setActive(
                false,
                options: .notifyOthersOnDeactivation
            )
        }
        #endif
    }
}
