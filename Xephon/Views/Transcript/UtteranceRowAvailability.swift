import Foundation
import Fusion

/// Per-row availability resolvers for `UtteranceRow`'s playback +
/// re-evaluate buttons. Live as static factories on the enums so
/// the transcript list can derive them inline from `recorder` and
/// the row's `UtteranceEstimate` instead of taking parent-supplied
/// closures.
extension UtteranceRow.PlaybackAvailability {
    /// Map the recorder's source/phase state onto the per-row
    /// playback availability. Hidden for mic sessions, disabled
    /// while a file analysis is still running (so the user knows
    /// playback is on the way), idle when ready, and playing for
    /// the one row that's currently mid-playback.
    @MainActor
    static func resolve(
        for u: UtteranceEstimate,
        recorder: RecordingController
    ) -> Self {
        guard recorder.playbackSourceURL != nil else { return .unavailable }
        if recorder.isRecording || recorder.isAnalyzing { return .disabled }
        // Disable playback across the list while a re-evaluation
        // is in flight — offline ASR + SER serialize naturally
        // and the user shouldn't be racing audio reads against
        // the re-analysis pass.
        if recorder.reevaluatingUtteranceID != nil { return .disabled }
        if recorder.playingUtteranceID == u.id { return .playing }
        return .idle
    }
}

extension UtteranceRow.ReevaluateAvailability {
    /// Re-evaluate availability matches playback's gating (no
    /// source audio → unavailable, recording/analyzing → disabled),
    /// plus a dedicated `.running` for the one row whose
    /// re-evaluation is in flight and `.completed` for rows whose
    /// `wasReevaluated` flag is set. Other rows get `.disabled`
    /// during a re-evaluation so the user can't queue overlapping
    /// passes. Reading the flag off the utterance itself means
    /// the green marker survives Save/Load and shows up in the
    /// JSON export alongside the affect data.
    ///
    /// `.completed` is checked **before** the session-busy
    /// `.disabled` branches so the green marker stays put while
    /// another row's re-evaluation runs. The controller's own
    /// guards (`reevaluatingUtteranceID == nil`, `phase == .idle`)
    /// keep the no-op safety net intact for any tap that arrives
    /// during the busy window.
    @MainActor
    static func resolve(
        for u: UtteranceEstimate,
        recorder: RecordingController
    ) -> Self {
        guard recorder.playbackSourceURL != nil else { return .unavailable }
        if recorder.reevaluatingUtteranceID == u.id { return .running }
        if u.wasReevaluated == true { return .completed }
        if recorder.isRecording || recorder.isAnalyzing { return .disabled }
        if recorder.reevaluatingUtteranceID != nil { return .disabled }
        return .idle
    }
}
