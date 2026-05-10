import ActivityKit
import Foundation

/// ActivityKit attributes for Xephon's Lock Screen / Dynamic Island
/// Live Activity. The static `attributes` (`sessionStartedAt`,
/// `sourceLabel`) are set once at `Activity.request` time; the
/// `ContentState` is what gets pushed via `Activity.update` whenever a
/// new utterance lands or the session phase changes.
///
/// This file ships in TWO targets — the Xephon app and the
/// XephonWidget extension. ActivityKit identifies the attributes type
/// by name + Codable shape, not by Swift module identity, so both
/// targets compiling their own copies work fine as long as the
/// definitions stay byte-for-byte identical.
public struct XephonActivityAttributes: ActivityAttributes {
    public struct ContentState: Codable, Hashable, Sendable {
        /// Session elapsed time in audio-time seconds (matches the
        /// in-app status line). Used by the widget to render the clock.
        public var elapsedSeconds: TimeInterval
        /// Number of finalized utterances accumulated so far.
        public var utteranceCount: Int
        /// Dominant emotion label so far, or nil while the running
        /// summary is still calibrating (< 3 utterances).
        public var topLabel: String?
        /// Mean valence centered to [-1, +1] — nil during calibration.
        public var valence: Float?
        /// Mean arousal centered to [-1, +1] — nil during calibration.
        public var arousal: Float?
        /// True while the session is finalizing (Stop tapped, analyzer
        /// flushing tail audio). Lets the widget swap the title state.
        public var isAnalyzing: Bool

        public init(
            elapsedSeconds: TimeInterval,
            utteranceCount: Int,
            topLabel: String?,
            valence: Float?,
            arousal: Float?,
            isAnalyzing: Bool
        ) {
            self.elapsedSeconds = elapsedSeconds
            self.utteranceCount = utteranceCount
            self.topLabel = topLabel
            self.valence = valence
            self.arousal = arousal
            self.isAnalyzing = isAnalyzing
        }
    }

    /// Wall-clock instant the session began. Allows the widget to
    /// drive a self-updating clock via `Text(timerInterval:)` if it
    /// chooses, in addition to the discrete elapsedSeconds updates.
    public let sessionStartedAt: Date
    /// "Microphone" or the file's display name. Shown as a subtitle on
    /// the Lock Screen so the user knows what's being analyzed.
    public let sourceLabel: String

    public init(sessionStartedAt: Date, sourceLabel: String) {
        self.sessionStartedAt = sessionStartedAt
        self.sourceLabel = sourceLabel
    }
}
