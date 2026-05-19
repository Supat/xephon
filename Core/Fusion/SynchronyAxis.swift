import Foundation

/// Toggle dimension for every synchrony view: fused valence vs
/// fused arousal. Shared by `AffectiveSynchronyCard` and
/// `SynchronyArcCard` so the user reads "synchrony on V" / "on A"
/// consistently across both. The cards' per-axis selectors all
/// route through this enum's extensions on the
/// `AffectiveSynchrony` result types below.
public enum SynchronyAxis: Hashable, Sendable {
    case valence, arousal
}

extension AffectiveSynchrony.PairResult {
    /// Correlation along the requested axis, or nil when that side
    /// of the pair has no usable signal. Replaces the per-card
    /// `axisValue(_:)` helpers that switched on a local Axis enum.
    public func correlation(on axis: SynchronyAxis) -> Double? {
        switch axis {
        case .valence: return valenceCorrelation
        case .arousal: return arousalCorrelation
        }
    }
}

extension AffectiveSynchrony.ArcBin {
    /// Session-mean value along the requested axis for this bin,
    /// or nil when no utterance in the bin produced a fused value.
    public func sessionMean(on axis: SynchronyAxis) -> Double? {
        switch axis {
        case .valence: return sessionMeanValence
        case .arousal: return sessionMeanArousal
        }
    }

    /// Per-speaker value along the requested axis for this bin.
    public func perSpeaker(on axis: SynchronyAxis) -> [String: Double] {
        switch axis {
        case .valence: return perSpeakerValence
        case .arousal: return perSpeakerArousal
        }
    }
}
