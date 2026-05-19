import Foundation
import Fusion

/// View-state companion for `AffectiveSynchronyCard`: the V/A axis
/// toggle plus the currently-inspected directed pair (drives the
/// tap-to-popover affordance), with a small `rankedPairs(from:)`
/// helper that re-sorts the synchrony result for whichever axis
/// the user has selected.
@MainActor
@Observable
final class AffectiveSynchronyViewModel {
    var axis: SynchronyAxis = .valence

    /// Pair whose inspector popover is currently open, if any.
    /// Tapping a pair row toggles it; the popover surfaces the
    /// full lag profile for both V and A — content that used to
    /// sit inline as a sparkline but overflowed the iPad portrait
    /// row budget.
    var inspectedPair: AffectiveSynchrony.DirectedPair?

    func toggleAxis() {
        axis = (axis == .valence) ? .arousal : .valence
    }

    /// Sort `result.pairs` by the absolute correlation along the
    /// current axis. Pairs with a value on this axis come first
    /// (descending by magnitude); pairs with no value sink to the
    /// bottom in their original relative order.
    func rankedPairs(
        from result: AffectiveSynchrony.Result
    ) -> [AffectiveSynchrony.PairResult] {
        result.pairs.sorted { lhs, rhs in
            let lhsValue = lhs.correlation(on: axis)
            let rhsValue = rhs.correlation(on: axis)
            switch (lhsValue, rhsValue) {
            case let (l?, r?): return abs(l) > abs(r)
            case (.some, .none): return true
            case (.none, .some): return false
            case (.none, .none): return false
            }
        }
    }
}
