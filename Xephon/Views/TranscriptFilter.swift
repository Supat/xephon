import Foundation
import Fusion

/// Fingerprint of the inputs that affect `filteredIndexedUtterances`
/// and `displayedSummary` in ContentView. `Equatable` so `FilterMemo`
/// can early-exit on no-change renders.
struct FilterDepsKey: Equatable {
    /// Already-normalized search query, so we don't re-tokenize the
    /// query string on every change-check.
    let normalizedQuery: String
    let labelFilter: String?
    let speakerFilter: String?
    /// When true, only utterances whose stored speaker disagrees
    /// with the cumulative-timeline majority survive the filter.
    /// Independent of `speakerFilter` (which selects a single
    /// speaker); the two never combine usefully so the chip bar
    /// makes the mismatch chip mutually exclusive with speaker
    /// chips.
    let mismatchOnly: Bool
    /// Utterance count handles appends (live streaming) and most
    /// resets. Paired with `utterancesVersion` for in-place mutations
    /// (re-evaluate, hand-edit, speaker rename) where the count
    /// doesn't change but the content did. Both are O(1) reads off
    /// the controller.
    let utteranceCount: Int
    let utterancesVersion: Int
    /// Diarization-timeline mutation version — bumped on every
    /// timeline assignment, including in-place rewrites that
    /// leave the segment count unchanged. Cumulative-timeline
    /// content drives the mismatch set, so the filter memo must
    /// invalidate whenever the timeline shifts.
    let timelineVersion: Int
}

/// Reference-typed memo for the filter + summary derivation in
/// ContentView. Held via `@State` so it survives view-body
/// re-evaluations; mutating its stored properties does NOT
/// re-trigger the body (which is what we want — the memo is read
/// during the current render pass).
@MainActor
final class FilterMemo {
    var lastKey: FilterDepsKey?
    var results: [(idx: Int, u: UtteranceEstimate)] = []
    var summary: ConversationSummary = ConversationSummary()
}

/// Reference-typed memo for the speaker-mismatch set in ContentView.
/// Same pattern as `FilterMemo` — held via `@State` so internal
/// mutation doesn't trigger body re-eval, and the per-render cost
/// drops to a key compare when nothing's changed.
@MainActor
final class MismatchMemo {
    struct Key: Equatable {
        let utterancesVersion: Int
        /// Same role as `FilterDepsKey.timelineVersion` —
        /// invalidates on every timeline mutation, including
        /// count-preserving rewrites.
        let timelineVersion: Int
        let utteranceCount: Int
    }
    var lastKey: Key?
    var set: Set<UUID> = []
}
