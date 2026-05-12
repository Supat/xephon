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
    /// Utterance count handles appends (live streaming) and most
    /// resets. Paired with `utterancesVersion` for in-place mutations
    /// (re-evaluate, hand-edit, speaker rename) where the count
    /// doesn't change but the content did. Both are O(1) reads off
    /// the controller.
    let utteranceCount: Int
    let utterancesVersion: Int
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
