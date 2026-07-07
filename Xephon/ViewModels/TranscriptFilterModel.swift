import Foundation
import Fusion
import XephonUtilities

/// Owns the transcript-list filter state and its derived caches.
///
/// Lives in Xephon/ViewModels/ — the app's presentation-model layer
/// (@MainActor @Observable companions owned by views via @State).
/// See docs/mvvm_audit.md for the layer's conventions: models here
/// derive presentation state from RecordingController reads, hold
/// sheet/workflow state, and memoize; they never touch AVFoundation
/// or the Core inference modules directly.
///
/// Extracted from `ContentView` so the view body stays focused on
/// layout. The filter inputs (search text, label / speaker chips,
/// mismatch toggle) feed two derived products — the filtered slice
/// of utterances and a per-slice `ConversationSummary` — both of
/// which read from a shared `FilterMemo` keyed on the inputs that
/// affect the outcome. Render-only state (selection, playback,
/// scroll) intentionally does not invalidate the memo.
///
/// The normalized-transcript cache and its background refresher
/// live here too so the search loop stays a dictionary lookup per
/// row even on the longest sessions.
@MainActor
@Observable
final class TranscriptFilterModel {
    /// Free-text filter applied to each utterance's transcript.
    /// Empty disables search filtering.
    var searchText: String = ""
    /// Nil = "All labels"; non-nil only shows utterances whose
    /// fused top label matches.
    var selectedLabelFilter: String?
    /// Nil = all speakers; non-nil only shows utterances stamped
    /// with the matching `speakerID`.
    var selectedSpeakerFilter: String?
    /// When true, only utterances whose stored speaker disagrees
    /// with the cumulative-timeline majority survive the filter.
    var showingMismatchOnly: Bool = false

    /// One cached normalized form alongside the raw transcript it
    /// was derived from. Used to invalidate entries whose
    /// underlying transcript has been edited in place — the ID
    /// stays the same across `commitHandEdit` and re-evaluation,
    /// so a plain `[UUID: String]` would happily hand back the
    /// pre-edit normalization forever. Comparing `raw` against
    /// `utterance.transcript` at lookup time catches the drift
    /// even if `refreshSearchCache` hasn't fired yet.
    struct NormalizedTranscript: Sendable {
        let raw: String
        let normalized: String
        /// Ordered per-token normalized forms. `normalized` is their
        /// join; the token list is what `NormalizedSearchQuery`'s
        /// boundary-form mode (single-letter / letter-name queries)
        /// needs, so we cache both off one tokenizer pass.
        let tokens: [String]
    }

    /// Normalized (Hepburn romaji, lowercased) form of each
    /// utterance's transcript, keyed by utterance ID. Populated
    /// off-MainActor in `refreshSearchCache` so the filter loop is
    /// a dictionary lookup per row instead of an N-per-keystroke
    /// CFStringTokenizer pass.
    @ObservationIgnored
    var normalizedTranscriptCache: [UUID: NormalizedTranscript] = [:]

    /// Look up the cached normalized form, fall back to inline
    /// normalization if missing OR if the cached entry is stale
    /// (raw text differs from the utterance's current transcript).
    /// Same call site for the filter loop and the keyword-count
    /// computation so they agree on staleness handling.
    private func normalizedForms(
        for utterance: UtteranceEstimate
    ) -> (normalized: String, tokens: [String]) {
        if let entry = normalizedTranscriptCache[utterance.id],
           entry.raw == utterance.transcript {
            return (entry.normalized, entry.tokens)
        }
        let tokens = JapaneseSearchNormalizer.normalizedTokens(utterance.transcript)
        return (tokens.joined(), tokens)
    }

    /// Background task that's currently rebuilding the cache.
    /// Cancelled and replaced on every utterance-count change so a
    /// fast-arriving stream of utterances doesn't spawn unbounded
    /// work.
    @ObservationIgnored
    private var searchCacheTask: Task<Void, Never>?

    /// Filter + summary memo. Reference type so we can mutate it
    /// from inside getters without triggering SwiftUI re-renders.
    @ObservationIgnored
    private let filterMemo = FilterMemo()

    /// Mismatch-set memo. Same reasoning as `filterMemo`.
    @ObservationIgnored
    private let mismatchMemo = MismatchMemo()

    /// Per-keyword occurrence-count memo. The Keywords page reads
    /// these to render a count badge per row; recomputing per
    /// body re-eval would be O(utterances × keywords) on every
    /// keystroke during keyword-add (every observed mutation fires
    /// a render across both panes), so memoize at the model
    /// level — same pattern as `mismatchMemo`.
    @ObservationIgnored
    private let keywordCountsMemo = KeywordCountsMemo()
    private let keywordTagMatchesMemo = KeywordTagMatchesMemo()
    @ObservationIgnored
    private let keywordHighlightMemo = KeywordHighlightMemo()

    // MARK: - Filter controls

    /// Display string for the label-filter menu trigger.
    var filterLabelDisplay: String {
        if let label = selectedLabelFilter {
            return label.capitalized(with: Locale(identifier: "en_US"))
        }
        return String(localized: "filter.label.all")
    }

    /// Reset every filter knob — called from the "Clear filters"
    /// button on the empty-results placeholder and from the
    /// session-boundary handler when the utterance list empties.
    /// Takes the keyword store because the keyword SELECTION is a
    /// fifth filter layer (`filteredIDs` ORs the selected keywords
    /// on top of the search field) that lives in `KeywordStore`,
    /// not here — clearing only the local knobs left a
    /// keyword-only filter stuck on the no-matches placeholder.
    func clearFilters(keywords: KeywordStore) {
        searchText = ""
        selectedLabelFilter = nil
        selectedSpeakerFilter = nil
        showingMismatchOnly = false
        keywords.selectedKeywordIDs.removeAll()
    }

    /// Drop the per-utterance normalized cache. Called when a new
    /// session begins (mic record / file analysis / `.xph` import)
    /// so the next session's renders aren't poisoned by stale IDs.
    func resetForNewSession() {
        normalizedTranscriptCache.removeAll(keepingCapacity: true)
    }

    // MARK: - Derived availability

    /// Distinct top labels seen so far this session, sorted
    /// alphabetically for stable menu order.
    func availableLabels(in utterances: [UtteranceEstimate]) -> [String] {
        Set(utterances.compactMap { $0.fusedTopLabel }).sorted()
    }

    /// Distinct speaker IDs in the order they first appear in the
    /// session. First-appearance order (rather than alphabetical)
    /// keeps the chip row stable as new utterances arrive — a new
    /// speaker is appended at the end instead of reshuffling
    /// existing chips.
    func availableSpeakers(in utterances: [UtteranceEstimate]) -> [String] {
        utterances.orderedSpeakerIDs
    }

    /// Utterance IDs whose stored `speakerID` disagrees with the
    /// cumulative diarization timeline's per-instant majority for
    /// that row's `[start, end]` window. Memoized on
    /// `(utterancesVersion, timelineVersion, utteranceCount)` —
    /// body re-eval fires often and the vote loop is
    /// O(N × samples × segments).
    func mismatchedUtteranceIDs(in recorder: RecordingController) -> Set<UUID> {
        let key = MismatchMemo.Key(
            utterancesVersion: recorder.utterancesVersion,
            timelineVersion: recorder.diarizationTimelineVersion,
            utteranceCount: recorder.utterances.count
        )
        if mismatchMemo.lastKey == key { return mismatchMemo.set }
        let timeline = recorder.diarizationTimeline
        var result: Set<UUID> = []
        if !timeline.isEmpty {
            // Pre-sort ONCE + windowed voting — same fix as
            // TranscriptList.speakerMismatchedIDs; the per-row
            // convenience form re-sorts the whole timeline per
            // utterance and goes quadratic on long sessions.
            let sorted = timeline.sorted { $0.start < $1.start }
            let maxSegmentDuration = timeline.lazy
                .map { $0.end - $0.start }
                .max() ?? 0
            for u in recorder.utterances {
                let dominant = AnalysisPipeline.dominantSpeakerInSegments(
                    sortedByStart: sorted,
                    maxSegmentDuration: maxSegmentDuration,
                    from: u.start,
                    to: u.end,
                    fallback: u.speakerID
                )
                if dominant != u.speakerID {
                    result.insert(u.id)
                }
            }
        }
        mismatchMemo.lastKey = key
        mismatchMemo.set = result
        return result
    }

    /// Per-keyword count of utterances whose normalized transcript
    /// contains that keyword's normalized form. Computed against
    /// the FULL utterance list (no other filter applied) so the
    /// badge tells the user how prevalent each keyword is in the
    /// session independent of any currently-active search /
    /// label / speaker filter — keeps the number stable across
    /// filter toggles. Empty keywords return 0; the empty map is
    /// returned when there are no keywords at all.
    ///
    /// Memoized on `(utterancesVersion, utteranceCount,
    /// keywordSignature)`; cached map is returned unchanged on
    /// renders that don't touch any of those inputs.
    func keywordOccurrenceCounts(
        in recorder: RecordingController
    ) -> [UUID: Int] {
        let kws = recorder.keywords.keywords
        let signature = kws.map { "\($0.id.uuidString)|\($0.text)" }
        let key = KeywordCountsMemo.Key(
            utterancesVersion: recorder.utterancesVersion,
            utteranceCount: recorder.utterances.count,
            keywordSignature: signature
        )
        if keywordCountsMemo.lastKey == key { return keywordCountsMemo.counts }
        var counts: [UUID: Int] = [:]
        guard !kws.isEmpty else {
            keywordCountsMemo.lastKey = key
            keywordCountsMemo.counts = counts
            return counts
        }
        // Pre-build each keyword's query once; reuse across every
        // utterance. Drops empties so a row of whitespace never
        // inflates every utterance's count. A single-letter keyword
        // expands to its spoken letter-name readings, same as the
        // search box.
        let keywordQueries: [(id: UUID, query: NormalizedSearchQuery)] = kws.compactMap {
            let q = NormalizedSearchQuery.build(from: $0.text)
            return q.isEmpty ? nil : (id: $0.id, query: q)
        }
        for u in recorder.utterances {
            let forms = normalizedForms(for: u)
            for entry in keywordQueries
            where entry.query.matches(normalized: forms.normalized, tokens: forms.tokens) {
                counts[entry.id, default: 0] += 1
            }
        }
        keywordCountsMemo.lastKey = key
        keywordCountsMemo.counts = counts
        return counts
    }

    /// utteranceID → ordered tags of the COLOR-TAGGED keywords
    /// matching that utterance, deduped by color (two same-colored
    /// keywords hitting one row paint one band). Match semantics =
    /// the counts above (NormalizedSearchQuery over the full
    /// utterance list). Memoized on the same inputs plus tagColor
    /// (a swatch change alters output without touching text).
    /// Drives KeywordTimelineStrip.
    func keywordTagMatches(
        in recorder: RecordingController
    ) -> [UUID: [KeywordTagColor]] {
        let kws = recorder.keywords.keywords
        let signature = kws.map {
            "\($0.id.uuidString)|\($0.text)|\($0.tagColor?.rawValue ?? "-")"
        }
        let key = KeywordTagMatchesMemo.Key(
            utterancesVersion: recorder.utterancesVersion,
            utteranceCount: recorder.utterances.count,
            keywordSignature: signature
        )
        if keywordTagMatchesMemo.lastKey == key { return keywordTagMatchesMemo.matches }
        var matches: [UUID: [KeywordTagColor]] = [:]
        let taggedQueries: [(tag: KeywordTagColor, query: NormalizedSearchQuery)] = kws.compactMap {
            guard let tag = $0.tagColor else { return nil }
            let q = NormalizedSearchQuery.build(from: $0.text)
            return q.isEmpty ? nil : (tag: tag, query: q)
        }
        if !taggedQueries.isEmpty {
            for u in recorder.utterances {
                let forms = normalizedForms(for: u)
                var tags: [KeywordTagColor] = []
                for entry in taggedQueries
                where entry.query.matches(normalized: forms.normalized, tokens: forms.tokens) {
                    if !tags.contains(entry.tag) { tags.append(entry.tag) }
                }
                if !tags.isEmpty { matches[u.id] = tags }
            }
        }
        keywordTagMatchesMemo.lastKey = key
        keywordTagMatchesMemo.matches = matches
        return matches
    }

    /// Memoized keyword-tag highlight rendering for one utterance
    /// row — see KeywordHighlightMemo for why per-render compute
    /// was too expensive. Lazy per-row fill: only rows the List
    /// actually materializes pay the tokenizer cost, once per
    /// (transcript edit × tagged-keyword change) generation.
    func highlightedTranscript(
        for utterance: UtteranceEstimate,
        in recorder: RecordingController
    ) -> AttributedString {
        let kws = recorder.keywords.keywords
        let signature: [String] = kws.compactMap {
            guard let tag = $0.tagColor else { return nil }
            return "\($0.id.uuidString)|\($0.text)|\(tag.rawValue)"
        }
        let key = KeywordHighlightMemo.Key(
            utterancesVersion: recorder.utterancesVersion,
            taggedKeywordSignature: signature
        )
        if keywordHighlightMemo.lastKey != key {
            keywordHighlightMemo.lastKey = key
            keywordHighlightMemo.rendered.removeAll(keepingCapacity: true)
        }
        if let cached = keywordHighlightMemo.rendered[utterance.id] {
            return cached
        }
        let rendered = KeywordHighlighter.attributedTranscript(
            utterance.transcript,
            keywords: kws
        )
        keywordHighlightMemo.rendered[utterance.id] = rendered
        return rendered
    }

    // MARK: - Filtered slice + summary

    /// `(originalIndex, utterance)` pairs surviving every active
    /// filter. The original index is preserved so each row's "#N"
    /// badge keeps the utterance's stable session number even when
    /// the list is filtered.
    func filteredIndexedUtterances(
        in recorder: RecordingController
    ) -> [(idx: Int, u: UtteranceEstimate)] {
        refreshFilterMemoIfNeeded(in: recorder)
        return filterMemo.results
    }

    /// IDs of the utterances surviving every active filter — the
    /// set-form companion to `filteredIndexedUtterances`, memoized
    /// under the same key so the per-render cost is a stored-
    /// property read. Used to prune `visibleUtteranceIDs` when the
    /// filter narrows (see TranscriptPaneView).
    func filteredUtteranceIDs(
        in recorder: RecordingController
    ) -> Set<UUID> {
        refreshFilterMemoIfNeeded(in: recorder)
        return filterMemo.resultIDs
    }

    /// `ConversationSummary` computed over the *filtered* slice, so
    /// the Summary and Statistics panels show the same slice the
    /// transcript list shows.
    func displayedSummary(
        in recorder: RecordingController
    ) -> ConversationSummary {
        refreshFilterMemoIfNeeded(in: recorder)
        return filterMemo.summary
    }

    /// Recompute the filter + summary memo when any input
    /// dependency has changed since the last render. No-op when
    /// nothing changed — the cached `results` and `summary` are
    /// returned as-is.
    private func refreshFilterMemoIfNeeded(in recorder: RecordingController) {
        // Build the search query once per refresh. A single Latin
        // letter expands to its spoken letter-name readings (G →
        // ジー / じー) plus the literal letter; everything else stays
        // a normalized substring match.
        let searchQuery = NormalizedSearchQuery.build(from: searchText)
        // Build every selected keyword's query once, deduped (the
        // user can have two keywords with the same surface form).
        let keywordQueries: Set<NormalizedSearchQuery> = Set(
            recorder.keywords.selectedKeywords
                .map { NormalizedSearchQuery.build(from: $0.text) }
                .filter { !$0.isEmpty }
        )
        let key = FilterDepsKey(
            searchQuery: searchQuery,
            keywordQueries: keywordQueries,
            labelFilter: selectedLabelFilter,
            speakerFilter: selectedSpeakerFilter,
            mismatchOnly: showingMismatchOnly,
            utteranceCount: recorder.utterances.count,
            utterancesVersion: recorder.utterancesVersion,
            timelineVersion: recorder.diarizationTimelineVersion
        )
        if filterMemo.lastKey == key { return }

        let mismatchSet: Set<UUID> = key.mismatchOnly
            ? mismatchedUtteranceIDs(in: recorder)
            : []
        let results: [(idx: Int, u: UtteranceEstimate)] = recorder
            .utterances
            .enumerated()
            .compactMap { idx, u in
                if let filterLabel = key.labelFilter,
                   u.fusedTopLabel != filterLabel {
                    return nil
                }
                if let filterSpeaker = key.speakerFilter,
                   u.speakerID != filterSpeaker {
                    return nil
                }
                if key.mismatchOnly, !mismatchSet.contains(u.id) {
                    return nil
                }
                // Compute normalized forms at most once per row even
                // when both the search query AND the keyword filter
                // are active. `normalizedForms(for:)` returns the
                // cached pair when the raw text still matches, and
                // re-normalizes inline when it doesn't (post-edit,
                // re-eval, or refresh-cache lag).
                let needsNormalized =
                    !key.searchQuery.isEmpty || !key.keywordQueries.isEmpty
                let forms = needsNormalized ? normalizedForms(for: u) : nil
                if !key.searchQuery.isEmpty,
                   let f = forms,
                   !key.searchQuery.matches(normalized: f.normalized, tokens: f.tokens) {
                    return nil
                }
                if !key.keywordQueries.isEmpty,
                   let f = forms,
                   !key.keywordQueries.contains(where: {
                       $0.matches(normalized: f.normalized, tokens: f.tokens)
                   }) {
                    return nil
                }
                return (idx, u)
            }

        var summary = ConversationSummary()
        for (_, u) in results {
            summary.update(with: u)
        }
        filterMemo.lastKey = key
        filterMemo.results = results
        filterMemo.resultIDs = Set(results.map(\.u.id))
        filterMemo.summary = summary
    }

    // MARK: - Search cache lifecycle

    /// Bring `normalizedTranscriptCache` up to date with the
    /// recorder's current utterance list. Re-normalizes utterances
    /// whose cached entry is missing OR whose raw text has
    /// diverged from the cached `raw` (in-place edits via
    /// `commitHandEdit` / `applyReevaluation` keep the ID but
    /// replace the transcript). Also evicts entries for ids that
    /// are no longer present so the dict can't grow unboundedly
    /// across session loads. Normalization runs concurrently
    /// across the to-rebuild set via `TaskGroup`, off the
    /// MainActor; completed results are merged back into the
    /// cache in one hop.
    func refreshSearchCache(for utterances: [UtteranceEstimate]) {
        let currentIDs = Set(utterances.map(\.id))
        if normalizedTranscriptCache.keys.contains(where: { !currentIDs.contains($0) }) {
            normalizedTranscriptCache = normalizedTranscriptCache.filter {
                currentIDs.contains($0.key)
            }
        }
        let cached = normalizedTranscriptCache
        let toRebuild: [(UUID, String)] = utterances.compactMap { u in
            if let entry = cached[u.id], entry.raw == u.transcript {
                return nil
            }
            return (u.id, u.transcript)
        }
        guard !toRebuild.isEmpty else { return }

        searchCacheTask?.cancel()
        searchCacheTask = Task.detached(priority: .userInitiated) { [weak self] in
            let normalized = await withTaskGroup(
                of: (UUID, String, [String]).self
            ) { group -> [(UUID, String, [String])] in
                for (id, raw) in toRebuild {
                    if Task.isCancelled { break }
                    group.addTask {
                        (id, raw, JapaneseSearchNormalizer.normalizedTokens(raw))
                    }
                }
                var out: [(UUID, String, [String])] = []
                for await triple in group {
                    out.append(triple)
                }
                return out
            }
            if Task.isCancelled { return }
            await self?.mergeNormalizedResults(normalized)
        }
    }

    private func mergeNormalizedResults(_ results: [(UUID, String, [String])]) {
        for (id, raw, tokens) in results {
            normalizedTranscriptCache[id] = NormalizedTranscript(
                raw: raw,
                normalized: tokens.joined(),
                tokens: tokens
            )
        }
    }
}
