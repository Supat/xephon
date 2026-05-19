import Foundation
import Fusion
import XephonUtilities

/// Owns the transcript-list filter state and its derived caches.
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

    /// Normalized (Hepburn romaji, lowercased) form of each
    /// utterance's transcript, keyed by utterance ID. Populated
    /// off-MainActor in `refreshSearchCache` so the filter loop is
    /// a dictionary lookup per row instead of an N-per-keystroke
    /// CFStringTokenizer pass.
    @ObservationIgnored
    var normalizedTranscriptCache: [UUID: String] = [:]

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
    func clearFilters() {
        searchText = ""
        selectedLabelFilter = nil
        selectedSpeakerFilter = nil
        showingMismatchOnly = false
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
            for u in recorder.utterances {
                let dominant = AnalysisPipeline.dominantSpeakerInSegments(
                    timeline,
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
        let trimmed = searchText.trimmingCharacters(in: .whitespacesAndNewlines)
        let key = FilterDepsKey(
            normalizedQuery: trimmed.isEmpty
                ? ""
                : JapaneseSearchNormalizer.normalize(trimmed),
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
                if !key.normalizedQuery.isEmpty {
                    // Fall back to inline normalization when the
                    // async cache hasn't caught up yet. The async
                    // refresher will populate it momentarily; the
                    // per-row inline call is rare.
                    let normalizedText = normalizedTranscriptCache[u.id]
                        ?? JapaneseSearchNormalizer.normalize(u.transcript)
                    if !normalizedText.contains(key.normalizedQuery) {
                        return nil
                    }
                }
                return (idx, u)
            }

        var summary = ConversationSummary()
        for (_, u) in results {
            summary.update(with: u)
        }
        filterMemo.lastKey = key
        filterMemo.results = results
        filterMemo.summary = summary
    }

    // MARK: - Search cache lifecycle

    /// Bring `normalizedTranscriptCache` up to date with the
    /// recorder's current utterance list. Only normalizes
    /// utterances that aren't already in the cache, so steady-state
    /// utterance arrivals each pay one normalize call (not N).
    /// Normalization runs concurrently across the missing entries
    /// via `TaskGroup`, off the MainActor; completed results are
    /// merged back into the cache in one hop.
    func refreshSearchCache(for utterances: [UtteranceEstimate]) {
        let cached = normalizedTranscriptCache
        let missing = utterances.filter { cached[$0.id] == nil }
        guard !missing.isEmpty else { return }

        searchCacheTask?.cancel()
        searchCacheTask = Task.detached(priority: .userInitiated) { [weak self] in
            let normalized = await withTaskGroup(
                of: (UUID, String).self
            ) { group -> [(UUID, String)] in
                for u in missing {
                    if Task.isCancelled { break }
                    group.addTask {
                        (u.id, JapaneseSearchNormalizer.normalize(u.transcript))
                    }
                }
                var out: [(UUID, String)] = []
                for await pair in group {
                    out.append(pair)
                }
                return out
            }
            if Task.isCancelled { return }
            await MainActor.run {
                guard let self else { return }
                for (id, text) in normalized {
                    self.normalizedTranscriptCache[id] = text
                }
            }
        }
    }
}
