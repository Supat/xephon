import Foundation
import Fusion
import XephonUtilities

/// Owns every piece of mutable state for the find-and-replace
/// sheet: the search + replace terms, per-utterance staged
/// replacements, per-utterance selected-match indices, the latest
/// computed match list, and the in-flight debounced search task.
///
/// `searchTerm` / `replaceTerm` / `matches` stay default-observed
/// so the sheet's bindings + match-list render react to changes.
/// The two dictionaries (`stagedReplacements`, `selectedMatches`)
/// are `@ObservationIgnored` because the only places that care
/// already re-render on `recorder.utterancesVersion` /
/// `matches` changes — observing them too would invalidate the
/// entire sheet on every per-row keystroke.
@MainActor
@Observable
final class SearchReplaceCoordinator {
    var searchTerm: String = ""
    var replaceTerm: String = ""

    /// User-toggled opt-in for fuzzy "Include similar" matching.
    /// When on, rows whose normalized transcript contains some
    /// substring within `similarMatchThreshold(for:)` edit
    /// operations of the normalized query also surface — but only
    /// for queries at or above `minQueryLengthForSimilar` normalized
    /// chars, since shorter queries explode into noise. Session-
    /// scoped (resets to off on sheet open / app launch).
    var includeSimilar: Bool = false

    /// Latest computed match list. Driven by `scheduleSearch(in:)`
    /// — never recomputed inline because `JapaneseSearchNormalizer`
    /// runs CFStringTokenizer per row and would block the keyboard
    /// on every keystroke for sessions with hundreds of utterances.
    var matches: [UtteranceEstimate] = []

    /// Staged replacement text per utterance. Populated by tapping
    /// Replace on a row; consumed by Commit which calls
    /// `commitHandEdit` and clears the entry. Survives view
    /// re-renders so the SER pipeline updating the row underneath
    /// doesn't blow away pending stages.
    @ObservationIgnored
    private var stagedReplacements: [UUID: String] = [:]

    /// Per-utterance set of match indices the user has picked for
    /// replacement. Empty (or absent) means "no explicit
    /// selection" — the Replace button treats that as "replace
    /// every match in this row", which is what most users want
    /// when there's only one match anyway. Indices reset after
    /// each Replace pass because the staged text's match
    /// positions no longer line up with the pre-staging ones.
    @ObservationIgnored
    private var selectedMatches: [UUID: Set<Int>] = [:]

    /// Set of utterance ids whose only reason for being in
    /// `matches` is the fuzzy "Include similar" pass — i.e. they
    /// failed both the raw substring and the cross-script normalized
    /// substring checks but were within the edit-distance threshold
    /// of some window of the normalized transcript. Drives the
    /// "Similar" badge in the sheet, distinct from the orange
    /// "Cross-script" badge. Replace stays disabled for these rows
    /// for the same reason it's disabled for cross-script-only
    /// rows: the raw substring isn't actually present, so there's
    /// nothing to swap. Repopulated alongside `matches` on every
    /// search pass; observation-ignored because the sheet already
    /// re-renders when `matches` changes.
    @ObservationIgnored
    private var similarMatchIDs: Set<UUID> = []

    /// In-flight search task, cancelled on every keystroke so a
    /// pile-up of normalizer passes doesn't trail behind the user.
    @ObservationIgnored
    private var searchTask: Task<Void, Never>?

    /// Minimum normalized-query length before the fuzzy pass is
    /// allowed to contribute. Shorter queries are a near-guarantee
    /// of noise: at length 3 with threshold 1 the matcher fires on
    /// any 2-of-3 character match anywhere in the transcript, which
    /// in Japanese romaji means almost every row. 4 is the smallest
    /// length where `floor(L / 4) = 1` still feels selective.
    /// `nonisolated` so the detached filter task can read it.
    nonisolated static let minQueryLengthForSimilar: Int = 4

    /// Edit-distance budget for the fuzzy pass at a given normalized
    /// query length. Linear in length so a 4-char query allows 1
    /// edit, an 8-char query 2, a 12-char query 3 — keeps the
    /// false-positive rate roughly stable across lengths.
    /// `nonisolated` so the detached filter task can call it.
    nonisolated static func similarMatchThreshold(for normalizedLength: Int) -> Int {
        max(1, normalizedLength / 4)
    }

    var trimmedSearch: String {
        searchTerm.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // MARK: - Per-row staged / selection accessors

    func staged(for utteranceID: UUID) -> String? {
        stagedReplacements[utteranceID]
    }

    func selection(for utteranceID: UUID) -> Set<Int> {
        selectedMatches[utteranceID] ?? []
    }

    func setSelection(_ indices: Set<Int>, for utteranceID: UUID) {
        selectedMatches[utteranceID] = indices
    }

    func toggleSelection(matchIndex: Int, for utteranceID: UUID) {
        var set = selectedMatches[utteranceID] ?? []
        if set.contains(matchIndex) {
            set.remove(matchIndex)
        } else {
            set.insert(matchIndex)
        }
        selectedMatches[utteranceID] = set
    }

    // MARK: - Debounced search

    /// Kick off a debounced, off-main search. Cancels any prior
    /// in-flight pass so a fast typist doesn't pile up normalizer
    /// passes behind the keyboard. The heavy filter (raw substring
    /// + Hepburn-normalized fallback) runs on a detached task so
    /// the main actor stays free for the TextField.
    ///
    /// Triggers: `searchTerm` change, plus utterance-list mutations
    /// (`utterancesVersion`, `utterances.count`) so a commit /
    /// re-evaluation refreshes the list without the user retyping.
    func scheduleSearch(in recorder: RecordingController) {
        searchTask?.cancel()
        let term = trimmedSearch
        guard !term.isEmpty else {
            matches = []
            similarMatchIDs = []
            return
        }
        let snapshot = recorder.utterances
        let allowSimilar = includeSimilar
        searchTask = Task { @MainActor in
            // Brief debounce. The sleep is cancellable, so each
            // new keystroke cancels the prior task before its
            // filter runs — only a real pause in typing produces
            // work.
            do {
                try await Task.sleep(for: .milliseconds(150))
            } catch {
                return
            }
            if Task.isCancelled { return }
            let result = await Self.filter(
                items: snapshot,
                term: term,
                allowSimilar: allowSimilar
            )
            if Task.isCancelled { return }
            matches = result.matches
            similarMatchIDs = result.similarIDs
        }
    }

    func cancelSearch() {
        searchTask?.cancel()
        searchTask = nil
    }

    /// Filter pass result. Separate `similarIDs` set lets the sheet
    /// label fuzzy-only rows distinctly from exact cross-script
    /// rows without rewalking the strings at render time.
    private struct FilterResult {
        let matches: [UtteranceEstimate]
        let similarIDs: Set<UUID>
    }

    /// Off-main filter. Runs three passes per row, falling through
    /// only when each one misses: raw substring → cross-script
    /// normalized substring → (when `allowSimilar` is on and the
    /// query is long enough) fuzzy substring on the normalized
    /// pair. Cancellation is wired through
    /// `withTaskCancellationHandler` so when the outer debounce
    /// task is cancelled the detached filter sees
    /// `Task.isCancelled == true` and bails on the next chunk
    /// boundary instead of running to completion against
    /// discarded input.
    private static func filter(
        items: [UtteranceEstimate],
        term: String,
        allowSimilar: Bool
    ) async -> FilterResult {
        let detached = Task.detached(priority: .userInitiated) {
            () -> FilterResult in
            let normalizedQuery = JapaneseSearchNormalizer.normalize(term)
            let doSimilar = allowSimilar
                && normalizedQuery.count >= Self.minQueryLengthForSimilar
            let threshold = Self.similarMatchThreshold(for: normalizedQuery.count)
            var out: [UtteranceEstimate] = []
            var similar: Set<UUID> = []
            for (idx, u) in items.enumerated() {
                // Check cancellation every 32 rows — enough to
                // keep the work cheap to abandon, infrequent
                // enough that the check itself isn't the
                // bottleneck.
                if idx & 0x1F == 0, Task.isCancelled {
                    return FilterResult(matches: [], similarIDs: [])
                }
                if u.transcript.localizedStandardRange(of: term) != nil {
                    out.append(u)
                    continue
                }
                if normalizedQuery.isEmpty { continue }
                let normalizedTranscript = JapaneseSearchNormalizer.normalize(u.transcript)
                if normalizedTranscript.contains(normalizedQuery) {
                    out.append(u)
                    continue
                }
                if doSimilar,
                   FuzzySubstringMatcher.hasSimilarSubstring(
                    query: normalizedQuery,
                    in: normalizedTranscript,
                    threshold: threshold
                   ) {
                    out.append(u)
                    similar.insert(u.id)
                }
            }
            return FilterResult(matches: out, similarIDs: similar)
        }
        return await withTaskCancellationHandler {
            await detached.value
        } onCancel: {
            detached.cancel()
        }
    }

    /// True when the row is only in `matches` because of the fuzzy
    /// pass — i.e. neither the raw substring nor the cross-script
    /// normalized substring matched. Drives the "Similar" badge in
    /// the sheet header.
    func isSimilarMatch(_ utterance: UtteranceEstimate) -> Bool {
        similarMatchIDs.contains(utterance.id)
    }

    // MARK: - Match enumeration + staging

    /// Enumerate every case-insensitive occurrence of `term`
    /// inside `text` as raw `String.Index` ranges. The cursor
    /// advances by `range.upperBound` so overlapping matches are
    /// skipped (which also stops the loop from spinning when
    /// `term` is empty inside `replaceSubrange`).
    func rawMatches(in text: String) -> [Range<String.Index>] {
        Self.rawMatches(in: text, term: trimmedSearch)
    }

    static func rawMatches(
        in text: String,
        term: String
    ) -> [Range<String.Index>] {
        guard !term.isEmpty else { return [] }
        var out: [Range<String.Index>] = []
        var cursor = text.startIndex
        while cursor < text.endIndex,
              let range = text.range(
                of: term,
                options: .caseInsensitive,
                range: cursor..<text.endIndex
              ) {
            out.append(range)
            cursor = range.upperBound
        }
        return out
    }

    /// True when this utterance contains the search term as a raw
    /// case-insensitive substring (the path that supports
    /// highlighting + Replace). False when the only reason the
    /// row surfaced is the cross-script normalized match — in
    /// which case Replace is disabled and the card shows a hint.
    func hasRawMatch(_ utterance: UtteranceEstimate) -> Bool {
        let term = trimmedSearch
        guard !term.isEmpty else { return false }
        return utterance.transcript.localizedStandardRange(of: term) != nil
    }

    func canStageReplace(for utterance: UtteranceEstimate) -> Bool {
        guard !trimmedSearch.isEmpty else { return false }
        let displayed = stagedReplacements[utterance.id] ?? utterance.transcript
        // Two reasons Replace stays disabled: (1) the displayed
        // text no longer contains the search term as a literal
        // substring (already replaced, or the row only matched
        // via cross-script normalization so we never had a raw
        // range to operate on); (2) trivially, the search is
        // empty.
        return displayed.localizedStandardRange(of: trimmedSearch) != nil
    }

    func stageReplace(for utterance: UtteranceEstimate) {
        let term = trimmedSearch
        guard !term.isEmpty else { return }
        let source = stagedReplacements[utterance.id] ?? utterance.transcript
        let matches = Self.rawMatches(in: source, term: term)
        guard !matches.isEmpty else { return }
        let selected = selectedMatches[utterance.id] ?? []
        // Empty selection = "replace every match", which matches
        // the no-friction default the user expects when there's
        // only one match. Non-empty selection = swap only those.
        let indicesToReplace: Set<Int> = selected.isEmpty
            ? Set(0..<matches.count)
            : selected.intersection(0..<matches.count)
        guard !indicesToReplace.isEmpty else { return }
        // Replace from highest index to lowest so each earlier
        // range stays valid as we mutate — `String.Index` would
        // otherwise dangle past an in-place mutation.
        var result = source
        for idx in indicesToReplace.sorted(by: >) {
            result.replaceSubrange(matches[idx], with: replaceTerm)
        }
        guard result != source else { return }
        stagedReplacements[utterance.id] = result
        // Clear the selection — its indices no longer correspond
        // to anything in the staged text. The next render will
        // recompute matches against the new source.
        selectedMatches[utterance.id] = []
    }

    func commit(for utterance: UtteranceEstimate, recorder: RecordingController) {
        guard let staged = stagedReplacements[utterance.id] else { return }
        let id = utterance.id
        let start = utterance.start
        let end = utterance.end
        Task {
            await recorder.commitHandEdit(
                utteranceID: id,
                newText: staged,
                newStart: start,
                newEnd: end
            )
            stagedReplacements.removeValue(forKey: id)
        }
    }
}
