import Foundation
import Fusion
import XephonUtilities

/// Owns every piece of mutable state for the find-and-replace
/// sheet: the search + replace terms, per-utterance staged
/// replacements, per-utterance selected-match indices, the latest
/// computed match list, and the in-flight debounced search task.
///
/// `searchTerm` / `replaceTerm` / `matches` / `stagedReplacements` /
/// `selectedMatches` are all default-observed so the card's
/// "Staged" badge, Commit-enabled state, and per-match selection
/// reflect every Replace tap or TextEditor keystroke without
/// waiting for the next search re-trigger. `similarMatchIDs` stays
/// ignored — it only changes in lockstep with `matches`, so any
/// dependent read already invalidates via the matches update.
/// `searchTask` is a task handle, not user-visible state.
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
    /// Replace on a row or typing into the manual-edit TextEditor;
    /// consumed by Commit which calls `commitHandEdit` and clears
    /// the entry. Observed so the card's Commit-button enable
    /// state, the "Staged" badge, and the highlighter mode all
    /// update synchronously with the write.
    private var stagedReplacements: [UUID: String] = [:]

    /// Per-utterance set of match indices the user has picked for
    /// replacement. Empty (or absent) means "no explicit
    /// selection" — the Replace button treats that as "replace
    /// every match in this row", which is what most users want
    /// when there's only one match anyway. Indices reset after
    /// each Replace pass because the staged text's match
    /// positions no longer line up with the pre-staging ones.
    /// Observed so the selection toggle updates the highlighter
    /// (green vs yellow per match) and the per-row counter in
    /// `selectionControls` immediately.
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

    /// Minimum contiguous-character run for the loose "wider
    /// variation" pass (longest common substring). Set to `max(3,
    /// ceil(L × 0.4))` so a 7-char query like "midiamu" needs a
    /// 3-char shared run (e.g. "dia" with "mediatte") while a
    /// 10-char query needs 4. The 3-char floor stops 2-char
    /// coincidences (e.g. "me" appearing somewhere unrelated)
    /// from firing. `nonisolated` so the detached filter task can
    /// call it.
    nonisolated static func wideVariationMinRun(for normalizedLength: Int) -> Int {
        max(3, (normalizedLength * 2 + 4) / 5)
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
                if doSimilar {
                    // Two passes under the same toggle:
                    //
                    // 1. Levenshtein near-match (tight) — catches
                    //    typos and homophone confusions where the
                    //    normalized strings differ by a few edits.
                    //
                    // 2. Longest common substring (loose) — catches
                    //    root-sharing words like ミディアム /
                    //    メディアって ("midiamu" / "mediatte" — both
                    //    contain "dia"), which Levenshtein at
                    //    `length/4` rejects. Wider net, more false
                    //    positives, but that's the explicit point of
                    //    "include similar" for the use case of
                    //    surfacing related rows rather than just typo
                    //    variants.
                    if FuzzySubstringMatcher.hasSimilarSubstring(
                        query: normalizedQuery,
                        in: normalizedTranscript,
                        threshold: threshold
                    ) {
                        out.append(u)
                        similar.insert(u.id)
                        continue
                    }
                    let minRun = Self.wideVariationMinRun(for: normalizedQuery.count)
                    if FuzzySubstringMatcher.hasLongCommonSubstring(
                        query: normalizedQuery,
                        in: normalizedTranscript,
                        minLength: minRun
                    ) {
                        out.append(u)
                        similar.insert(u.id)
                    }
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

    /// Original-text character ranges to paint purple in a match
    /// card — every chunk whose normalized form contains a fuzzy
    /// or cross-script hit on the query, EXCLUDING chunks that
    /// overlap a raw substring match (those get the yellow raw
    /// highlight instead). Works for every row in `matches`, not
    /// just `isSimilarMatch` ones, so a card that surfaced via a
    /// raw match can still show purple on a separate similar /
    /// cross-script region elsewhere in the same utterance.
    ///
    /// Per-token check rather than a single normalized-window
    /// sweep: it's the only way to surface multiple non-raw
    /// regions in one transcript, and it composes naturally with
    /// the raw-overlap skip. Granularity stays chunk-level — for a
    /// multi-character katakana loanword tokenized as one chunk
    /// (e.g. メディアって → "mediatte") the highlight covers the
    /// whole word, not just the shared "dia" substring, because
    /// the tokenizer doesn't expose intra-chunk character
    /// correspondences.
    ///
    /// Returns [] when the query is too short for the fuzzy pass
    /// (below `minQueryLengthForSimilar`) or when no token clears
    /// either matcher.
    func similarMatchRanges(for utterance: UtteranceEstimate) -> [Range<String.Index>] {
        let normalizedQuery = JapaneseSearchNormalizer.normalize(trimmedSearch)
        guard normalizedQuery.count >= Self.minQueryLengthForSimilar else { return [] }

        let tokens = JapaneseSearchNormalizer.tokens(utterance.transcript)
        guard !tokens.isEmpty else { return [] }

        // Raw match ranges in the original transcript — yellow
        // highlights cover them, so skip any token whose original
        // range intersects one of them. The overlap check is
        // generous on purpose (any character intersection counts)
        // so partial coverage doesn't double-up.
        let rawRanges = Self.rawMatches(in: utterance.transcript, term: trimmedSearch)
        let levThreshold = Self.similarMatchThreshold(for: normalizedQuery.count)
        let minRun = Self.wideVariationMinRun(for: normalizedQuery.count)

        var ranges: [Range<String.Index>] = []
        for token in tokens {
            if rawRanges.contains(where: { $0.overlaps(token.originalRange) }) {
                continue
            }
            if token.normalized.isEmpty { continue }
            // Tight Levenshtein first, loose LCS fallback — same
            // cascade the filter pass uses for "Include similar".
            let hit = FuzzySubstringMatcher.hasSimilarSubstring(
                query: normalizedQuery,
                in: token.normalized,
                threshold: levThreshold
            ) || FuzzySubstringMatcher.hasLongCommonSubstring(
                query: normalizedQuery,
                in: token.normalized,
                minLength: minRun
            )
            if hit {
                ranges.append(token.originalRange)
            }
        }
        return ranges
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

    /// Stage `text` for `utteranceID` as if it were the result of
    /// a Replace pass. Used by the sheet's editable TextEditor on
    /// "edit manually" rows (cross-script / similar matches), where
    /// there's no raw substring to swap so the user types the
    /// correction directly. Clears the staging entry when `text`
    /// equals `original` so the Commit button greys out the moment
    /// the user reverts their edit. Selection state is cleared
    /// alongside the stage write because match indices computed
    /// against the prior text no longer line up.
    func setManualStaged(_ text: String, for utteranceID: UUID, original: String) {
        if text == original {
            stagedReplacements.removeValue(forKey: utteranceID)
        } else {
            stagedReplacements[utteranceID] = text
        }
        selectedMatches[utteranceID] = []
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
            // Refresh the match list against the post-commit
            // utterance snapshot. `.onChange(of: utterancesVersion)`
            // in the sheet body doesn't fire reliably because the
            // body reads `coord.matches` (not the recorder's
            // utterances), so the version-bump observation
            // dependency isn't established. Calling scheduleSearch
            // here closes the loop deterministically so the card
            // either drops out of the list (term no longer present)
            // or re-renders against the new transcript.
            scheduleSearch(in: recorder)
        }
    }
}
