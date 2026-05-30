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

    /// Same per-token Levenshtein/LCS sweep `similarMatchRanges`
    /// uses, but returns Bool — used by the filter to gate row
    /// inclusion so we don't surface non-raw matches that no token
    /// can highlight. Without this gate, rows whose match only
    /// exists on the concatenated normalized text (spanning
    /// tokenizer chunks too small to clear the per-token min-run
    /// individually) would appear in the list as bare cards with
    /// no visual indication of what triggered the match.
    /// `nonisolated` so the detached filter task can call it.
    nonisolated static func hasHighlightableSimilarToken(
        in transcript: String,
        normalizedQuery: String,
        threshold: Int,
        minRun: Int
    ) -> Bool {
        guard !normalizedQuery.isEmpty else { return false }
        let tokens = JapaneseSearchNormalizer.tokens(transcript)
        for token in tokens {
            if token.normalized.isEmpty { continue }
            if FuzzySubstringMatcher.hasSimilarSubstring(
                query: normalizedQuery,
                in: token.normalized,
                threshold: threshold
            ) { return true }
            if FuzzySubstringMatcher.hasLongCommonSubstring(
                query: normalizedQuery,
                in: token.normalized,
                minLength: minRun
            ) { return true }
        }
        return false
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
            let minRun = Self.wideVariationMinRun(for: normalizedQuery.count)
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
                let isCrossScript = normalizedTranscript.contains(normalizedQuery)
                // Fuzzy passes only run under the "Include similar"
                // toggle. Two layers under the same flag:
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
                //    "include similar".
                let fuzzyHit = doSimilar && !isCrossScript && (
                    FuzzySubstringMatcher.hasSimilarSubstring(
                        query: normalizedQuery,
                        in: normalizedTranscript,
                        threshold: threshold
                    ) || FuzzySubstringMatcher.hasLongCommonSubstring(
                        query: normalizedQuery,
                        in: normalizedTranscript,
                        minLength: minRun
                    )
                )
                guard isCrossScript || fuzzyHit else { continue }
                // Highlight-availability gate. The sheet's
                // similarMatchRanges paints purple per-token, and a
                // row whose only match exists on the concatenated
                // normalized text (no single token clears the
                // per-token Levenshtein/LCS thresholds) would
                // surface as a card with neither yellow nor purple
                // highlight — unexplained noise. Skip those.
                guard Self.hasHighlightableSimilarToken(
                    in: u.transcript,
                    normalizedQuery: normalizedQuery,
                    threshold: threshold,
                    minRun: minRun
                ) else { continue }
                out.append(u)
                if !isCrossScript {
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

    /// Stage a replacement for a row that has NO raw substring
    /// match — i.e. cross-script or similar matches surfaced via
    /// the per-token highlight pass. Operates on
    /// `utterance.transcript` (the original) and swaps every
    /// chunk range that `similarMatchRanges` would paint purple
    /// for the replace term. Sorted high-to-low so earlier ranges
    /// stay valid as we mutate.
    ///
    /// Intentionally bypasses any current staging: the "Replace
    /// Anyway" affordance is a one-shot from the original text.
    /// Manual edits (via the TextEditor) are the right tool once
    /// the user has started shaping the staged form themselves.
    /// No-op when the search or replace term is empty, or when no
    /// chunk range exists for this row.
    func stageReplaceAnyway(for utterance: UtteranceEstimate) {
        guard !trimmedSearch.isEmpty else { return }
        let ranges = similarMatchRanges(for: utterance)
        guard !ranges.isEmpty else { return }
        var result = utterance.transcript
        for range in ranges.sorted(by: { $0.lowerBound > $1.lowerBound }) {
            result.replaceSubrange(range, with: replaceTerm)
        }
        guard result != utterance.transcript else { return }
        stagedReplacements[utterance.id] = result
        selectedMatches[utterance.id] = []
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

    /// True while a `commitAll` Task is in flight. Drives the
    /// "Commit All" button's disabled state alongside
    /// `hasStagedAny` so a second tap during the loop can't
    /// snapshot a half-processed dict and re-fire commits for
    /// rows the first call hasn't drained yet.
    private(set) var commitAllInflight: Bool = false

    /// True when at least one row is staged AND no Commit All is
    /// currently running. Drives the "Commit All" button's
    /// enabled state in the sheet header.
    var canCommitAll: Bool {
        !stagedReplacements.isEmpty && !commitAllInflight
    }

    /// True when at least one row is staged. Kept around for
    /// older readers; new code should prefer `canCommitAll` for
    /// button gating.
    var hasStagedAny: Bool {
        !stagedReplacements.isEmpty
    }

    /// Commit every staged row in one pass. Snapshots
    /// `stagedReplacements` synchronously, looks up each
    /// utterance's `[start, end]` from the recorder's current
    /// list, then awaits `commitHandEdit` serially (parallel
    /// would race the recorder's actor-side bookkeeping). Each
    /// id is removed from the staging dict as soon as its commit
    /// completes, so a mid-flight cancellation still leaves the
    /// staging dict consistent with what was actually committed.
    /// Final `scheduleSearch` rebuilds the matches list against
    /// the post-commit utterances.
    ///
    /// Re-entry-guarded via `commitAllInflight`: a second call
    /// while a previous Task is still draining the dict is a
    /// no-op. Without this guard the second snapshot would
    /// include ids the first call hadn't committed yet and
    /// commitHandEdit would re-run SER+fusion on already-
    /// committed text, plus two `scheduleSearch` debounces would
    /// race the matches list.
    func commitAll(recorder: RecordingController) {
        guard !commitAllInflight else { return }
        let snapshot = stagedReplacements
        guard !snapshot.isEmpty else { return }
        let pending: [(id: UUID, text: String, start: TimeInterval, end: TimeInterval)] =
            snapshot.compactMap { id, text in
                guard let u = recorder.utterances.first(where: { $0.id == id }) else {
                    return nil
                }
                return (id, text, u.start, u.end)
            }
        commitAllInflight = true
        Task {
            defer { commitAllInflight = false }
            for entry in pending {
                await recorder.commitHandEdit(
                    utteranceID: entry.id,
                    newText: entry.text,
                    newStart: entry.start,
                    newEnd: entry.end
                )
                stagedReplacements.removeValue(forKey: entry.id)
            }
            scheduleSearch(in: recorder)
        }
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
