import Foundation

/// Approximate-substring search via Levenshtein distance.
///
/// Used by `SearchReplaceCoordinator`'s "Include similar" path:
/// after the raw-substring and cross-script normalized passes both
/// miss, this matcher slides a window over the normalized
/// transcript and reports a hit when any window's edit distance to
/// the normalized query is at or below `threshold`. Bounded
/// Levenshtein with row-minimum early termination keeps the inner
/// loop cheap even when most windows are far off.
///
/// The matcher works on `Character` arrays so callers can pre-
/// normalize both sides via `JapaneseSearchNormalizer` (Hepburn
/// romaji) and reuse the same script-neutral comparison the rest of
/// the search pipeline does. Empty inputs return false.
enum FuzzySubstringMatcher {
    /// True when some substring of `text` is within `threshold` edit
    /// operations of `query`. Window lengths cover
    /// `[query.count - threshold, query.count + threshold]` so a
    /// missing or extra character in the transcript still produces a
    /// hit. When `text` is shorter than the minimum window the full
    /// strings are compared directly so short transcripts don't
    /// silently miss.
    static func hasSimilarSubstring(
        query: String,
        in text: String,
        threshold: Int
    ) -> Bool {
        findSimilarSubstring(query: query, in: text, threshold: threshold) != nil
    }

    /// True when `query` and `text` share a contiguous run of at
    /// least `minLength` characters. Complements `hasSimilarSubstring`
    /// — that one is tight (typo-tolerant near-matches), this one is
    /// loose (root-sharing words like メディア / ミディアム or
    /// medium / media that share a stem but diverge in the suffix).
    /// Useful when the user wants the search to surface conceptual
    /// relatives, not just typo variants.
    ///
    /// O(query.count × text.count) DP; early-exits the instant the
    /// running match length crosses `minLength`. Empty inputs or
    /// either side shorter than `minLength` return false.
    static func hasLongCommonSubstring(
        query: String,
        in text: String,
        minLength: Int
    ) -> Bool {
        findLongCommonSubstring(query: query, in: text, minLength: minLength) != nil
    }

    /// Same as `hasLongCommonSubstring` but returns the
    /// `[start, end)` integer position range in `text` of the first
    /// run of length ≥ `minLength` that was found. Used by the
    /// find-and-replace highlighter to map a normalized-space hit
    /// back to original tokens. The returned positions index into
    /// `text` as a `Character` array, NOT into the original
    /// `String.Index` space — callers map back via the
    /// `JapaneseSearchNormalizer.Token` chunk offsets.
    static func findLongCommonSubstring(
        query: String,
        in text: String,
        minLength: Int
    ) -> Range<Int>? {
        guard !query.isEmpty, !text.isEmpty, minLength > 0 else { return nil }
        let q = Array(query)
        let t = Array(text)
        let m = q.count
        let n = t.count
        if m < minLength || n < minLength { return nil }
        var prev = Array(repeating: 0, count: n + 1)
        for i in 1...m {
            var curr = Array(repeating: 0, count: n + 1)
            for j in 1...n {
                if q[i - 1] == t[j - 1] {
                    let next = prev[j - 1] + 1
                    if next >= minLength {
                        return (j - next)..<j
                    }
                    curr[j] = next
                }
            }
            prev = curr
        }
        return nil
    }

    /// Same as `hasSimilarSubstring` but returns the `[start, end)`
    /// integer position range in `text` of the first window within
    /// `threshold` edits of `query`. Used by the highlighter for
    /// fuzzy-match range reconstruction. Returns nil when nothing
    /// fits the threshold.
    static func findSimilarSubstring(
        query: String,
        in text: String,
        threshold: Int
    ) -> Range<Int>? {
        guard !query.isEmpty, !text.isEmpty, threshold >= 0 else { return nil }
        let q = Array(query)
        let t = Array(text)
        let m = q.count
        let n = t.count
        // Semi-global (Sellers) alignment with Damerau
        // transpositions: row 0 is all zeros so a match may START at
        // any text position; the best END is the column with the
        // lowest final-row value ≤ threshold. Each cell carries the
        // start index its alignment began at, so the hit range comes
        // out of the same single O(m·n) pass — replacing the old
        // window-sliding loop that recomputed a full Levenshtein per
        // (window length × start) pair, O((2t+1)·n·m²)-ish. Same
        // semantics, orders of magnitude cheaper on long texts,
        // which is what makes whole-transcript fuzzy passes and
        // joined-token highlighting affordable.
        var prevPrevDist: [Int] = []
        var prevPrevStart: [Int] = []
        var prevDist = [Int](repeating: 0, count: n + 1)
        var prevStart = [Int](0...n)          // row 0: start = own column
        var currDist = [Int](repeating: 0, count: n + 1)
        var currStart = [Int](repeating: 0, count: n + 1)

        var bestDist = threshold + 1
        var bestRange: Range<Int>?

        for i in 1...m {
            currDist[0] = i
            currStart[0] = 0
            var rowMin = currDist[0]
            for j in 1...n {
                let cost = q[i - 1] == t[j - 1] ? 0 : 1
                // substitution / deletion / insertion
                var d = prevDist[j - 1] + cost
                var s = prevStart[j - 1]
                let del = prevDist[j] + 1
                if del < d { d = del; s = prevStart[j] }
                let ins = currDist[j - 1] + 1
                if ins < d { d = ins; s = currStart[j - 1] }
                // Damerau adjacent transposition (R6): "ie" ↔ "ei"
                // costs one edit instead of two.
                if i > 1, j > 1,
                   q[i - 1] == t[j - 2], q[i - 2] == t[j - 1] {
                    let tr = prevPrevDist[j - 2] + 1
                    if tr < d { d = tr; s = prevPrevStart[j - 2] }
                }
                currDist[j] = d
                currStart[j] = s
                if d < rowMin { rowMin = d }
            }
            if rowMin > threshold { return nil }
            prevPrevDist = prevDist
            prevPrevStart = prevStart
            swap(&prevDist, &currDist)
            swap(&prevStart, &currStart)
        }
        for j in 1...n where prevDist[j] < bestDist {
            bestDist = prevDist[j]
            bestRange = prevStart[j]..<j
        }
        return bestRange
    }
}
