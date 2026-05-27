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
        guard !query.isEmpty, !text.isEmpty, threshold >= 0 else { return false }
        let qChars = Array(query)
        let tChars = Array(text)
        let qLen = qChars.count
        let tLen = tChars.count
        let minW = max(1, qLen - threshold)
        let maxW = min(qLen + threshold, tLen)
        if tLen < minW {
            return levenshtein(qChars, tChars, threshold: threshold) <= threshold
        }
        for w in minW...maxW {
            let lastStart = tLen - w
            if lastStart < 0 { continue }
            var start = 0
            while start <= lastStart {
                let slice = Array(tChars[start..<(start + w)])
                if levenshtein(qChars, slice, threshold: threshold) <= threshold {
                    return true
                }
                start += 1
            }
        }
        return false
    }

    /// Bounded Levenshtein. Returns `threshold + 1` (i.e. "too far")
    /// as soon as the row minimum exceeds `threshold`, so the caller
    /// can early-exit without a wrong-sentinel ambiguity. Standard
    /// two-row DP otherwise.
    private static func levenshtein(
        _ a: [Character],
        _ b: [Character],
        threshold: Int
    ) -> Int {
        let m = a.count
        let n = b.count
        if abs(m - n) > threshold { return threshold + 1 }
        if m == 0 { return n }
        if n == 0 { return m }
        var prev = Array(0...n)
        var curr = Array(repeating: 0, count: n + 1)
        for i in 1...m {
            curr[0] = i
            var rowMin = curr[0]
            for j in 1...n {
                let cost = a[i - 1] == b[j - 1] ? 0 : 1
                let del = prev[j] + 1
                let ins = curr[j - 1] + 1
                let sub = prev[j - 1] + cost
                let v = min(del, min(ins, sub))
                curr[j] = v
                if v < rowMin { rowMin = v }
            }
            if rowMin > threshold { return threshold + 1 }
            swap(&prev, &curr)
        }
        return prev[n]
    }
}
