import Foundation
import SwiftUI

/// Paints keyword tag colors over transcript text in utterance
/// rows. Only COLOR-TAGGED keywords highlight — untagged keywords
/// (the default) keep the row visually quiet and rely on the
/// occurrence badge in the Keywords card.
///
/// Two match passes per tagged keyword, mirroring the transcript
/// filter's semantics:
///  1. Raw case-insensitive substring — precise character ranges.
///  2. Cross-script (kanji ↔ kana ↔ romaji) via
///     `JapaneseSearchNormalizer`: scan the joined normalized form
///     and map hits back through the tokens' `originalRange`
///     offsets — the same reconstruction the find-and-replace
///     highlighter uses. Granularity is chunk-level (the tokenizer
///     exposes no intra-chunk correspondences), so a hit inside a
///     long loanword paints the whole word.
///
/// First-painted wins on overlap: keywords earlier in the user's
/// list take precedence, and within one keyword raw hits are
/// painted before cross-script hits.
///
/// Computed per row render, uncached — transcripts are short (a
/// conversational utterance) and `CFStringTokenizer` on them costs
/// tens of microseconds; the search sheet's match cards already
/// re-tokenize per render at equal or larger scale. Revisit with a
/// memo keyed on (utterancesVersion, keyword mutation) only if
/// scroll profiling ever names this.
enum KeywordHighlighter {
    /// Highlight ranges of `transcript`, tagged with their color.
    static func matches(
        transcript: String,
        keywords: [Keyword]
    ) -> [(range: Range<String.Index>, tag: KeywordTagColor)] {
        guard !transcript.isEmpty else { return [] }
        let tagged = keywords.filter { $0.tagColor != nil }
        guard !tagged.isEmpty else { return [] }

        var results: [(range: Range<String.Index>, tag: KeywordTagColor)] = []
        // Tokenization deferred until a cross-script pass actually
        // needs it (raw-only sessions skip the tokenizer entirely).
        var tokens: [JapaneseSearchNormalizer.Token]?

        for keyword in tagged {
            guard let tag = keyword.tagColor else { continue }
            let term = keyword.text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !term.isEmpty else { continue }

            // Pass 1 — raw case-insensitive occurrences.
            var cursor = transcript.startIndex
            while cursor < transcript.endIndex,
                  let hit = transcript.range(
                    of: term,
                    options: [.caseInsensitive],
                    range: cursor..<transcript.endIndex
                  ) {
                if !results.contains(where: { $0.range.overlaps(hit) }) {
                    results.append((hit, tag))
                }
                cursor = hit.upperBound
            }

            // Pass 2 — cross-script hits in normalized space.
            let normTerm = JapaneseSearchNormalizer.normalize(term)
            guard !normTerm.isEmpty else { continue }
            if tokens == nil {
                tokens = JapaneseSearchNormalizer.tokens(transcript)
            }
            guard let toks = tokens, !toks.isEmpty else { continue }
            // Cumulative Character-offsets of each token's start in
            // the joined normalized string.
            var starts: [Int] = []
            starts.reserveCapacity(toks.count)
            var total = 0
            for t in toks {
                starts.append(total)
                total += t.normalized.count
            }
            let joined = toks.map(\.normalized).joined()
            var searchStart = joined.startIndex
            while searchStart < joined.endIndex,
                  let hit = joined.range(of: normTerm, range: searchStart..<joined.endIndex) {
                let s = joined.distance(from: joined.startIndex, to: hit.lowerBound)
                let e = joined.distance(from: joined.startIndex, to: hit.upperBound)
                searchStart = hit.upperBound
                guard let firstTok = toks.indices.last(where: { starts[$0] <= s }) else { continue }
                var lastTok = firstTok
                while lastTok + 1 < toks.count,
                      starts[lastTok] + toks[lastTok].normalized.count < e {
                    lastTok += 1
                }
                let original = toks[firstTok].originalRange.lowerBound
                    ..< toks[lastTok].originalRange.upperBound
                if !results.contains(where: { $0.range.overlaps(original) }) {
                    results.append((original, tag))
                }
            }
        }
        return results
    }

    /// `transcript` with each tagged-keyword hit painted as a
    /// low-opacity background run — reads as a highlighter pen
    /// without hurting text legibility in either color scheme.
    static func attributedTranscript(
        _ transcript: String,
        keywords: [Keyword]
    ) -> AttributedString {
        var attributed = AttributedString(transcript)
        for match in matches(transcript: transcript, keywords: keywords) {
            if let r = Range(match.range, in: attributed) {
                attributed[r].backgroundColor = match.tag.color.opacity(0.28)
            }
        }
        return attributed
    }
}
