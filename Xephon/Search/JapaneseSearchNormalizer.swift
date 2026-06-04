import Foundation

/// Cross-script search normalization for Japanese text.
///
/// Converts any mix of kanji, hiragana, katakana, or romaji into a
/// single comparable form — Hepburn-style romaji, lowercased, with
/// whitespace stripped — using `CFStringTokenizer`'s latin
/// transcription attribute. The tokenizer consults the same reading
/// dictionary the system IME uses, so:
///
/// - 渋谷 → "shibuya"
/// - しぶや → "shibuya"
/// - シブヤ → "shibuya"
/// - Shibuya → "shibuya"
///
/// All four collapse to the same key, and substring matching on the
/// normalized form is symmetric across scripts.
///
/// Caveat: kanji with multiple readings produce the tokenizer's
/// best contextual guess (usually the most common reading); rare
/// proper-noun readings can miss.
enum JapaneseSearchNormalizer {
    /// One CFStringTokenizer chunk with the original character
    /// range it covered and the lowercased, whitespace-stripped
    /// latin transcription it produced. Exposed so callers that
    /// need to map a position in the concatenated normalized form
    /// back to an original range (e.g. the find-and-replace
    /// highlighter for fuzzy / cross-script hits) can walk the
    /// chunks rather than re-tokenizing or guessing.
    struct Token: Sendable, Hashable {
        let originalRange: Range<String.Index>
        let normalized: String
    }

    /// Normalized form suitable for substring search. Empty input
    /// returns "" so callers can compare against it directly.
    static func normalize(_ input: String) -> String {
        tokens(input).map(\.normalized).joined()
    }

    /// Per-token normalized forms, in order. The join of this array
    /// equals `normalize(input)`; whole-token / token-sequence
    /// matchers want the array (`NormalizedSearchQuery`'s token mode)
    /// while substring matchers want the join, so callers that need
    /// both compute this once and join when required.
    static func normalizedTokens(_ input: String) -> [String] {
        tokens(input).map(\.normalized)
    }

    /// Per-chunk view of `normalize(input)`. The concatenation of
    /// the returned tokens' `normalized` strings equals
    /// `normalize(input)` byte-for-byte. Tokens whose chunk
    /// produced no characters after lowercasing and whitespace
    /// stripping are dropped so the offsets stay tight.
    static func tokens(_ input: String) -> [Token] {
        guard !input.isEmpty else { return [] }
        let nsInput = input as NSString
        let mutable = NSMutableString(string: input) as CFMutableString
        let range = CFRangeMake(0, CFStringGetLength(mutable))
        let tokenizer = CFStringTokenizerCreate(
            kCFAllocatorDefault,
            mutable,
            range,
            kCFStringTokenizerUnitWordBoundary,
            Locale(identifier: "ja") as CFLocale
        )

        var out: [Token] = []
        var type = CFStringTokenizerAdvanceToNextToken(tokenizer)
        while type != [] {
            let tokenRange = CFStringTokenizerGetCurrentTokenRange(tokenizer)
            let raw: String
            if let latin = CFStringTokenizerCopyCurrentTokenAttribute(
                tokenizer,
                kCFStringTokenizerAttributeLatinTranscription
            ) as? String {
                raw = latin
            } else if tokenRange.length > 0 {
                // No latin attribute (punctuation, digit run, or
                // ASCII the tokenizer doesn't retranscribe).
                // Reuse the raw chunk so numbers and existing
                // romaji still participate in matching.
                let nsRange = NSRange(
                    location: tokenRange.location,
                    length: tokenRange.length
                )
                raw = nsInput.substring(with: nsRange)
            } else {
                raw = ""
            }
            let normalized = raw
                .lowercased()
                .components(separatedBy: .whitespacesAndNewlines)
                .joined()
            if !normalized.isEmpty, tokenRange.length > 0 {
                let nsRange = NSRange(
                    location: tokenRange.location,
                    length: tokenRange.length
                )
                if let origRange = Range(nsRange, in: input) {
                    out.append(Token(
                        originalRange: origRange,
                        normalized: normalized
                    ))
                }
            }
            type = CFStringTokenizerAdvanceToNextToken(tokenizer)
        }
        return out
    }
}
