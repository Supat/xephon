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
        tokens(input, allowRetranscription: true)
    }

    private static func tokens(
        _ input: String,
        allowRetranscription: Bool
    ) -> [Token] {
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
            var normalized = raw
                // R2: NFKC compatibility fold BEFORE casing — folds
                // full-width ASCII (ＡＢＣ１２３ → ABC123) and
                // half-width katakana to their canonical forms so
                // pasted/typed text can't silently miss. Applied to
                // the per-token OUTPUT (not the tokenizer input) so
                // `originalRange` keeps indexing the caller's string.
                .precomposedStringWithCompatibilityMapping
                .lowercased()
                .components(separatedBy: .whitespacesAndNewlines)
                .joined()
            // Half-width katakana chunks carry no latin attribute
            // (the tokenizer treats them as symbols), so the fold
            // above leaves KANA in the normalized stream where
            // every other path produces romaji. One nested pass
            // over the folded text picks up the transcription.
            // HARD depth limit of one: the old "the fold changed
            // something" equality guard couldn't terminate a
            // fold/transcription 1-cycle — for 、 the tokenizer's
            // latin attribute yields a variant form that NFKC folds
            // straight back, so tokens("、") recursed on itself
            // until the main thread froze (seen on-device, ~1800
            // frames). One level is always sufficient: fold → kana
            // → latin/ASCII.
            if allowRetranscription,
               normalized != raw.lowercased()
                .components(separatedBy: .whitespacesAndNewlines).joined(),
               normalized.contains(where: { !$0.isASCII }) {
                let retranscribed = tokens(normalized, allowRetranscription: false)
                    .map(\.normalized).joined()
                if !retranscribed.isEmpty { normalized = retranscribed }
            }
            normalized = Self.canonicalizeRomaji(normalized)
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

    /// R1: fold romaji spelling variants onto the tokenizer's
    /// Hepburn-style output so typed Kunrei/Nihon-shiki queries
    /// (`si`, `ti`, `tu`, `sya`…) and macron long vowels exact-match
    /// instead of burning Levenshtein budget on systematic spelling
    /// variance. Applied to EVERY normalized token — including plain
    /// English passthrough, where it "corrupts" symmetrically
    /// (query and transcript see the same fold, so English↔English
    /// matching is unaffected; the normalized string is never
    /// displayed). Order matters: trigraph digraphs before the
    /// two-letter rules that would otherwise bite their substrings.
    static func canonicalizeRomaji(_ input: String) -> String {
        // Fast path: pure-romaji output without any of the trigger
        // characters skips the replacement cascade.
        guard input.contains(where: { "āīūēōâîûêôsyzjtdh".contains($0) }) else {
            return input
        }
        var s = input
        let rules: [(String, String)] = [
            // Macron / circumflex long vowels → kana-spelling style.
            ("ā", "aa"), ("ī", "ii"), ("ū", "uu"), ("ē", "ee"), ("ō", "ou"),
            ("â", "aa"), ("î", "ii"), ("û", "uu"), ("ê", "ee"), ("ô", "ou"),
            // Kunrei palatalized rows first (contain the two-letter
            // patterns below as substrings).
            ("sya", "sha"), ("syu", "shu"), ("syo", "sho"),
            ("tya", "cha"), ("tyu", "chu"), ("tyo", "cho"),
            ("zya", "ja"), ("zyu", "ju"), ("zyo", "jo"),
            ("jya", "ja"), ("jyu", "ju"), ("jyo", "jo"),
            ("dya", "ja"), ("dyu", "ju"), ("dyo", "jo"),
            // Kunrei base rows.
            ("si", "shi"), ("ti", "chi"), ("tu", "tsu"),
            ("hu", "fu"), ("zi", "ji"), ("di", "ji"), ("du", "zu"),
        ]
        for (from, to) in rules where s.contains(from) {
            s = s.replacingOccurrences(of: from, with: to)
        }
        return s
    }
}
