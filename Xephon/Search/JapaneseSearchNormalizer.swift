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
    /// Normalized form suitable for substring search. Empty input
    /// returns "" so callers can compare against it directly.
    static func normalize(_ input: String) -> String {
        guard !input.isEmpty else { return "" }
        let mutable = NSMutableString(string: input) as CFMutableString
        let range = CFRangeMake(0, CFStringGetLength(mutable))
        let tokenizer = CFStringTokenizerCreate(
            kCFAllocatorDefault,
            mutable,
            range,
            kCFStringTokenizerUnitWordBoundary,
            Locale(identifier: "ja") as CFLocale
        )

        var out = ""
        var type = CFStringTokenizerAdvanceToNextToken(tokenizer)
        while type != [] {
            if let latin = CFStringTokenizerCopyCurrentTokenAttribute(
                tokenizer,
                kCFStringTokenizerAttributeLatinTranscription
            ) as? String {
                out.append(latin)
            } else {
                // Token had no latin attribute (e.g. punctuation,
                // a digit run, or ASCII that the tokenizer doesn't
                // re-transcribe). Append the raw token text so
                // numbers and existing romaji still participate in
                // matching.
                let tokenRange = CFStringTokenizerGetCurrentTokenRange(tokenizer)
                if tokenRange.length > 0 {
                    let nsRange = NSRange(location: tokenRange.location, length: tokenRange.length)
                    out.append((input as NSString).substring(with: nsRange))
                }
            }
            type = CFStringTokenizerAdvanceToNextToken(tokenizer)
        }
        return out
            .lowercased()
            .components(separatedBy: .whitespacesAndNewlines)
            .joined()
    }
}
