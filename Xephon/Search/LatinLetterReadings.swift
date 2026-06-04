import Foundation

/// Spoken letter-name readings for the 26 Latin letters, in
/// Japanese katakana (e.g. G → ジー). Used to expand a single-letter
/// search query so it also surfaces utterances where the letter was
/// *spoken aloud* and transcribed phonetically — searching "G" finds
/// ジー / じー, not just a literal "G" / "g".
///
/// Why both katakana and a derived hiragana form (see
/// `readingForms(for:)`): `CFStringTokenizer` does NOT romanize the
/// two identically for every letter. Most collapse (ビー / びー both →
/// "bī"), but a long mark after a voiced hiragana doesn't merge into
/// a macron the way it does after katakana — じー tokenizes to
/// ["ji", "ー"] while ジー tokenizes to ["jī"]. Carrying both forms,
/// matched as token *sequences* rather than a single joined string,
/// covers that asymmetry and also covers readings that are inherently
/// multi-token (F → ["e", "fu"], H → ["ei", "chi"], J → ["je", "ー"],
/// N → ["e", "nu"]).
enum LatinLetterReadings {
    /// Canonical katakana reading per lowercase Latin letter.
    static let katakana: [Character: String] = [
        "a": "エー", "b": "ビー", "c": "シー", "d": "ディー",
        "e": "イー", "f": "エフ", "g": "ジー", "h": "エイチ",
        "i": "アイ", "j": "ジェー", "k": "ケー", "l": "エル",
        "m": "エム", "n": "エヌ", "o": "オー", "p": "ピー",
        "q": "キュー", "r": "アール", "s": "エス", "t": "ティー",
        "u": "ユー", "v": "ブイ", "w": "ダブリュー", "x": "エックス",
        "y": "ワイ", "z": "ゼット",
    ]

    /// Surface forms of a letter's spoken reading to expand a query
    /// into: the katakana form plus its hiragana transliteration.
    /// Deduped by the caller (most letters' two forms normalize to
    /// the same token sequence). Empty if `letter` isn't A–Z.
    static func readingForms(for letter: Character) -> [String] {
        guard let kata = katakana[letter] else { return [] }
        let hira = katakanaToHiragana(kata)
        return hira == kata ? [kata] : [kata, hira]
    }

    /// Katakana → hiragana by direct scalar shift (U+30A1…U+30F6 →
    /// U+3041…U+3096). Done by hand rather than via
    /// `StringTransform.hiraganaToKatakana`, which "normalizes" the
    /// prolonged-sound mark by expanding ー into a vowel (ジー → じい),
    /// destroying the very long-vowel form the user types as じー.
    /// The shift leaves ー (U+30FC) — and anything outside the kana
    /// block — untouched, so ジー → じー exactly.
    private static func katakanaToHiragana(_ s: String) -> String {
        String(String.UnicodeScalarView(s.unicodeScalars.map { scalar in
            (0x30A1...0x30F6).contains(scalar.value)
                ? (Unicode.Scalar(scalar.value - 0x60) ?? scalar)
                : scalar
        }))
    }

    /// If `query` is exactly one Latin letter — ASCII or fullwidth,
    /// any case, ignoring surrounding whitespace — return it folded
    /// to a lowercase ASCII letter. Otherwise nil. Fullwidth Ａ–Ｚ /
    /// ａ–ｚ are folded so a query typed on a Japanese full-width
    /// keyboard behaves the same as the ASCII form.
    static func singleLetter(in query: String) -> Character? {
        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.count == 1,
              let scalar = trimmed.unicodeScalars.first else {
            return nil
        }
        let folded: UInt32
        switch scalar.value {
        case 0xFF21...0xFF3A, 0xFF41...0xFF5A: // fullwidth A–Z / a–z
            folded = scalar.value - 0xFEE0
        default:
            folded = scalar.value
        }
        switch folded {
        case 0x41...0x5A: // A–Z
            return Character(Unicode.Scalar(folded + 0x20)!)
        case 0x61...0x7A: // a–z
            return Character(Unicode.Scalar(folded)!)
        default:
            return nil
        }
    }
}
