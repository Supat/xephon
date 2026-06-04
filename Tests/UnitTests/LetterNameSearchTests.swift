import Foundation
import Testing
@testable import Xephon

/// Behavioral pinning for single Latin-letter search expansion.
///
/// A one-letter query ("G") must surface utterances containing the
/// letter's spoken Japanese reading (ジー / じー) and a literal "G"/"g",
/// matched at token boundaries — without the romaji-substring
/// explosion that a bare "g" → が / ご would otherwise produce. These
/// tests exercise the pure query layer (`NormalizedSearchQuery` +
/// `LatinLetterReadings`) against the real `JapaneseSearchNormalizer`.
@Suite("Single-letter (letter-name) search")
struct LetterNameSearchTests {

    /// Does `query` match `transcript` under the production matcher?
    private func matches(_ query: String, _ transcript: String) -> Bool {
        let q = NormalizedSearchQuery.build(from: query)
        let tokens = JapaneseSearchNormalizer.normalizedTokens(transcript)
        return q.matches(normalized: tokens.joined(), tokens: tokens)
    }

    @Test("Katakana spoken reading surfaces")
    func katakanaReading() {
        #expect(matches("G", "それはジーです"))
        #expect(matches("g", "それはジーです")) // case-insensitive
    }

    @Test("Hiragana spoken reading surfaces at a clean boundary")
    func hiraganaReading() {
        #expect(matches("G", "これはじー"))
    }

    @Test("Literal isolated letter surfaces")
    func literalLetter() {
        #expect(matches("G", "5Gの話"))
    }

    @Test("Reading buried in a loanword does NOT match (whole-token)")
    func loanwordExcluded() {
        #expect(!matches("G", "ジーパンを買った")) // jeans, not the letter
    }

    @Test("Unrelated romaji 'g' does NOT match (no substring explosion)")
    func noSubstringExplosion() {
        #expect(!matches("G", "ありがとうございます")) // arigatō contains 'g'
        #expect(!matches("G", "Googleで検索"))         // gūguru
    }

    @Test("Multi-token readings match regardless of context splitting")
    func multiTokenReadings() {
        #expect(matches("H", "これはエイチです")) // ei+chi vs eichi
        #expect(matches("F", "エフですか"))
        #expect(matches("N", "エヌが好き"))
        #expect(matches("J", "ジェーリーグ"))
    }

    @Test("Every letter's katakana reading surfaces in a sentence")
    func everyLetter() {
        for (letter, reading) in LatinLetterReadings.katakana {
            #expect(
                matches(String(letter), "それは\(reading)だ"),
                "letter \(letter) reading \(reading) should match"
            )
        }
    }

    @Test("Multi-character queries keep ordinary cross-script substring behavior")
    func multiCharUnaffected() {
        #expect(matches("渋谷", "昨日は渋谷に行った"))
        #expect(matches("しぶや", "昨日はシブヤに行った")) // cross-script
        #expect(matches("ABC", "ABCを見た"))
        #expect(!matches("渋谷", "今日は良い天気"))
    }

    @Test("Empty / whitespace query matches nothing")
    func emptyQuery() {
        #expect(NormalizedSearchQuery.build(from: "").isEmpty)
        #expect(NormalizedSearchQuery.build(from: "   ").isEmpty)
    }

    @Test("Fullwidth letter folds to ASCII behavior")
    func fullwidthLetter() {
        #expect(matches("Ｇ", "それはジーです"))
    }

    @Test("Token-aligned run helper is boundary-exact")
    func tokenAlignedRun() {
        #expect(NormalizedSearchQuery.containsTokenAlignedRun(["ei", "chi"], "eichi"))
        #expect(NormalizedSearchQuery.containsTokenAlignedRun(["kore", "ha", "jī"], "jī"))
        #expect(!NormalizedSearchQuery.containsTokenAlignedRun(["jīpan"], "jī"))
        #expect(!NormalizedSearchQuery.containsTokenAlignedRun([], "jī"))
        #expect(!NormalizedSearchQuery.containsTokenAlignedRun(["jī"], ""))
    }
}
