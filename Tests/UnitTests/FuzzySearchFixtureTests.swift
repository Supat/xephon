import Foundation
import Testing
@testable import Xephon

/// Labeled fixtures for the fuzzy-search stack (fuzzy-search audit
/// R7): every recall/precision-shifting change to the normalizer,
/// phonetic key, or matcher must keep these green — they pin the
/// behaviors the audit implementations (R1–R6) added, plus the
/// false-positive traps they must not reopen. Pure-layer tests only
/// (normalizer / PhoneticKey / FuzzySubstringMatcher /
/// NormalizedSearchQuery); no RecordingController.
@Suite("Fuzzy-search fixtures")
struct FuzzySearchFixtureTests {

    private func matches(_ query: String, _ transcript: String) -> Bool {
        let q = NormalizedSearchQuery.build(from: query)
        let tokens = JapaneseSearchNormalizer.normalizedTokens(transcript)
        return q.matches(normalized: tokens.joined(), tokens: tokens)
    }

    // MARK: R1 — romaji canonicalization

    @Test("Kunrei digraphs fold to Hepburn")
    func kunreiFolds() {
        #expect(JapaneseSearchNormalizer.canonicalizeRomaji("sinbun") == "shinbun")
        #expect(JapaneseSearchNormalizer.canonicalizeRomaji("tizu") == "chizu")
        #expect(JapaneseSearchNormalizer.canonicalizeRomaji("tuki") == "tsuki")
        #expect(JapaneseSearchNormalizer.canonicalizeRomaji("huzi") == "fuji")
        #expect(JapaneseSearchNormalizer.canonicalizeRomaji("syashin") == "shashin")
        #expect(JapaneseSearchNormalizer.canonicalizeRomaji("zyouhou") == "jouhou")
    }

    @Test("Macron long vowels fold to kana spelling")
    func macronFolds() {
        #expect(JapaneseSearchNormalizer.canonicalizeRomaji("tōkyō") == "toukyou")
        #expect(JapaneseSearchNormalizer.canonicalizeRomaji("kūkō") == "kuukou")
    }

    @Test("Kunrei-typed query matches kana transcript")
    func kunreiQueryCrossScript() {
        #expect(matches("sinbun", "しんぶんを読んだ"))
    }

    @Test("Hepburn output untouched")
    func hepburnStable() {
        #expect(JapaneseSearchNormalizer.canonicalizeRomaji("shinbun") == "shinbun")
        #expect(JapaneseSearchNormalizer.canonicalizeRomaji("chotto") == "chotto")
        #expect(JapaneseSearchNormalizer.canonicalizeRomaji("tsunami") == "tsunami")
    }

    // MARK: R2 — compatibility folding

    @Test("Full-width ASCII folds")
    func fullWidthASCII() {
        #expect(JapaneseSearchNormalizer.normalize("ＡＢＣ１２３") == "abc123")
    }

    @Test("Half-width katakana matches its full-width reading")
    func halfWidthKatakana() {
        #expect(
            JapaneseSearchNormalizer.normalize("ｼﾌﾞﾔ")
                == JapaneseSearchNormalizer.normalize("シブヤ")
        )
    }

    // MARK: R3 — phonetic key

    @Test("Key collapses ASR confusion classes")
    func keyEquivalences() {
        // long/short vowel
        #expect(PhoneticKey.key(fromCanonicalRomaji: "toukyou")
            == PhoneticKey.key(fromCanonicalRomaji: "tokyo"))
        // geminate presence
        #expect(PhoneticKey.key(fromCanonicalRomaji: "kitte")
            == PhoneticKey.key(fromCanonicalRomaji: "kite"))
        // voicing
        #expect(PhoneticKey.key(fromCanonicalRomaji: "kagi")
            == PhoneticKey.key(fromCanonicalRomaji: "kaki"))
        // n/m assimilation
        #expect(PhoneticKey.key(fromCanonicalRomaji: "shimbun")
            == PhoneticKey.key(fromCanonicalRomaji: "shinbun"))
        // ei long vowel
        #expect(PhoneticKey.key(fromCanonicalRomaji: "sensei")
            == PhoneticKey.key(fromCanonicalRomaji: "sense"))
    }

    @Test("Key keeps genuinely different words apart")
    func keyDiscrimination() {
        #expect(PhoneticKey.key(fromCanonicalRomaji: "kasa")
            != PhoneticKey.key(fromCanonicalRomaji: "kani"))
        #expect(PhoneticKey.key(fromCanonicalRomaji: "shiryou")
            != PhoneticKey.key(fromCanonicalRomaji: "shorui"))
        #expect(PhoneticKey.key(fromCanonicalRomaji: "hana")
            != PhoneticKey.key(fromCanonicalRomaji: "hara"))
    }

    // MARK: R4/R6 — semi-global matcher with transpositions

    @Test("Semi-global finds an embedded near-match with its range")
    func semiGlobalRange() {
        let range = FuzzySubstringMatcher.findSimilarSubstring(
            query: "shiryou", in: "kaiginoshiryoudesu", threshold: 1
        )
        #expect(range != nil)
        if let range {
            let t = Array("kaiginoshiryoudesu")
            let hit = String(t[range])
            #expect(hit.contains("shiryo"))
        }
    }

    @Test("One-edit miss inside a longer text is found")
    func oneEditEmbedded() {
        // "shiryou" vs embedded "shiryuu" — single substitution.
        #expect(FuzzySubstringMatcher.hasSimilarSubstring(
            query: "shiryou", in: "sonoshiryuuwo", threshold: 1
        ))
    }

    @Test("Adjacent transposition costs one edit (Damerau)")
    func damerauTransposition() {
        #expect(FuzzySubstringMatcher.hasSimilarSubstring(
            query: "taberu", in: "tabreu", threshold: 1
        ))
        // The classic Levenshtein cost would be 2 — pin that a
        // tighter threshold still rejects a DOUBLE transposition.
        #expect(!FuzzySubstringMatcher.hasSimilarSubstring(
            query: "taberu", in: "atbred", threshold: 1
        ))
    }

    @Test("Far-off strings stay rejected")
    func rejection() {
        #expect(!FuzzySubstringMatcher.hasSimilarSubstring(
            query: "shiryou", in: "kondonokaigi", threshold: 1
        ))
    }

    // MARK: false-positive traps (must stay red-flags)

    @Test("Short queries don't fuzzy-match everywhere")
    func shortQueryTrap() {
        // 2-char queries are below minQueryLengthForSimilar; pin the
        // constant so a threshold loosening is a conscious choice.
        #expect(SearchReplaceCoordinator.minQueryLengthForSimilar >= 4)
    }

    @Test("Unrelated words don't cross-script match")
    func unrelatedTrap() {
        #expect(!matches("かさ", "会議の資料です"))
    }
}
