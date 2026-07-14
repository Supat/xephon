import Foundation
import Testing
import Fusion
@testable import EvalFormPlugin

/// Pins the deterministic tier of the eval-form pipeline — the
/// spoken-score grammar, candidate matching, the merge policy, and
/// the pack/draft codables. These carry the anti-hallucination
/// guarantees (stated-only, quantization gate, evidence validity),
/// so every rule gets an explicit fixture. LLM-tier behavior is
/// covered at the prompt/parse level; live generation is evaluated
/// on device against ground-truth forms (research doc §6).
@Suite("EvalForm deterministic pipeline")
struct EvalFormPluginTests {

    private let template = EvalFormTemplate.a1StraightRoad
    private var allowed: [Double] { template.strengthScale.allowedValues }

    private func utterance(_ text: String, speaker: String = "S01") -> UtteranceEstimate {
        UtteranceEstimate(
            speakerID: speaker,
            start: 0, end: 1,
            transcript: text,
            asrConfidence: 0.9,
            dimensional: nil,
            acousticCategorical: nil,
            plutchik: nil,
            fusedValence: nil,
            fusedArousal: nil,
            fusedDominance: nil,
            fusedTopLabel: nil
        )
    }

    // MARK: - Spoken-score grammar

    @Test func statedScoreCapturesSignedQuantizedValues() {
        #expect(values("ヒョコヒョコはマイナス0.25かな") == [-0.25])
        #expect(values("プラス0.5ぐらいだね") == [0.5])
        #expect(values("−0.125だと思う") == [-0.125])
        #expect(values("これはプラス1だ") == [1.0])
    }

    @Test func statedScoreFoldsFullWidthDigits() {
        #expect(values("マイナス０．５ですね") == [-0.5])
    }

    @Test func longVowelMarkReadsAsMinusBeforeDigits() {
        // ASR frequently renders a spoken マイナス as the long-vowel
        // mark directly before the number.
        #expect(values("ー0.25ぐらい") == [-0.25])
    }

    @Test func plusMinusZeroFamilyMapsToZero() {
        #expect(values("プラマイゼロかな") == [0])
        #expect(values("±0ですね") == [0])
        #expect(values("プラスマイナスゼロ") == [0])
    }

    @Test func unsignedNumbersAreNotScores() {
        // Speeds, road names, counts — never sheet entries.
        #expect(values("F路60キロで行きます") == [])
        #expect(values("0.5ぐらい違う") == [])
    }

    @Test func offScaleValuesAreRejected() {
        // Not a legal 0.125 step → conversation, not a score.
        #expect(values("マイナス0.3ぐらいかな") == [])
        #expect(values("マイナス1.5だね") == [])
    }

    @Test func multipleScoresInOneUtteranceAllCapture() {
        #expect(values("最初マイナス0.5だったけど今はマイナス0.25") == [-0.5, -0.25])
    }

    @Test func preferenceNeedsBothDigitTenAndPreferenceWord() {
        #expect(preference("これは好きだね、7点") == 7)
        #expect(preference("嫌いだな、2点ぐらい") == 2)
        #expect(preference("7点ですね") == nil)          // no 好き/嫌い
        #expect(preference("これは好きだね") == nil)      // no digit+点
    }

    private func values(_ text: String) -> [Double] {
        SpokenScoreParser.statedStrengths(in: text, allowedValues: allowed)
            .map(\.value)
    }

    private func preference(_ text: String) -> Int? {
        SpokenScoreParser.statedPreference(in: text, minimum: 1, maximum: 9)
    }

    // MARK: - Candidate matching

    @Test func candidateRowsMatchVocabularyAcrossKanaForms() {
        let utterances = [
            utterance("ヒョコヒョコが気になるね"),           // katakana
            utterance("ちょっとひょこひょこするね"),         // hiragana variant
            utterance("今日はいい天気だ"),                   // unrelated
            utterance("ゴツゴツ感が強い"),                   // other item
        ]
        let hyoko = template.items.first { $0.id == "11_hyokohyoko" }!
        #expect(EvalFormExtractor.candidateRowNumbers(
            for: hyoko, utterances: utterances
        ) == [1, 2])
    }

    // MARK: - Merge policy

    @Test func singleStatedScoreWinsWithoutConflict() {
        let item = template.items[1]
        let det = EvalFormExtractor.DeterministicFindings(
            statedScores: [(row: 5, value: -0.25)],
            statedPreference: nil
        )
        let merged = EvalFormExtractor.merge(
            item: item, template: template,
            deterministic: det, wire: nil, validRowNumbers: [5]
        )
        #expect(merged.strengthScore == -0.25)
        #expect(merged.conflicts.isEmpty)
        #expect(merged.evidenceRows == [5])
    }

    @Test func revisedScoreTakesLastAndFlagsTrail() {
        let item = template.items[1]
        let det = EvalFormExtractor.DeterministicFindings(
            statedScores: [(row: 5, value: -0.5), (row: 9, value: -0.25)],
            statedPreference: nil
        )
        let merged = EvalFormExtractor.merge(
            item: item, template: template,
            deterministic: det, wire: nil, validRowNumbers: [5, 9]
        )
        #expect(merged.strengthScore == -0.25)
        #expect(merged.conflicts.count == 1)
        #expect(merged.evidenceRows == [5, 9])
    }

    @Test func modelStatedScoreOnlyFillsGapsAndMismatchIsFlagged() {
        let item = template.items[1]
        // Gap fill: deterministic empty, model stated legal.
        let gap = EvalFormExtractor.merge(
            item: item, template: template,
            deterministic: .init(statedScores: [], statedPreference: nil),
            wire: .init(statedScore: -0.5, inferredScore: nil,
                        likeDislike: nil, comment: "c", evidenceRows: [3]),
            validRowNumbers: [3]
        )
        #expect(gap.strengthScore == -0.5)

        // Mismatch: deterministic wins, conflict recorded.
        let clash = EvalFormExtractor.merge(
            item: item, template: template,
            deterministic: .init(statedScores: [(row: 2, value: -0.25)], statedPreference: nil),
            wire: .init(statedScore: -0.75, inferredScore: nil,
                        likeDislike: nil, comment: nil, evidenceRows: [2]),
            validRowNumbers: [2]
        )
        #expect(clash.strengthScore == -0.25)
        #expect(clash.conflicts.count == 1)
    }

    @Test func inferredScoreGatedByStatedAndQuantization() {
        let item = template.items[1]
        // Accepted: nothing stated, legal step.
        let ok = EvalFormExtractor.merge(
            item: item, template: template,
            deterministic: .init(statedScores: [], statedPreference: nil),
            wire: .init(statedScore: nil, inferredScore: -0.25,
                        likeDislike: nil, comment: nil, evidenceRows: []),
            validRowNumbers: []
        )
        #expect(ok.strengthScoreInferred == -0.25)

        // Rejected: off-scale suggestion.
        let offScale = EvalFormExtractor.merge(
            item: item, template: template,
            deterministic: .init(statedScores: [], statedPreference: nil),
            wire: .init(statedScore: nil, inferredScore: -0.3,
                        likeDislike: nil, comment: nil, evidenceRows: []),
            validRowNumbers: []
        )
        #expect(offScale.strengthScoreInferred == nil)

        // Suppressed: a stated score exists.
        let suppressed = EvalFormExtractor.merge(
            item: item, template: template,
            deterministic: .init(statedScores: [(row: 1, value: -0.5)], statedPreference: nil),
            wire: .init(statedScore: nil, inferredScore: -0.25,
                        likeDislike: nil, comment: nil, evidenceRows: [1]),
            validRowNumbers: [1]
        )
        #expect(suppressed.strengthScoreInferred == nil)
    }

    @Test func evidenceRowsFilterToValidNumbers() {
        let item = template.items[0]
        let merged = EvalFormExtractor.merge(
            item: item, template: template,
            deterministic: .init(statedScores: [], statedPreference: nil),
            wire: .init(statedScore: nil, inferredScore: nil, likeDislike: nil,
                        comment: "c", evidenceRows: [2, 99]),  // 99 invented
            validRowNumbers: [1, 2, 3]
        )
        #expect(merged.evidenceRows == [2])
    }

    // MARK: - Response parsing

    @Test func responseParserTolerantOfFencesAndProse() {
        let fenced = """
        Sure, here's the JSON:
        ```json
        {"statedScore":-0.25,"inferredScore":null,"likeDislike":null,"comment":"収まりが悪い","evidenceRows":[4]}
        ```
        """
        let wire = EvalFormExtractor.parseItemResponse(fenced)
        #expect(wire?.statedScore == -0.25)
        #expect(wire?.evidenceRows == [4])
        #expect(EvalFormExtractor.parseItemResponse("no json here") == nil)
    }

    // MARK: - Codables

    @Test func templatePackRoundTripsThroughJSON() throws {
        let data = try JSONEncoder().encode(template)
        let decoded = try EvalFormTemplate.decode(data)
        #expect(decoded == template)
    }

    @Test func draftRoundTripsThroughJSON() throws {
        var draft = EvalFormDraft(templateID: template.id, generatedAtUtterancesVersion: 7)
        draft.items = [
            .init(itemID: "11_hyokohyoko", strengthScore: -0.25,
                  comment: "収まり悪い", evidenceRows: [3, 5])
        ]
        draft.metadata = ["評価車両": "ティグアン"]
        let decoded = try EvalFormDraft.decode(try draft.encoded())
        #expect(decoded == draft)
    }
}
