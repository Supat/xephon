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

    // MARK: - Context window

    @Test func contextWindowPullsNeighborsAroundMentions() {
        let rows = [
            utterance("それでは行きます"),               // 1: context (−2)
            utterance("次の区間に入ります"),             // 2: context (−1)
            utterance("ヒョコヒョコが気になりますね"),   // 3: mention
            utterance("マイナス0.25ぐらいかな"),         // 4: context (+1)
            utterance("そうですね"),                     // 5: context (+2)
            utterance("今日は天気がいい"),               // 6: outside
        ]
        let hyoko = template.items.first { $0.id == "11_hyokohyoko" }!
        #expect(EvalFormExtractor.contextExpandedRows(
            for: hyoko, template: template, utterances: rows
        ) == [1, 2, 3, 4, 5])
    }

    @Test func contextRowAttachesToNearestItemOnly() {
        let rows = [
            utterance("ヒョコヒョコが強いですね"),       // 1: hyoko mention
            utterance("マイナス0.25ぐらいかな"),         // 2: 1 from hyoko, 2 from gotsu
            utterance("うん、そう思います"),             // 3: 2 from hyoko, 1 from gotsu
            utterance("ゴツゴツも見ておきましょう"),     // 4: gotsu mention
        ]
        let hyoko = template.items.first { $0.id == "11_hyokohyoko" }!
        let gotsu = template.items.first { $0.id == "13_gotsugotsu" }!
        // Row 2 is nearer hyoko; row 3 nearer gotsu; each other's
        // mention rows (distance 0 to their own item) never leak.
        #expect(EvalFormExtractor.contextExpandedRows(
            for: hyoko, template: template, utterances: rows
        ) == [1, 2])
        #expect(EvalFormExtractor.contextExpandedRows(
            for: gotsu, template: template, utterances: rows
        ) == [3, 4])
    }

    @Test func equidistantContextRowAttachesToBoth() {
        let rows = [
            utterance("ヒョコヒョコが強いですね"),       // 1: hyoko
            utterance("マイナス0.25ぐらいかな"),         // 2: tie (1 from each)
            utterance("ゴツゴツはどうでしょう"),         // 3: gotsu
        ]
        let hyoko = template.items.first { $0.id == "11_hyokohyoko" }!
        let gotsu = template.items.first { $0.id == "13_gotsugotsu" }!
        #expect(EvalFormExtractor.contextExpandedRows(
            for: hyoko, template: template, utterances: rows
        ) == [1, 2])
        #expect(EvalFormExtractor.contextExpandedRows(
            for: gotsu, template: template, utterances: rows
        ) == [2, 3])
    }

    @Test func statedScoreInFollowUpRowIsCaptured() {
        // The recall gap that motivated the window: keyword in one
        // ASR segment, the spoken score in the next.
        let rows = [
            utterance("ヒョコヒョコはどうですか"),
            utterance("マイナス0.25ぐらいですね"),
        ]
        let hyoko = template.items.first { $0.id == "11_hyokohyoko" }!
        let expanded = EvalFormExtractor.contextExpandedRows(
            for: hyoko, template: template, utterances: rows
        )
        let findings = EvalFormExtractor.deterministicFindings(
            candidateRows: expanded,
            utterances: rows,
            template: template
        )
        #expect(findings.statedScores.count == 1)
        #expect(findings.statedScores.first?.value == -0.25)
        #expect(findings.statedScores.first?.row == 2)
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

    @Test func promptCarriesRubricAndPolarityRule() {
        let item = template.items[1]
        let prompt = EvalFormExtractor.extractionPrompt(
            item: item,
            template: template,
            rows: [(number: 1, speakerID: "S01", transcript: "t")],
            deterministic: .init(statedScores: [], statedPreference: nil)
        )
        #expect(prompt.contains("POLARITY"))
        #expect(prompt.contains("まったく違う"))                 // rubric anchors injected
        #expect(prompt.contains("敏感な人が分かるレベル"))
    }

    @Test func inferredPolarityContradictionIsFlagged() {
        let item = template.items[1]
        // Evidence says STRONGER; model inferred the weak (+) side.
        let flipped = EvalFormExtractor.merge(
            item: item, template: template,
            deterministic: .init(statedScores: [], statedPreference: nil),
            wire: .init(statedScore: nil, inferredScore: 0.375,
                        likeDislike: nil, comment: "c", evidenceRows: [2]),
            validRowNumbers: [2],
            transcriptForRow: { _ in "ひょこひょこ感がかなり強い" }
        )
        #expect(flipped.conflicts.contains { $0.contains("極性要確認") })

        // Evidence says WEAKER/gone; inferred the strong (−) side.
        let flipped2 = EvalFormExtractor.merge(
            item: item, template: template,
            deterministic: .init(statedScores: [], statedPreference: nil),
            wire: .init(statedScore: nil, inferredScore: -0.5,
                        likeDislike: nil, comment: "c", evidenceRows: [2]),
            validRowNumbers: [2],
            transcriptForRow: { _ in "ビリビリ感はなくなっている" }
        )
        #expect(flipped2.conflicts.contains { $0.contains("極性要確認") })

        // Consistent sign → no flag.
        let consistent = EvalFormExtractor.merge(
            item: item, template: template,
            deterministic: .init(statedScores: [], statedPreference: nil),
            wire: .init(statedScore: nil, inferredScore: -0.5,
                        likeDislike: nil, comment: "c", evidenceRows: [2]),
            validRowNumbers: [2],
            transcriptForRow: { _ in "ひょこひょこ感がかなり強い" }
        )
        #expect(!consistent.conflicts.contains { $0.contains("極性要確認") })

        // Mixed direction words → ambiguous, no flag; and a STATED
        // score is never polarity-checked (evaluator's own words).
        let stated = EvalFormExtractor.merge(
            item: item, template: template,
            deterministic: .init(statedScores: [(row: 2, value: 0.25)], statedPreference: nil),
            wire: nil,
            validRowNumbers: [2],
            transcriptForRow: { _ in "強い" }
        )
        #expect(!stated.conflicts.contains { $0.contains("極性要確認") })
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

    @Test func responseParserAcceptsStringTypedNumbers() {
        // The quantized-model reality: numbers as strings, full-
        // width minus, string row numbers, "null" strings.
        let loose = """
        {"statedScore":"−0.25","inferredScore":"null","likeDislike":"7",
         "comment":"c","evidenceRows":["84","158"]}
        """
        let wire = EvalFormExtractor.parseItemResponse(loose)
        #expect(wire?.statedScore == -0.25)
        #expect(wire?.inferredScore == nil)
        #expect(wire?.likeDislike == 7)
        #expect(wire?.evidenceRows == [84, 158])
    }

    @Test func responseParserRepairsTruncatedOutput() {
        // Cut off mid-comment (token cap) — the repair closes the
        // string and braces; fields before the cut survive.
        let truncated = """
        {"statedScore":null,"inferredScore":-0.25,"likeDislike":null,
         "comment":"ゴツゴツ感が増えており、突き上げ
        """
        let wire = EvalFormExtractor.parseItemResponse(truncated)
        #expect(wire?.inferredScore == -0.25)
        #expect(wire?.comment?.hasPrefix("ゴツゴツ感") == true)

        // Cut off right after a key's colon → null-completed.
        let danglingKey = "{\"statedScore\":-0.5,\"inferredScore\":"
        let wire2 = EvalFormExtractor.parseItemResponse(danglingKey)
        #expect(wire2?.statedScore == -0.5)
        #expect(wire2?.inferredScore == nil)

        // Balanced garbage stays nil — repair is for truncation,
        // not for inventing structure.
        #expect(EvalFormExtractor.repairTruncatedJSON("{\"a\":1}") == nil)
        #expect(EvalFormExtractor.repairTruncatedJSON("{\"a\":1]") == nil)
    }

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
        draft.reviewedItemIDs = ["11_hyokohyoko"]
        let decoded = try EvalFormDraft.decode(try draft.encoded())
        #expect(decoded == draft)
    }

    // MARK: - Payload migration (v1 → v2)

    @Test func v1PayloadMigratesWithEmptyReviewState() {
        // A frozen v1 wire payload — no `reviewedItemIDs` key.
        let v1JSON = """
        {"templateID":"a1-straight-v1","generatedAtUtterancesVersion":7,
         "items":[{"itemID":"11_hyokohyoko","strengthScore":-0.25,
         "evidenceRows":[3],"conflicts":[]}],
         "metadata":{"評価車両":"ティグアン"}}
        """
        let restored = EvalFormDraft.restore(
            data: Data(v1JSON.utf8), storedVersion: 1
        )
        #expect(restored?.items.first?.strengthScore == -0.25)
        #expect(restored?.metadata["評価車両"] == "ティグアン")
        #expect(restored?.reviewedItemIDs == [])
    }

    @Test func newerPayloadVersionIsLeftAlone() throws {
        let draft = EvalFormDraft(templateID: template.id)
        let data = try draft.encoded()
        // A payload stamped by a FUTURE plugin build must not be
        // guess-decoded — nil means "show no draft, keep the bytes".
        #expect(EvalFormDraft.restore(data: data, storedVersion: 3) == nil)
        #expect(EvalFormDraft.restore(data: data, storedVersion: 2) != nil)
    }

    // MARK: - Road sections

    @Test func templateDerivesRoadNamesInFirstAppearanceOrder() {
        #expect(template.roadNames == [
            "E3路", "D路", "G路", "F路", "段差路", "H路", "スペイン歩道",
        ])
    }

    @Test func roadCalloutsSegmentTheSession() {
        let rows = [
            utterance("それでは始めます"),          // before any callout
            utterance("D路40キロで入ります"),        // D路 opens
            utterance("ゴツゴツ強いね"),
            utterance("次、F路60キロ"),              // F路 opens
            utterance("ブルブル来る"),
            utterance("もう一度D路に戻ります"),      // D路 second visit
        ]
        let proposals = EvalFormExtractor.roadSectionProposals(
            utterances: rows,
            roadNames: template.roadNames
        )
        #expect(proposals.map(\.title) == ["D路", "F路", "D路 (2)"])
        #expect(proposals[0].startUtteranceID == rows[1].id)
        #expect(proposals[0].endUtteranceID == rows[2].id)
        #expect(proposals[1].startUtteranceID == rows[3].id)
        #expect(proposals[1].endUtteranceID == rows[4].id)
        #expect(proposals[2].startUtteranceID == rows[5].id)
        #expect(proposals[2].endUtteranceID == rows[5].id)
    }

    @Test func noCalloutsMeansNoProposals() {
        let rows = [utterance("ゴツゴツするね"), utterance("そうですね")]
        #expect(EvalFormExtractor.roadSectionProposals(
            utterances: rows,
            roadNames: template.roadNames
        ).isEmpty)
    }

    // MARK: - Supplementary + metadata candidates

    @Test func supplementaryCandidatesSkipClaimedAndBackchannels() {
        let rows = [
            utterance("ヒョコヒョコが気になりますね"),   // 1: claimed
            utterance("うん"),                          // 2: backchannel
            utterance("ステアリングの戻りが少し重いように感じます"), // 3: substantive
            utterance("そう"),                          // 4: backchannel
            utterance("リアシートだと突き上げがもっとはっきり出ます"), // 5: substantive
        ]
        let candidates = EvalFormExtractor.supplementaryCandidateRows(
            utterances: rows,
            claimedRows: [1]
        )
        #expect(candidates == [3, 5])
    }

    @Test func metadataCandidatesUnionOpeningAndCueHits() {
        var rows = (1...25).map { utterance("走行コメント その\($0)ですね") }
        rows.append(utterance("ここでアブソーバーを評価仕様に交換します"))  // row 26, cue hit
        let candidates = EvalFormExtractor.metadataCandidateRows(
            utterances: rows,
            cues: template.metadataCues,
            openingCount: 5,
            limit: 30
        )
        #expect(candidates.prefix(5).elementsEqual(1...5))
        #expect(candidates.contains(26))
    }

    @Test func draftWithoutSupplementaryEvidenceKeyDecodesNil() throws {
        // Optional-tolerant v2 addition: a v2 payload written before
        // the field existed still decodes.
        let draft = EvalFormDraft(templateID: template.id, supplementaryComment: "x")
        var object = try JSONSerialization.jsonObject(
            with: draft.encoded()
        ) as! [String: Any]
        object.removeValue(forKey: "supplementaryEvidenceRows")
        let stripped = try JSONSerialization.data(withJSONObject: object)
        let decoded = try EvalFormDraft.decode(stripped)
        #expect(decoded.supplementaryComment == "x")
        #expect(decoded.supplementaryEvidenceRows == nil)
    }

    // MARK: - Scale geometry

    @Test func scalePositionsNormalizeAndClamp() {
        #expect(EvalScaleAxis.normalizedPosition(-1, minimum: -1, maximum: 1) == 0)
        #expect(EvalScaleAxis.normalizedPosition(0, minimum: -1, maximum: 1) == 0.5)
        #expect(EvalScaleAxis.normalizedPosition(1, minimum: -1, maximum: 1) == 1)
        #expect(abs(EvalScaleAxis.normalizedPosition(-0.25, minimum: -1, maximum: 1) - 0.375) < 0.0001)
        // Out-of-range values clamp instead of drawing off-track.
        #expect(EvalScaleAxis.normalizedPosition(5, minimum: -1, maximum: 1) == 1)
        #expect(EvalScaleAxis.normalizedPosition(5, minimum: 1, maximum: 9) == 0.5)
    }

    // MARK: - Undetected coverage

    /// A draft mirroring the first real trial's shape: some items
    /// inferred-only, one mentioned without fields, some untouched,
    /// partial metadata.
    private var gappyDraft: EvalFormDraft {
        var draft = EvalFormDraft(templateID: template.id)
        draft.items = [
            // Stated score + comment, no preference.
            .init(itemID: "11_hyokohyoko", strengthScore: -0.25,
                  comment: "c", evidenceRows: [3]),
            // Inferred-only + comment.
            .init(itemID: "12_buruburu", strengthScoreInferred: -0.75,
                  comment: "c", evidenceRows: [5]),
            // Evidence but nothing filled — mentioned, all missing.
            .init(itemID: "13_gotsugotsu", evidenceRows: [7]),
            // 10_flat, 13_biribiri, 14_harshness untouched.
        ]
        draft.metadata = ["評価車両": "ティグアン"]
        return draft
    }

    @Test func coverageListsMissingTargetsPerScope() {
        let undetected = EvalFormCoverage.undetected(
            draft: gappyDraft,
            template: template
        )
        // Header: every field except the one extracted.
        #expect(!undetected.missingHeaderFields.contains("評価車両"))
        #expect(undetected.missingHeaderFields.contains("天気"))

        let byID = Dictionary(
            uniqueKeysWithValues: undetected.items.map { ($0.itemID, $0) }
        )
        // Complete miss on stated score is qualified by an inferred
        // suggestion when one exists.
        #expect(EvalFormCoverage.gapPhrase(byID["11_hyokohyoko"]!) == "好き嫌い")
        #expect(EvalFormCoverage.gapPhrase(byID["12_buruburu"]!)
            == "評点（推定のみ・要確認）, 好き嫌い")
        #expect(EvalFormCoverage.gapPhrase(byID["13_gotsugotsu"]!)
            == "評点, 好き嫌い, コメント")
        // Untouched items read as not mentioned at all.
        #expect(EvalFormCoverage.gapPhrase(byID["14_harshness"]!) == "言及なし")
        #expect(byID["10_flat"]?.mentioned == false)
    }

    @Test func rendersIncludeUndetectedSection() {
        let markdown = EvalFormMarkdown.render(
            draft: gappyDraft,
            template: template,
            sessionTitle: ""
        )
        #expect(markdown.contains("## 未検出（要手動記入）"))
        #expect(markdown.contains("- ヘッダ: "))
        #expect(markdown.contains("14. ハーシュネス（ショック・ノイズ・減衰）: 言及なし"))

        let csv = EvalFormCSV.render(draft: gappyDraft, template: template)
        #expect(csv.contains("undetected,scope,missing"))
        #expect(csv.contains("undetected,14,言及なし"))
        #expect(csv.contains("undetected,12,\"評点（推定のみ・要確認）, 好き嫌い\""))
    }

    // MARK: - CSV

    @Test func csvEscapesQuotesCommasAndNewlines() {
        #expect(EvalFormCSV.escaped("plain") == "plain")
        #expect(EvalFormCSV.escaped("a,b") == "\"a,b\"")
        #expect(EvalFormCSV.escaped("say \"hi\"") == "\"say \"\"hi\"\"\"")
        #expect(EvalFormCSV.escaped("two\nlines") == "\"two\nlines\"")
    }

    @Test func csvRendersItemRowsAndSupplementary() {
        var draft = EvalFormDraft(templateID: template.id)
        draft.items = [
            .init(itemID: "11_hyokohyoko", strengthScore: -0.25,
                  comment: "収まり, 悪い", evidenceRows: [3, 5])
        ]
        draft.reviewedItemIDs = ["11_hyokohyoko"]
        draft.supplementaryComment = "路面続き"
        draft.supplementaryEvidenceRows = [9]
        let csv = EvalFormCSV.render(draft: draft, template: template)
        #expect(csv.contains("11,ヒョコヒョコ（過減衰感、バネ上の動き）,-0.25,,,\"収まり, 悪い\",3 5,yes,"))
        #expect(csv.contains("supplementaryComment,路面続き"))
        #expect(csv.contains("supplementaryEvidenceRows,9"))
    }
}
