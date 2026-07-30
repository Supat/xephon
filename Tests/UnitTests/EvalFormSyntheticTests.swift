import Foundation
import Testing
import XephonPluginKit
@testable import VehiclePerformanceMetricsA_1_Straight

/// Pins the synthetic known-answer harness itself plus the
/// deterministic-tier BASELINE: with the LLM tier failing entirely,
/// the pipeline must still recover every planted stated score with
/// zero false fills — that floor is what makes the live eval's
/// numbers interpretable (anything above it is what the model
/// adds; any false fill is something the model broke).
@Suite("EvalForm synthetic harness")
struct EvalFormSyntheticTests {

    private let template = EvalFormTemplate.a1StraightRoad

    /// Inference that always fails — deterministic tier only.
    struct FailingInference: InferenceService {
        @MainActor var availability: InferenceAvailability { .available }
        func generate(
            prompt: String,
            schemaJSON: String?,
            maxOutputTokens: Int
        ) async throws -> String {
            throw PluginInferenceError.unavailable(reason: "offline baseline")
        }
    }

    private var spec: EvalFormSynthetic.Spec {
        EvalFormSynthetic.Spec(
            facts: [
                "11_hyokohyoko": .statedScore(-0.25, preference: 6),
                "12_buruburu": .statedScore(-0.5),
                "13_gotsugotsu": .qualitativeOnly,
                // remaining items absent
            ],
            metadata: ["評価車両": "ティグアン", "天気": "晴れ"],
            distractorCount: 10,
            seed: 42
        )
    }

    @Test func generationIsDeterministicPerSeed() {
        let a = EvalFormSynthetic.generate(template: template, spec: spec)
        let b = EvalFormSynthetic.generate(template: template, spec: spec)
        #expect(a.utterances.map(\.transcript) == b.utterances.map(\.transcript))
        #expect(a.expected.factRows == b.expected.factRows)

        var reseeded = spec
        reseeded.seed = 43
        let c = EvalFormSynthetic.generate(template: template, spec: reseeded)
        #expect(a.utterances.map(\.transcript) != c.utterances.map(\.transcript))
    }

    @Test func expectedSheetMatchesSpecByConstruction() {
        let session = EvalFormSynthetic.generate(template: template, spec: spec)
        #expect(session.expected.statedScores == [
            "11_hyokohyoko": -0.25, "12_buruburu": -0.5,
        ])
        #expect(session.expected.preferences == ["11_hyokohyoko": 6])
        #expect(session.expected.discussedItems == [
            "11_hyokohyoko", "12_buruburu", "13_gotsugotsu",
        ])
        // Every fact row's transcript actually contains its item's
        // vocabulary (generator self-check).
        for (itemID, rows) in session.expected.factRows {
            let item = template.items.first { $0.id == itemID }!
            for row in rows {
                let text = session.utterances[row - 1].transcript
                #expect(item.vocabulary.contains { text.contains($0) })
            }
        }
    }

    @Test func deterministicBaselineRecoversAllStatedScoresWithNoFalseFills() async throws {
        let session = EvalFormSynthetic.generate(template: template, spec: spec)
        let draft = try await EvalFormRunner.fill(
            template: template,
            utterances: session.utterances,
            utterancesVersion: nil,
            inference: FailingInference()
        )
        let report = EvalFormSynthetic.score(
            draft: draft,
            expected: session.expected,
            template: template
        )
        // The floor: regex tier alone recovers every planted stated
        // score and preference, invents nothing, and cites the
        // planted rows.
        #expect(report.scoreExact == 2)
        #expect(report.scoreExpected == 2)
        #expect(report.falseScoreFills == 0)
        #expect(report.missedScores == 0)
        #expect(report.preferenceExact == 1)
        #expect(report.evidenceValid == report.evidenceChecked)
        #expect(report.evidenceChecked == 2)
        // And, without a model: no comments, no metadata.
        #expect(report.commentsPresent == 0)
        #expect(report.metadataCorrect == 0)
        #expect(report.falseMetadata == 0)
    }

    @Test func scorerCountsFalseFillsAndMisses() {
        let session = EvalFormSynthetic.generate(template: template, spec: spec)
        var draft = EvalFormDraft(templateID: template.id)
        draft.items = [
            // Correct stated score with valid evidence.
            .init(itemID: "11_hyokohyoko", strengthScore: -0.25,
                  likeDislike: 6, comment: "c",
                  evidenceRows: session.expected.factRows["11_hyokohyoko"] ?? []),
            // Missed stated score.
            .init(itemID: "12_buruburu"),
            // FALSE fill: stated score on a qualitative-only item.
            .init(itemID: "13_gotsugotsu", strengthScore: -0.5, comment: "c",
                  evidenceRows: session.expected.factRows["13_gotsugotsu"] ?? []),
            // FALSE comment on an absent item.
            .init(itemID: "14_harshness", comment: "invented"),
        ]
        draft.metadata = ["評価車両": "ティグアン", "評価者": "誰か"]  // 2nd is invented
        let report = EvalFormSynthetic.score(
            draft: draft,
            expected: session.expected,
            template: template
        )
        #expect(report.scoreExact == 1)
        #expect(report.missedScores == 1)
        #expect(report.falseScoreFills == 1)
        #expect(report.preferenceExact == 1)
        #expect(report.falseComments == 1)
        #expect(report.metadataCorrect == 1)
        #expect(report.falseMetadata == 1)
    }
}
