import Foundation
import Testing
import XephonPluginKit
import Summarizer
@testable import VehiclePerformanceMetricsA_1_Straight

/// Live model eval over synthetic known-answer sessions — the
/// model-independent-of-human-evaluator harness. Runs the FULL
/// pipeline (deterministic + LLM tiers) against an LM Studio
/// server and scores against the by-construction truth.
///
/// Gated on `XEPHON_LMSTUDIO_URL` (e.g. `http://127.0.0.1:1234`);
/// set `XEPHON_LMSTUDIO_MODEL` to pick the served model — this is
/// how the backend bake-off runs: same harness, different served
/// model, compare the printed reports (numbers → docs/eval_log.md).
///
/// Hard assertions cover only what the deterministic tier
/// guarantees REGARDLESS of model (stated-score recovery, no
/// regression below the offline floor); everything the model adds
/// (comments, metadata, inferred suggestions) is reported, not
/// asserted — models differ, the report is the deliverable.
@Suite(
    "EvalForm live model eval",
    .enabled(if: ProcessInfo.processInfo.environment["XEPHON_LMSTUDIO_URL"] != nil)
)
struct EvalFormLiveEvalTests {

    struct LMStudioEvalInference: InferenceService {
        let client: LMStudioClient

        @MainActor var availability: InferenceAvailability { .available }

        func generate(
            prompt: String,
            schemaJSON: String?,
            maxOutputTokens: Int
        ) async throws -> String {
            var responseFormat: Data?
            if let schemaJSON,
               let schema = try? JSONSerialization.jsonObject(
                   with: Data(schemaJSON.utf8)
               ) {
                responseFormat = try? JSONSerialization.data(withJSONObject: [
                    "type": "json_schema",
                    "json_schema": ["name": "eval_output", "schema": schema],
                ])
            }
            return try await client.chat(
                userMessage: prompt,
                temperature: 0.0,  // greedy — match the app's plugin path
                maxTokens: maxOutputTokens,
                responseFormatJSON: responseFormat
            )
        }
    }

    private func makeInference() throws -> (LMStudioEvalInference, String) {
        let env = ProcessInfo.processInfo.environment
        guard let urlString = env["XEPHON_LMSTUDIO_URL"],
              let url = URL(string: urlString) else {
            throw TestSkipError()
        }
        let model = env["XEPHON_LMSTUDIO_MODEL"] ?? "local-model"
        let client = LMStudioClient(configuration: .init(
            baseURL: url,
            modelID: model,
            requestTimeoutSeconds: 180
        ))
        return (LMStudioEvalInference(client: client), model)
    }

    struct TestSkipError: Error {}

    /// The eval battery: several seeded specs covering the fact
    /// mix (stated / qualitative / absent, preferences, metadata).
    private static let specs: [(label: String, spec: EvalFormSynthetic.Spec)] = [
        ("mixed-1", EvalFormSynthetic.Spec(
            facts: [
                "11_hyokohyoko": .statedScore(-0.25, preference: 6),
                "12_buruburu": .statedScore(-0.5),
                "13_gotsugotsu": .qualitativeOnly,
            ],
            metadata: ["評価車両": "ティグアン", "天気": "晴れ"],
            distractorCount: 12,
            seed: 42
        )),
        ("mixed-2", EvalFormSynthetic.Spec(
            facts: [
                "10_flat": .qualitativeOnly,
                "13_biribiri": .statedScore(0.25),
                "14_harshness": .statedScore(-0.75, preference: 3),
            ],
            metadata: ["路面状況": "dry", "気温": "22℃"],
            distractorCount: 16,
            seed: 7
        )),
        ("sparse", EvalFormSynthetic.Spec(
            facts: [
                "12_buruburu": .qualitativeOnly,
            ],
            metadata: [:],
            distractorCount: 20,
            seed: 99
        )),
    ]

    @Test(.timeLimit(.minutes(30)))
    func liveModelAgainstSyntheticTruth() async throws {
        let (inference, model) = try makeInference()
        let template = EvalFormTemplate.a1StraightRoad
        var lines: [String] = ["=== EvalForm live eval — model: \(model) ==="]

        for (label, spec) in Self.specs {
            let session = EvalFormSynthetic.generate(template: template, spec: spec)
            let draft = try await EvalFormRunner.fill(
                template: template,
                utterances: session.utterances,
                utterancesVersion: nil,
                inference: inference
            )
            let report = EvalFormSynthetic.score(
                draft: draft,
                expected: session.expected,
                template: template
            )
            lines.append(report.render(label: label))

            // Deterministic-tier guarantees hold under ANY model:
            // every planted stated score recovered exactly (the
            // regex tier owns these and the merge policy protects
            // them from the model).
            #expect(
                report.scoreExact == report.scoreExpected,
                "\(label): stated-score recovery regressed below the deterministic floor"
            )
            #expect(report.missedScores == 0, "\(label)")
        }

        // The report is the deliverable — copy into docs/eval_log.md.
        print(lines.joined(separator: "\n\n"))
    }
}
