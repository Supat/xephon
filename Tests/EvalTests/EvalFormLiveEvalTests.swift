import Foundation
import Testing
import XephonPluginKit
import Summarizer
@testable import VehiclePerformanceMetricsA_1_Straight

/// Live model eval over synthetic known-answer sessions — the
/// model-independent-of-human-evaluator harness. Runs the FULL
/// pipeline (deterministic + LLM tiers) against a real model and
/// scores against the by-construction truth.
///
/// Two backends, selected by environment:
/// - `XEPHON_MLX_MODEL_DIR=<dir>` — the PRODUCTION inference path
///   (`MLXQwenSummarizer.generateRaw`, prompt-contract schema, KV
///   prefix cache and all), running MLX directly in the test host.
///   Point it at a Qwen3-8B-4bit directory (the app's
///   Application Support install, `Models/qwen3-8b-4bit` from
///   `fetch_models.sh --with-summarizer`, or a HF snapshot). Needs
///   real Metal — physical device or a Designed-for-iPad run on an
///   Apple silicon Mac; the iOS Simulator won't do.
/// - `XEPHON_LMSTUDIO_URL` (e.g. `http://127.0.0.1:1234`) + optional
///   `XEPHON_LMSTUDIO_MODEL` — any served model via the LM Studio
///   client (native json_schema enforcement). This is the
///   cross-model bake-off path.
/// MLX wins when both are set. Reports print to the test log —
/// numbers go to docs/eval_log.md.
///
/// Hard assertions cover only what the deterministic tier
/// guarantees REGARDLESS of model (stated-score recovery, no
/// regression below the offline floor); everything the model adds
/// (comments, metadata, inferred suggestions) is reported, not
/// asserted — models differ, the report is the deliverable.
@Suite(
    "EvalForm live model eval",
    .enabled(if: ProcessInfo.processInfo.environment["XEPHON_LMSTUDIO_URL"] != nil
        || ProcessInfo.processInfo.environment["XEPHON_MLX_MODEL_DIR"] != nil)
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

    /// The production on-device path: same actor, same greedy
    /// settings, same prompt-contract schema appendix the
    /// coordinator's `pluginGenerate` uses for MLX backends — so
    /// harness numbers transfer to the iPad, and prompt/KV-cache
    /// changes are exercised exactly as shipped.
    struct MLXEvalInference: InferenceService {
        let actor: MLXQwenSummarizer

        @MainActor var availability: InferenceAvailability { .available }

        func generate(
            prompt: String,
            schemaJSON: String?,
            maxOutputTokens: Int
        ) async throws -> String {
            var effectivePrompt = prompt
            if let schemaJSON {
                // Byte-for-byte the coordinator's non-native schema
                // appendix (SummarizerCoordinator.pluginGenerate).
                effectivePrompt += """


                Return ONLY a valid JSON object conforming to this JSON Schema. \
                The FIRST character of your output MUST be `{`. No prose.
                \(schemaJSON)
                """
            }
            return try await actor.generateRaw(
                prompt: effectivePrompt,
                maxOutputTokens: maxOutputTokens
            )
        }
    }

    enum EvalBackend {
        case mlx(MLXEvalInference)
        case lmStudio(LMStudioEvalInference)

        var service: any InferenceService {
            switch self {
            case .mlx(let s): return s
            case .lmStudio(let s): return s
            }
        }
    }

    private func makeInference() throws -> (EvalBackend, String) {
        let env = ProcessInfo.processInfo.environment
        if let dir = env["XEPHON_MLX_MODEL_DIR"] {
            let url = URL(fileURLWithPath: (dir as NSString).expandingTildeInPath)
            let actor = MLXQwenSummarizer(
                modelIdentifier: "qwen3-8b-4bit",
                modelDirectory: url
            )
            return (.mlx(.init(actor: actor)), "mlx:qwen3-8b-4bit")
        }
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
        return (.lmStudio(.init(client: client)), model)
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
        // Tier 3a battery: positive + negative evaluative wording
        // must land in-band; measurement-only wording must leave
        // the inferred cell nil (false-positive check); a stated
        // score with no preference must not grow an inferred one
        // from measurement talk.
        ("tier3a", EvalFormSynthetic.Spec(
            facts: [
                "10_flat": .qualitativePositive,
                "11_hyokohyoko": .qualitativeOnly,
                "12_buruburu": .measurementOnly,
                "14_harshness": .statedScore(-0.25),
            ],
            metadata: [:],
            distractorCount: 14,
            seed: 3131
        )),
    ]

    @Test(.timeLimit(.minutes(60)))
    func liveModelAgainstSyntheticTruth() async throws {
        let (backend, model) = try makeInference()
        defer {
            // Free the ~4.6 GB MLX weights before the next suite in
            // the host process.
            if case .mlx(let mlx) = backend {
                Task { await mlx.actor.unload() }
            }
        }
        let inference = backend.service
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
