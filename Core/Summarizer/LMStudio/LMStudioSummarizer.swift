import Foundation
import Fusion
import XephonLogging

/// `SessionSummarizer` that off-loads inference to a user-configured
/// LM Studio server over the local network. Pairs with
/// `LMStudioTranscriptionReviewer` and slots alongside Apple FM
/// + MLX in `SummarizerCoordinator`'s backend dispatch.
///
/// **Per CLAUDE.md's local-first / remote-open posture:** the
/// `.lmStudio` backend is opt-in via the Settings card's Remote
/// LLM Server toggle. Audio never leaves the device — only the
/// post-ASR transcript text — but local-network transmission is
/// still off-device in spirit, so privacy.md and a visible Remote
/// badge on the toolbar back the opt-in up.
///
/// **Prompt reuse.** Builds the user message from `MLXQwenSpec`'s
/// `buildPrompt` so the structure stays in lock-step with the
/// MLX path. LM Studio applies the loaded model's chat template
/// server-side, just like MLX-LM does on-device, so a Qwen3
/// loaded in LM Studio receives the same effective prompt as the
/// in-process `MLXQwenSummarizer`. Llama-served servers tolerate
/// the `/no_think` directive (literal text, ignored by Llama);
/// if that ever proves noisy we can split into per-served-family
/// specs.
///
/// **Mode coverage.** Phase 1 supports `.trailing` and
/// `.heuristic` (single-pass selection mirrored from
/// `MLXLLMSummarizerCore.summarizeSinglePass`). `.deep` is best-
/// effort-honored by falling back to `.trailing` — the per-window
/// + merge orchestration is a larger change we can layer in
/// later when there's a concrete deep-mode use case for the
/// remote path.
public actor LMStudioSummarizer: SessionSummarizer {
    public let modelIdentifier: String
    private let client: LMStudioClient
    private let spec = MLXQwenSpec()
    /// When true, send an OpenAI-format `response_format` with
    /// the summary JSON schema attached. The coordinator passes
    /// `LMStudioSettings.useStructuredOutput` through here at
    /// construction time so a settings flip doesn't affect an
    /// in-flight inference.
    private let useStructuredOutput: Bool

    public init(
        modelIdentifier: String,
        client: LMStudioClient,
        useStructuredOutput: Bool = false
    ) {
        // `modelIdentifier` is what gets stamped into the
        // resulting `SessionSummary.model` for attribution; we
        // pass the LM Studio server's model id verbatim so the
        // produced summary reads e.g.
        // `model: "lmstudio:qwen3-8b-instruct"`. Empty model is
        // valid (LM Studio falls back to its loaded model) — we
        // still produce an attribution string in that case so
        // the JSON output isn't blank.
        if modelIdentifier.isEmpty {
            self.modelIdentifier = "lmstudio"
        } else {
            self.modelIdentifier = "lmstudio:\(modelIdentifier)"
        }
        self.client = client
        self.useStructuredOutput = useStructuredOutput
    }

    public var isReady: Bool {
        // We can't verify reachability without a round-trip, and
        // the picker UI gates the LM Studio option on the Settings
        // toggle separately. Returning `true` here keeps the
        // coordinator's `ready` check from blocking based on a
        // network probe that would fire on every body re-render.
        true
    }

    public func summarize(
        utterances: [UtteranceEstimate],
        speakerNames: [String: String],
        mode: SummarizeMode,
        boostedUtteranceIDs: Set<UUID>
    ) async throws -> SessionSummary {
        guard !utterances.isEmpty else {
            return SessionSummary(
                inferredSetting: nil,
                topic: "",
                overallMood: "",
                perSpeaker: [],
                model: modelIdentifier,
                generatedAt: Date()
            )
        }

        // Phase 1: deep falls back to trailing. Document the
        // demotion in the log so the user can correlate a
        // "Summary looks shorter than I expected on a long
        // session" question with the actual selection used.
        let resolvedMode: SummarizeMode
        let selection: MLXLLMSelection
        switch mode {
        case .trailing:
            resolvedMode = .trailing
            selection = .trailing
        case .heuristic:
            resolvedMode = .heuristic
            selection = .heuristicTopN
        case .deep:
            AppLog.app.info("LMStudioSummarizer: .deep requested → demoted to .trailing (phase 1)")
            resolvedMode = .trailing
            selection = .trailing
        }

        let (promptUtterances, truncatedFrom) = selectUtterances(
            from: utterances,
            cap: spec.maxPromptUtterances,
            selection: selection,
            boostedUtteranceIDs: boostedUtteranceIDs
        )
        let prompt = spec.buildPrompt(
            utterances: promptUtterances,
            speakerNames: speakerNames,
            truncatedFromTotal: truncatedFrom,
            selection: selection
        )
        AppLog.app.info(
            "LMStudioSummarizer summarizing \(promptUtterances.count, privacy: .public) utterances (prompt \(prompt.count, privacy: .public) chars)"
        )
        let raw: String
        do {
            let responseFormatJSON = useStructuredOutput
                ? try? JSONEncoder().encode(LMStudioResponseFormat.jsonSchema(
                    name: "session_summary",
                    schema: LMStudioSchemas.summarySchema
                ))
                : nil
            raw = try await client.chat(
                userMessage: prompt,
                temperature: 0.2,
                maxTokens: spec.maxOutputTokens,
                responseFormatJSON: responseFormatJSON
            )
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            // Bubble up as SummarizerError so the coordinator's
            // existing error-surfacing path treats it uniformly
            // with the on-device backends' failures.
            throw SummarizerError.inferenceFailed(reason: String(describing: error))
        }
        if Task.isCancelled { throw CancellationError() }
        AppLog.app.info(
            "LMStudioSummarizer raw output: \(raw.count, privacy: .public) chars"
        )
        return try MLXLLMSummarizerCore.parse(
            raw: raw,
            speakerNames: speakerNames,
            modelIdentifier: modelIdentifier,
            mode: resolvedMode,
            expectedSpeakerIDs: promptUtterances.orderedSpeakerIDs
        )
    }

    /// Mirrors the trailing / heuristic selection block inside
    /// `MLXLLMSummarizerCore.summarizeSinglePass` so the prompt
    /// the LM Studio server receives matches what the on-device
    /// MLX path would have sent for the same inputs. Kept inline
    /// (rather than refactoring the MLX path to share) because
    /// that helper takes a `ModelContainer` and the surgery to
    /// extract just the selection logic is wider than this
    /// adapter warrants.
    private func selectUtterances(
        from utterances: [UtteranceEstimate],
        cap: Int,
        selection: MLXLLMSelection,
        boostedUtteranceIDs: Set<UUID>
    ) -> (selected: [UtteranceEstimate], truncatedFromTotal: Int?) {
        guard utterances.count > cap else { return (utterances, nil) }
        switch selection {
        case .trailing:
            return (Array(utterances.suffix(cap)), utterances.count)
        case .heuristicTopN:
            let topIDs = Informativeness.topNBalancedBySpeaker(
                cap,
                utterances: utterances,
                boostedIDs: boostedUtteranceIDs
            )
            return (utterances.filter { topIDs.contains($0.id) }, utterances.count)
        }
    }
}
