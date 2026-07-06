import Foundation
import Fusion
import XephonLogging
import MLX
import MLXLLM
import MLXLMCommon

/// MLX-backed transcription reviewer for the Qwen3-8B family.
/// Pairs with `MLXLlamaTranscriptionReviewer`; both are thin
/// actors over the shared orchestration in
/// `MLXLLMReviewerCore`. This file supplies the Qwen-specific
/// spec (English prompt + Chinese-anchoring, `/no_think`, no
/// repetition penalty, single extra EOS) and owns the per-
/// actor lifecycle (load / unload of the `ModelContainer`).
///
/// Lifecycle mirrors the summarizer actors: `load()` brings
/// the weights in, `review(...)` chunks the input and runs
/// per-chunk inference, `unload()` releases the working set
/// when the review sheet dismisses.
public actor MLXQwenTranscriptionReviewer:
    TranscriptionReviewer, MLXLLMReviewerActor
{
    public let modelIdentifier: String
    private let modelDirectory: URL
    private var container: ModelContainer?
    private let spec = MLXQwenReviewerSpec()

    public init(modelIdentifier: String, modelDirectory: URL) {
        self.modelIdentifier = modelIdentifier
        self.modelDirectory = modelDirectory
    }

    public var isReady: Bool {
        container != nil
    }

    public func load() async throws {
        if container != nil { return }
        AppLog.app.info(
            "MLXQwenTranscriptionReviewer loading from \(self.modelDirectory.path, privacy: .public)"
        )
        // Matches MLXQwenSummarizer — see its `load()` for the
        // 128 MB rationale (SER actors torn down before this
        // runs).
        MLX.GPU.set(cacheLimit: 128 * 1024 * 1024)
        do {
            let configuration = ModelConfiguration(
                directory: modelDirectory,
                extraEOSTokens: spec.extraEOSTokens
            )
            container = try await LLMModelFactory.shared.loadContainer(
                configuration: configuration
            )
            AppLog.app.info("MLXQwenTranscriptionReviewer loaded")
        } catch {
            throw TranscriptionReviewError.modelLoadFailed(
                reason: String(describing: error)
            )
        }
    }

    public func unload() {
        container = nil
        AppLog.app.info("MLXQwenTranscriptionReviewer unloaded")
    }

    public func review(
        utterances: [UtteranceEstimate],
        speakerNames: [String: String],
        language: ReviewLanguage
    ) async throws -> [TranscriptionIssue] {
        try await load()
        guard let container else {
            throw TranscriptionReviewError.modelNotInstalled
        }
        return try await MLXLLMReviewerCore.review(
            container: container,
            utterances: utterances,
            speakerNames: speakerNames,
            language: language,
            spec: spec
        )
    }
}

// MARK: - Qwen reviewer spec

/// Qwen3-specific configuration + prompt builder for the
/// transcription reviewer.
internal struct MLXQwenReviewerSpec: MLXLLMReviewerSpec {
    let family: LLMModelFamily = .qwen

    /// Cap on utterances per chunk. Review needs more textual
    /// context per row than summarization to evaluate
    /// homophone candidates against neighbours, but Qwen3's
    /// 32k context easily fits 80 rows at the compact format.
    /// Long sessions get chunked, not truncated — see
    /// `MLXLLMReviewerCore.review`.
    let maxPromptUtterances = 80

    /// Cap on output tokens. Each issue is ~50–80 tokens of
    /// JSON; 4096 fits 60+ entries even on a noisy chunk; the
    /// tolerant parser salvages the prefix if a chunk still
    /// runs over.
    let maxOutputTokens = 4096

    let extraEOSTokens: Set<String> = ["<|endoftext|>"]

    let repetitionPenalty: Float? = nil

    /// Previous-chunk rows replayed as unindexed context — enough
    /// turns to carry the topic across the boundary without
    /// meaningfully eating into the 80-row chunk budget.
    let contextOverlapRows = 6

    func buildPrompt(
        utterances: [UtteranceEstimate],
        speakerNames: [String: String],
        language: ReviewLanguage,
        chunkIndex: Int,
        totalChunks: Int,
        contextPrefix: [UtteranceEstimate]
    ) -> String {
        var lines: [String] = []
        lines.reserveCapacity(utterances.count + 22)
        lines.append("You are a transcription proofreader for a multi-speaker conversation.")
        // Language anchoring up front — Qwen3 was trained on a
        // Chinese-dominant corpus and otherwise reads kanji as
        // Mandarin (suggesting Chinese-style replacements
        // meaningless to the user). Putting this above the
        // task description biases the rest of the prompt into
        // the right language frame from the first token.
        lines.append(language.qwenAnchor)
        lines.append("")
        lines.append("Each utterance below has a 1-based row index, speaker id, time, and transcript.")
        lines.append("Find rows whose transcript is likely WRONG because of:")
        lines.append("  - a misrecognized homophone or near-homophone,")
        lines.append("  - a sentence that does not fit the session context (non-sequitur),")
        lines.append("  - a clear grammar slip that reads as an ASR error, not a stylistic choice.")
        lines.append("Do NOT flag rows that are merely informal, dialectal, or unusual but coherent.")
        lines.append("You are NOT reviewing content, opinions, emotions, or facts. The ONLY question is whether the transcription matches what was likely SAID. Anything else is out of scope and must not be flagged.")
        lines.append("")
        lines.append("A flagged row MUST have a SPECIFIC plausible alternative reading in mind — a different word or phrase the ASR could have confused with what's written. If no specific alternative comes to mind, OMIT the row entirely. NEVER write a reason of the form \"X may be a misinterpretation of X\" where X is the same phrase as the row's transcript — that's a tautology and not an issue. Omitting rows is ALWAYS preferred over flagging without a real candidate.")
        lines.append("")
        lines.append("Return ONLY a JSON object with one field:")
        lines.append("  \"issues\": array of { \"rowIndex\": int, \"kind\": one of \"homophone\"|\"contextual\"|\"grammar\"|\"other\", \"excerpt\": the EXACT substring of that row's transcript that looks wrong — copied verbatim, no paraphrase (empty string only for a contextual issue about the whole row), \"reason\": one short sentence describing what looks wrong, \"confidence\": number 0.0–1.0 }")
        lines.append("Issues whose excerpt does not appear character-for-character in the row's text are DISCARDED automatically — copy, never retype.")
        lines.append("Calibrate confidence: 0.9 = near-certain misrecognition, 0.6 = plausible. Do NOT emit issues you would score below 0.5.")
        lines.append(Self.fewShotExample(language: language))
        lines.append("DO NOT propose a corrected transcript — the human user will edit the row themselves. Just identify which rows look wrong and why.")
        lines.append("Omit rows that read correctly.")
        lines.append("Return ONLY valid JSON, no prose before or after.")
        // Pin the freeform `reason` field's language to the
        // user's iPadOS app-language pick.
        lines.append("The \"reason\" text in each issue MUST be written in this language: \(SummarizerLocale.responseLanguageNameInEnglish). No other language is acceptable.")
        // Qwen3's `/no_think` directive disables its thinking-
        // mode chain-of-thought for a single turn. Llama
        // ignores this token and would emit it verbatim, hence
        // family-gated.
        lines.append("/no_think")
        if totalChunks > 1 {
            lines.append("")
            lines.append("NOTE: This is chunk \(chunkIndex + 1) of \(totalChunks) of the conversation's review pass. Earlier and later utterances are reviewed in their own chunks; do not flag rows as non-sequitur just because you can't see the broader topic context.")
        }
        lines.append("")
        if !contextPrefix.isEmpty {
            lines.append("Context from the previous chunk — for topic continuity ONLY. These rows have no row index, were already reviewed in their own chunk, and MUST NOT be flagged:")
            for u in contextPrefix {
                lines.append(MLXLLMReviewerRendering.contextLine(
                    for: u,
                    speakerNames: speakerNames
                ))
            }
            lines.append("")
        }
        lines.append("Utterances:")
        for (idx, u) in utterances.enumerated() {
            lines.append(MLXLLMReviewerRendering.compactLine(
                rowIndex: idx + 1,
                for: u,
                speakerNames: speakerNames
            ))
        }
        // Instruction sandwich — restate the directive after
        // the utterance list so the model's recent attention
        // has "produce JSON" rather than the last utterance
        // line. See `MLXQwenSpec.buildPrompt` for the failure
        // mode this prevents.
        lines.append("")
        lines.append("---")
        lines.append("IMPORTANT: Follow the instructions above and produce exactly one valid JSON object with an `issues` field. The FIRST character of your output MUST be `{`. Do NOT echo the utterance list above; do NOT add any prose. Omit rows that read correctly.")
        // Recency restatement of the two rules the models drift on
        // most: task scope and the reason language (same fix the
        // summarizer prompts got — the early copies sit thousands
        // of tokens back once the utterance list is long).
        lines.append("Flag ONLY likely transcription misrecognitions with a verbatim excerpt; when unsure, omit the row. The \"reason\" text MUST be written in \(SummarizerLocale.responseLanguageNameInEnglish); no other language is acceptable.")
        return lines.joined(separator: "\n")
    }

    /// One worked example + one non-example, language-matched to
    /// the conversation. Small quantized models follow the schema
    /// and — critically — the excerpt-grounding rule far more
    /// reliably with a concrete demonstration than with rules
    /// alone. The reason string follows the app-language pick so
    /// the example can't contradict the reason-language directive.
    static func fewShotExample(language: ReviewLanguage) -> String {
        let reasonInJapanese = SummarizerLocale.responseLanguageNameInEnglish == "Japanese"
        switch language {
        case .japanese:
            let reason = reasonInJapanese
                ? "「資料」(しりょう)を「飼料」と誤認識した可能性"
                : "Likely misrecognition of 資料 (shiryō, documents) as 飼料 (animal feed)"
            return """
                Example (format demonstration ONLY — these rows are NOT in your input; never copy them into your output):
                  input row:   - row=3 speaker=S01 t=42.1s text="会議の飼料を配りました"
                  output item: { "rowIndex": 3, "kind": "homophone", "excerpt": "飼料", "reason": "\(reason)", "confidence": 0.85 }
                  A row like "うん、そうだね" is coherent conversational Japanese — do NOT flag it.
                """
        case .english:
            let reason = reasonInJapanese
                ? "「their」を「there」と誤認識した可能性"
                : "Likely misrecognition of 'there' for 'their'"
            return """
                Example (format demonstration ONLY — these rows are NOT in your input; never copy them into your output):
                  input row:   - row=3 speaker=S01 t=42.1s text="we left there bags at the station"
                  output item: { "rowIndex": 3, "kind": "homophone", "excerpt": "there", "reason": "\(reason)", "confidence": 0.85 }
                  A row like "yeah, exactly" is coherent conversational English — do NOT flag it.
                """
        }
    }
}
