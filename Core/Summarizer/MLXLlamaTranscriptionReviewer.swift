import Foundation
import Fusion
import XephonLogging
import MLX
import MLXLLM
import MLXLMCommon

/// MLX-backed transcription reviewer for the Llama-3.1-Swallow
/// family. Pairs with `MLXQwenTranscriptionReviewer`; both are
/// thin actors over the shared orchestration in
/// `MLXLLMReviewerCore`. This file supplies the Llama-specific
/// spec (Japanese-language prompt, EOS-token triple,
/// repetition penalty).
///
/// Lifecycle mirrors `MLXQwenTranscriptionReviewer` exactly —
/// only the spec passed to `MLXLLMReviewerCore.review` differs.
public actor MLXLlamaTranscriptionReviewer:
    TranscriptionReviewer, MLXLLMReviewerActor
{
    public let modelIdentifier: String
    private let modelDirectory: URL
    private var container: ModelContainer?
    private let spec = MLXLlamaReviewerSpec()

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
            "MLXLlamaTranscriptionReviewer loading from \(self.modelDirectory.path, privacy: .public)"
        )
        MLX.GPU.set(cacheLimit: 128 * 1024 * 1024)
        do {
            let configuration = ModelConfiguration(
                directory: modelDirectory,
                extraEOSTokens: spec.extraEOSTokens
            )
            container = try await LLMModelFactory.shared.loadContainer(
                configuration: configuration
            )
            AppLog.app.info("MLXLlamaTranscriptionReviewer loaded")
        } catch {
            throw TranscriptionReviewError.modelLoadFailed(
                reason: String(describing: error)
            )
        }
    }

    public func unload() {
        container = nil
        AppLog.app.info("MLXLlamaTranscriptionReviewer unloaded")
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

// MARK: - Llama reviewer spec

/// Llama-3.1-Swallow-specific configuration + prompt builder
/// for the transcription reviewer.
internal struct MLXLlamaReviewerSpec: MLXLLMReviewerSpec {
    let family: LLMModelFamily = .llama

    /// Same chunk size as Qwen — reviewer rows are minimal
    /// (speaker / time / text only) regardless of family, so
    /// the token cost per chunk is comparable.
    let maxPromptUtterances = 80

    let maxOutputTokens = 4096

    /// All three Llama 3.1 stop tokens. See
    /// `MLXLlamaSpec.extraEOSTokens` for the rationale: MLX-LM
    /// only honors a single `eos_token` from
    /// `tokenizer_config.json` but Llama 3.1 declares three.
    let extraEOSTokens: Set<String> = [
        "<|end_of_text|>",
        "<|eom_id|>",
        "<|eot_id|>",
    ]

    /// 1.05 — the lightest touch that breaks the repetition
    /// loops Llama-Swallow falls into on English-language
    /// JSON-schema prompts. Same value used in the
    /// summarizer.
    let repetitionPenalty: Float? = 1.05

    func buildPrompt(
        utterances: [UtteranceEstimate],
        speakerNames: [String: String],
        language: ReviewLanguage,
        chunkIndex: Int,
        totalChunks: Int
    ) -> String {
        var lines: [String] = []
        lines.reserveCapacity(utterances.count + 22)
        lines.append("あなたは複数話者の会話の文字起こし校正者です。")
        lines.append(language.qwenAnchor)
        lines.append("")
        lines.append("以下の各発話には、1始まりの行インデックス、話者ID、時刻、文字起こしが含まれています。")
        lines.append("次の理由で文字起こしが誤っている可能性が高い行を見つけてください：")
        lines.append("  - 同音異義語または類似音の誤認識、")
        lines.append("  - セッションの文脈に合わない文（non sequitur）、")
        lines.append("  - スタイル的選択ではなくASRエラーと読める明確な文法ミス。")
        lines.append("単にカジュアル、方言、または異例だが整合性のある行はフラグしないでください。")
        lines.append("")
        lines.append("次の1つのフィールドを持つJSONオブジェクトのみを返してください：")
        lines.append("  \"issues\": { \"rowIndex\": int, \"kind\": \"homophone\"|\"contextual\"|\"grammar\"|\"other\" のいずれか, \"reason\": 何が間違って見えるかを1文で, \"confidence\": 0.0〜1.0 の数値 } の配列")
        lines.append("修正後の文字起こしを提案しないでください — 人間ユーザーが自分で行を編集します。どの行が間違って見えるか、なぜそう思うかだけを特定してください。")
        lines.append("正しく読める行は省略してください。")
        lines.append("有効なJSONのみを返し、前後に散文を含めないでください。")
        lines.append("各issueの \"reason\" テキストは必ず次の言語で書いてください：\(SummarizerLocale.responseLanguageNameInEnglish)。他の言語は許容されません。")
        if totalChunks > 1 {
            lines.append("")
            lines.append("注意：これは会話の校閲パスのチャンク\(chunkIndex + 1)/\(totalChunks)です。前後の発話は別のチャンクで校閲されます。広いトピックの文脈が見えないという理由だけでnon-sequiturとしてフラグしないでください。")
        }
        lines.append("")
        lines.append("発話：")
        for (idx, u) in utterances.enumerated() {
            lines.append(MLXLLMReviewerRendering.compactLine(
                rowIndex: idx + 1,
                for: u,
                speakerNames: speakerNames
            ))
        }
        lines.append("")
        lines.append("---")
        lines.append("重要：上記の指示に従い、issuesフィールドを持つ有効なJSONオブジェクトを1つだけ生成してください。出力の最初の文字は必ず `{` でなければなりません。上記の発話リストをエコーしないでください。散文も一切含めないでください。正しく読める行は省略してください。")
        return lines.joined(separator: "\n")
    }
}
