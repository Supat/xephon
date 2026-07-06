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

    /// See `MLXQwenReviewerSpec.contextOverlapRows`.
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
        lines.append("あなたは複数話者の会話の文字起こし校正者です。")
        lines.append(language.qwenAnchor)
        lines.append("")
        lines.append("以下の各発話には、1始まりの行インデックス、話者ID、時刻、文字起こしが含まれています。")
        lines.append("次の理由で文字起こしが誤っている可能性が高い行を見つけてください：")
        lines.append("  - 同音異義語または類似音の誤認識、")
        lines.append("  - セッションの文脈に合わない文（non sequitur）、")
        lines.append("  - スタイル的選択ではなくASRエラーと読める明確な文法ミス。")
        lines.append("単にカジュアル、方言、または異例だが整合性のある行はフラグしないでください。")
        lines.append("内容・意見・感情・事実の正否を校閲するのではありません。唯一の問いは「文字起こしが実際の発話と一致しているか」です。それ以外は範囲外であり、フラグしてはいけません。")
        lines.append("")
        lines.append("フラグする行には、必ず具体的な代替読みを念頭に置いてください — ASRが書かれている内容と混同した可能性のある別の単語またはフレーズです。具体的な代替が思い浮かばない場合は、行を完全に省略してください。「XはXの誤認識かもしれない」(Xは同じフレーズ) のような同義反復の理由は絶対に書かないでください。具体的な候補がない場合は省略を優先してください。")
        lines.append("")
        lines.append("次の1つのフィールドを持つJSONオブジェクトのみを返してください：")
        lines.append("  \"issues\": { \"rowIndex\": int, \"kind\": \"homophone\"|\"contextual\"|\"grammar\"|\"other\" のいずれか, \"excerpt\": その行の文字起こしから誤りと思われる部分を一字一句そのままコピーした文字列（行全体に関わるcontextualの場合のみ空文字列可）, \"reason\": 何が間違って見えるかを1文で, \"confidence\": 0.0〜1.0 の数値 } の配列")
        lines.append("excerptが行のテキストに一字一句一致しないissueは自動的に破棄されます — 打ち直さず、必ずコピーしてください。")
        lines.append("confidenceの基準：0.9 = ほぼ確実な誤認識、0.6 = あり得る。0.5未満と判断するissueは出力しないでください。")
        lines.append(MLXQwenReviewerSpec.fewShotExample(language: language))
        lines.append("修正後の文字起こしを提案しないでください — 人間ユーザーが自分で行を編集します。どの行が間違って見えるか、なぜそう思うかだけを特定してください。")
        lines.append("正しく読める行は省略してください。")
        lines.append("有効なJSONのみを返し、前後に散文を含めないでください。")
        lines.append("各issueの \"reason\" テキストは必ず次の言語で書いてください：\(SummarizerLocale.responseLanguageNameInEnglish)。他の言語は許容されません。")
        if totalChunks > 1 {
            lines.append("")
            lines.append("注意：これは会話の校閲パスのチャンク\(chunkIndex + 1)/\(totalChunks)です。前後の発話は別のチャンクで校閲されます。広いトピックの文脈が見えないという理由だけでnon-sequiturとしてフラグしないでください。")
        }
        lines.append("")
        if !contextPrefix.isEmpty {
            lines.append("前のチャンクからの文脈 — トピックの連続性のためだけのものです。以下の行には行インデックスがなく、別のチャンクで校閲済みです。絶対にフラグしないでください：")
            for u in contextPrefix {
                lines.append(MLXLLMReviewerRendering.contextLine(
                    for: u,
                    speakerNames: speakerNames
                ))
            }
            lines.append("")
        }
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
        // 直近性による再表明 — スコープとreason言語はモデルが最も
        // 逸脱しやすい2規則（サマライザープロンプトと同じ修正）。
        lines.append("フラグするのは一字一句のexcerptを伴う文字起こしの誤認識のみ。確信が持てない行は省略してください。\"reason\" は必ず\(SummarizerLocale.responseLanguageNameInEnglish)で書いてください。他の言語は許容されません。")
        return lines.joined(separator: "\n")
    }
}
