import Foundation
import Fusion
import SERAcoustic
import SERText
import XephonLogging
import MLX
import MLXLLM
import MLXLMCommon

/// MLX-backed summarizer for the Llama 3.1 Swallow 8B
/// (4-bit) family — Tokyo Tech's Japanese fine-tune of
/// Llama 3.1. Pairs with `MLXQwenSummarizer`; both are thin
/// actors over the shared orchestration in
/// `MLXLLMSummarizerCore`. This file supplies the Llama-
/// specific spec (Japanese-language prompts, EOS-token
/// triple, SER-stripped row format, repetition penalty).
///
/// Lifecycle mirrors `MLXQwenSummarizer` exactly — only the
/// spec passed to `MLXLLMSummarizerCore.summarize` differs.
public actor MLXLlamaSummarizer: SessionSummarizer, MLXLLMSummarizerActor {
    public let modelIdentifier: String
    private let modelDirectory: URL
    private var container: ModelContainer?
    private let spec = MLXLlamaSpec()

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
            "MLXLlamaSummarizer loading from \(self.modelDirectory.path, privacy: .public)"
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
            AppLog.app.info("MLXLlamaSummarizer loaded")
        } catch {
            throw SummarizerError.modelLoadFailed(
                reason: String(describing: error)
            )
        }
    }

    public func unload() {
        container = nil
        AppLog.app.info("MLXLlamaSummarizer unloaded")
    }

    public func summarize(
        utterances: [UtteranceEstimate],
        speakerNames: [String: String],
        mode: SummarizeMode,
        boostedUtteranceIDs: Set<UUID>,
        glossaryTerms: [String]
    ) async throws -> SessionSummary {
        try await load()
        guard let container else {
            throw SummarizerError.modelNotInstalled
        }
        return try await MLXLLMSummarizerCore.summarize(
            container: container,
            modelIdentifier: modelIdentifier,
            utterances: utterances,
            speakerNames: speakerNames,
            mode: mode,
            boostedUtteranceIDs: boostedUtteranceIDs,
            glossaryTerms: glossaryTerms,
            spec: spec
        )
    }
}

// MARK: - Llama spec

/// Llama-3.1-Swallow-specific configuration + prompt
/// builders. Used by `MLXLlamaSummarizer` and consumed by
/// `MLXLLMSummarizerCore`.
internal struct MLXLlamaSpec: MLXLLMSpec {
    let family: LLMModelFamily = .llama

    /// Same cap as Qwen on paper, but Llama rows are
    /// drastically shorter (SER-stripped, see `compactLine`),
    /// so the effective token cost is ~4× lower — a 100-row
    /// Llama prompt lands at ~4k tokens vs Qwen's ~14k.
    let maxPromptUtterances = 100

    /// Larger than Qwen's 50 because Llama's `compactLine`
    /// is stripped of SER detail — same raw token budget
    /// fits more rows per window. Fewer chunks → fewer
    /// merge-stage compression cascades, the dominant
    /// source of detail loss in deep mode. A 200-utterance
    /// session goes from 4 windows (at 50) to 3 windows
    /// (at 75); each window covers more of the conversation
    /// so the merge synthesizes from richer intermediates.
    let deepWindowSize = 75

    /// Bumped from 1280 → 2560 specifically for Llama after the
    /// schema loosening (per-speaker notes 1-2 → 3-5 sentences)
    /// and window size bump (50 → 75) stacked enough additional
    /// output demand that the prior cap saturated mid-`perSpeaker`
    /// on speaker-heavy windows. With ~8 speakers × ~5 sentences
    /// × ~30 tokens = ~1200 tokens just for the notes, plus
    /// topic/mood/modalityFlags overhead, 1280 left no headroom
    /// and forced the `recoverTruncatedWindowIntermediate` path
    /// or — worse — the placeholder synthesis fallback. 2560
    /// gives the verbose chunks room to finish naturally;
    /// recovery still salvages anything that spills.
    let deepWindowOutputTokens = 2560

    let maxOutputTokens = 4096

    /// Llama 3.1's `config.json` declares three valid stop
    /// tokens (`<|end_of_text|>`, `<|eom_id|>`, `<|eot_id|>`)
    /// but the tokenizer only registers one as
    /// `eos_token_id`. The Swallow fine-tune has been
    /// observed emitting the base-model EOS instead of the
    /// chat-template EOT, so without all three listed here
    /// generation can run all the way to `maxOutputTokens`
    /// on the EOS the tokenizer doesn't recognize.
    let extraEOSTokens: Set<String> = [
        "<|end_of_text|>",
        "<|eom_id|>",
        "<|eot_id|>",
    ]

    /// 1.05 — the lightest touch that breaks the repetition
    /// loops Llama-Swallow falls into when given English-
    /// language JSON-schema instructions, without warping
    /// legitimate repeated phrases (speaker names, numeric
    /// scores).
    let repetitionPenalty: Float? = 1.05

    func buildPrompt(
        utterances: [UtteranceEstimate],
        speakerNames: [String: String],
        truncatedFromTotal: Int?,
        selection: MLXLLMSelection
    ) -> String {
        let speakers = utterances.orderedSpeakerIDs
        let speakerList = speakers.joined(separator: ", ")
        var lines: [String] = []
        lines.reserveCapacity(utterances.count + 14)
        lines.append("あなたは複数話者の会話を要約するアナリストです。")
        lines.append("以下のすべての発話を読み、次の4つのフィールドを持つJSONオブジェクトを1つ生成してください：")
        lines.append("  \"setting\" — 会話の状況・場面・レジスタを1文で示してください（例：「友人同士のカジュアルな電話」「就職面接」「教室での議論」）。具体的な場所や機関を創作せず、一般的に保ってください。最初に出力し、以降の内容と一貫させてください。")
        lines.append("  \"topic\" — 会話の主題を1〜2文で記述してください。")
        lines.append("  \"overallMood\" — 推定された状況と一貫した、セッション全体の感情的な雰囲気を1段落で記述してください。")
        lines.append("  \"perSpeaker\" — 配列。次の話者IDごとに1エントリ：\(speakerList)。")
        lines.append("各perSpeakerエントリの形式：{ \"speakerID\": <id>, \"summary\": <1段落>, \"dominantMood\": <短いフレーズ> }。")
        // Llama rows are stripped of SER detail (see
        // `compactLine`) so no aP/tP/V-A-D schema description
        // here — describing fields the model won't see in the
        // data confuses it. Speaker, time, and transcript
        // only.
        lines.append("各行には話者ID、時刻、文字起こしのみが含まれます。感情は文字起こしの内容から推測してください。")
        lines.append("以下の話者人口統計ブロックに性別が記載されている場合、その話者の代名詞として全体で使用してください — 女性は「she/her」、男性は「he/him」、子供または性別が記載されていない場合は「they/them」。（日本語のように代名詞を省略する言語では無関係です。）")
        lines.append("有効なJSONのみを返し、前後に散文を含めないでください。")
        lines.append(SummarizerLocale.responseLanguageInstruction)
        if let total = truncatedFromTotal {
            lines.append("")
            switch selection {
            case .trailing:
                lines.append("注意：この会話は全体で\(total)発話ありますが、以下には最新の\(utterances.count)発話のみが表示されています。overallMoodはセッション全体の弧ではなく、後半部分として位置付けてください。")
            case .heuristicTopN:
                lines.append("注意：この会話は全体で\(total)発話あります。以下には最新ではなく、セッション内TF-IDFで最も特徴的な\(utterances.count)発話を時系列順で表示しています。overallMoodは連続した末尾区間ではなく、セッション全体を代表するサンプルとして記述してください — 行間のギャップは想定内です。")
            }
        }
        let demographics = SpeakerDemographicsDigest
            .build(from: utterances)
            .renderForPrompt(speakerIDs: speakers, speakerNames: speakerNames)
        if !demographics.isEmpty {
            lines.append("")
            lines.append(demographics)
        }
        lines.append("")
        lines.append("発話：")
        for u in utterances {
            lines.append(compactLine(for: u, speakerNames: speakerNames))
        }
        lines.append("")
        lines.append("---")
        lines.append("重要：上記の指示に従い、setting、topic、overallMood、perSpeaker の4フィールドを持つ有効なJSONオブジェクトを1つだけ生成してください。出力の最初の文字は必ず `{` でなければなりません。上記の発話リストをエコーしないでください。散文も一切含めないでください。")
        // Restate the language directive LAST — recency wins on
        // long prompts (same reason the JSON-shape rule is
        // restated in this sandwich). The copy mid-prompt sits
        // thousands of tokens back, above the utterance list, and
        // the small quantized models drift to the transcript's
        // (or the prompt's own) language without this reminder —
        // Apple FM honored the early copy, these did not.
        lines.append(SummarizerLocale.responseLanguageInstruction)
        return lines.joined(separator: "\n")
    }

    func buildDeepWindowPrompt(
        utterances: [UtteranceEstimate],
        speakerNames: [String: String],
        windowIndex: Int,
        totalWindows: Int
    ) -> String {
        let speakers = utterances.orderedSpeakerIDs
        let speakerList = speakers.joined(separator: ", ")
        let timeStart = utterances.first?.start ?? 0
        let timeEnd = utterances.last?.end ?? 0
        let tStartStr = String(format: "%.1f", timeStart)
        let tEndStr = String(format: "%.1f", timeEnd)
        var lines: [String] = []
        lines.reserveCapacity(utterances.count + 20)
        lines.append("あなたは長い複数話者の会話のうち1つのウィンドウを要約するアナリストです。")
        lines.append("これはウィンドウ\(windowIndex + 1)/\(totalWindows)で、t=\(tStartStr)秒からt=\(tEndStr)秒の発話を対象とします。")
        lines.append("JSONオブジェクトを1つ生成してください — 最終要約ではなく、マージパスが他のウィンドウと結合する簡潔な中間データです。")
        lines.append("フィールド：")
        lines.append("  \"windowIndex\": \(windowIndex)（この数字をコピー）、")
        lines.append("  \"timeStart\": \(tStartStr)、")
        lines.append("  \"timeEnd\": \(tEndStr)、")
        lines.append("  \"topicSnapshot\": このウィンドウの話題を短いフレーズで、")
        lines.append("  \"moodSnapshot\": このウィンドウの感情的な雰囲気を短いフレーズで、")
        lines.append("  \"perSpeaker\": { \"speakerID\": <id>, \"notes\": この話者の本ウィンドウでの貢献を3〜5文でしっかり記述してください（重要な発言、トピック、感情の変化を含めて）, \"dominantMood\": 短いフレーズ } の配列。次の話者ごとに1エントリ：\(speakerList)。")
        // modalityFlags field omitted — Llama rows don't
        // carry aP/tP, so the model has no signal to populate
        // it. `MLXLLMDeepWindowIntermediate.modalityFlags` is
        // optional; the decoder treats nil as "no flags this
        // window."
        lines.append("各行には話者ID、時刻、文字起こしのみが含まれます。感情は文字起こしの内容から推測してください。")
        lines.append("有効なJSONのみを返し、前後に散文を含めないでください。")
        lines.append(SummarizerLocale.responseLanguageInstruction)
        let demographics = SpeakerDemographicsDigest
            .build(from: utterances)
            .renderForPrompt(speakerIDs: speakers, speakerNames: speakerNames)
        if !demographics.isEmpty {
            lines.append("")
            lines.append(demographics)
        }
        lines.append("")
        lines.append("発話：")
        for u in utterances {
            lines.append(compactLine(for: u, speakerNames: speakerNames))
        }
        lines.append("")
        lines.append("---")
        lines.append("重要：上記の指示に従い、windowIndex、timeStart、timeEnd、topicSnapshot、moodSnapshot、perSpeaker のフィールドを持つウィンドウ中間JSONオブジェクトを1つだけ生成してください。出力の最初の文字は必ず `{` でなければなりません。上記の発話リストをエコーしないでください。散文も一切含めないでください。")
        // Restate the language directive LAST — recency wins on
        // long prompts (same reason the JSON-shape rule is
        // restated in this sandwich). The copy mid-prompt sits
        // thousands of tokens back, above the utterance list, and
        // the small quantized models drift to the transcript's
        // (or the prompt's own) language without this reminder —
        // Apple FM honored the early copy, these did not.
        lines.append(SummarizerLocale.responseLanguageInstruction)
        return lines.joined(separator: "\n")
    }

    func buildDeepMergePrompt(
        intermediates: [MLXLLMDeepWindowIntermediate],
        allUtterances: [UtteranceEstimate],
        speakerNames: [String: String]
    ) -> String {
        let allSpeakers = allUtterances.orderedSpeakerIDs
        let speakerList = allSpeakers.joined(separator: ", ")
        var lines: [String] = []
        lines.reserveCapacity(intermediates.count + 24)
        lines.append("あなたは複数話者の会話の最終要約を作成するアナリストです。")
        lines.append("以下には2種類の補完的な入力があります：(a) 会話の\(intermediates.count)個の連続するウィンドウのJSON要約（時系列順）、(b) セッション全体から選ばれた最も特徴的な発話の抜粋。会話には合計\(allUtterances.count)発話があります。")
        lines.append("ウィンドウ要約を統合して話者ごとの一貫した弧を作成し（各話者はウィンドウを跨いで1人の人物として扱い、弧をウィンドウごとに分割しないでください）、話者ごとの要約を書く際は発話の抜粋を引用や具体的な表現の根拠として活用してください。")
        lines.append("")
        lines.append("次の4つのフィールドを持つJSONオブジェクトを1つ生成してください：")
        lines.append("  \"setting\" — 会話の状況・場面・レジスタを1文で示してください（例：「友人同士のカジュアルな電話」「就職面接」「教室での議論」）。具体的な場所や機関を創作せず、一般的に保ってください。最初に出力し、以降の内容と一貫させてください。")
        lines.append("  \"topic\" — 会話の主題を1〜2文で（すべてのウィンドウの話題スナップショットを考慮）。")
        lines.append("  \"overallMood\" — セッション全体の感情的な雰囲気を1段落で — 末尾のウィンドウだけでなく、弧全体を記述してください。")
        lines.append("  \"perSpeaker\" — 配列、次の話者IDごとに1エントリ：\(speakerList)。")
        lines.append("各perSpeakerエントリの形式：{ \"speakerID\": <id>, \"summary\": <セッション全体にわたる1段落>, \"dominantMood\": <短いフレーズ> }。")
        // No modalityFlags guidance — Llama window
        // intermediates omit that field because the row data
        // doesn't carry the per-modality vectors.
        lines.append("以下の話者人口統計ブロックに性別が記載されている場合、その話者の代名詞として全体で使用してください — 女性は「she/her」、男性は「he/him」、子供または性別が記載されていない場合は「they/them」。（日本語のように代名詞を省略する言語では無関係です。）")
        lines.append("有効なJSONのみを返し、前後に散文を含めないでください。")
        lines.append(SummarizerLocale.responseLanguageInstruction)
        let demographics = SpeakerDemographicsDigest
            .build(from: allUtterances)
            .renderForPrompt(speakerIDs: allSpeakers, speakerNames: speakerNames)
        if !demographics.isEmpty {
            lines.append("")
            lines.append(demographics)
        }
        lines.append("")
        lines.append("ウィンドウ要約(各JSONオブジェクト)：")
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.withoutEscapingSlashes]
        for intermediate in intermediates {
            if let data = try? encoder.encode(intermediate),
               let json = String(data: data, encoding: .utf8) {
                lines.append(json)
            }
        }
        // Hybrid merge — see `MLXQwenSpec.buildDeepMergePrompt`
        // for the rationale. Without raw utterances the merge
        // is "summarizing summaries"; injecting top-N by
        // speaker-balanced TF-IDF gives it verbatim quotes
        // and specific phrasing to draw on.
        let supplementalCap = Self.deepMergeSupplementalUtterances
        let topIDs = Informativeness.topNBalancedBySpeaker(
            supplementalCap,
            utterances: allUtterances
        )
        let supplemental = allUtterances.filter { topIDs.contains($0.id) }
        if !supplemental.isEmpty {
            lines.append("")
            lines.append("発話の抜粋（全\(allUtterances.count)発話のうち最も特徴的な\(supplemental.count)発話 — 話者ごとの記述で引用や具体的な内容に使ってください）：")
            for u in supplemental {
                lines.append(compactLine(for: u, speakerNames: speakerNames))
            }
        }
        lines.append("")
        lines.append("---")
        lines.append("重要：上記の指示に従い、setting、topic、overallMood、perSpeaker の4フィールドを持つ最終要約JSONオブジェクトを1つだけ生成してください。出力の最初の文字は必ず `{` でなければなりません。上記のウィンドウ要約や発話の抜粋をエコーしないでください。散文も一切含めないでください。")
        // Restate the language directive LAST — recency wins on
        // long prompts (same reason the JSON-shape rule is
        // restated in this sandwich). The copy mid-prompt sits
        // thousands of tokens back, above the utterance list, and
        // the small quantized models drift to the transcript's
        // (or the prompt's own) language without this reminder —
        // Apple FM honored the early copy, these did not.
        lines.append(SummarizerLocale.responseLanguageInstruction)
        return lines.joined(separator: "\n")
    }

    /// Count of raw utterances to append to the deep-merge
    /// prompt as quoted-detail context. Same value as Qwen
    /// (50) — Llama's `compactLine` is shorter per row so
    /// the supplemental section costs even less token-wise
    /// here.
    private static let deepMergeSupplementalUtterances = 50

    /// Stripped row format: speaker / time / transcript only.
    /// The full Qwen-shape row (label + V/A/D + aP + tP)
    /// caused Llama-Swallow to echo the per-row input
    /// pattern back as its output on long prompts — see
    /// `MLXQwenSpec.compactLine` for the contrast. Stripping
    /// drops per-row token cost ~70% (160 → 25-45 tokens) and
    /// removes the rich format the model was echoing. The
    /// trade-off is losing the modality-disagreement signal
    /// on this backend; the user picks Qwen if they need it.
    func compactLine(
        for u: UtteranceEstimate,
        speakerNames: [String: String]
    ) -> String {
        var fields: [String] = []
        fields.append("speaker=\(u.speakerID)")
        if let name = speakerNames[u.speakerID], !name.isEmpty {
            fields.append("name=\(name)")
        }
        fields.append(String(format: "t=%.1fs", u.start))
        fields.append("text=\"\(MLXLLMRendering.escapedTranscript(u.transcript))\"")
        return "- " + fields.joined(separator: " ")
    }
}
