import Foundation
import Fusion
import XephonLogging
import MLX
import MLXLLM
import MLXLMCommon

/// MLX-backed transcription reviewer using the same Qwen3-8B 4-bit
/// weights as `MLXQwenSummarizer`. Owns its own `ModelContainer` —
/// the controller is responsible for unloading whichever Qwen-backed
/// actor isn't currently in use, since loading both simultaneously
/// would blow the per-app memory ceiling at ~9 GB of resident
/// weights.
///
/// Lifecycle mirrors the summarizer: `load()` brings the weights in,
/// `review(...)` runs one pass, `unload()` releases.
public actor MLXQwenTranscriptionReviewer: TranscriptionReviewer {
    public let modelIdentifier: String
    private let modelDirectory: URL
    /// Model family — gates a few prompt-token differences (Qwen3's
    /// `/no_think`, mainly). The actual model loading is family-
    /// agnostic via the MLX-LM factory.
    private let family: LLMModelFamily
    private var container: ModelContainer?

    /// Cap on output tokens. Each issue is ~50–80 tokens of JSON
    /// (rowIndex + kind + reason + confidence — no suggested text)
    /// and a real-world Japanese session can produce 30+ flagged
    /// rows. 4096 fits 60+ entries comfortably even for the noisiest
    /// sessions; the tolerant parser below recovers the prefix if
    /// the model still runs past it.
    private static let maxOutputTokens = 4096

    /// Cap on utterances per review pass. Qwen3-8B's 32k context
    /// is far roomier than Apple FM's, but prefill KV-cache cost
    /// scales linearly and we're already paying for 4.6 GB of
    /// resident weights — keep the working set tight. Trailing
    /// window mirrors the summarizer's strategy for long sessions.
    private static let maxPromptUtterances = 80

    public init(
        modelIdentifier: String,
        modelDirectory: URL,
        family: LLMModelFamily = .qwen
    ) {
        self.modelIdentifier = modelIdentifier
        self.modelDirectory = modelDirectory
        self.family = family
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
        // 128 MB rationale (SER actors torn down before this runs).
        MLX.GPU.set(cacheLimit: 128 * 1024 * 1024)
        do {
            // Family-specific `extraEOSTokens` — without all three
            // Llama 3.1 stop tokens (`<|end_of_text|>`,
            // `<|eom_id|>`, `<|eot_id|>`) generation runs to the
            // 4096-token cap because MLX-LM only honors a single
            // `eos_token` from tokenizer_config.json while
            // Llama-Swallow has been observed emitting the base-
            // model EOS instead of the chat EOT. See
            // `LLMModelFamily.extraEOSTokens`.
            let configuration = ModelConfiguration(
                directory: modelDirectory,
                extraEOSTokens: family.extraEOSTokens
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
        guard !utterances.isEmpty else { return [] }

        let promptUtterances: [UtteranceEstimate]
        let truncatedFrom: Int?
        if utterances.count > Self.maxPromptUtterances {
            promptUtterances = Array(utterances.suffix(Self.maxPromptUtterances))
            truncatedFrom = utterances.count
        } else {
            promptUtterances = utterances
            truncatedFrom = nil
        }
        let indexToID: [Int: UUID] = Dictionary(
            uniqueKeysWithValues: promptUtterances
                .enumerated()
                .map { ($0.offset + 1, $0.element.id) }
        )

        let prompt = Self.buildPrompt(
            utterances: promptUtterances,
            speakerNames: speakerNames,
            truncatedFromTotal: truncatedFrom,
            language: language,
            family: family
        )
        AppLog.app.info(
            "MLXQwenTranscriptionReviewer reviewing \(promptUtterances.count, privacy: .public) utterances (prompt \(prompt.count, privacy: .public) chars)"
        )

        let raw: String
        let family = self.family
        do {
            raw = try await container.perform { context -> String in
                let userInput = UserInput(prompt: prompt)
                let lmInput = try await context.processor.prepare(input: userInput)
                AppLog.app.info(
                    "MLXQwenTranscriptionReviewer prepared input: \(lmInput.text.tokens.size, privacy: .public) prompt tokens"
                )
                var parameters = GenerateParameters(
                    maxTokens: Self.maxOutputTokens,
                    temperature: 0.2
                )
                // Llama-Swallow's Japanese fine-tune occasionally
                // loops on English-language JSON instructions —
                // see the matching note in `MLXQwenSummarizer`.
                if family == .llama {
                    parameters.repetitionPenalty = 1.05
                }
                // Matches MLXQwenSummarizer — see its generate
                // block for the 128-vs-64 watchdog rationale.
                parameters.prefillStepSize = 128
                let startTime = Date()
                var firstTokenTime: Date? = nil
                let result = try MLXLMCommon.generate(
                    input: lmInput,
                    parameters: parameters,
                    context: context,
                    // Cooperative cancellation — see the matching
                    // hook in `MLXQwenSummarizer` for the rationale.
                    didGenerate: { (tokens: [Int]) -> GenerateDisposition in
                        if firstTokenTime == nil {
                            firstTokenTime = Date()
                            let prefillSec = firstTokenTime!.timeIntervalSince(startTime)
                            let msg = String(
                                format: "MLXQwenTranscriptionReviewer prefill done in %.2f s; starting decode",
                                prefillSec
                            )
                            AppLog.app.info("\(msg, privacy: .public)")
                        } else if tokens.count.isMultiple(of: 64) {
                            let elapsed = Date().timeIntervalSince(firstTokenTime!)
                            let tps = elapsed > 0 ? Double(tokens.count) / elapsed : 0
                            let msg = String(
                                format: "MLXQwenTranscriptionReviewer decode: %d tokens in %.1f s (%.1f t/s)",
                                tokens.count, elapsed, tps
                            )
                            AppLog.app.info("\(msg, privacy: .public)")
                        }
                        return Task.isCancelled ? .stop : .more
                    }
                )
                let totalSec = Date().timeIntervalSince(startTime)
                let finishMsg = String(
                    format: "MLXQwenTranscriptionReviewer generate finished: %d tokens in %.1f s total",
                    result.tokens.count, totalSec
                )
                AppLog.app.info("\(finishMsg, privacy: .public)")
                return result.output
            }
        } catch let error as TranscriptionReviewError {
            throw error
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            AppLog.app.error(
                "MLXQwenTranscriptionReviewer.generate failed: \(String(describing: error), privacy: .public)"
            )
            throw TranscriptionReviewError.inferenceFailed(
                reason: String(describing: error)
            )
        }
        if Task.isCancelled { throw CancellationError() }
        AppLog.app.info(
            "MLXQwenTranscriptionReviewer raw output: \(raw.count, privacy: .public) chars"
        )

        return try Self.parse(raw: raw, indexToID: indexToID)
    }

    // MARK: - Prompt + parser

    private static func buildPrompt(
        utterances: [UtteranceEstimate],
        speakerNames: [String: String],
        truncatedFromTotal: Int?,
        language: ReviewLanguage,
        family: LLMModelFamily
    ) -> String {
        var lines: [String] = []
        lines.reserveCapacity(utterances.count + 24)
        // Instruction language follows the model family — Llama-
        // Swallow (Japanese fine-tune) is more concise + reliable
        // when prompted in Japanese; the JSON keys and "kind"
        // enum values stay English in both copies because they
        // are parsed back by Swift. The `language.qwenAnchor`
        // line stays anchored regardless — that's about the
        // transcript content's language, not the prompt's.
        switch family {
        case .qwen:
            lines.append("You are a transcription proofreader for a multi-speaker conversation.")
            // Language anchoring up front — Qwen3 was trained on
            // a Chinese-dominant corpus and otherwise reads kanji
            // as Mandarin (suggesting Chinese-style replacements
            // that are meaningless to the user). Putting this
            // above the task description biases the rest of the
            // prompt into the right language frame from the
            // first token.
            lines.append(language.qwenAnchor)
            lines.append("")
            lines.append("Each utterance below has a 1-based row index, speaker id, time, and transcript.")
            lines.append("Find rows whose transcript is likely WRONG because of:")
            lines.append("  - a misrecognized homophone or near-homophone,")
            lines.append("  - a sentence that does not fit the session context (non-sequitur),")
            lines.append("  - a clear grammar slip that reads as an ASR error, not a stylistic choice.")
            lines.append("Do NOT flag rows that are merely informal, dialectal, or unusual but coherent.")
            lines.append("")
            lines.append("Return ONLY a JSON object with one field:")
            lines.append("  \"issues\": array of { \"rowIndex\": int, \"kind\": one of \"homophone\"|\"contextual\"|\"grammar\"|\"other\", \"reason\": one short sentence describing what looks wrong, \"confidence\": number 0.0–1.0 }")
            lines.append("DO NOT propose a corrected transcript — the human user will edit the row themselves. Just identify which rows look wrong and why.")
            lines.append("Omit rows that read correctly.")
            lines.append("Return ONLY valid JSON, no prose before or after.")
            // Pin the freeform `reason` field's language to the
            // user's iPadOS app-language pick.
            lines.append("The \"reason\" text in each issue MUST be written in this language: \(SummarizerLocale.responseLanguageNameInEnglish). No other language is acceptable.")
        case .llama:
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
        }
        if family == .qwen {
            lines.append("/no_think")
        }
        if let total = truncatedFromTotal {
            lines.append("")
            switch family {
            case .qwen:
                lines.append("NOTE: This conversation has \(total) utterances total; only the most recent \(utterances.count) are shown below. Review only these rows.")
            case .llama:
                lines.append("注意：この会話は全体で\(total)発話ありますが、以下には最新の\(utterances.count)発話のみが表示されています。これらの行のみを校閲してください。")
            }
        }
        lines.append("")
        switch family {
        case .qwen:  lines.append("Utterances:")
        case .llama: lines.append("発話：")
        }
        for (idx, u) in utterances.enumerated() {
            lines.append(compactLine(rowIndex: idx + 1, for: u, speakerNames: speakerNames))
        }
        // Instruction sandwich — restate the directive after the
        // utterance list so the model's recent attention has
        // "produce JSON" rather than the last utterance line. See
        // `MLXQwenSummarizer.buildPrompt` for the failure mode
        // this prevents (Llama echoing the input format).
        lines.append("")
        lines.append("---")
        switch family {
        case .qwen:
            lines.append("IMPORTANT: Follow the instructions above and produce exactly one valid JSON object with an `issues` field. The FIRST character of your output MUST be `{`. Do NOT echo the utterance list above; do NOT add any prose. Omit rows that read correctly.")
        case .llama:
            lines.append("重要：上記の指示に従い、issuesフィールドを持つ有効なJSONオブジェクトを1つだけ生成してください。出力の最初の文字は必ず `{` でなければなりません。上記の発話リストをエコーしないでください。散文も一切含めないでください。正しく読める行は省略してください。")
        }
        return lines.joined(separator: "\n")
    }

    private static func compactLine(
        rowIndex: Int,
        for u: UtteranceEstimate,
        speakerNames: [String: String]
    ) -> String {
        var fields: [String] = []
        fields.append("row=\(rowIndex)")
        fields.append("speaker=\(u.speakerID)")
        if let name = speakerNames[u.speakerID], !name.isEmpty {
            fields.append("name=\(name)")
        }
        fields.append(String(format: "t=%.1fs", u.start))
        let escaped = u.transcript
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
        fields.append("text=\"\(escaped)\"")
        return "- " + fields.joined(separator: " ")
    }

    /// Decode the LLM's JSON output. Tolerates `<think>` blocks
    /// (defensive even though `/no_think` is in the prompt),
    /// ```fence``` wrappers, AND truncated output — when the model
    /// ran past the output-token cap mid-array we recover everything
    /// up to the last complete issue entry rather than throwing the
    /// whole list away. Drops issues whose `rowIndex` isn't a row
    /// we sent — better silent drop than aliasing the flag onto the
    /// wrong row and confusing the user.
    private static func parse(
        raw: String,
        indexToID: [Int: UUID]
    ) throws -> [TranscriptionIssue] {
        let dethought = stripThinkBlocks(raw)
        let stripped = stripCodeFence(dethought)
        guard let braceStart = stripped.firstIndex(of: "{") else {
            throw TranscriptionReviewError.decodeFailed(reason: "no JSON object found")
        }
        struct Wire: Decodable {
            struct Entry: Decodable {
                let rowIndex: Int
                let kind: String
                let reason: String
                let confidence: Float?
            }
            let issues: [Entry]
        }
        // First try strict — the whole output between the first `{`
        // and the last `}`. Works whenever the model closed its
        // JSON cleanly.
        let strict: String? = {
            guard let braceEnd = stripped.lastIndex(of: "}") else { return nil }
            return String(stripped[braceStart...braceEnd])
        }()
        let decoded: Wire
        if let strict, let data = strict.data(using: .utf8),
           let ok = try? JSONDecoder().decode(Wire.self, from: data) {
            decoded = ok
        } else {
            // Truncated output: walk the suffix from the array's
            // opening `[`, find the last balanced entry object, and
            // close the array + outer object manually so the JSON
            // parses. The in-flight (broken) entry is discarded;
            // every complete one is preserved.
            guard let recovered = recoverTruncatedIssues(
                stripped: String(stripped[braceStart...])
            ) else {
                throw TranscriptionReviewError.decodeFailed(
                    reason: "no JSON object found"
                )
            }
            guard let data = recovered.data(using: .utf8) else {
                throw TranscriptionReviewError.decodeFailed(
                    reason: "non-utf8 output"
                )
            }
            AppLog.app.info(
                "MLXQwenTranscriptionReviewer recovered truncated JSON (\(data.count, privacy: .public) bytes)"
            )
            do {
                decoded = try JSONDecoder().decode(Wire.self, from: data)
            } catch {
                throw TranscriptionReviewError.decodeFailed(
                    reason: String(describing: error)
                )
            }
        }
        return decoded.issues.compactMap { entry -> TranscriptionIssue? in
            guard let utteranceID = indexToID[entry.rowIndex] else { return nil }
            let kind = TranscriptionIssue.Kind(rawValue: entry.kind) ?? .other
            return TranscriptionIssue(
                utteranceID: utteranceID,
                kind: kind,
                reason: entry.reason,
                confidence: entry.confidence
            )
        }
    }

    private static func stripThinkBlocks(_ raw: String) -> String {
        var s = raw
        while let openRange = s.range(of: "<think>") {
            if let closeRange = s.range(
                of: "</think>",
                range: openRange.upperBound..<s.endIndex
            ) {
                s.removeSubrange(openRange.lowerBound..<closeRange.upperBound)
            } else {
                s.removeSubrange(openRange.lowerBound..<s.endIndex)
                break
            }
        }
        if let strayClose = s.range(of: "</think>") {
            s.removeSubrange(s.startIndex..<strayClose.upperBound)
        }
        return s
    }

    /// Salvage a truncated `{"issues":[...]}` blob by scanning
    /// forward, tracking string-literal and brace-nesting state, and
    /// remembering the position immediately after the *most recent
    /// top-level object that closed* inside the array. When the
    /// scan hits end-of-input mid-entry, we cut the string at that
    /// remembered position, then synthesize `]}` to close the array
    /// and outer object. The resulting JSON contains every complete
    /// issue the model managed to emit before the token budget ran
    /// out; the in-flight entry is discarded.
    ///
    /// Returns nil when the input doesn't look like our expected
    /// `{"issues":[…` shape — let the caller surface the original
    /// parse error in that case rather than silently returning an
    /// empty list.
    private static func recoverTruncatedIssues(stripped: String) -> String? {
        guard let arrayOpenRange = stripped.range(of: "[") else { return nil }
        // Sanity check: there should be an `"issues"` token before
        // the bracket. If not, this isn't the shape we expect and
        // recovery would be guessing.
        let prefix = stripped[..<arrayOpenRange.lowerBound]
        guard prefix.contains("issues") else { return nil }

        var depth = 0
        var inString = false
        var escape = false
        var lastCompleteEntryEnd: String.Index? = nil

        var i = arrayOpenRange.upperBound
        while i < stripped.endIndex {
            let ch = stripped[i]
            if escape {
                escape = false
            } else if inString {
                if ch == "\\" {
                    escape = true
                } else if ch == "\"" {
                    inString = false
                }
            } else {
                switch ch {
                case "\"":
                    inString = true
                case "{":
                    depth += 1
                case "}":
                    depth -= 1
                    if depth == 0 {
                        // Just closed a top-level entry inside the
                        // array. Record the position right after
                        // this `}` so we can rewind here if the
                        // scan runs out mid-next-entry.
                        lastCompleteEntryEnd = stripped.index(after: i)
                    }
                case "]" where depth == 0:
                    // The array closed normally — the strict
                    // parser should have handled this; fall
                    // through to nil so the caller surfaces the
                    // original error.
                    return nil
                default:
                    break
                }
            }
            i = stripped.index(after: i)
        }

        guard let cutEnd = lastCompleteEntryEnd else { return nil }
        // `stripped[..<cutEnd]` ends at "...},". Strip any trailing
        // comma/whitespace, then close the array + object.
        var truncated = String(stripped[..<cutEnd])
        while let last = truncated.last,
              last == "," || last.isWhitespace {
            truncated.removeLast()
        }
        truncated.append("]}")
        return truncated
    }

    private static func stripCodeFence(_ raw: String) -> String {
        var s = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if s.hasPrefix("```") {
            if let firstNewline = s.firstIndex(of: "\n") {
                s = String(s[s.index(after: firstNewline)...])
            }
        }
        if s.hasSuffix("```") {
            s = String(s.dropLast(3)).trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return s
    }
}
