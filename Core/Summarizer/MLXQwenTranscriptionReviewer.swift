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
    private var container: ModelContainer?

    /// Cap on output tokens. Review JSON tends to be longer per
    /// row than summary JSON (each entry carries its own reason
    /// paragraph), but the input is bounded — `maxPromptUtterances`
    /// rows × ~4 fields each. 1536 leaves comfortable headroom
    /// without inviting unbounded runs that would push memory.
    private static let maxOutputTokens = 1536

    /// Cap on utterances per review pass. Qwen3-8B's 32k context
    /// is far roomier than Apple FM's, but prefill KV-cache cost
    /// scales linearly and we're already paying for 4.6 GB of
    /// resident weights — keep the working set tight. Trailing
    /// window mirrors the summarizer's strategy for long sessions.
    private static let maxPromptUtterances = 80

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
        MLX.GPU.set(cacheLimit: 32 * 1024 * 1024)
        do {
            let configuration = ModelConfiguration(directory: modelDirectory)
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
    ) async throws -> [TranscriptionSuggestion] {
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
            language: language
        )
        AppLog.app.info(
            "MLXQwenTranscriptionReviewer reviewing \(promptUtterances.count, privacy: .public) utterances (prompt \(prompt.count, privacy: .public) chars)"
        )

        let raw: String
        do {
            raw = try await container.perform { context -> String in
                let userInput = UserInput(prompt: prompt)
                let lmInput = try await context.processor.prepare(input: userInput)
                var parameters = GenerateParameters(
                    maxTokens: Self.maxOutputTokens,
                    temperature: 0.2
                )
                parameters.prefillStepSize = 64
                let result = try MLXLMCommon.generate(
                    input: lmInput,
                    parameters: parameters,
                    context: context,
                    // Cooperative cancellation — see the matching
                    // hook in `MLXQwenSummarizer` for the rationale.
                    didGenerate: { (_: [Int]) -> GenerateDisposition in
                        Task.isCancelled ? .stop : .more
                    }
                )
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

        return try Self.parse(
            raw: raw,
            indexToID: indexToID,
            originalsByID: Dictionary(
                uniqueKeysWithValues: promptUtterances.map { ($0.id, $0.transcript) }
            )
        )
    }

    // MARK: - Prompt + parser

    private static func buildPrompt(
        utterances: [UtteranceEstimate],
        speakerNames: [String: String],
        truncatedFromTotal: Int?,
        language: ReviewLanguage
    ) -> String {
        var lines: [String] = []
        lines.reserveCapacity(utterances.count + 24)
        lines.append("You are a transcription proofreader for a multi-speaker conversation.")
        // Language anchoring up front — Qwen3 was trained on a
        // Chinese-dominant corpus and otherwise reads kanji as
        // Mandarin (suggesting Chinese-style replacements that are
        // meaningless to the user). Putting this above the task
        // description biases the rest of the prompt into the right
        // language frame from the first token.
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
        lines.append("  \"suggestions\": array of { \"rowIndex\": int, \"kind\": one of \"homophone\"|\"contextual\"|\"grammar\"|\"other\", \"suggestedText\": string, \"reason\": one short sentence, \"confidence\": number 0.0–1.0 }")
        lines.append("Omit rows that read correctly. If you cannot propose a concrete fix, set suggestedText to \"\" and still include the reason.")
        lines.append("CRITICAL: suggestedText MUST differ from the original transcript. Never echo the original text back unchanged — if the only fix you can think of is identical to the original, omit the row entirely.")
        lines.append("Return ONLY valid JSON, no prose before or after.")
        lines.append("/no_think")
        if let total = truncatedFromTotal {
            lines.append("")
            lines.append("NOTE: This conversation has \(total) utterances total; only the most recent \(utterances.count) are shown below. Review only these rows.")
        }
        lines.append("")
        lines.append("Utterances:")
        for (idx, u) in utterances.enumerated() {
            lines.append(compactLine(rowIndex: idx + 1, for: u, speakerNames: speakerNames))
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
    /// (defensive even though `/no_think` is in the prompt) and
    /// ```fence``` wrappers. Drops suggestions whose `rowIndex`
    /// isn't a row we sent — better silent drop than aliasing onto
    /// the wrong row and corrupting the user's transcript.
    private static func parse(
        raw: String,
        indexToID: [Int: UUID],
        originalsByID: [UUID: String]
    ) throws -> [TranscriptionSuggestion] {
        let dethought = stripThinkBlocks(raw)
        let stripped = stripCodeFence(dethought)
        guard let braceStart = stripped.firstIndex(of: "{"),
              let braceEnd = stripped.lastIndex(of: "}") else {
            throw TranscriptionReviewError.decodeFailed(reason: "no JSON object found")
        }
        let jsonString = String(stripped[braceStart...braceEnd])
        guard let data = jsonString.data(using: .utf8) else {
            throw TranscriptionReviewError.decodeFailed(reason: "non-utf8 output")
        }
        struct Wire: Decodable {
            struct Entry: Decodable {
                let rowIndex: Int
                let kind: String
                let suggestedText: String
                let reason: String
                let confidence: Float?
            }
            let suggestions: [Entry]
        }
        let decoded: Wire
        do {
            decoded = try JSONDecoder().decode(Wire.self, from: data)
        } catch {
            throw TranscriptionReviewError.decodeFailed(
                reason: String(describing: error)
            )
        }
        return decoded.suggestions.compactMap { entry -> TranscriptionSuggestion? in
            guard let utteranceID = indexToID[entry.rowIndex] else { return nil }
            let kind = TranscriptionSuggestion.Kind(rawValue: entry.kind) ?? .other
            let original = originalsByID[utteranceID] ?? ""
            // Drop no-op suggestions where the model echoed the
            // original verbatim. Qwen frequently emits these even
            // when told "omit rows that read correctly" — the
            // schema constraint plus the model's tendency to fill
            // every output slot beats the instruction. Compare on
            // normalized form so a stray trailing space doesn't
            // count as a real change.
            let trimmedSuggestion = entry.suggestedText
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let trimmedOriginal = original
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmedSuggestion == trimmedOriginal { return nil }
            return TranscriptionSuggestion(
                utteranceID: utteranceID,
                kind: kind,
                originalText: original,
                suggestedText: entry.suggestedText,
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
