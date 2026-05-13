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

    /// Cap on output tokens. Each suggestion is ~100–150 tokens of
    /// JSON (rowIndex + kind + suggestedText + reason + confidence)
    /// and a real-world Japanese session can produce 30+ flagged
    /// rows, which overshoots the previous 1536 cap and lands the
    /// parser at "Unexpected end of file" every time. 4096 fits
    /// ~25–35 entries comfortably; the tolerant parser below
    /// recovers the prefix if the model still runs past it on a
    /// particularly noisy session.
    private static let maxOutputTokens = 4096

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
    /// (defensive even though `/no_think` is in the prompt),
    /// ```fence``` wrappers, AND truncated output — when the model
    /// ran past the output-token cap mid-array we recover everything
    /// up to the last complete suggestion entry rather than throwing
    /// the whole list away. Drops suggestions whose `rowIndex` isn't
    /// a row we sent — better silent drop than aliasing onto the
    /// wrong row and corrupting the user's transcript.
    private static func parse(
        raw: String,
        indexToID: [Int: UUID],
        originalsByID: [UUID: String]
    ) throws -> [TranscriptionSuggestion] {
        let dethought = stripThinkBlocks(raw)
        let stripped = stripCodeFence(dethought)
        guard let braceStart = stripped.firstIndex(of: "{") else {
            throw TranscriptionReviewError.decodeFailed(reason: "no JSON object found")
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
            guard let recovered = recoverTruncatedSuggestions(
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

    /// Salvage a truncated `{"suggestions":[...]}` blob by scanning
    /// forward, tracking string-literal and brace-nesting state, and
    /// remembering the position immediately after the *most recent
    /// top-level object that closed* inside the array. When the
    /// scan hits end-of-input mid-entry, we cut the string at that
    /// remembered position, then synthesize `]}` to close the array
    /// and outer object. The resulting JSON contains every complete
    /// suggestion the model managed to emit before the token budget
    /// ran out; the in-flight entry is discarded.
    ///
    /// Returns nil when the input doesn't look like our expected
    /// `{"suggestions":[…` shape — let the caller surface the
    /// original parse error in that case rather than silently
    /// returning an empty list.
    private static func recoverTruncatedSuggestions(stripped: String) -> String? {
        guard let arrayOpenRange = stripped.range(of: "[") else { return nil }
        // Sanity check: there should be a `"suggestions"` token
        // before the bracket. If not, this isn't the shape we
        // expect and recovery would be guessing.
        let prefix = stripped[..<arrayOpenRange.lowerBound]
        guard prefix.contains("suggestions") else { return nil }

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
