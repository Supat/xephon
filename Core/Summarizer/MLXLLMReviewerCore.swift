import Foundation
import Fusion
import XephonLogging
import MLX
import MLXLLM
import MLXLMCommon

/// Shared infrastructure behind `MLXQwenTranscriptionReviewer`
/// and `MLXLlamaTranscriptionReviewer`. Both actors are thin
/// wrappers around `MLXLLMReviewerCore`'s static orchestration;
/// the only per-family differences are encapsulated in the
/// `MLXLLMReviewerSpec` an actor hands to the core.
///
/// Mirrors the structure of `MLXLLMSummarizerCore` for the
/// reviewer side. `runInference`, `stripThinkBlocks`, and
/// `stripCodeFence` are duplicated rather than shared with the
/// summarizer core — they're self-contained ~60-line helpers
/// and the two paths could legitimately diverge later (different
/// timing-log labels, cancellation semantics, etc.).

// MARK: - Actor type unification

/// Storage type for the coordinator's resident MLX reviewer.
/// Both `MLXQwenTranscriptionReviewer` and
/// `MLXLlamaTranscriptionReviewer` conform — each is a thin
/// actor wrapping a per-family `MLXLLMReviewerSpec`. Surfaces
/// just the lifecycle bits (`load`, `unload`) the coordinator
/// drives directly; the `review` call comes through
/// `TranscriptionReviewer`.
public protocol MLXLLMReviewerActor: TranscriptionReviewer {
    func load() async throws
    func unload() async
}

// MARK: - Spec contract

/// Per-family behavior the shared reviewer orchestration needs.
/// Concrete types: `MLXQwenReviewerSpec` and
/// `MLXLlamaReviewerSpec`. Sendable because instances are
/// handed across actor boundaries from each per-family actor
/// into the static orchestration functions below.
internal protocol MLXLLMReviewerSpec: Sendable {
    var family: LLMModelFamily { get }

    /// Cap on utterances per chunk. Long sessions get chunked
    /// rather than truncated — review needs per-utterance
    /// proofreading on every row, so silently skipping a
    /// trailing-window prefix isn't acceptable.
    var maxPromptUtterances: Int { get }

    /// Output-token cap per chunk inference.
    var maxOutputTokens: Int { get }

    /// Rows of the PREVIOUS chunk replayed at the top of each
    /// chunk's prompt as unindexed, non-flaggable context. Gives
    /// the `contextual` (non-sequitur) kind real topic continuity
    /// at chunk boundaries instead of a cold start. 0 disables.
    var contextOverlapRows: Int { get }

    /// Stop tokens for `ModelConfiguration.extraEOSTokens`.
    /// Same shape + rationale as the summarizer spec.
    var extraEOSTokens: Set<String> { get }

    /// Family-specific repetition penalty for the generator,
    /// or nil to use the library default.
    var repetitionPenalty: Float? { get }

    /// Build the per-chunk prompt for one review pass.
    /// Includes a chunk-aware note (when `totalChunks > 1`)
    /// telling the model not to over-flag rows as
    /// non-sequiturs just because earlier / later context
    /// isn't visible in this chunk.
    func buildPrompt(
        utterances: [UtteranceEstimate],
        speakerNames: [String: String],
        language: ReviewLanguage,
        chunkIndex: Int,
        totalChunks: Int,
        contextPrefix: [UtteranceEstimate]
    ) -> String
}

// MARK: - Orchestration

internal enum MLXLLMReviewerCore {
    /// Top-level entry point. Always chunks the input so every
    /// utterance gets a chance to be flagged; concatenates
    /// per-chunk issue lists into the final return value.
    static func review(
        container: ModelContainer,
        utterances: [UtteranceEstimate],
        speakerNames: [String: String],
        language: ReviewLanguage,
        spec: any MLXLLMReviewerSpec
    ) async throws -> [TranscriptionIssue] {
        guard !utterances.isEmpty else { return [] }
        // Chunk starts (not materialized chunks) so each iteration
        // can also carve the previous rows as boundary context.
        let chunkStarts = Array(stride(from: 0, to: utterances.count, by: spec.maxPromptUtterances))
        AppLog.app.info(
            "MLX reviewer (\(spec.family.rawValue, privacy: .public)) reviewing \(utterances.count, privacy: .public) utterances in \(chunkStarts.count, privacy: .public) chunk(s)"
        )
        var allIssues: [TranscriptionIssue] = []
        for (chunkIndex, start) in chunkStarts.enumerated() {
            if Task.isCancelled { throw CancellationError() }
            let end = min(start + spec.maxPromptUtterances, utterances.count)
            let chunk = Array(utterances[start..<end])
            // Boundary context: the tail of the previous chunk,
            // replayed unindexed (prompted as non-flaggable; also
            // absent from indexToID, so a flag against it can't
            // parse into an issue).
            let ctxStart = max(0, start - spec.contextOverlapRows)
            let contextPrefix = start == 0 ? [] : Array(utterances[ctxStart..<start])
            let issues = try await reviewChunk(
                container: container,
                chunk: chunk,
                speakerNames: speakerNames,
                language: language,
                chunkIndex: chunkIndex,
                totalChunks: chunkStarts.count,
                contextPrefix: contextPrefix,
                spec: spec
            )
            allIssues.append(contentsOf: issues)
            AppLog.app.info(
                "MLX reviewer chunk \(chunkIndex + 1, privacy: .public)/\(chunkStarts.count, privacy: .public) yielded \(issues.count, privacy: .public) issue(s)"
            )
        }
        // Post-parse grounding/sanity filter — see
        // ReviewIssueValidator for the failure classes it kills.
        return ReviewIssueValidator.filter(allIssues, utterances: utterances)
    }

    /// Single-chunk inference + parse. 1-based row indices are
    /// local to the chunk; the per-chunk `indexToID` map remaps
    /// them back to the session-wide utterance UUIDs the
    /// returned `TranscriptionIssue`s carry.
    static func reviewChunk(
        container: ModelContainer,
        chunk: [UtteranceEstimate],
        speakerNames: [String: String],
        language: ReviewLanguage,
        chunkIndex: Int,
        totalChunks: Int,
        contextPrefix: [UtteranceEstimate],
        spec: any MLXLLMReviewerSpec
    ) async throws -> [TranscriptionIssue] {
        let indexToID: [Int: UUID] = Dictionary(
            uniqueKeysWithValues: chunk.enumerated().map { ($0.offset + 1, $0.element.id) }
        )
        let prompt = spec.buildPrompt(
            utterances: chunk,
            speakerNames: speakerNames,
            language: language,
            chunkIndex: chunkIndex,
            totalChunks: totalChunks,
            contextPrefix: contextPrefix
        )
        let chunkLabel = "MLX reviewer (\(spec.family.rawValue)) chunk \(chunkIndex + 1)/\(totalChunks)"
        AppLog.app.info(
            "\(chunkLabel, privacy: .public) reviewing \(chunk.count, privacy: .public) utterances (prompt \(prompt.count, privacy: .public) chars)"
        )
        let raw = try await runInference(
            container: container,
            prompt: prompt,
            maxTokens: spec.maxOutputTokens,
            repetitionPenalty: spec.repetitionPenalty,
            label: chunkLabel
        )
        if Task.isCancelled { throw CancellationError() }
        AppLog.app.info(
            "\(chunkLabel, privacy: .public) raw output: \(raw.count, privacy: .public) chars"
        )
        return try parse(raw: raw, indexToID: indexToID)
    }

    // MARK: Inference helper

    /// Single inference call against the loaded container.
    /// Mirrors `MLXLLMSummarizerCore.runInference` — kept
    /// separate so the two paths can evolve their logging /
    /// cancellation independently if needed.
    static func runInference(
        container: ModelContainer,
        prompt: String,
        maxTokens: Int,
        repetitionPenalty: Float?,
        label: String
    ) async throws -> String {
        do {
            return try await container.perform { context -> String in
                let userInput = UserInput(prompt: prompt)
                let lmInput = try await context.processor.prepare(input: userInput)
                let preparedMsg = String(
                    format: "%@ prepared input: %d prompt tokens",
                    label, lmInput.text.tokens.size
                )
                AppLog.app.info("\(preparedMsg, privacy: .public)")
                var parameters = GenerateParameters(
                    maxTokens: maxTokens,
                    temperature: 0.2
                )
                if let penalty = repetitionPenalty {
                    parameters.repetitionPenalty = penalty
                }
                parameters.prefillStepSize = 128
                let startTime = Date()
                // Cancellable prefill — same rationale as
                // MLXLLMSummarizerCore.runInference (see
                // MLXCancellablePrefill's doc for the backgrounding
                // crash this prevents).
                let (remaining, iterator) = try MLXCancellablePrefill.primedIterator(
                    input: lmInput,
                    context: context,
                    parameters: parameters
                )
                var firstTokenTime: Date? = nil
                let result = MLXLMCommon.generate(
                    input: remaining,
                    context: context,
                    iterator: iterator,
                    didGenerate: { (tokens: [Int]) -> GenerateDisposition in
                        if firstTokenTime == nil {
                            firstTokenTime = Date()
                            let prefillSec = firstTokenTime!
                                .timeIntervalSince(startTime)
                            let msg = String(
                                format: "%@ prefill done in %.2f s; starting decode",
                                label, prefillSec
                            )
                            AppLog.app.info("\(msg, privacy: .public)")
                        } else if tokens.count.isMultiple(of: 64) {
                            let elapsed = Date()
                                .timeIntervalSince(firstTokenTime!)
                            let tps = elapsed > 0
                                ? Double(tokens.count) / elapsed
                                : 0
                            let msg = String(
                                format: "%@ decode: %d tokens in %.1f s (%.1f t/s)",
                                label, tokens.count, elapsed, tps
                            )
                            AppLog.app.info("\(msg, privacy: .public)")
                        }
                        return Task.isCancelled ? .stop : .more
                    }
                )
                let totalSec = Date().timeIntervalSince(startTime)
                let finishMsg = String(
                    format: "%@ generate finished: %d tokens in %.1f s total",
                    label, result.tokens.count, totalSec
                )
                AppLog.app.info("\(finishMsg, privacy: .public)")
                return result.output
            }
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            AppLog.app.error(
                "\(label, privacy: .public) generate failed: \(String(describing: error), privacy: .public)"
            )
            throw TranscriptionReviewError.inferenceFailed(
                reason: String(describing: error)
            )
        }
    }

    // MARK: Parser

    /// Decode the LLM's `{"issues": [...]}` output into
    /// `[TranscriptionIssue]`. Tolerant of `<think>` blocks,
    /// ```` ```json ``` ```` fences, and truncated output
    /// (`recoverTruncatedIssues` salvages every complete entry
    /// from a token-capped run). Drops issues whose
    /// `rowIndex` isn't in the chunk's `indexToID` map —
    /// better to lose a flag than alias onto the wrong row.
    static func parse(
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
                let excerpt: String?
            }
            let issues: [Entry]
        }
        let strict: String? = {
            // braceEnd >= braceStart: a stray `}` in prose BEFORE the
            // JSON opens, with the output cap cutting generation off
            // before any closing brace, inverts the pair — and the
            // ClosedRange subscript would trap, not throw.
            guard let braceEnd = stripped.lastIndex(of: "}"),
                  braceEnd >= braceStart else { return nil }
            return String(stripped[braceStart...braceEnd])
        }()
        let decoded: Wire
        if let strict, let data = strict.data(using: .utf8),
           let ok = try? JSONDecoder().decode(Wire.self, from: data) {
            decoded = ok
        } else {
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
                "MLX reviewer recovered truncated JSON (\(data.count, privacy: .public) bytes)"
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
                confidence: entry.confidence,
                excerpt: entry.excerpt
            )
        }
    }

    /// Salvage a truncated `{"issues":[...]}` blob by scanning
    /// forward, tracking string-literal and brace-nesting
    /// state, remembering the position after the most recent
    /// top-level object that closed inside the array. Cut +
    /// append `]}` to close cleanly. Returns nil for shapes
    /// that don't match the expected prefix.
    static func recoverTruncatedIssues(stripped: String) -> String? {
        guard let arrayOpenRange = stripped.range(of: "[") else { return nil }
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
                        lastCompleteEntryEnd = stripped.index(after: i)
                    }
                case "]" where depth == 0:
                    return nil
                default:
                    break
                }
            }
            i = stripped.index(after: i)
        }
        guard let cutEnd = lastCompleteEntryEnd else { return nil }
        var truncated = String(stripped[..<cutEnd])
        while let last = truncated.last,
              last == "," || last.isWhitespace {
            truncated.removeLast()
        }
        truncated.append("]}")
        return truncated
    }

    // MARK: Output sanitizers

    static func stripThinkBlocks(_ raw: String) -> String {
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

    static func stripCodeFence(_ raw: String) -> String {
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

// MARK: - Per-row rendering helper

/// Shared escape helper used by both reviewers' compactLine
/// implementations. Lifted out of each spec so backslash/quote
/// escaping is uniform.
internal enum MLXLLMReviewerRendering {
    static func escapedTranscript(_ raw: String) -> String {
        raw
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
    }

    /// One per-utterance line in the format the reviewer
    /// prompts expect — `- row=N speaker=S01 name=Alice
    /// t=12.3s text="..."`. Used identically by both family
    /// specs; only the surrounding prompt prose differs.
    static func compactLine(
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
        fields.append("text=\"\(escapedTranscript(u.transcript))\"")
        return "- " + fields.joined(separator: " ")
    }

    /// Unindexed variant for boundary-context rows: `- ctx
    /// speaker=S01 t=12.3s text="..."`. No row index by design —
    /// context rows must not be flaggable, and a model that flags
    /// one anyway has no index that maps in `indexToID`.
    static func contextLine(
        for u: UtteranceEstimate,
        speakerNames: [String: String]
    ) -> String {
        var fields: [String] = ["ctx"]
        fields.append("speaker=\(u.speakerID)")
        if let name = speakerNames[u.speakerID], !name.isEmpty {
            fields.append("name=\(name)")
        }
        fields.append(String(format: "t=%.1fs", u.start))
        fields.append("text=\"\(escapedTranscript(u.transcript))\"")
        return "- " + fields.joined(separator: " ")
    }
}
