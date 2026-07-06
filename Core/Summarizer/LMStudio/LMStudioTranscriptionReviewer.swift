import Foundation
import Fusion
import XephonLogging

/// `TranscriptionReviewer` that off-loads inference to the
/// user-configured LM Studio server. Same orchestration pattern
/// as `MLXLLMReviewerCore`: chunk the utterance list so every
/// row gets a chance to be flagged, run one chat completion per
/// chunk, parse + remap row indices back to utterance UUIDs,
/// accumulate.
///
/// Prompts come from `MLXQwenReviewerSpec.buildPrompt` so the
/// LM Studio path stays in lock-step with the on-device MLX
/// Qwen reviewer. Llama-served LM Studio servers pick up the
/// Qwen-anchored language directive; if Llama-specific behaviour
/// becomes necessary we can split into per-served-family specs
/// later.
public actor LMStudioTranscriptionReviewer: TranscriptionReviewer {
    public let modelIdentifier: String
    private let client: LMStudioClient
    private let spec = MLXQwenReviewerSpec()
    private let useStructuredOutput: Bool

    public init(
        modelIdentifier: String,
        client: LMStudioClient,
        useStructuredOutput: Bool = false
    ) {
        if modelIdentifier.isEmpty {
            self.modelIdentifier = "lmstudio"
        } else {
            self.modelIdentifier = "lmstudio:\(modelIdentifier)"
        }
        self.client = client
        self.useStructuredOutput = useStructuredOutput
    }

    public var isReady: Bool { true }

    public func review(
        utterances: [UtteranceEstimate],
        speakerNames: [String: String],
        language: ReviewLanguage
    ) async throws -> [TranscriptionIssue] {
        guard !utterances.isEmpty else { return [] }
        // Same chunking rule as `MLXLLMReviewerCore.review` so
        // the LM Studio path produces the same coverage on long
        // sessions. Reviewer's purpose is per-utterance
        // proofreading, so silently dropping a trailing
        // prefix wouldn't be acceptable.
        let chunkStarts = Array(stride(from: 0, to: utterances.count, by: spec.maxPromptUtterances))
        AppLog.app.info(
            "LMStudioReviewer reviewing \(utterances.count, privacy: .public) utterances in \(chunkStarts.count, privacy: .public) chunk(s)"
        )

        var allIssues: [TranscriptionIssue] = []
        for (chunkIndex, start) in chunkStarts.enumerated() {
            if Task.isCancelled { throw CancellationError() }
            let end = min(start + spec.maxPromptUtterances, utterances.count)
            let chunk = Array(utterances[start..<end])
            // Boundary context — same carve as MLXLLMReviewerCore.
            let ctxStart = max(0, start - spec.contextOverlapRows)
            let contextPrefix = start == 0 ? [] : Array(utterances[ctxStart..<start])
            let issues = try await reviewChunk(
                chunk: chunk,
                speakerNames: speakerNames,
                language: language,
                chunkIndex: chunkIndex,
                totalChunks: chunkStarts.count,
                contextPrefix: contextPrefix
            )
            allIssues.append(contentsOf: issues)
            AppLog.app.info(
                "LMStudioReviewer chunk \(chunkIndex + 1, privacy: .public)/\(chunkStarts.count, privacy: .public) yielded \(issues.count, privacy: .public) issue(s)"
            )
        }
        // Post-parse grounding/sanity filter — see
        // ReviewIssueValidator for the failure classes it kills.
        return ReviewIssueValidator.filter(allIssues, utterances: utterances)
    }

    private func reviewChunk(
        chunk: [UtteranceEstimate],
        speakerNames: [String: String],
        language: ReviewLanguage,
        chunkIndex: Int,
        totalChunks: Int,
        contextPrefix: [UtteranceEstimate]
    ) async throws -> [TranscriptionIssue] {
        let indexToID: [Int: UUID] = Dictionary(
            uniqueKeysWithValues: chunk
                .enumerated()
                .map { ($0.offset + 1, $0.element.id) }
        )
        let prompt = spec.buildPrompt(
            utterances: chunk,
            speakerNames: speakerNames,
            language: language,
            chunkIndex: chunkIndex,
            totalChunks: totalChunks,
            contextPrefix: contextPrefix
        )
        AppLog.app.info(
            "LMStudioReviewer chunk \(chunkIndex + 1, privacy: .public)/\(totalChunks, privacy: .public): \(chunk.count, privacy: .public) utterances (prompt \(prompt.count, privacy: .public) chars)"
        )
        let raw: String
        do {
            let responseFormatJSON = useStructuredOutput
                ? try? JSONEncoder().encode(LMStudioResponseFormat.jsonSchema(
                    name: "transcription_review",
                    schema: LMStudioSchemas.reviewSchema
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
            throw TranscriptionReviewError.inferenceFailed(
                reason: String(describing: error)
            )
        }
        if Task.isCancelled { throw CancellationError() }
        let preview = raw.count > 400
            ? String(raw.prefix(400)) + "…[truncated]"
            : raw
        AppLog.app.info(
            "LMStudioReviewer chunk \(chunkIndex + 1, privacy: .public) raw: \(raw.count, privacy: .public) chars, preview: \(preview, privacy: .public)"
        )
        return try MLXLLMReviewerCore.parse(raw: raw, indexToID: indexToID)
    }
}
