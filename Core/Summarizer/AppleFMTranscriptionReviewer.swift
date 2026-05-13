import Foundation
import FoundationModels
import Fusion
import XephonLogging

/// Transcription reviewer backed by Apple Foundation Models. Uses
/// the same `LanguageModelSession` + `@Generable` pattern as
/// `AppleFMSummarizer` — fresh session per call, constrained
/// decoding for the suggestion list, schema kept out of the prompt
/// to stay under the 4096-token context.
///
/// The on-device model addresses the row by 1-based `rowIndex` (so
/// the prompt doesn't have to carry full UUIDs); we map back to
/// `UtteranceEstimate.id` after decoding. Index → UUID lookup is
/// tight enough that an out-of-range index from the model just
/// drops that suggestion silently — better to lose a row than to
/// corrupt the user's transcript by aliasing onto the wrong one.
public actor AppleFMTranscriptionReviewer: TranscriptionReviewer {
    public let modelIdentifier = "apple-foundation-models"

    public var isReady: Bool {
        SystemLanguageModel.default.isAvailable
    }

    public init() {}

    /// Cap on prompt utterances. Review needs more textual context
    /// per row than summarization to evaluate homophone candidates
    /// against neighbours, but the 4096-token shared budget caps
    /// what fits. 20 strikes the same balance the summarizer's 15
    /// does — comfortable schema + response headroom, trailing
    /// window when the session is long.
    private static let maxPromptUtterances = 20

    public func review(
        utterances: [UtteranceEstimate],
        speakerNames: [String: String],
        language: ReviewLanguage
    ) async throws -> [TranscriptionSuggestion] {
        guard SystemLanguageModel.default.isAvailable else {
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
        // 1-based row index → utterance id. The LLM's output refers
        // back to these indices; we look them up to construct
        // `TranscriptionSuggestion`s keyed by the canonical UUID.
        let indexToID: [Int: UUID] = Dictionary(
            uniqueKeysWithValues: promptUtterances
                .enumerated()
                .map { ($0.offset + 1, $0.element.id) }
        )

        let lines = promptUtterances.enumerated().map { idx, u in
            Self.compactLine(rowIndex: idx + 1, for: u, speakerNames: speakerNames)
        }.joined(separator: "\n")
        let preface: String
        if let total = truncatedFrom {
            preface = "Showing the most recent \(promptUtterances.count) of \(total) utterances; review only these.\n\n"
        } else {
            preface = ""
        }
        let userMessage = """
            The conversation is in \(language.label). Reason about meaning and homophones in \(language.label) only.

            \(preface)Utterances (rowIndex speaker t=time text):
            \(lines)
            """

        AppLog.app.info(
            "AppleFMTranscriptionReviewer reviewing \(promptUtterances.count, privacy: .public) utterances"
        )
        let session = LanguageModelSession(instructions: Self.instructions)
        let response: LanguageModelSession.Response<GenerableReviewList>
        do {
            response = try await session.respond(
                to: userMessage,
                generating: GenerableReviewList.self,
                includeSchemaInPrompt: false
            )
        } catch let error as TranscriptionReviewError {
            throw error
        } catch {
            AppLog.app.error(
                "AppleFMTranscriptionReviewer.respond failed: \(String(describing: error), privacy: .public)"
            )
            throw TranscriptionReviewError.inferenceFailed(
                reason: String(describing: error)
            )
        }
        return response.content.suggestions.compactMap { entry -> TranscriptionSuggestion? in
            guard let utteranceID = indexToID[entry.rowIndex] else { return nil }
            // `originalText` is stamped from the live utterance list
            // (not the LLM's echo) — defensive against the model
            // paraphrasing the original in its response, which would
            // make the staleness check at accept time meaningless.
            let original = promptUtterances
                .first { $0.id == utteranceID }?
                .transcript ?? ""
            let kind = TranscriptionSuggestion.Kind(rawValue: entry.kind) ?? .other
            // Drop no-op suggestions where the model echoed the row
            // verbatim. Constrained decoding plus a schema slot for
            // `suggestedText` makes "leave it empty for a no-op"
            // unstable across runs — the model often fills it with
            // the original. Normalize whitespace before comparing.
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

    // MARK: - Prompt + helpers

    private static let instructions = """
        You are a transcription proofreader for a multi-speaker conversation.
        Find rows whose transcript is likely wrong because of (a) a
        misrecognized homophone or near-homophone, (b) a sentence that does
        not fit the session context, or (c) a clear grammar slip. Do not
        flag rows that are merely informal or unusual but coherent.
        For each issue, return rowIndex, a short kind tag from
        ["homophone","contextual","grammar","other"], the suggested fix,
        a one-sentence reason, and a 0.0–1.0 confidence. If you cannot
        propose a concrete fix, leave suggestedText empty and still emit
        the reason. Skip the field entirely when a row reads correctly.
        """

    /// `1 S01 t=12.3 「テキスト」` — minimal so the prompt fits.
    private static func compactLine(
        rowIndex: Int,
        for u: UtteranceEstimate,
        speakerNames: [String: String]
    ) -> String {
        var parts: [String] = []
        parts.append(String(rowIndex))
        if let name = speakerNames[u.speakerID], !name.isEmpty {
            parts.append("\(u.speakerID)(\(name))")
        } else {
            parts.append(u.speakerID)
        }
        parts.append(String(format: "t=%.1f", u.start))
        let escaped = u.transcript
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
        parts.append("\"\(escaped)\"")
        return parts.joined(separator: " ")
    }
}

@Generable
private struct GenerableSuggestion {
    @Guide(description: "1-based row index from the input list")
    var rowIndex: Int
    @Guide(description: "Issue kind: homophone, contextual, grammar, or other")
    var kind: String
    @Guide(description: "Proposed corrected transcript; empty if no concrete fix")
    var suggestedText: String
    @Guide(description: "One short sentence explaining the issue")
    var reason: String
    @Guide(description: "Self-reported confidence in this fix, 0.0–1.0")
    var confidence: Float
}

@Generable
private struct GenerableReviewList {
    @Guide(description: "Zero or more flagged rows; omit rows that read correctly")
    var suggestions: [GenerableSuggestion]
}
