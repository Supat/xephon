import Foundation
import FoundationModels
import Fusion
import XephonLogging

/// Transcription reviewer backed by Apple Foundation Models. Uses
/// the same `LanguageModelSession` + `@Generable` pattern as
/// `AppleFMSummarizer` — fresh session per call, constrained
/// decoding for the issue list, schema kept out of the prompt to
/// stay under the 4096-token context.
///
/// The on-device model addresses the row by 1-based `rowIndex` (so
/// the prompt doesn't have to carry full UUIDs); we map back to
/// `UtteranceEstimate.id` after decoding. An out-of-range index
/// from the model just drops that issue silently — better to lose
/// a flag than to alias onto the wrong row and confuse the user.
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
    ) async throws -> [TranscriptionIssue] {
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
        // `TranscriptionIssue`s keyed by the canonical UUID.
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
        // Reason in the conversation's language so meaning + homophone
        // analysis works, but emit the `reason` field in the user's
        // app-language pick so it reads naturally in the review sheet.
        let userMessage = """
            The conversation is in \(language.label). Reason about meaning and homophones in \(language.label) only.
            Write each issue's "reason" field in \(SummarizerLocale.responseLanguageNameInEnglish). Use no other language for the reason text.

            \(preface)Utterances (rowIndex speaker t=time text):
            \(lines)
            """

        AppLog.app.info(
            "AppleFMTranscriptionReviewer reviewing \(promptUtterances.count, privacy: .public) utterances"
        )
        let session = LanguageModelSession(instructions: Self.instructions)
        let response: LanguageModelSession.Response<GenerableIssueList>
        do {
            response = try await session.respond(
                to: userMessage,
                generating: GenerableIssueList.self,
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
        return response.content.issues.compactMap { entry -> TranscriptionIssue? in
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

    // MARK: - Prompt + helpers

    private static let instructions = """
        You are a transcription proofreader for a multi-speaker conversation.
        Find rows whose transcript is likely wrong because of (a) a
        misrecognized homophone or near-homophone, (b) a sentence that does
        not fit the session context, or (c) a clear grammar slip. Do not
        flag rows that are merely informal or unusual but coherent.
        For each issue, return rowIndex, a short kind tag from
        ["homophone","contextual","grammar","other"], a one-sentence reason
        explaining what looks wrong, and a 0.0–1.0 confidence. DO NOT
        propose a corrected transcript — the human user will edit the row
        themselves. Skip rows that read correctly.
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
private struct GenerableIssue {
    @Guide(description: "1-based row index from the input list")
    var rowIndex: Int
    @Guide(description: "Issue kind: homophone, contextual, grammar, or other")
    var kind: String
    @Guide(description: "One short sentence explaining what looks wrong")
    var reason: String
    @Guide(description: "Self-reported confidence in this flag, 0.0–1.0")
    var confidence: Float
}

@Generable
private struct GenerableIssueList {
    @Guide(description: "Zero or more flagged rows; omit rows that read correctly")
    var issues: [GenerableIssue]
}
