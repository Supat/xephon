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

    /// Previous-chunk rows replayed as unindexed context. Smaller
    /// than the MLX specs' 6 — the 4096-token shared budget is
    /// tight and each context row costs the same as a review row.
    private static let contextOverlapRows = 4

    public func review(
        utterances: [UtteranceEstimate],
        speakerNames: [String: String],
        language: ReviewLanguage
    ) async throws -> [TranscriptionIssue] {
        guard SystemLanguageModel.default.isAvailable else {
            throw TranscriptionReviewError.modelNotInstalled
        }
        guard !utterances.isEmpty else { return [] }

        // Always-chunk: review's purpose is per-utterance
        // proofreading, so every utterance must get a chance
        // to be flagged. The trailing-window truncation we
        // used before silently skipped the prefix of long
        // sessions. We pay multiple `session.respond` calls
        // on long sessions to keep the per-utterance guarantee.
        let chunks = stride(from: 0, to: utterances.count, by: Self.maxPromptUtterances).map {
            offset -> [UtteranceEstimate] in
            let end = min(offset + Self.maxPromptUtterances, utterances.count)
            return Array(utterances[offset..<end])
        }
        AppLog.app.info(
            "AppleFMTranscriptionReviewer reviewing \(utterances.count, privacy: .public) utterances in \(chunks.count, privacy: .public) chunk(s)"
        )

        var allIssues: [TranscriptionIssue] = []
        for (chunkIndex, chunk) in chunks.enumerated() {
            if Task.isCancelled { throw CancellationError() }
            // Boundary context — tail of the previous chunk,
            // replayed unindexed (see the ctx block in reviewChunk).
            let start = chunkIndex * Self.maxPromptUtterances
            let ctxStart = max(0, start - Self.contextOverlapRows)
            let contextPrefix = start == 0 ? [] : Array(utterances[ctxStart..<start])
            let issues = try await reviewChunk(
                chunk: chunk,
                speakerNames: speakerNames,
                language: language,
                chunkIndex: chunkIndex,
                totalChunks: chunks.count,
                contextPrefix: contextPrefix
            )
            allIssues.append(contentsOf: issues)
            AppLog.app.info(
                "AppleFMTranscriptionReviewer chunk \(chunkIndex + 1, privacy: .public)/\(chunks.count, privacy: .public) yielded \(issues.count, privacy: .public) issue(s)"
            )
        }
        // Constrained decoding guarantees SHAPE, not grounding —
        // the validator still applies (verbatim-excerpt check,
        // confidence floor, dedupe).
        return ReviewIssueValidator.filter(allIssues, utterances: utterances)
    }

    /// Inference on a single chunk. Row indices are 1-based
    /// local to the chunk; the per-chunk `indexToID` remaps
    /// back to the session-wide utterance UUIDs the
    /// `TranscriptionIssue` carries. Fresh
    /// `LanguageModelSession` per chunk so the 4 k context
    /// doesn't accumulate across calls.
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
        let lines = chunk.enumerated().map { idx, u in
            Self.compactLine(rowIndex: idx + 1, for: u, speakerNames: speakerNames)
        }.joined(separator: "\n")
        // Tell the model this is one chunk of a longer pass
        // when there's more than one — keeps it from treating
        // "missing prior topic context" as a non-sequitur
        // signal. Single-chunk sessions skip the preface.
        let preface: String
        if totalChunks > 1 {
            preface = "This is chunk \(chunkIndex + 1) of \(totalChunks) of the conversation's review pass. Earlier and later utterances are reviewed separately; do not flag rows as non-sequitur just because broader topic context isn't visible here.\n\n"
        } else {
            preface = ""
        }
        // Reason in the conversation's language so meaning +
        // homophone analysis works, but emit the `reason`
        // field in the user's app-language pick.
        // Context block: unindexed, non-flaggable rows from the
        // previous chunk for topic continuity. A model flag against
        // one can't parse anyway (no rowIndex in indexToID).
        let contextBlock: String
        if contextPrefix.isEmpty {
            contextBlock = ""
        } else {
            let ctxLines = contextPrefix.map { u in
                "ctx " + Self.compactLine(rowIndex: 0, for: u, speakerNames: speakerNames)
                    .split(separator: " ", maxSplits: 1)
                    .dropFirst()
                    .joined()
            }.joined(separator: "\n")
            contextBlock = """
                Context from the previous chunk — for topic continuity ONLY; these rows have no row index and MUST NOT be flagged:
                \(ctxLines)


                """
        }
        let userMessage = """
            The conversation is in \(language.label). Reason about meaning and homophones in \(language.label) only.
            Write each issue's "reason" field in \(SummarizerLocale.responseLanguageNameInEnglish). Use no other language for the reason text.

            \(preface)\(contextBlock)Utterances (rowIndex speaker t=time text):
            \(lines)
            """

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
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            AppLog.app.error(
                "AppleFMTranscriptionReviewer chunk \(chunkIndex + 1, privacy: .public)/\(totalChunks, privacy: .public) respond failed: \(String(describing: error), privacy: .public)"
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
                confidence: entry.confidence,
                excerpt: entry.excerpt.isEmpty ? nil : entry.excerpt
            )
        }
    }

    // MARK: - Prompt + helpers

    internal static let instructions = """
        You are a transcription proofreader for a multi-speaker conversation.
        Find rows whose transcript is likely wrong because of (a) a
        misrecognized homophone or near-homophone, (b) a sentence that does
        not fit the session context, or (c) a clear grammar slip. Do not
        flag rows that are merely informal or unusual but coherent.

        A flagged row MUST have a SPECIFIC plausible alternative reading in
        mind — a different word or phrase the ASR could have confused with
        what's written. If no specific alternative comes to mind, OMIT the
        row entirely. NEVER write a reason of the form "X may be a
        misinterpretation of X" where X is the same phrase as the row's
        transcript — that's a tautology and not an issue. Omitting rows is
        ALWAYS preferred over flagging without a real candidate.

        You are NOT reviewing content, opinions, emotions, or facts. The
        ONLY question is whether the transcription matches what was likely
        said. Anything else is out of scope and must not be flagged.

        For each issue, return rowIndex, a short kind tag from
        ["homophone","contextual","grammar","other"], the EXACT substring
        of that row's transcript that looks wrong (copied verbatim — issues
        whose excerpt doesn't appear character-for-character in the row are
        discarded automatically; empty string only for a contextual issue
        about the whole row), a one-sentence reason explaining what looks
        wrong, and a 0.0–1.0 confidence (0.9 = near-certain, 0.6 =
        plausible; do not emit below 0.5). DO NOT propose a corrected
        transcript — the human user will edit the row themselves. Skip rows
        that read correctly.
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
    @Guide(description: "EXACT substring of the flagged row's transcript that looks wrong, copied verbatim; empty string only for a whole-row contextual issue")
    var excerpt: String
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
