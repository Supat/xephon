import Foundation
import FoundationModels
import Fusion
import XephonLogging

/// Session summarizer backed by Apple Foundation Models (iPadOS
/// 26+, ~3B parameters, system-managed). Uses the same
/// `LanguageModelSession` + `@Generable` structured-output pattern
/// `FoundationModelsSER` already exercises for per-utterance
/// Plutchik scoring — fresh session per call so the 4096-token
/// context window doesn't accumulate across runs.
///
/// Lighter than the Qwen path: no 4 GB download, no MLX runtime,
/// no per-app memory orchestration required. The trade-off is
/// model size (3B vs 7B) and context (4k vs 32k) — long sessions
/// get aggressively truncated to fit, and multi-speaker reasoning
/// quality is meaningfully lower on subtle arcs.
public actor AppleFMSummarizer: SessionSummarizer {
    public let modelIdentifier = "apple-foundation-models"

    public var isReady: Bool {
        SystemLanguageModel.default.isAvailable
    }

    public init() {}

    /// Cap on prompt utterances. Apple FM's 4096-token window is
    /// shared by instructions, the schema for constrained
    /// decoding, the utterance lines, AND the generated output —
    /// the reserved response budget eats into "what we can pass
    /// in" even though our prompt strictly looks smaller than
    /// 4096 tokens. At 30 utterances we were tripping
    /// `exceededContextWindowSize`; 15 leaves comfortable
    /// headroom for a ~500-token response and the schema. The
    /// trailing edge of a session has the most actionable arc
    /// anyway.
    private static let maxPromptUtterances = 15

    public func summarize(
        utterances: [UtteranceEstimate],
        speakerNames: [String: String]
    ) async throws -> SessionSummary {
        guard SystemLanguageModel.default.isAvailable else {
            throw SummarizerError.modelNotInstalled
        }
        guard !utterances.isEmpty else {
            return SessionSummary(
                topic: "",
                overallMood: "",
                perSpeaker: [],
                model: modelIdentifier,
                generatedAt: Date()
            )
        }

        let promptUtterances: [UtteranceEstimate]
        let truncatedFrom: Int?
        if utterances.count > Self.maxPromptUtterances {
            promptUtterances = Array(utterances.suffix(Self.maxPromptUtterances))
            truncatedFrom = utterances.count
        } else {
            promptUtterances = utterances
            truncatedFrom = nil
        }

        let speakers = Self.orderedSpeakerIDs(from: promptUtterances)
        let utteranceLines = promptUtterances
            .map { Self.compactLine(for: $0, speakerNames: speakerNames) }
            .joined(separator: "\n")
        let truncationNote: String
        if let total = truncatedFrom {
            truncationNote = "\n\n(Showing the most recent \(promptUtterances.count) of \(total) utterances; frame overall mood as the trailing portion.)"
        } else {
            truncationNote = ""
        }
        let userMessage = """
            Speakers present: \(speakers.joined(separator: ", ")).

            Utterances:
            \(utteranceLines)\(truncationNote)
            """

        AppLog.app.info(
            "AppleFMSummarizer summarizing \(promptUtterances.count, privacy: .public) utterances"
        )
        let session = LanguageModelSession(instructions: Self.instructions)
        do {
            // `includeSchemaInPrompt: false` keeps the schema out
            // of the textual prompt — constrained decoding still
            // enforces the shape, but the nested
            // `perSpeaker: [GenerableSpeakerSummary]` schema is
            // bulky enough to push a 30-utterance prompt past the
            // 4096-token window. Worth ~300+ tokens back.
            let response = try await session.respond(
                to: userMessage,
                generating: GenerableSummary.self,
                includeSchemaInPrompt: false
            )
            let g = response.content
            let perSpeaker = g.perSpeaker.map { entry in
                SessionSummary.SpeakerSummary(
                    speakerID: entry.speakerID,
                    speakerName: speakerNames[entry.speakerID],
                    summary: entry.summary,
                    dominantMood: entry.dominantMood
                )
            }
            return SessionSummary(
                topic: g.topic,
                overallMood: g.overallMood,
                perSpeaker: perSpeaker,
                model: modelIdentifier,
                generatedAt: Date()
            )
        } catch let error as SummarizerError {
            throw error
        } catch {
            AppLog.app.error(
                "AppleFMSummarizer.respond failed: \(String(describing: error), privacy: .public)"
            )
            throw SummarizerError.inferenceFailed(
                reason: String(describing: error)
            )
        }
    }

    // MARK: - Prompt + helpers

    private static let instructions = """
        Summarize a multi-speaker conversation. Each input line has
        speaker, time, fused emotion label, valence V (0..1, 0.5 = neutral),
        and arousal A (0..1, higher = stronger affect), then the transcript.
        Produce: a one-sentence topic, a one-paragraph overall mood, and one
        per-speaker entry (short paragraph + dominant-mood phrase) for every
        speaker id in the input. Do not invent speakers.
        """

    private static func orderedSpeakerIDs(
        from utterances: [UtteranceEstimate]
    ) -> [String] {
        var seen: Set<String> = []
        var ordered: [String] = []
        for u in utterances where !seen.contains(u.speakerID) {
            seen.insert(u.speakerID)
            ordered.append(u.speakerID)
        }
        return ordered
    }

    /// Tight per-utterance line tuned for the 4k context.
    /// Drops dominance (least-used affect axis) and the
    /// optional `name=` field (we re-map names from the
    /// `speakerNames` dictionary onto the result anyway).
    /// Format: `S01 12.3s joy V0.42 A0.71 "text"`
    private static func compactLine(
        for u: UtteranceEstimate,
        speakerNames: [String: String]
    ) -> String {
        var parts: [String] = []
        parts.append(u.speakerID)
        parts.append(String(format: "%.1fs", u.start))
        if let label = u.fusedTopLabel { parts.append(label) }
        if let v = u.fusedValence { parts.append(String(format: "V%.2f", v)) }
        if let a = u.fusedArousal { parts.append(String(format: "A%.2f", a)) }
        let escaped = u.transcript
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
        parts.append("\"\(escaped)\"")
        return parts.joined(separator: " ")
    }
}

@Generable
private struct GenerableSpeakerSummary {
    @Guide(description: "Canonical speaker id (e.g. S01, S02) — copy from the input")
    var speakerID: String
    @Guide(description: "One paragraph (3–5 sentences) on this speaker's emotional arc through the session")
    var summary: String
    @Guide(description: "Short phrase (1–6 words) capturing the speaker's dominant mood")
    var dominantMood: String
}

@Generable
private struct GenerableSummary {
    @Guide(description: "One or two sentences on what the conversation is about")
    var topic: String
    @Guide(description: "One paragraph on the session's overall emotional tone, factoring V/A/D and labels")
    var overallMood: String
    @Guide(description: "Per-speaker emotional arcs, one entry per distinct speaker id in the input")
    var perSpeaker: [GenerableSpeakerSummary]
}
