import Foundation
import Fusion

/// Errors a `TranscriptionReviewer` can raise. Mirrors
/// `SummarizerError` because the two surfaces share the same
/// failure modes (model not present, OOM, decode mismatch).
public enum TranscriptionReviewError: Error, CustomStringConvertible {
    case modelLoadFailed(reason: String)
    case inferenceFailed(reason: String)
    case decodeFailed(reason: String)
    case modelNotInstalled

    public var description: String {
        switch self {
        case .modelLoadFailed(let r): return "Reviewer model load failed: \(r)"
        case .inferenceFailed(let r): return "Reviewer inference failed: \(r)"
        case .decodeFailed(let r):    return "Reviewer output didn't match the expected schema: \(r)"
        case .modelNotInstalled:      return "Reviewer model isn't installed yet"
        }
    }
}

/// Which natural language the reviewer should treat the transcript
/// as. Qwen3 is Chinese-dominant in pretraining and will silently
/// reinterpret kanji-only utterances as Mandarin unless the prompt
/// pins the language; the same anchoring helps Apple FM produce
/// homophone candidates that match the locale (their/there vs.
/// 橋/箸). Kept in the Summarizer module so the protocol can stay
/// independent of the app's `SessionLanguage` type.
public enum ReviewLanguage: Sendable, Hashable {
    case japanese
    case english

    /// Human-readable label used in the prompt body ("The
    /// conversation is in <label>.").
    public var label: String {
        switch self {
        case .japanese: return "Japanese"
        case .english:  return "English"
        }
    }

    /// Extra anchoring sentences the Qwen prompt needs because the
    /// base model defaults to Chinese readings for CJK characters
    /// unless explicitly told otherwise. Apple FM doesn't have this
    /// failure mode so the AFM reviewer just uses `label`.
    public var qwenAnchor: String {
        switch self {
        case .japanese:
            return """
                The conversation is in JAPANESE. All transcripts use Japanese script
                (kanji, hiragana, katakana). DO NOT interpret characters as Chinese
                or propose Chinese readings. Common Japanese ASR errors to look for:
                - misrecognized homophones
                - wrong kanji selection for the same yomi
                - wrong particle
                - missing or extra long-vowel marker
                Only cite homophone candidates that actually share the reading of text
                in the utterance. Do not invent candidates that don't match the yomi.
                Reason about meaning, grammar, and naturalness in Japanese only.
                """
        case .english:
            return """
                The conversation is in ENGLISH. Common ASR errors to look for:
                - homophones
                - missing/extra plural -s, wrong tense
                - wrong word choice that sounds similar
                Only cite homophone candidates that actually sound like text in the
                utterance. Do not invent candidates that don't match the pronunciation.
                Reason about meaning and grammar in English only.
                """
        }
    }
}

/// Abstract interface a transcription reviewer conforms to. Same
/// shape as `SessionSummarizer`: async, throwing, two backends (Apple
/// FM + MLX Qwen). The reviewer takes the same utterance list the
/// summarizer does and returns a flat list of issues located by
/// `utteranceID` — order doesn't matter for the consumer.
public protocol TranscriptionReviewer: Sendable {
    var modelIdentifier: String { get async }
    var isReady: Bool { get async }

    /// Walk `utterances` and return any transcription issues the
    /// model can identify. `speakerNames` lets the prompt refer to
    /// renamed speakers by their friendly names so the model's
    /// reasoning reads naturally; the returned issues are keyed by
    /// `UtteranceEstimate.id` only — names are presentation.
    /// `language` pins the prompt to the session's natural language
    /// — critical for the Qwen path, which otherwise interprets
    /// kanji as Mandarin and emits useless issue reasons.
    func review(
        utterances: [UtteranceEstimate],
        speakerNames: [String: String],
        language: ReviewLanguage
    ) async throws -> [TranscriptionIssue]
}
