import Foundation
import Fusion
import XephonLogging

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

/// Post-parse validation shared by every reviewer backend. The
/// LLMs — especially the small quantized ones — regularly emit
/// flags outside the prompt's scope: content/opinion commentary,
/// tautologies, invented text, low-conviction noise. Prompts alone
/// can't stop that; this filter makes the core failure classes
/// mechanically detectable and drops them before the UI:
///
/// - GROUNDING: word-level kinds (homophone / grammar) must quote
///   the offending span (`excerpt`) and the quote must appear
///   verbatim in the target row's transcript. A model commenting
///   on content or hallucinating text can't produce a passing
///   quote; row-index drift fails it too. Whole-row kinds
///   (contextual / other) may omit the excerpt, but a present
///   excerpt must still match.
/// - CONFIDENCE FLOOR: self-reported confidence below
///   `confidenceFloor` reads as noise. nil passes (not every
///   backend emits one).
/// - REASON SANITY: empty reasons and transcript echoes drop.
/// - DEDUPE: one issue per (row, kind); first (model-order) wins.
public enum ReviewIssueValidator {
    /// Below this the model itself says it's guessing; matches the
    /// prompt's "do not emit below 0.5" calibration line with a
    /// little slack for backends that calibrate low.
    public static let confidenceFloor: Float = 0.4

    public static func filter(
        _ issues: [TranscriptionIssue],
        utterances: [UtteranceEstimate]
    ) -> [TranscriptionIssue] {
        guard !issues.isEmpty else { return issues }
        let transcriptByID = Dictionary(
            uniqueKeysWithValues: utterances.map { ($0.id, $0.transcript) }
        )
        var ungrounded = 0, lowConfidence = 0, emptyReason = 0, duplicate = 0
        var seen = Set<String>()
        var kept: [TranscriptionIssue] = []
        for issue in issues {
            guard let transcript = transcriptByID[issue.utteranceID] else {
                ungrounded += 1
                continue
            }
            let reason = issue.reason.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !reason.isEmpty, reason != transcript else {
                emptyReason += 1
                continue
            }
            if let c = issue.confidence, c < Self.confidenceFloor {
                lowConfidence += 1
                continue
            }
            let excerpt = issue.excerpt?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            if !excerpt.isEmpty {
                guard transcript.contains(excerpt) else {
                    ungrounded += 1
                    continue
                }
            } else if issue.kind == .homophone || issue.kind == .grammar {
                ungrounded += 1
                continue
            }
            guard seen.insert("\(issue.utteranceID)|\(issue.kind.rawValue)").inserted else {
                duplicate += 1
                continue
            }
            kept.append(issue)
        }
        let droppedTotal = issues.count - kept.count
        if droppedTotal > 0 {
            AppLog.app.info("review validator: kept \(kept.count, privacy: .public)/\(issues.count, privacy: .public) (ungrounded=\(ungrounded, privacy: .public) lowConf=\(lowConfidence, privacy: .public) emptyReason=\(emptyReason, privacy: .public) dupe=\(duplicate, privacy: .public))")
        }
        return kept
    }
}
