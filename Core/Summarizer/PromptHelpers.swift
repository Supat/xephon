import Foundation
import Fusion

/// Prompt-building helpers shared between the LLM backends. Each
/// backend (AppleFM + MLXQwen, for both the summarizer and the
/// reviewer roles) builds its own compact per-row line — the
/// formats genuinely differ because of context-window pressure and
/// the per-task fields the model needs to see — but two pieces of
/// glue are identical across all four:
///
/// - `orderedSpeakerIDs(from:)`: first-appearance order over the
///   utterance list, used by the summarizers so the `perSpeaker`
///   array tracks the order the user reads the chips in.
/// - `escapeForQuotedLiteral(_:)`: `\\` and `"` escaping so a
///   transcript that contains `"` doesn't break the compact JSON-
///   ish line shape `text="..."`.
///
/// Kept as a tiny namespace rather than a base class because the
/// backends don't share enough to justify inheritance, but they
/// do benefit from one source of truth on these two corners.
public enum PromptHelpers {

    /// Order distinct speaker ids in their first-appearance order
    /// across `utterances`. Stable for fed-back prompts so the
    /// per-speaker output matches the reading order in the UI.
    public static func orderedSpeakerIDs(
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

    /// Escape a transcript for embedding inside a `"..."` literal.
    /// Backslashes first, then quotes — otherwise the quote escape
    /// would emit `\\"` which the second pass would re-escape to
    /// `\\\\"`.
    public static func escapeForQuotedLiteral(_ s: String) -> String {
        s.replacingOccurrences(of: "\\", with: "\\\\")
         .replacingOccurrences(of: "\"", with: "\\\"")
    }
}
