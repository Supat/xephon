import Foundation

/// Text-cleanup helpers shared by the MLX-backed summarizer and the
/// MLX-backed transcription reviewer. Both call into `stripThinkBlocks`
/// and `stripCodeFence` before handing the LLM's raw output to a JSON
/// decoder; centralizing them here means a future MLX-backed task
/// (or a model-family swap that changes the reasoning-block markers)
/// is a one-file edit instead of having to keep parallel copies in
/// sync.
public enum MLXOutputCleaner {

    /// Remove any `<think>...</think>` reasoning blocks Qwen3 emits
    /// when its thinking mode is engaged. Greedy across newlines.
    /// Also drops a stray closing `</think>` if the model elided
    /// the opening tag (occasionally seen with `/no_think`).
    public static func stripThinkBlocks(_ raw: String) -> String {
        var s = raw
        while let openRange = s.range(of: "<think>") {
            if let closeRange = s.range(
                of: "</think>",
                range: openRange.upperBound..<s.endIndex
            ) {
                s.removeSubrange(openRange.lowerBound..<closeRange.upperBound)
            } else {
                // Unterminated block — drop everything from the
                // opener forward; the JSON, if any, was supposed
                // to come after a `</think>` we never saw.
                s.removeSubrange(openRange.lowerBound..<s.endIndex)
                break
            }
        }
        // Tolerate a stray closing tag without an opener.
        if let strayClose = s.range(of: "</think>") {
            s.removeSubrange(s.startIndex..<strayClose.upperBound)
        }
        return s
    }

    /// Strip ```json ... ``` fences a chat-tuned model might emit
    /// even after being told "JSON only." Leaves bare JSON alone.
    public static func stripCodeFence(_ raw: String) -> String {
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
