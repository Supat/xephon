import SwiftUI
import Summarizer

/// One-line live status under the generating spinner: prefill
/// percentage while the prompt is being read, then token count +
/// decode rate. Shared by the session/section summary and
/// transcription-review sheets. Renders nothing when there is no
/// progress to show (non-MLX backends never emit any).
struct LLMLiveProgressLabel: View {
    let progress: MLXGenerationProgress?

    var body: some View {
        if let progress {
            Text(Self.line(for: progress))
                .font(.footnote.monospacedDigit())
                .foregroundStyle(.tertiary)
                .accessibilityLabel(Self.line(for: progress))
        }
    }

    private static func line(for progress: MLXGenerationProgress) -> String {
        switch progress.phase {
        case .prefill(let processed, let total):
            let percent = total > 0 ? processed * 100 / total : 0
            return String(
                format: String(localized: "llm.progress.prefill"),
                percent
            )
        case .decoding(let tokens, let tokensPerSecond):
            return String(
                format: String(localized: "llm.progress.decoding"),
                tokens, tokensPerSecond
            )
        }
    }
}
