import Foundation
import Fusion
import SERText

/// Public read-only catalog of the prompts the on-device LLMs
/// receive. Used by the app's "Prompts" card to expose what each
/// model is being asked to do — non-editable today, just visible.
///
/// MLX entries call the real `buildPrompt` / `buildDeepWindowPrompt`
/// / `buildDeepMergePrompt` on the actual spec types with a small
/// canned utterance list so the user sees the exact structure
/// (header + utterance lines + footer) that lands at the model.
/// Apple Foundation Models entries reach into the static
/// instruction constants on `AppleFMSummarizer` /
/// `AppleFMTranscriptionReviewer` and pair them with a sample
/// user message because the FM path is "instructions + per-call
/// user message" rather than one monolithic prompt.
///
/// Single source of truth: the same spec types and constants the
/// live inference path uses. If the prompt drifts, the catalog
/// drifts with it; nothing to keep in sync manually.
public enum PromptCatalog {

    public struct PromptEntry: Sendable, Identifiable {
        public let id: String
        public let title: String
        public let body: String

        public init(id: String, title: String, body: String) {
            self.id = id
            self.title = title
            self.body = body
        }
    }

    // MARK: - Top-level groupings

    /// Every summarizer prompt: one entry per (backend × mode).
    /// MLX backends contribute three entries each (single-pass
    /// trailing, deep-window, deep-merge). Apple FM contributes
    /// three instruction blocks plus a sample user message.
    public static func summarizerPrompts() -> [PromptEntry] {
        var entries: [PromptEntry] = []

        // Apple FM uses static instruction strings + a per-call
        // user message constructed by the inference call site.
        // Surface the instructions alone — the user message is
        // just the utterance list interpolated into a fixed
        // shell, which is the same shape across modes.
        entries.append(PromptEntry(
            id: "appleFM.summarizer.singlePass",
            title: "Apple FM · Summarize · Trailing / Heuristic",
            body: AppleFMSummarizer.instructions
        ))
        entries.append(PromptEntry(
            id: "appleFM.summarizer.deepWindow",
            title: "Apple FM · Summarize · Deep · Per-window",
            body: AppleFMSummarizer.windowInstructions
        ))
        entries.append(PromptEntry(
            id: "appleFM.summarizer.deepMerge",
            title: "Apple FM · Summarize · Deep · Merge",
            body: AppleFMSummarizer.mergeInstructions
        ))

        // Qwen3 (MLX).
        let qwen = MLXQwenSpec()
        entries.append(PromptEntry(
            id: "qwen.summarizer.singlePass",
            title: "Qwen3 · Summarize · Trailing / Heuristic",
            body: qwen.buildPrompt(
                utterances: sampleUtterances,
                speakerNames: sampleSpeakerNames,
                truncatedFromTotal: nil,
                selection: .trailing
            )
        ))
        entries.append(PromptEntry(
            id: "qwen.summarizer.deepWindow",
            title: "Qwen3 · Summarize · Deep · Per-window",
            body: qwen.buildDeepWindowPrompt(
                utterances: sampleUtterances,
                speakerNames: sampleSpeakerNames,
                windowIndex: 0,
                totalWindows: 1
            )
        ))
        entries.append(PromptEntry(
            id: "qwen.summarizer.deepMerge",
            title: "Qwen3 · Summarize · Deep · Merge",
            body: qwen.buildDeepMergePrompt(
                intermediates: sampleDeepIntermediates,
                allUtterances: sampleUtterances,
                speakerNames: sampleSpeakerNames
            )
        ))

        // Llama-3-Swallow (MLX).
        let llama = MLXLlamaSpec()
        entries.append(PromptEntry(
            id: "llama.summarizer.singlePass",
            title: "Llama-3-Swallow · Summarize · Trailing / Heuristic",
            body: llama.buildPrompt(
                utterances: sampleUtterances,
                speakerNames: sampleSpeakerNames,
                truncatedFromTotal: nil,
                selection: .trailing
            )
        ))
        entries.append(PromptEntry(
            id: "llama.summarizer.deepWindow",
            title: "Llama-3-Swallow · Summarize · Deep · Per-window",
            body: llama.buildDeepWindowPrompt(
                utterances: sampleUtterances,
                speakerNames: sampleSpeakerNames,
                windowIndex: 0,
                totalWindows: 1
            )
        ))
        entries.append(PromptEntry(
            id: "llama.summarizer.deepMerge",
            title: "Llama-3-Swallow · Summarize · Deep · Merge",
            body: llama.buildDeepMergePrompt(
                intermediates: sampleDeepIntermediates,
                allUtterances: sampleUtterances,
                speakerNames: sampleSpeakerNames
            )
        ))

        return entries
    }

    /// Every transcription-reviewer prompt: one entry per backend.
    public static func reviewerPrompts() -> [PromptEntry] {
        var entries: [PromptEntry] = []

        entries.append(PromptEntry(
            id: "appleFM.reviewer",
            title: "Apple FM · Reviewer",
            body: AppleFMTranscriptionReviewer.instructions
        ))

        let qwen = MLXQwenReviewerSpec()
        entries.append(PromptEntry(
            id: "qwen.reviewer",
            title: "Qwen3 · Reviewer",
            body: qwen.buildPrompt(
                utterances: sampleUtterances,
                speakerNames: sampleSpeakerNames,
                language: .japanese,
                chunkIndex: 0,
                totalChunks: 1
            )
        ))

        let llama = MLXLlamaReviewerSpec()
        entries.append(PromptEntry(
            id: "llama.reviewer",
            title: "Llama-3-Swallow · Reviewer",
            body: llama.buildPrompt(
                utterances: sampleUtterances,
                speakerNames: sampleSpeakerNames,
                language: .japanese,
                chunkIndex: 0,
                totalChunks: 1
            )
        ))

        return entries
    }

    /// Text-SER prompts. WRIME (the fine-tuned regressor) has no
    /// prompt, so only the FoundationModels entry appears here.
    /// The live prompt sandwiches a language-specific opener
    /// ("You are a Japanese-language affect annotator.") around
    /// the static instruction body; the preview shows the body
    /// alone with a placeholder opener so the language pick
    /// substitution is visible.
    public static func textSERPrompts() -> [PromptEntry] {
        return [
            PromptEntry(
                id: "foundationModels.textSER",
                title: "Apple FM · Text SER (Plutchik)",
                body: "You are a <language>-language affect annotator.\n\n"
                    + FoundationModelsSER.instructionsBody
            )
        ]
    }

    // MARK: - Sample data

    /// Two-utterance sample used to render the dynamic
    /// `buildPrompt` outputs. Keeps the surrounding
    /// instructions / schema visible while showing the exact
    /// per-line format the model receives.
    private static let sampleUtterances: [UtteranceEstimate] = [
        UtteranceEstimate(
            speakerID: "S01",
            start: 0.0,
            end: 3.2,
            transcript: "Hello, this is a sample utterance.",
            asrConfidence: 0.95,
            dimensional: nil,
            acousticCategorical: nil,
            plutchik: nil,
            fusedValence: 0.55,
            fusedArousal: 0.45,
            fusedDominance: nil,
            fusedTopLabel: "happy"
        ),
        UtteranceEstimate(
            speakerID: "S02",
            start: 3.2,
            end: 6.0,
            transcript: "And this is a second sample utterance.",
            asrConfidence: 0.91,
            dimensional: nil,
            acousticCategorical: nil,
            plutchik: nil,
            fusedValence: 0.50,
            fusedArousal: 0.42,
            fusedDominance: nil,
            fusedTopLabel: "neutral"
        ),
    ]

    private static let sampleSpeakerNames: [String: String] = [
        "S01": "Alice",
        "S02": "Bob",
    ]

    /// Sample deep-window intermediate for the merge-prompt
    /// preview. One intermediate keeps the preview short while
    /// still showing the merge prompt's per-window block format.
    private static let sampleDeepIntermediates: [MLXLLMDeepWindowIntermediate] = [
        MLXLLMDeepWindowIntermediate(
            windowIndex: 0,
            timeStart: 0.0,
            timeEnd: 6.0,
            topicSnapshot: "casual catch-up",
            moodSnapshot: "warm, mildly excited",
            perSpeaker: [
                MLXLLMDeepWindowIntermediate.PerSpeakerNote(
                    speakerID: "S01",
                    notes: "Opened the conversation, sounded pleased.",
                    dominantMood: "happy"
                )
            ],
            modalityFlags: nil
        )
    ]
}
