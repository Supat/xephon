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
        entries.append(PromptEntry(
            id: "appleFM.summarizer.meeting",
            title: "Apple FM · Summarize · Meeting",
            body: AppleFMSummarizer.meetingInstructionsClassic
        ))
        entries.append(PromptEntry(
            id: "appleFM.summarizer.meetingExperimental",
            title: "Apple FM · Summarize · Meeting (Experiment)",
            body: AppleFMSummarizer.meetingInstructionsExperimental
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
        // Meeting prompts are family-parameterized statics shared by
        // both MLX specs (rows are text-only, so the specs have
        // nothing to vary beyond the per-family turn directives).
        entries.append(contentsOf: mlxMeetingEntries(
            family: .qwen,
            idPrefix: "qwen",
            titlePrefix: "Qwen3"
        ))

        // LM Studio (remote backend, OpenAI-compatible HTTP).
        // Uses the same MLXQwenSpec.buildPrompt the on-device
        // Qwen path uses — LM Studio applies the loaded model's
        // chat template server-side, so the structure shown
        // here is what the user message will look like.
        entries.append(PromptEntry(
            id: "lmStudio.summarizer.singlePass",
            title: "LM Studio · Summarize · Trailing / Heuristic",
            body: qwen.buildPrompt(
                utterances: sampleUtterances,
                speakerNames: sampleSpeakerNames,
                truncatedFromTotal: nil,
                selection: .trailing
            )
        ))
        // Meeting modes have their OWN prompt on this backend
        // (unlike trailing/heuristic above, which reuses the Qwen
        // spec) — surface the real LMStudioSummarizer builders.
        entries.append(PromptEntry(
            id: "lmStudio.summarizer.meeting",
            title: "LM Studio · Summarize · Meeting",
            body: LMStudioSummarizer.buildMeetingPromptClassic(
                utterances: sampleUtterances,
                speakerNames: sampleSpeakerNames,
                truncatedFromTotal: nil
            )
        ))
        entries.append(PromptEntry(
            id: "lmStudio.summarizer.meetingExperimental",
            title: "LM Studio · Summarize · Meeting (Experiment)",
            body: LMStudioSummarizer.buildMeetingPromptExperimental(
                utterances: sampleUtterances,
                speakerNames: sampleSpeakerNames,
                glossaryTerms: sampleGlossaryTerms,
                truncatedFromTotal: nil
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
        entries.append(contentsOf: mlxMeetingEntries(
            family: .llama,
            idPrefix: "llama",
            titlePrefix: "Llama-3-Swallow"
        ))

        return entries
    }

    /// The three meeting entries one MLX family contributes:
    /// classic single-pass, experimental single-pass (the same
    /// builder also produces the map-reduce window prompts — they
    /// differ only by a window-context line), and the experimental
    /// map-reduce merge. Classic has no merge: it is single-pass
    /// only (over-cap sessions fall back to TF-IDF sampling).
    private static func mlxMeetingEntries(
        family: LLMModelFamily,
        idPrefix: String,
        titlePrefix: String
    ) -> [PromptEntry] {
        [
            PromptEntry(
                id: "\(idPrefix).summarizer.meeting",
                title: "\(titlePrefix) · Summarize · Meeting",
                body: MLXLLMSummarizerCore.buildMeetingPromptClassic(
                    utterances: sampleUtterances,
                    speakerNames: sampleSpeakerNames,
                    truncatedFromTotal: nil,
                    family: family
                )
            ),
            PromptEntry(
                id: "\(idPrefix).summarizer.meetingExperimental",
                title: "\(titlePrefix) · Summarize · Meeting (Experiment)",
                body: MLXLLMSummarizerCore.buildMeetingPrompt(
                    utterances: sampleUtterances,
                    firstRowNumber: 1,
                    speakerNames: sampleSpeakerNames,
                    glossaryTerms: sampleGlossaryTerms,
                    windowContext: nil,
                    family: family
                )
            ),
            PromptEntry(
                id: "\(idPrefix).summarizer.meetingExperimentalMerge",
                title: "\(titlePrefix) · Summarize · Meeting (Experiment) · Merge",
                body: MLXLLMSummarizerCore.buildMeetingMergePrompt(
                    intermediates: sampleMeetingIntermediates,
                    allUtterances: sampleUtterances,
                    speakerNames: sampleSpeakerNames,
                    glossaryTerms: sampleGlossaryTerms,
                    family: family
                )
            ),
        ]
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
                totalChunks: 1,
                contextPrefix: []
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
                totalChunks: 1,
                contextPrefix: []
            )
        ))

        // LM Studio reviewer — same MLXQwenReviewerSpec the
        // adapter uses at request time.
        entries.append(PromptEntry(
            id: "lmStudio.reviewer",
            title: "LM Studio · Reviewer",
            body: qwen.buildPrompt(
                utterances: sampleUtterances,
                speakerNames: sampleSpeakerNames,
                language: .japanese,
                chunkIndex: 0,
                totalChunks: 1,
                contextPrefix: []
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

    // MARK: - Debug (uncapped, single-prompt) builders

    /// Build the full Summary + Reviewer prompt for every model
    /// using `utterances` as-is — no per-spec cap, no chunking,
    /// no per-window splitting. Intended for the Prompts card's
    /// Debug section so the developer can see exactly what each
    /// model would receive if the live path didn't truncate.
    /// One entry per (model × {Summary, Reviewer}); returns an
    /// empty list when `utterances` is empty (nothing to inspect).
    ///
    /// The resulting prompts will routinely exceed each model's
    /// context window — that's expected. These are inspection
    /// artifacts, not something to feed back into inference.
    public static func debugPrompts(
        utterances: [UtteranceEstimate],
        speakerNames: [String: String],
        language: ReviewLanguage
    ) -> [PromptEntry] {
        guard !utterances.isEmpty else { return [] }
        var entries: [PromptEntry] = []

        entries.append(PromptEntry(
            id: "debug.appleFM.summarizer",
            title: "Apple FM · Summary · All utterances",
            body: appleFMSummarizerDebugPrompt(
                utterances: utterances,
                speakerNames: speakerNames
            )
        ))
        entries.append(PromptEntry(
            id: "debug.appleFM.reviewer",
            title: "Apple FM · Reviewer · All utterances",
            body: appleFMReviewerDebugPrompt(
                utterances: utterances,
                speakerNames: speakerNames,
                language: language
            )
        ))

        entries.append(PromptEntry(
            id: "debug.qwen.summarizer",
            title: "Qwen3 · Summary · All utterances",
            body: MLXQwenSpec().buildPrompt(
                utterances: utterances,
                speakerNames: speakerNames,
                truncatedFromTotal: nil,
                selection: .trailing
            )
        ))
        entries.append(PromptEntry(
            id: "debug.qwen.reviewer",
            title: "Qwen3 · Reviewer · All utterances",
            body: MLXQwenReviewerSpec().buildPrompt(
                utterances: utterances,
                speakerNames: speakerNames,
                language: language,
                chunkIndex: 0,
                totalChunks: 1,
                contextPrefix: []
            )
        ))

        entries.append(PromptEntry(
            id: "debug.llama.summarizer",
            title: "Llama-3-Swallow · Summary · All utterances",
            body: MLXLlamaSpec().buildPrompt(
                utterances: utterances,
                speakerNames: speakerNames,
                truncatedFromTotal: nil,
                selection: .trailing
            )
        ))
        entries.append(PromptEntry(
            id: "debug.llama.reviewer",
            title: "Llama-3-Swallow · Reviewer · All utterances",
            body: MLXLlamaReviewerSpec().buildPrompt(
                utterances: utterances,
                speakerNames: speakerNames,
                language: language,
                chunkIndex: 0,
                totalChunks: 1,
                contextPrefix: []
            )
        ))

        return entries
    }

    /// Mirrors `AppleFMSummarizer.summarizeSinglePass` exactly, but
     /// with the per-spec cap bypassed (every utterance fed in
     /// instead of the trailing / heuristic-selected slice). Same
     /// language directive, same demographics block, same
     /// `compactLine` shape, no truncation note (nothing was
     /// truncated). See [[live-singlepass]] in
     /// AppleFMSummarizer.swift for the source path.
    private static func appleFMSummarizerDebugPrompt(
        utterances: [UtteranceEstimate],
        speakerNames: [String: String]
    ) -> String {
        let speakers = utterances.orderedSpeakerIDs
        let utteranceLines = utterances
            .map { appleFMSummarizerCompactLine(for: $0, speakerNames: speakerNames) }
            .joined(separator: "\n")
        let demographicsBlock = SpeakerDemographicsDigest
            .build(from: utterances)
            .renderForPrompt(speakerIDs: speakers, speakerNames: speakerNames)
        let demographicsLine = demographicsBlock.isEmpty
            ? ""
            : "\n\n\(demographicsBlock)"
        let languageDirective = SummarizerLocale.responseLanguageInstruction
        let userMessage = """
            \(languageDirective)

            Speakers present: \(speakers.joined(separator: ", ")).\(demographicsLine)

            Utterances:
            \(utteranceLines)
            """
        return [
            "=== SYSTEM (instructions) ===",
            AppleFMSummarizer.instructions,
            "",
            "=== USER (per-call message) ===",
            userMessage
        ].joined(separator: "\n")
    }

    /// Mirrors `AppleFMTranscriptionReviewer.reviewChunk` with
     /// `chunkIndex = 0, totalChunks = 1` (so the multi-chunk
     /// preface stays off) and every utterance in one shot —
     /// the "as if unlimited" view of what the live path would
     /// have sent if the 20-utterance chunk cap didn't exist.
    private static func appleFMReviewerDebugPrompt(
        utterances: [UtteranceEstimate],
        speakerNames: [String: String],
        language: ReviewLanguage
    ) -> String {
        let lines = utterances.enumerated().map { rowIdx, u in
            appleFMReviewerCompactLine(
                rowIndex: rowIdx + 1,
                for: u,
                speakerNames: speakerNames
            )
        }.joined(separator: "\n")
        let userMessage = """
            The conversation is in \(language.label). Reason about meaning and homophones in \(language.label) only.
            Write each issue's "reason" field in \(SummarizerLocale.responseLanguageNameInEnglish). Use no other language for the reason text.

            Utterances (rowIndex speaker t=time text):
            \(lines)
            """
        return [
            "=== SYSTEM (instructions) ===",
            AppleFMTranscriptionReviewer.instructions,
            "",
            "=== USER (per-call message) ===",
            userMessage
        ].joined(separator: "\n")
    }

    /// Mirrors the private `AppleFMSummarizer.compactLine`:
     /// `S01 12.3s joy V0.42 A0.71 "text"` (label / V / A are
     /// emitted only when present; transcript is quoted and
     /// backslash/quote-escaped). Reproduced here because the
     /// summarizer's helper is `private static`.
    private static func appleFMSummarizerCompactLine(
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

    // MARK: - Real-prompt generation

    /// Build the prompt(s) that would actually be sent to the
    /// model for `entryID` using `utterances` from the live
    /// session. Returns multiple strings when the live path
    /// chunks (reviewer per-chunk, summarizer deep-window per-
    /// window); the caller concatenates them with separators.
    /// Returns nil when the entry can't be fully realized
    /// without running the live model (summarizer deep-merge
    /// needs per-window intermediates from a real generate
    /// pass).
    ///
    /// MLX entries call the same spec types' `buildPrompt`
    /// methods the inference path uses, after applying the
    /// same selection (trailing / heuristic) or
    /// chunking / windowing the live path applies. Apple FM
    /// entries reproduce the user message that the live
    /// `summarize` / `review` constructs inline (formatting
    /// helpers + speakers roster + utterance lines + language
    /// directive).
    /// `glossaryTerms` feeds only the meeting-experimental
    /// entries (the live path folds the keyword bank + custom
    /// glossary into those prompts); every other entry ignores it,
    /// and the default keeps existing call sites source-compatible.
    public static func realPrompts(
        forEntryID entryID: String,
        utterances: [UtteranceEstimate],
        speakerNames: [String: String],
        language: ReviewLanguage,
        glossaryTerms: [String] = []
    ) -> [String]? {
        switch entryID {
        // Apple FM
        case "appleFM.summarizer.singlePass":
            return [appleFMSinglePassRealPrompt(
                utterances: utterances,
                speakerNames: speakerNames
            )]
        case "appleFM.summarizer.deepWindow":
            return appleFMDeepWindowRealPrompts(
                utterances: utterances,
                speakerNames: speakerNames
            )
        case "appleFM.summarizer.deepMerge":
            // Live merge prompt depends on per-window
            // intermediates that only exist after a real
            // generate pass — can't construct deterministically.
            return nil
        case "appleFM.summarizer.meeting":
            return [appleFMMeetingRealPrompt(
                utterances: utterances,
                speakerNames: speakerNames,
                experimental: false,
                glossaryTerms: []
            )]
        case "appleFM.summarizer.meetingExperimental":
            return [appleFMMeetingRealPrompt(
                utterances: utterances,
                speakerNames: speakerNames,
                experimental: true,
                glossaryTerms: glossaryTerms
            )]
        case "appleFM.reviewer":
            return appleFMReviewerRealPrompts(
                utterances: utterances,
                speakerNames: speakerNames,
                language: language
            )
        case "foundationModels.textSER":
            return foundationModelsTextSERRealPrompts(
                utterances: utterances
            )

        // Qwen3
        case "qwen.summarizer.singlePass":
            return [mlxSinglePassRealPrompt(
                spec: MLXQwenSpec(),
                utterances: utterances,
                speakerNames: speakerNames
            )]
        case "qwen.summarizer.deepWindow":
            return mlxDeepWindowRealPrompts(
                spec: MLXQwenSpec(),
                utterances: utterances,
                speakerNames: speakerNames
            )
        case "qwen.summarizer.deepMerge":
            return nil
        case "qwen.summarizer.meeting":
            return [mlxMeetingClassicRealPrompt(
                spec: MLXQwenSpec(),
                utterances: utterances,
                speakerNames: speakerNames
            )]
        case "qwen.summarizer.meetingExperimental":
            return mlxMeetingExperimentalRealPrompts(
                spec: MLXQwenSpec(),
                utterances: utterances,
                speakerNames: speakerNames,
                glossaryTerms: glossaryTerms
            )
        case "qwen.summarizer.meetingExperimentalMerge":
            // Like the deep merges: the live merge prompt embeds
            // per-window intermediates that only exist after a real
            // generate pass — can't construct deterministically.
            return nil
        case "qwen.reviewer":
            return mlxReviewerRealPrompts(
                spec: MLXQwenReviewerSpec(),
                utterances: utterances,
                speakerNames: speakerNames,
                language: language
            )

        // Llama-3-Swallow
        case "llama.summarizer.singlePass":
            return [mlxSinglePassRealPrompt(
                spec: MLXLlamaSpec(),
                utterances: utterances,
                speakerNames: speakerNames
            )]
        case "llama.summarizer.deepWindow":
            return mlxDeepWindowRealPrompts(
                spec: MLXLlamaSpec(),
                utterances: utterances,
                speakerNames: speakerNames
            )
        case "llama.summarizer.deepMerge":
            return nil
        case "llama.summarizer.meeting":
            return [mlxMeetingClassicRealPrompt(
                spec: MLXLlamaSpec(),
                utterances: utterances,
                speakerNames: speakerNames
            )]
        case "llama.summarizer.meetingExperimental":
            return mlxMeetingExperimentalRealPrompts(
                spec: MLXLlamaSpec(),
                utterances: utterances,
                speakerNames: speakerNames,
                glossaryTerms: glossaryTerms
            )
        case "llama.summarizer.meetingExperimentalMerge":
            return nil
        case "llama.reviewer":
            return mlxReviewerRealPrompts(
                spec: MLXLlamaReviewerSpec(),
                utterances: utterances,
                speakerNames: speakerNames,
                language: language
            )

        // LM Studio (meeting modes only — the trailing/heuristic
        // entry reuses the Qwen spec and predates real-prompt
        // support for this backend).
        case "lmStudio.summarizer.meeting":
            return [lmStudioMeetingRealPrompt(
                utterances: utterances,
                speakerNames: speakerNames,
                experimental: false,
                glossaryTerms: []
            )]
        case "lmStudio.summarizer.meetingExperimental":
            return [lmStudioMeetingRealPrompt(
                utterances: utterances,
                speakerNames: speakerNames,
                experimental: true,
                glossaryTerms: glossaryTerms
            )]

        default:
            return nil
        }
    }

    // MARK: - MLX real-prompt builders

    private static func mlxSinglePassRealPrompt(
        spec: any MLXLLMSpec,
        utterances: [UtteranceEstimate],
        speakerNames: [String: String]
    ) -> String {
        let (selected, truncatedFromTotal) = applySinglePassSelection(
            utterances: utterances,
            cap: spec.maxPromptUtterances,
            selection: .trailing
        )
        return spec.buildPrompt(
            utterances: selected,
            speakerNames: speakerNames,
            truncatedFromTotal: truncatedFromTotal,
            selection: .trailing
        )
    }

    private static func mlxDeepWindowRealPrompts(
        spec: any MLXLLMSpec,
        utterances: [UtteranceEstimate],
        speakerNames: [String: String]
    ) -> [String] {
        let windows = splitWindows(utterances, size: spec.deepWindowSize)
        return windows.enumerated().map { idx, window in
            spec.buildDeepWindowPrompt(
                utterances: window,
                speakerNames: speakerNames,
                windowIndex: idx,
                totalWindows: windows.count
            )
        }
    }

    private static func mlxReviewerRealPrompts(
        spec: any MLXLLMReviewerSpec,
        utterances: [UtteranceEstimate],
        speakerNames: [String: String],
        language: ReviewLanguage
    ) -> [String] {
        let chunks = splitWindows(utterances, size: spec.maxPromptUtterances)
        return chunks.enumerated().map { idx, chunk in
            spec.buildPrompt(
                utterances: chunk,
                speakerNames: speakerNames,
                language: language,
                chunkIndex: idx,
                totalChunks: chunks.count,
                contextPrefix: []
            )
        }
    }

    // MARK: - Meeting real-prompt builders

    // Over-cap selection for every meeting preview is the shared
    // `applySinglePassSelection(_, cap:, .heuristicTopN)` — the
    // same speaker-balanced TF-IDF the live paths run. The live
    // paths additionally boost keyword-matched rows
    // (`boostedUtteranceIDs`); the preview can't know the boost
    // set from this static call site, so the unboosted selection
    // is the representative case — same approximation precedent
    // as `appleFMSinglePassRealPrompt`'s trailing default.

    private static func mlxMeetingClassicRealPrompt(
        spec: any MLXLLMSpec,
        utterances: [UtteranceEstimate],
        speakerNames: [String: String]
    ) -> String {
        let (selected, truncatedFromTotal) = applySinglePassSelection(
            utterances: utterances,
            cap: spec.meetingMaxPromptUtterances,
            selection: .heuristicTopN
        )
        return MLXLLMSummarizerCore.buildMeetingPromptClassic(
            utterances: selected,
            speakerNames: speakerNames,
            truncatedFromTotal: truncatedFromTotal,
            family: spec.family
        )
    }

    /// Mirrors `LMStudioSummarizer.summarizeMeetingClassic` /
    /// `summarizeMeetingExperimental`: heuristic top-N over this
    /// backend's own cap, then the matching prompt builder.
    private static func lmStudioMeetingRealPrompt(
        utterances: [UtteranceEstimate],
        speakerNames: [String: String],
        experimental: Bool,
        glossaryTerms: [String]
    ) -> String {
        let (selected, truncatedFromTotal) = applySinglePassSelection(
            utterances: utterances,
            cap: LMStudioSummarizer.meetingMaxPromptUtterances,
            selection: .heuristicTopN
        )
        if experimental {
            return LMStudioSummarizer.buildMeetingPromptExperimental(
                utterances: selected,
                speakerNames: speakerNames,
                glossaryTerms: glossaryTerms,
                truncatedFromTotal: truncatedFromTotal
            )
        }
        return LMStudioSummarizer.buildMeetingPromptClassic(
            utterances: selected,
            speakerNames: speakerNames,
            truncatedFromTotal: truncatedFromTotal
        )
    }

    /// Mirrors `MLXLLMSummarizerCore.summarizeMeeting`: at or under
    /// the single-pass cap → one prompt (no window context); over
    /// it → the map-reduce's per-window prompts with globally-
    /// numbered rows. The merge prompt is excluded for the same
    /// reason as deep-merge (needs live intermediates).
    private static func mlxMeetingExperimentalRealPrompts(
        spec: any MLXLLMSpec,
        utterances: [UtteranceEstimate],
        speakerNames: [String: String],
        glossaryTerms: [String]
    ) -> [String] {
        guard utterances.count > spec.meetingMaxPromptUtterances else {
            return [MLXLLMSummarizerCore.buildMeetingPrompt(
                utterances: utterances,
                firstRowNumber: 1,
                speakerNames: speakerNames,
                glossaryTerms: glossaryTerms,
                windowContext: nil,
                family: spec.family
            )]
        }
        let windows = splitWindows(utterances, size: spec.meetingDeepWindowSize)
        var firstRowNumber = 1
        return windows.enumerated().map { idx, window in
            let prompt = MLXLLMSummarizerCore.buildMeetingPrompt(
                utterances: window,
                firstRowNumber: firstRowNumber,
                speakerNames: speakerNames,
                glossaryTerms: glossaryTerms,
                windowContext: (index: idx, total: windows.count),
                family: spec.family
            )
            firstRowNumber += window.count
            return prompt
        }
    }

    /// Mirrors `AppleFMSummarizer.summarizeMeetingClassic` /
    /// `summarizeMeetingExperimental`'s user-message construction:
    /// language directive, speakers roster, friendly-name roster,
    /// (experimental only) domain-terms line, text-only rows, and
    /// the TF-IDF truncation note.
    private static func appleFMMeetingRealPrompt(
        utterances: [UtteranceEstimate],
        speakerNames: [String: String],
        experimental: Bool,
        glossaryTerms: [String]
    ) -> String {
        let (selected, truncatedFrom) = applySinglePassSelection(
            utterances: utterances,
            cap: AppleFMSummarizer.meetingMaxPromptUtterances,
            selection: .heuristicTopN
        )
        let speakers = orderedSpeakerIDs(selected)
        let utteranceLines = experimental
            ? selected.enumerated()
                .map { AppleFMSummarizer.meetingLineExperimental(index: $0.offset + 1, for: $0.element) }
                .joined(separator: "\n")
            : selected
                .map { AppleFMSummarizer.meetingLineClassic(for: $0) }
                .joined(separator: "\n")
        let nameRoster = speakers
            .compactMap { id -> String? in
                guard let name = speakerNames[id], !name.isEmpty else { return nil }
                return "\(id) = \(name)"
            }
            .joined(separator: ", ")
        let nameRosterLine = nameRoster.isEmpty
            ? ""
            : "\n\nSpeaker names: \(nameRoster)."
        let glossaryLine = (experimental && !glossaryTerms.isEmpty)
            ? "\n\nDomain terms: \(glossaryTerms.joined(separator: ", "))."
            : ""
        let truncationNote: String
        if let total = truncatedFrom {
            truncationNote = "\n\n(Showing the \(selected.count) most distinctive of \(total) utterances by session-relative TF-IDF, in chronological order; treat as a representative sample of the meeting.)"
        } else {
            truncationNote = ""
        }
        let userMessage = """
            \(SummarizerLocale.responseLanguageInstruction)

            Speakers present: \(speakers.joined(separator: ", ")).\(nameRosterLine)\(glossaryLine)

            Utterances:
            \(utteranceLines)\(truncationNote)
            """
        let instructions = experimental
            ? AppleFMSummarizer.meetingInstructionsExperimental
            : AppleFMSummarizer.meetingInstructionsClassic
        return [
            "=== SYSTEM (instructions) ===",
            instructions,
            "",
            "=== USER (per-call message) ===",
            userMessage
        ].joined(separator: "\n")
    }

    // MARK: - Apple FM real-prompt builders

    /// Mirrors `AppleFMSummarizer.summarizeSinglePass`'s prompt
    /// construction: instructions + user message with speakers
    /// roster + utterance lines + truncation note. Selection is
    /// `.trailing` for the preview (the live path uses the
    /// mode the user picked; we don't have access to the user's
    /// pick from this static call site, and trailing is the
    /// default mode so it's the representative case).
    private static func appleFMSinglePassRealPrompt(
        utterances: [UtteranceEstimate],
        speakerNames: [String: String]
    ) -> String {
        let cap = 15
        let (selected, truncatedFromTotal) = applySinglePassSelection(
            utterances: utterances,
            cap: cap,
            selection: .trailing
        )
        let speakers = orderedSpeakerIDs(selected)
        let lines = selected
            .map { appleFMCompactLine(for: $0, speakerNames: speakerNames) }
            .joined(separator: "\n")
        var trunc = ""
        if let total = truncatedFromTotal {
            trunc = "\n\n(Showing the most recent \(selected.count) of \(total) utterances; frame overall mood as the trailing portion.)"
        }
        let userMessage = """
            <language-directive>

            Speakers present: \(speakers.joined(separator: ", ")).

            Utterances:
            \(lines)\(trunc)
            """
        return [
            "=== SYSTEM (instructions) ===",
            AppleFMSummarizer.instructions,
            "",
            "=== USER (per-call message) ===",
            userMessage
        ].joined(separator: "\n")
    }

    private static func appleFMDeepWindowRealPrompts(
        utterances: [UtteranceEstimate],
        speakerNames: [String: String]
    ) -> [String] {
        let windows = splitWindows(utterances, size: 15)
        return windows.enumerated().map { idx, window in
            let lines = window
                .map { appleFMCompactLine(for: $0, speakerNames: speakerNames) }
                .joined(separator: "\n")
            let userMessage = """
                Window \(idx + 1) of \(windows.count).

                Utterances:
                \(lines)
                """
            return [
                "=== SYSTEM (instructions) ===",
                AppleFMSummarizer.windowInstructions,
                "",
                "=== USER (per-window message) ===",
                userMessage
            ].joined(separator: "\n")
        }
    }

    private static func appleFMReviewerRealPrompts(
        utterances: [UtteranceEstimate],
        speakerNames: [String: String],
        language: ReviewLanguage
    ) -> [String] {
        let cap = 20
        let chunks = splitWindows(utterances, size: cap)
        return chunks.enumerated().map { idx, chunk in
            let lines = chunk.enumerated().map { rowIdx, u in
                appleFMReviewerCompactLine(
                    rowIndex: rowIdx + 1,
                    for: u,
                    speakerNames: speakerNames
                )
            }.joined(separator: "\n")
            let preface = chunks.count > 1
                ? "This is chunk \(idx + 1) of \(chunks.count) of the conversation's review pass. Earlier and later utterances are reviewed separately; do not flag rows as non-sequitur just because broader topic context isn't visible here.\n\n"
                : ""
            let userMessage = """
                The conversation is in \(language.label). Reason about meaning and homophones in \(language.label) only.

                \(preface)Utterances (rowIndex speaker t=time text):
                \(lines)
                """
            return [
                "=== SYSTEM (instructions) ===",
                AppleFMTranscriptionReviewer.instructions,
                "",
                "=== USER (per-chunk message) ===",
                userMessage
            ].joined(separator: "\n")
        }
    }

    private static func foundationModelsTextSERRealPrompts(
        utterances: [UtteranceEstimate]
    ) -> [String] {
        // The live path sends one prompt per utterance —
        // instructions + "Utterance: <text>". Empty / filler
        // rows are skipped upstream in `AnalysisPipeline.runText`;
        // we mirror that here so the export reflects what's
        // actually sent.
        let instructions = "You are a <language>-language affect annotator.\n\n"
            + FoundationModelsSER.instructionsBody
        return utterances.compactMap { u in
            let trimmed = u.transcript.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { return nil }
            return [
                "=== SYSTEM (instructions) ===",
                instructions,
                "",
                "=== USER (per-utterance message) ===",
                "Utterance: \(trimmed)"
            ].joined(separator: "\n")
        }
    }

    // MARK: - Shared helpers

    private static func applySinglePassSelection(
        utterances: [UtteranceEstimate],
        cap: Int,
        selection: MLXLLMSelection
    ) -> (selected: [UtteranceEstimate], truncatedFromTotal: Int?) {
        guard utterances.count > cap else { return (utterances, nil) }
        switch selection {
        case .trailing:
            return (Array(utterances.suffix(cap)), utterances.count)
        case .heuristicTopN:
            let ids = Informativeness.topNBalancedBySpeaker(
                cap, utterances: utterances
            )
            return (utterances.filter { ids.contains($0.id) }, utterances.count)
        }
    }

    private static func splitWindows(
        _ utterances: [UtteranceEstimate],
        size: Int
    ) -> [[UtteranceEstimate]] {
        guard size > 0, !utterances.isEmpty else {
            return utterances.isEmpty ? [] : [utterances]
        }
        return stride(from: 0, to: utterances.count, by: size).map { offset in
            let end = min(offset + size, utterances.count)
            return Array(utterances[offset..<end])
        }
    }

    private static func orderedSpeakerIDs(
        _ utterances: [UtteranceEstimate]
    ) -> [String] {
        var seen: Set<String> = []
        var ordered: [String] = []
        for u in utterances where !seen.contains(u.speakerID) {
            seen.insert(u.speakerID)
            ordered.append(u.speakerID)
        }
        return ordered
    }

    /// Mirrors `AppleFMSummarizer.compactLine` (the same line
    /// format the live summarizer feeds into its user message).
    private static func appleFMCompactLine(
        for u: UtteranceEstimate,
        speakerNames: [String: String]
    ) -> String {
        let speaker: String
        if let name = speakerNames[u.speakerID]?
            .trimmingCharacters(in: .whitespacesAndNewlines),
           !name.isEmpty {
            speaker = "\(u.speakerID)(\(name))"
        } else {
            speaker = u.speakerID
        }
        let label = u.fusedTopLabel ?? "—"
        let v = u.fusedValence.map { String(format: "%.2f", $0) } ?? "—"
        let a = u.fusedArousal.map { String(format: "%.2f", $0) } ?? "—"
        return "\(speaker) t=\(String(format: "%.1f", u.start)) \(label) V=\(v) A=\(a) \(u.transcript)"
    }

    /// Mirrors `AppleFMTranscriptionReviewer.compactLine`.
    private static func appleFMReviewerCompactLine(
        rowIndex: Int,
        for u: UtteranceEstimate,
        speakerNames: [String: String]
    ) -> String {
        let speaker: String
        if let name = speakerNames[u.speakerID], !name.isEmpty {
            speaker = "\(u.speakerID)(\(name))"
        } else {
            speaker = u.speakerID
        }
        let escaped = u.transcript
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
        return "\(rowIndex) \(speaker) t=\(String(format: "%.1f", u.start)) \"\(escaped)\""
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

    /// Sample glossary for the meeting-experimental previews. Two
    /// terms keep the "Domain terms" line visible in the rendered
    /// prompt; sessions without keywords/glossary entries simply
    /// omit that line at inference time.
    private static let sampleGlossaryTerms: [String] = ["Xephon", "diarization"]

    /// Sample per-window PARTIAL minutes for the meeting merge-
    /// prompt preview — same role as `sampleDeepIntermediates`
    /// below, shaped like one decoded map-reduce window.
    private static let sampleMeetingIntermediates: [MLXLLMSummarizerCore.MeetingWire] = [
        MLXLLMSummarizerCore.MeetingWire(
            topic: "Kick-off scheduling and venue budget.",
            topics: [
                MLXLLMSummarizerCore.MeetingWire.Topic(
                    title: "Kick-off date",
                    raisedBy: "S01",
                    evidence: MLXLLMSummarizerCore.MeetingWire.Evidence(numbers: [1]),
                    positions: [
                        MLXLLMSummarizerCore.MeetingWire.Position(
                            speaker: "S02",
                            stance: "Prefers the later date to allow prep time.",
                            evidence: MLXLLMSummarizerCore.MeetingWire.Evidence(numbers: [2])
                        )
                    ]
                )
            ],
            perSpeaker: [
                MLXLLMSummarizerCore.MeetingWire.PerSpeaker(
                    speakerID: "S01",
                    talkingPoints: ["proposed kick-off date"]
                ),
                MLXLLMSummarizerCore.MeetingWire.PerSpeaker(
                    speakerID: "S02",
                    talkingPoints: ["asked for prep time"]
                ),
            ]
        )
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
