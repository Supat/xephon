import Foundation
import Fusion
import SERAcoustic
import SERText
import XephonLogging
import MLX
import MLXLLM
import MLXLMCommon

/// MLX-backed summarizer for the Qwen3-8B-Instruct (4-bit)
/// family. Pairs with `MLXLlamaSummarizer` for the Llama 3.1
/// Swallow family. Both are thin actors over the shared
/// orchestration in `MLXLLMSummarizerCore`; this file
/// supplies the Qwen-specific spec (prompts, EOS tokens,
/// row format) and owns the per-actor lifecycle (load /
/// unload of the `ModelContainer`).
///
/// Lifecycle:
///   1. Construct with `modelIdentifier` + `modelDirectory`
///      (handled by `SummarizerCoordinator` against the
///      `ModelStore`).
///   2. `load()` brings the 4-bit weights into memory
///      (~5–10 s on M4 iPad Pro). Idempotent; lazy-called
///      by `summarize(...)` on first use.
///   3. `summarize(...)` runs one of the three
///      `SummarizeMode` paths via `MLXLLMSummarizerCore`.
///   4. `unload()` releases the ~4.6 GB working set when the
///      summary sheet dismisses.
public actor MLXQwenSummarizer: SessionSummarizer, MLXLLMSummarizerActor {
    public let modelIdentifier: String
    private let modelDirectory: URL
    private var container: ModelContainer?
    private let spec = MLXQwenSpec()
    /// KV reuse across consecutive plugin calls — today that means
    /// verbatim parse-failure retries (see MLXPromptPrefixCache's
    /// doc for the reverted shared-head history). Plugin path only —
    /// summarize/review prompts don't share prefixes worth the held
    /// KV memory (~0.6 GB for a 2k-token prompt, held for the
    /// batch envelope's lifetime).
    private let pluginPrefixCache = MLXPromptPrefixCache()

    public init(modelIdentifier: String, modelDirectory: URL) {
        self.modelIdentifier = modelIdentifier
        self.modelDirectory = modelDirectory
    }

    public var isReady: Bool {
        container != nil
    }

    public func load() async throws {
        if container != nil { return }
        AppLog.app.info(
            "MLXQwenSummarizer loading from \(self.modelDirectory.path, privacy: .public)"
        )
        // Cap MLX's buffer cache to keep the working set
        // tight. The default is bounded by Metal's
        // recommendedMaxWorkingSetSize, which on a 16 GB iPad
        // sits high enough to push the process over the
        // Jetsam ceiling once Qwen weights and the SER
        // pipeline coexist. 128 MB lets MLX keep more
        // prefill/decode scratch buffers resident,
        // measurably improving generation throughput without
        // re-introducing Jetsam pressure (the SER actors are
        // torn down by `SummarizerCoordinator` before
        // `summarize` runs, so we have the headroom).
        MLX.GPU.set(cacheLimit: 128 * 1024 * 1024)
        do {
            let configuration = ModelConfiguration(
                directory: modelDirectory,
                extraEOSTokens: spec.extraEOSTokens
            )
            container = try await LLMModelFactory.shared.loadContainer(
                configuration: configuration
            )
            AppLog.app.info("MLXQwenSummarizer loaded")
        } catch {
            throw SummarizerError.modelLoadFailed(
                reason: String(describing: error)
            )
        }
    }

    public func unload() {
        container = nil
        // The held KV state references model-sized MLX buffers —
        // never outlive the weights.
        pluginPrefixCache.reset()
        AppLog.app.info("MLXQwenSummarizer unloaded")
    }

    public func generateRaw(prompt: String, maxOutputTokens: Int) async throws -> String {
        try await load()
        guard let container else {
            throw SummarizerError.modelNotInstalled
        }
        // Family turn directives are non-negotiable for raw
        // generation: without /no_think, Qwen3 spends the entire
        // output budget inside a think block and the caller gets
        // truncated non-JSON (observed on-device: every plugin
        // call ran to the token cap and failed to parse).
        var effectivePrompt = prompt
        let directives = MLXLLMSpecDirectives.turnDirectives(family: spec.family)
        if !directives.isEmpty {
            effectivePrompt += "\n" + directives.joined(separator: "\n")
        }
        let raw = try await MLXLLMSummarizerCore.runInference(
            container: container,
            prompt: effectivePrompt,
            maxTokens: maxOutputTokens,
            repetitionPenalty: spec.repetitionPenalty,
            label: "MLX[\(spec.family.rawValue)] plugin",
            // Greedy: plugin calls fill evaluation sheets — repeat
            // runs on the same session must reproduce.
            temperature: 0,
            prefixCache: pluginPrefixCache
        )
        // Belt to the directive's braces: /no_think still emits an
        // empty think block on some checkpoints, and a think block
        // containing braces would poison the caller's first-{ to
        // last-} slice.
        return MLXLLMSummarizerCore.stripThinkBlocks(raw)
    }

    public func summarize(
        utterances: [UtteranceEstimate],
        speakerNames: [String: String],
        mode: SummarizeMode,
        boostedUtteranceIDs: Set<UUID>,
        glossaryTerms: [String]
    ) async throws -> SessionSummary {
        try await load()
        guard let container else {
            throw SummarizerError.modelNotInstalled
        }
        return try await MLXLLMSummarizerCore.summarize(
            container: container,
            modelIdentifier: modelIdentifier,
            utterances: utterances,
            speakerNames: speakerNames,
            mode: mode,
            boostedUtteranceIDs: boostedUtteranceIDs,
            glossaryTerms: glossaryTerms,
            spec: spec
        )
    }
}

// MARK: - Qwen spec

/// Qwen3-specific configuration + prompt builders. Used by
/// `MLXQwenSummarizer` and consumed by
/// `MLXLLMSummarizerCore`.
internal struct MLXQwenSpec: MLXLLMSpec {
    let family: LLMModelFamily = .qwen

    /// Cap on the number of utterances we feed into a single
    /// summary pass. Each row carries the full acoustic
    /// 9-class softmax + Plutchik 8-class intensity in
    /// addition to the fused label/V/A/D — ~2.3× the per-row
    /// token cost vs. a fused-only line. 100 utterances at
    /// this richer format ≈ 10–14k input tokens, comfortably
    /// inside Qwen3's context window and the per-app memory
    /// budget. Sessions longer than this go through the
    /// `.heuristic` or `.deep` paths to handle the surplus.
    let maxPromptUtterances = 100

    /// Per-window utterance count for `.deep` mode. Smaller
    /// than `maxPromptUtterances` because deep mode runs many
    /// windows back-to-back; tight windows bound per-call
    /// prefill cost and give the merge pass uniformly-sized
    /// inputs. 50 lands at ~5–7k prompt tokens per window.
    let deepWindowSize = 50

    /// Output-token cap for per-window intermediate summaries.
    /// Bumped from 768 → 1280 after a 5-speaker window
    /// saturated the prior cap mid-emit and forced placeholder
    /// synthesis. 1280 fits verbose windows with headroom;
    /// `MLXLLMSummarizerCore.recoverTruncatedWindowIntermediate`
    /// salvages anything that still spills.
    let deepWindowOutputTokens = 1280

    /// Hard cap on tokens the LLM may emit per fast /
    /// heuristic / merge summary. 4096 fits the realistic max
    /// for a busy session's per-speaker arcs; the tolerant
    /// parser salvages the prefix when the model still runs
    /// over.
    let maxOutputTokens = 4096

    /// Qwen3's tokenizer already has `<|im_end|>` as
    /// `eos_token`, which handles the chat-template stop.
    /// `<|endoftext|>` is added defensively in case a future
    /// quant drops the chat template.
    let extraEOSTokens: Set<String> = ["<|endoftext|>"]

    /// Qwen3's instruction tuning is strong enough at 0.2
    /// temperature; no repetition-penalty needed.
    let repetitionPenalty: Float? = nil

    func buildPrompt(
        utterances: [UtteranceEstimate],
        speakerNames: [String: String],
        truncatedFromTotal: Int?,
        selection: MLXLLMSelection
    ) -> String {
        let speakers = utterances.orderedSpeakerIDs
        let speakerList = speakers.joined(separator: ", ")
        var lines: [String] = []
        lines.reserveCapacity(utterances.count + 14)
        lines.append("You are an analyst summarizing a multi-speaker conversation.")
        lines.append("Read every utterance below and produce a JSON object with four fields:")
        lines.append("  \"setting\" — one short sentence identifying the conversation's setting / situation / register (e.g. 'casual phone catchup between friends', 'job interview', 'classroom discussion'). Stay general — do not invent specific locations or institutions. Emit this FIRST so the rest stays consistent with it.")
        lines.append("  \"topic\" — one or two sentences on what the conversation is about.")
        lines.append("  \"overallMood\" — one paragraph on the session's overall emotional tone, consistent with the inferred setting.")
        lines.append("  \"perSpeaker\" — array, one entry per speaker id in this list: \(speakerList).")
        lines.append("Each perSpeaker entry has: { \"speakerID\": <id>, \"summary\": <one paragraph>, \"dominantMood\": <one short phrase> }.")
        lines.append("Each row carries: fused label and fused V/A/D (valence/arousal/dominance, 0–1), plus the raw per-modality probability vectors:")
        lines.append("  aP = acoustic 9-class softmax (angry, disgusted, fearful, happy, neutral, other, sad, surprised, unknown)")
        lines.append("  tP = text 8-class Plutchik intensity (joy, sadness, anticipation, surprise, anger, fear, disgust, trust)")
        lines.append("Use these to judge confidence and to flag modality disagreement — e.g. a row where aP says sad but tP says joy is worth calling out per-speaker; the fused label hides that signal.")
        lines.append("When the speaker demographics block below lists a gender, use it as the canonical pronoun for that speaker throughout the summary — 'she/her' for female, 'he/him' for male, 'they/them' for child or when no gender is listed. (Moot for languages that drop pronouns, like Japanese.)")
        lines.append("Return ONLY valid JSON, no prose before or after.")
        lines.append(SummarizerLocale.responseLanguageInstruction)
        // Qwen3 ships with a "thinking" mode that emits a
        // `<think>...</think>` chain-of-thought block before
        // the actual response. The `/no_think` directive
        // disables it for a single turn.
        lines.append("/no_think")
        if let total = truncatedFromTotal {
            lines.append("")
            switch selection {
            case .trailing:
                lines.append("NOTE: This conversation has \(total) utterances total; only the most recent \(utterances.count) are shown below. Frame the overall mood as the trailing portion of the session, not the whole arc.")
            case .heuristicTopN:
                lines.append("NOTE: This conversation has \(total) utterances total; the \(utterances.count) most distinctive utterances (chosen by session-relative TF-IDF, NOT the most recent) are shown below in chronological order. Frame the overall mood as a representative sample of the whole session, not a continuous trailing segment — gaps between rows are expected.")
            }
        }
        let demographics = SpeakerDemographicsDigest
            .build(from: utterances)
            .renderForPrompt(speakerIDs: speakers, speakerNames: speakerNames)
        if !demographics.isEmpty {
            lines.append("")
            lines.append(demographics)
        }
        lines.append("")
        lines.append("Utterances:")
        for u in utterances {
            lines.append(compactLine(for: u, speakerNames: speakerNames))
        }
        // Instruction sandwich — restate the directive right
        // before generation so the model's recent attention
        // has the "produce JSON" cue, not the last
        // utterance's `- speaker=…` line.
        lines.append("")
        lines.append("---")
        lines.append("IMPORTANT: Follow the instructions above and produce exactly one valid JSON object with fields setting, topic, overallMood, perSpeaker. The FIRST character of your output MUST be `{`. Do NOT echo the utterance list above; do NOT add any prose.")
        // Restate the language directive LAST — recency wins on
        // long prompts (same reason the JSON-shape rule is
        // restated in this sandwich). The copy mid-prompt sits
        // thousands of tokens back, above the utterance list, and
        // the small quantized models drift to the transcript's
        // (or the prompt's own) language without this reminder —
        // Apple FM honored the early copy, these did not.
        lines.append(SummarizerLocale.responseLanguageInstruction)
        return lines.joined(separator: "\n")
    }

    func buildDeepWindowPrompt(
        utterances: [UtteranceEstimate],
        speakerNames: [String: String],
        windowIndex: Int,
        totalWindows: Int
    ) -> String {
        let speakers = utterances.orderedSpeakerIDs
        let speakerList = speakers.joined(separator: ", ")
        let timeStart = utterances.first?.start ?? 0
        let timeEnd = utterances.last?.end ?? 0
        let tStartStr = String(format: "%.1f", timeStart)
        let tEndStr = String(format: "%.1f", timeEnd)
        var lines: [String] = []
        lines.reserveCapacity(utterances.count + 20)
        lines.append("You are an analyst summarizing ONE WINDOW of a longer multi-speaker conversation.")
        lines.append("This is window \(windowIndex + 1) of \(totalWindows), covering utterances from t=\(tStartStr)s to t=\(tEndStr)s.")
        lines.append("Produce a JSON object — NOT the final summary, just a compact intermediate that a merge pass will combine with the other windows.")
        lines.append("Fields:")
        lines.append("  \"windowIndex\": \(windowIndex) (copy this number),")
        lines.append("  \"timeStart\": \(tStartStr),")
        lines.append("  \"timeEnd\": \(tEndStr),")
        lines.append("  \"topicSnapshot\": one short phrase on this window's topic,")
        lines.append("  \"moodSnapshot\": one short phrase on this window's emotional tone,")
        lines.append("  \"perSpeaker\": array of { \"speakerID\": <id>, \"notes\": three to five sentences fleshing out this speaker's contribution in this window (notable statements, topics, emotional shifts — give the merge pass enough material to write a real per-speaker arc), \"dominantMood\": short phrase }, one entry per speaker in: \(speakerList),")
        lines.append("  \"modalityFlags\": array of short strings, each flagging one row where the acoustic and text classifiers notably disagreed (e.g. \"S03 at 42.3s: acoustic=sad, text=joy\"). Empty array if no notable disagreement.")
        lines.append("Each row carries: fused label and fused V/A/D (valence/arousal/dominance, 0–1), plus the raw per-modality probability vectors:")
        lines.append("  aP = acoustic 9-class softmax (angry, disgusted, fearful, happy, neutral, other, sad, surprised, unknown)")
        lines.append("  tP = text 8-class Plutchik intensity (joy, sadness, anticipation, surprise, anger, fear, disgust, trust)")
        lines.append("Return ONLY valid JSON, no prose before or after.")
        lines.append(SummarizerLocale.responseLanguageInstruction)
        lines.append("/no_think")
        let demographics = SpeakerDemographicsDigest
            .build(from: utterances)
            .renderForPrompt(speakerIDs: speakers, speakerNames: speakerNames)
        if !demographics.isEmpty {
            lines.append("")
            lines.append(demographics)
        }
        lines.append("")
        lines.append("Utterances:")
        for u in utterances {
            lines.append(compactLine(for: u, speakerNames: speakerNames))
        }
        lines.append("")
        lines.append("---")
        lines.append("IMPORTANT: Follow the instructions above and produce exactly one window-intermediate JSON object with fields windowIndex, timeStart, timeEnd, topicSnapshot, moodSnapshot, perSpeaker, modalityFlags. The FIRST character of your output MUST be `{`. Do NOT echo the utterance list above; do NOT add any prose.")
        // Restate the language directive LAST — recency wins on
        // long prompts (same reason the JSON-shape rule is
        // restated in this sandwich). The copy mid-prompt sits
        // thousands of tokens back, above the utterance list, and
        // the small quantized models drift to the transcript's
        // (or the prompt's own) language without this reminder —
        // Apple FM honored the early copy, these did not.
        lines.append(SummarizerLocale.responseLanguageInstruction)
        return lines.joined(separator: "\n")
    }

    func buildDeepMergePrompt(
        intermediates: [MLXLLMDeepWindowIntermediate],
        allUtterances: [UtteranceEstimate],
        speakerNames: [String: String]
    ) -> String {
        let allSpeakers = allUtterances.orderedSpeakerIDs
        let speakerList = allSpeakers.joined(separator: ", ")
        var lines: [String] = []
        lines.reserveCapacity(intermediates.count + 24)
        lines.append("You are an analyst producing the FINAL summary of a multi-speaker conversation.")
        lines.append("Below you'll see two complementary inputs: (a) JSON summaries of \(intermediates.count) consecutive windows of the conversation (in chronological order), and (b) a sample of the most distinctive raw utterances from across the session. The conversation has \(allUtterances.count) utterances total.")
        lines.append("Synthesize the window summaries into a single structured per-speaker arc — each speaker treated as one person across windows, not split into per-window sections — and lean on the raw utterances for verbatim quotes and specific phrasing when writing the per-speaker summaries.")
        lines.append("")
        lines.append("Produce a JSON object with four fields:")
        lines.append("  \"setting\" — one short sentence identifying the conversation's setting / situation / register (e.g. 'casual phone catchup between friends', 'job interview', 'classroom discussion'). Stay general — do not invent specific locations or institutions. Emit this FIRST so the rest stays consistent with it.")
        lines.append("  \"topic\" — one or two sentences on what the conversation is about (factor topic snapshots across all windows).")
        lines.append("  \"overallMood\" — one paragraph on the session's overall emotional tone — describe the arc, not just the trailing window.")
        lines.append("  \"perSpeaker\" — array, one entry per speaker id in this list: \(speakerList).")
        lines.append("Each perSpeaker entry has: { \"speakerID\": <id>, \"summary\": <one paragraph spanning the whole session>, \"dominantMood\": <one short phrase> }.")
        lines.append("The window intermediates include a \"modalityFlags\" array — surface meaningful acoustic↔text disagreements in the per-speaker write-ups when they recur or look notable. Don't enumerate every flag.")
        lines.append("When the speaker demographics block below lists a gender, use it as the canonical pronoun for that speaker throughout the summary — 'she/her' for female, 'he/him' for male, 'they/them' for child or when no gender is listed. (Moot for languages that drop pronouns, like Japanese.)")
        lines.append("Return ONLY valid JSON, no prose before or after.")
        lines.append(SummarizerLocale.responseLanguageInstruction)
        lines.append("/no_think")
        let demographics = SpeakerDemographicsDigest
            .build(from: allUtterances)
            .renderForPrompt(speakerIDs: allSpeakers, speakerNames: speakerNames)
        if !demographics.isEmpty {
            lines.append("")
            lines.append(demographics)
        }
        lines.append("")
        lines.append("Window summaries (each is a JSON object):")
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.withoutEscapingSlashes]
        for intermediate in intermediates {
            if let data = try? encoder.encode(intermediate),
               let json = String(data: data, encoding: .utf8) {
                lines.append(json)
            }
        }
        // Hybrid merge: also feed the top-N most informative
        // RAW utterances as quoted-detail context. Window
        // intermediates are already summaries; without raw
        // signal the merge's per-speaker write-ups are
        // "summarizing summaries" and lose verbatim quotes /
        // specific phrasing. Top-N selected by the same
        // speaker-balanced TF-IDF ranker heuristic mode uses,
        // so every speaker is represented.
        let supplementalCap = Self.deepMergeSupplementalUtterances
        let topIDs = Informativeness.topNBalancedBySpeaker(
            supplementalCap,
            utterances: allUtterances
        )
        let supplemental = allUtterances.filter { topIDs.contains($0.id) }
        if !supplemental.isEmpty {
            lines.append("")
            lines.append("Key raw utterances (\(supplemental.count) of \(allUtterances.count), chosen for distinctiveness — use these for verbatim quotes and specific detail in the per-speaker write-ups):")
            for u in supplemental {
                lines.append(compactLine(for: u, speakerNames: speakerNames))
            }
        }
        lines.append("")
        lines.append("---")
        lines.append("IMPORTANT: Follow the instructions above and produce exactly one final-summary JSON object with fields setting, topic, overallMood, perSpeaker. The FIRST character of your output MUST be `{`. Do NOT echo the window summaries or raw utterances above; do NOT add any prose.")
        // Restate the language directive LAST — recency wins on
        // long prompts (same reason the JSON-shape rule is
        // restated in this sandwich). The copy mid-prompt sits
        // thousands of tokens back, above the utterance list, and
        // the small quantized models drift to the transcript's
        // (or the prompt's own) language without this reminder —
        // Apple FM honored the early copy, these did not.
        lines.append(SummarizerLocale.responseLanguageInstruction)
        return lines.joined(separator: "\n")
    }

    /// Count of raw utterances to append to the deep-merge
    /// prompt as quoted-detail context (alongside the window
    /// intermediates). Half of `maxPromptUtterances` keeps
    /// the merge prompt's total token budget comfortably
    /// below Qwen3's 32k context even when combined with
    /// per-window intermediates.
    private static let deepMergeSupplementalUtterances = 50

    /// Qwen rows carry the FULL SER block (label + V/A/D +
    /// aP + tP). Qwen3-8B can actually use this signal to
    /// flag cross-modality disagreement in its per-speaker
    /// summaries; the verbose row format pays for itself.
    /// (Llama gets a stripped version — see
    /// `MLXLlamaSpec.compactLine`.)
    func compactLine(
        for u: UtteranceEstimate,
        speakerNames: [String: String]
    ) -> String {
        var fields: [String] = []
        fields.append("speaker=\(u.speakerID)")
        if let name = speakerNames[u.speakerID], !name.isEmpty {
            fields.append("name=\(name)")
        }
        fields.append(String(format: "t=%.1fs", u.start))
        if let label = u.fusedTopLabel { fields.append("label=\(label)") }
        if let v = u.fusedValence { fields.append(String(format: "V=%.2f", v)) }
        if let a = u.fusedArousal { fields.append(String(format: "A=%.2f", a)) }
        if let d = u.fusedDominance { fields.append(String(format: "D=%.2f", d)) }
        if let acoustic = u.acousticCategorical {
            fields.append("aP={\(MLXLLMRendering.acoustic(acoustic))}")
        }
        if let plutchik = u.plutchik {
            fields.append("tP={\(MLXLLMRendering.plutchik(plutchik))}")
        }
        fields.append("text=\"\(MLXLLMRendering.escapedTranscript(u.transcript))\"")
        return "- " + fields.joined(separator: " ")
    }
}
