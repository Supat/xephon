import Foundation
import Fusion
import SERAcoustic
import SERText
import XephonLogging
import MLX
import MLXLLM
import MLXLMCommon

/// Shared infrastructure behind `MLXQwenSummarizer` and
/// `MLXLlamaSummarizer`. Both actors are thin wrappers around
/// `MLXLLMSummarizerCore`'s static orchestration; the only
/// per-family differences are encapsulated in the
/// `MLXLLMSpec` an actor hands to the core.
///
/// **Why core + spec instead of one actor with `family`
/// branches.** The previous design had a single
/// `MLXQwenSummarizer` actor with `switch family` in every
/// prompt builder + a few `if family == .qwen` flags in
/// inference params. The name was misleading (Llama also
/// ran through it) and adding a third family would have
/// doubled the per-method branch surface. The split lifts the
/// per-family pieces into their own spec types so each
/// actor's file reads as "the Qwen path" or "the Llama path"
/// without family-conditional logic in the orchestration.

// MARK: - Actor type unification

/// Storage type for the coordinator's resident MLX summarizer.
/// Both `MLXQwenSummarizer` and `MLXLlamaSummarizer` conform —
/// each is a thin actor wrapping a per-family
/// `MLXLLMSpec`. The protocol surfaces just the lifecycle bits
/// (`load`, `unload`) the coordinator drives directly; the
/// `summarize` call comes through `SessionSummarizer`.
public protocol MLXLLMSummarizerActor: SessionSummarizer {
    func load() async throws
    func unload() async
}

// MARK: - Spec contract

/// Per-family behavior the shared orchestration needs.
/// Concrete types: `MLXQwenSpec` and `MLXLlamaSpec`. Sendable
/// because instances are handed across actor boundaries from
/// each per-family actor into the static orchestration
/// functions below.
internal protocol MLXLLMSpec: Sendable {
    var family: LLMModelFamily { get }

    /// Cap on prompt utterances for single-pass modes
    /// (`.trailing` / `.heuristic`). When the session exceeds
    /// this, the caller's selection strategy (trailing /
    /// heuristic top-N) decides which window to keep.
    var maxPromptUtterances: Int { get }

    /// Per-window utterance count for `.deep` mode's
    /// map-reduce.
    var deepWindowSize: Int { get }

    /// Output-token cap for per-window intermediate summaries
    /// in `.deep` mode.
    var deepWindowOutputTokens: Int { get }

    /// Output-token cap for trailing / heuristic / merge passes
    /// (i.e. anything that emits a full `SessionSummary`).
    var maxOutputTokens: Int { get }

    /// Stop tokens to pass into
    /// `ModelConfiguration.extraEOSTokens`. MLX-LM only
    /// honors a single `eos_token` from
    /// `tokenizer_config.json`, but families with multiple
    /// declared stop tokens (Llama 3.1 in particular) need
    /// the rest enumerated here so generation stops cleanly
    /// instead of running to `maxOutputTokens`.
    var extraEOSTokens: Set<String> { get }

    /// Family-specific repetition penalty for the generator,
    /// or `nil` to use the library default. Llama's Japanese
    /// fine-tune (Swallow) gets 1.05 to break out of pattern-
    /// echo loops it falls into on English-language JSON
    /// prompts; Qwen3 doesn't need it.
    var repetitionPenalty: Float? { get }

    /// Single-pass prompt: the chat-style message body the
    /// model receives for `.trailing` and `.heuristic` modes.
    func buildPrompt(
        utterances: [UtteranceEstimate],
        speakerNames: [String: String],
        truncatedFromTotal: Int?,
        selection: MLXLLMSelection
    ) -> String

    /// Per-window prompt for `.deep` mode. Asks for a compact
    /// intermediate JSON (NOT a full `SessionSummary`) that
    /// `buildDeepMergePrompt` will fuse.
    func buildDeepWindowPrompt(
        utterances: [UtteranceEstimate],
        speakerNames: [String: String],
        windowIndex: Int,
        totalWindows: Int
    ) -> String

    /// Merge prompt for `.deep` mode. Output schema MUST be
    /// the canonical `SessionSummary` JSON so the same parser
    /// works for trailing / heuristic / merge.
    func buildDeepMergePrompt(
        intermediates: [MLXLLMDeepWindowIntermediate],
        allUtterances: [UtteranceEstimate],
        speakerNames: [String: String]
    ) -> String

    /// One per-utterance line in whichever compact format the
    /// family expects. Qwen keeps the full SER block (label +
    /// V/A/D + aP + tP); Llama strips to speaker / time /
    /// text only (see `MLXLlamaSpec.compactLine` for the
    /// rationale).
    func compactLine(
        for utterance: UtteranceEstimate,
        speakerNames: [String: String]
    ) -> String
}

extension MLXLLMSpec {
    /// Cap on prompt utterances for `.meeting` mode. Meeting rows
    /// are TEXT-ONLY (speaker label + transcript — no fused label,
    /// V/A/D, acoustic/Plutchik vectors, or demographics block), so
    /// each row costs a fraction of an affect row. That freed token
    /// budget buys a much larger one-pass cap: ~250 vs the affect
    /// modes' `maxPromptUtterances` (~100). Sessions AT or under
    /// the cap run single-pass; longer ones go through
    /// `summarizeMeetingDeep`'s map-reduce so every utterance is
    /// read (the previous TF-IDF sampling systematically dropped
    /// the short decisive rows — agreements, retractions — that
    /// minutes exist to capture). Don't raise the cap instead:
    /// past ~10k prompt tokens single-pass coverage degrades.
    var meetingMaxPromptUtterances: Int { 250 }

    /// Per-window utterance count for the meeting map-reduce.
    /// Smaller than the single-pass cap so each window's prompt
    /// stays well inside the instruction-following sweet spot
    /// (~6k tokens of text-only rows) and the per-window JSON
    /// stays focused. Shared by both families — meeting rows are
    /// the same text-only shape for both.
    var meetingDeepWindowSize: Int { 150 }
}

// MARK: - Selection strategy for single-pass modes

/// How to pick which utterances feed a single-pass prompt
/// when the session exceeds `maxPromptUtterances`.
internal enum MLXLLMSelection {
    /// Most-recent N utterances (`utterances.suffix(N)`).
    /// Default for `.trailing` mode.
    case trailing
    /// Top-N by `Informativeness.topNBalancedBySpeaker`,
    /// restored to chronological order at the call site.
    /// Default for `.heuristic` mode.
    case heuristicTopN
}

// MARK: - Per-window intermediate (deep mode)

/// One window's compact summary. In-memory only — never
/// persisted, never crosses the public boundary. `modalityFlags`
/// is optional so a recovered (truncated) intermediate that
/// omitted the trailing field still decodes, and so families
/// (Llama) that omit the field from their per-row data can
/// drop it from the schema without breaking the merge pass.
internal struct MLXLLMDeepWindowIntermediate: Codable, Sendable {
    let windowIndex: Int
    let timeStart: Double
    let timeEnd: Double
    let topicSnapshot: String
    let moodSnapshot: String
    let perSpeaker: [PerSpeakerNote]
    let modalityFlags: [String]?

    struct PerSpeakerNote: Codable, Sendable {
        let speakerID: String
        let notes: String
        let dominantMood: String
    }
}

// MARK: - Orchestration

internal enum MLXLLMSummarizerCore {
    /// Top-level dispatcher. Receives the loaded
    /// `ModelContainer` from the actor and the actor's
    /// `spec`; returns the structured `SessionSummary`.
    static func summarize(
        container: ModelContainer,
        modelIdentifier: String,
        utterances: [UtteranceEstimate],
        speakerNames: [String: String],
        mode: SummarizeMode,
        boostedUtteranceIDs: Set<UUID>,
        glossaryTerms: [String],
        spec: any MLXLLMSpec
    ) async throws -> SessionSummary {
        guard !utterances.isEmpty else {
            return SessionSummary(
                inferredSetting: nil,
                topic: "",
                overallMood: "",
                perSpeaker: [],
                model: modelIdentifier,
                generatedAt: Date()
            )
        }
        switch mode {
        case .trailing:
            return try await summarizeSinglePass(
                container: container,
                modelIdentifier: modelIdentifier,
                utterances: utterances,
                speakerNames: speakerNames,
                selection: .trailing,
                mode: .trailing,
                boostedUtteranceIDs: boostedUtteranceIDs,
                spec: spec
            )
        case .heuristic:
            return try await summarizeSinglePass(
                container: container,
                modelIdentifier: modelIdentifier,
                utterances: utterances,
                speakerNames: speakerNames,
                selection: .heuristicTopN,
                mode: .heuristic,
                boostedUtteranceIDs: boostedUtteranceIDs,
                spec: spec
            )
        case .deep:
            return try await summarizeDeep(
                container: container,
                modelIdentifier: modelIdentifier,
                utterances: utterances,
                speakerNames: speakerNames,
                spec: spec
            )
        case .meeting:
            // The trusted baseline schema (no evidence, TF-IDF
            // sampling over the cap) — see SummarizeMode.meeting
            // vs .meetingExperimental.
            return try await summarizeMeetingClassic(
                container: container,
                modelIdentifier: modelIdentifier,
                utterances: utterances,
                speakerNames: speakerNames,
                boostedUtteranceIDs: boostedUtteranceIDs,
                spec: spec
            )
        case .meetingExperimental:
            return try await summarizeMeeting(
                container: container,
                modelIdentifier: modelIdentifier,
                utterances: utterances,
                speakerNames: speakerNames,
                boostedUtteranceIDs: boostedUtteranceIDs,
                glossaryTerms: glossaryTerms,
                spec: spec
            )
        case .all:
            // `.all` is the LM-Studio-only "no cap, no chunking"
            // mode — on-device backends can't fit it in context
            // and would OOM or 4096-token-cap mid-emit. Demote
            // to `.trailing` (best-effort honor per the
            // SessionSummarizer protocol contract) and log the
            // demotion so a user who switches backend after
            // picking `.all` understands why the output looks
            // truncated.
            AppLog.app.info(
                "MLX summarizer (\(spec.family.rawValue, privacy: .public)): .all unsupported on-device → demoted to .trailing"
            )
            return try await summarizeSinglePass(
                container: container,
                modelIdentifier: modelIdentifier,
                utterances: utterances,
                speakerNames: speakerNames,
                selection: .trailing,
                mode: .trailing,
                boostedUtteranceIDs: boostedUtteranceIDs,
                spec: spec
            )
        }
    }

    // MARK: Single-pass (fast / heuristic)

    static func summarizeSinglePass(
        container: ModelContainer,
        modelIdentifier: String,
        utterances: [UtteranceEstimate],
        speakerNames: [String: String],
        selection: MLXLLMSelection,
        mode: SummarizeMode,
        boostedUtteranceIDs: Set<UUID>,
        spec: any MLXLLMSpec
    ) async throws -> SessionSummary {
        let promptUtterances: [UtteranceEstimate]
        let truncatedFrom: Int?
        if utterances.count > spec.maxPromptUtterances {
            switch selection {
            case .trailing:
                promptUtterances = Array(utterances.suffix(spec.maxPromptUtterances))
            case .heuristicTopN:
                // `topNBalancedBySpeaker` guarantees each
                // speaker at least one slot (when the budget
                // allows) and proportions the rest by total
                // informativeness, with `boostedUtteranceIDs`
                // (caller-flagged keyword hits) getting a 4×
                // boost. Returned IDs are unordered; restore
                // chronology via `filter` so the LLM sees the
                // rows in time order.
                let topIDs = Informativeness.topNBalancedBySpeaker(
                    spec.maxPromptUtterances,
                    utterances: utterances,
                    boostedIDs: boostedUtteranceIDs
                )
                promptUtterances = utterances.filter { topIDs.contains($0.id) }
            }
            truncatedFrom = utterances.count
        } else {
            promptUtterances = utterances
            truncatedFrom = nil
        }
        let prompt = spec.buildPrompt(
            utterances: promptUtterances,
            speakerNames: speakerNames,
            truncatedFromTotal: truncatedFrom,
            selection: selection
        )
        if let truncatedFrom {
            let kind: String
            switch selection {
            case .trailing:      kind = "trailing"
            case .heuristicTopN: kind = "heuristic top-N"
            }
            AppLog.app.info(
                "MLX summarizer (\(spec.family.rawValue, privacy: .public)) selecting \(promptUtterances.count, privacy: .public) of \(truncatedFrom, privacy: .public) utterances (\(kind, privacy: .public))"
            )
        }
        AppLog.app.info(
            "MLX summarizer (\(spec.family.rawValue, privacy: .public)) summarizing \(promptUtterances.count, privacy: .public) utterances (prompt \(prompt.count, privacy: .public) chars)"
        )
        let raw = try await runInference(
            container: container,
            prompt: prompt,
            maxTokens: spec.maxOutputTokens,
            repetitionPenalty: spec.repetitionPenalty,
            label: "MLX[\(spec.family.rawValue)]"
        )
        if Task.isCancelled { throw CancellationError() }
        let outputPreview = String(raw.prefix(500))
        AppLog.app.info(
            "MLX summarizer (\(spec.family.rawValue, privacy: .public)) raw output: \(raw.count, privacy: .public) chars, preview: \(outputPreview, privacy: .public)"
        )
        return try parse(
            raw: raw,
            speakerNames: speakerNames,
            modelIdentifier: modelIdentifier,
            mode: mode,
            expectedSpeakerIDs: promptUtterances.orderedSpeakerIDs
        )
    }

    // MARK: Meeting (single-pass, content-only)

    /// Meeting-minutes pass. Same speaker-balanced heuristic
    /// selection as `.heuristic` (restored to chronological
    /// order) but with the larger `meetingMaxPromptUtterances`
    /// cap — meeting rows drop all emotion so far more of the
    /// conversation fits one pass. Builds text-only rows, runs a
    /// single inference, and parses the topic / topics /
    /// per-speaker-talking-points schema. Single pass only: no
    /// map-reduce variant (see the `meetingMaxPromptUtterances`
    /// ponytail note for the upgrade path).
    static func summarizeMeeting(
        container: ModelContainer,
        modelIdentifier: String,
        utterances: [UtteranceEstimate],
        speakerNames: [String: String],
        boostedUtteranceIDs: Set<UUID>,
        glossaryTerms: [String],
        spec: any MLXLLMSpec
    ) async throws -> SessionSummary {
        // Over the single-pass cap → map-reduce so EVERY utterance
        // is read. The old TF-IDF sampling dropped exactly the
        // short decisive rows (agreements, retractions) minutes
        // exist to capture; `boostedUtteranceIDs` is unused here
        // now for the same reason `.deep` ignores it — no
        // selection happens, so there's nothing to boost.
        if utterances.count > spec.meetingMaxPromptUtterances {
            return try await summarizeMeetingDeep(
                container: container,
                modelIdentifier: modelIdentifier,
                utterances: utterances,
                speakerNames: speakerNames,
                glossaryTerms: glossaryTerms,
                spec: spec
            )
        }
        let prompt = buildMeetingPrompt(
            utterances: utterances,
            firstRowNumber: 1,
            speakerNames: speakerNames,
            glossaryTerms: glossaryTerms,
            windowContext: nil,
            family: spec.family
        )
        AppLog.app.info(
            "MLX meeting (\(spec.family.rawValue, privacy: .public)) summarizing \(utterances.count, privacy: .public) utterances (prompt \(prompt.count, privacy: .public) chars)"
        )
        let raw = try await runInference(
            container: container,
            prompt: prompt,
            maxTokens: spec.maxOutputTokens,
            repetitionPenalty: spec.repetitionPenalty,
            label: "MLX[\(spec.family.rawValue)] meeting"
        )
        if Task.isCancelled { throw CancellationError() }
        let outputPreview = String(raw.prefix(500))
        AppLog.app.info(
            "MLX meeting (\(spec.family.rawValue, privacy: .public)) raw output: \(raw.count, privacy: .public) chars, preview: \(outputPreview, privacy: .public)"
        )
        return try parseMeeting(
            raw: raw,
            rows: utterances,
            speakerNames: speakerNames,
            modelIdentifier: modelIdentifier,
            expectedSpeakerIDs: utterances.orderedSpeakerIDs
        )
    }

    // MARK: Meeting (classic baseline)

    static func summarizeMeetingClassic(
        container: ModelContainer,
        modelIdentifier: String,
        utterances: [UtteranceEstimate],
        speakerNames: [String: String],
        boostedUtteranceIDs: Set<UUID>,
        spec: any MLXLLMSpec
    ) async throws -> SessionSummary {
        let cap = spec.meetingMaxPromptUtterances
        let promptUtterances: [UtteranceEstimate]
        let truncatedFrom: Int?
        if utterances.count > cap {
            // Same selection policy as `.heuristic` — top-N by
            // speaker-balanced TF-IDF with the caller's keyword
            // boost — just a bigger budget. Restore chronology
            // via `filter` so the model sees rows in time order.
            let topIDs = Informativeness.topNBalancedBySpeaker(
                cap,
                utterances: utterances,
                boostedIDs: boostedUtteranceIDs
            )
            promptUtterances = utterances.filter { topIDs.contains($0.id) }
            truncatedFrom = utterances.count
        } else {
            promptUtterances = utterances
            truncatedFrom = nil
        }
        let prompt = buildMeetingPromptClassic(
            utterances: promptUtterances,
            speakerNames: speakerNames,
            truncatedFromTotal: truncatedFrom,
            family: spec.family
        )
        if let truncatedFrom {
            AppLog.app.info(
                "MLX meeting (\(spec.family.rawValue, privacy: .public)) selecting \(promptUtterances.count, privacy: .public) of \(truncatedFrom, privacy: .public) utterances (heuristic top-N)"
            )
        }
        AppLog.app.info(
            "MLX meeting (\(spec.family.rawValue, privacy: .public)) summarizing \(promptUtterances.count, privacy: .public) utterances (prompt \(prompt.count, privacy: .public) chars)"
        )
        let raw = try await runInference(
            container: container,
            prompt: prompt,
            maxTokens: spec.maxOutputTokens,
            repetitionPenalty: spec.repetitionPenalty,
            label: "MLX[\(spec.family.rawValue)] meeting"
        )
        if Task.isCancelled { throw CancellationError() }
        let outputPreview = String(raw.prefix(500))
        AppLog.app.info(
            "MLX meeting (\(spec.family.rawValue, privacy: .public)) raw output: \(raw.count, privacy: .public) chars, preview: \(outputPreview, privacy: .public)"
        )
        // Shared parser: the classic prompt requests no evidence,
        // so the wire's evidence fields decode nil and the summary
        // carries none — byte-compatible with the pre-experiment
        // output.
        return try parseMeeting(
            raw: raw,
            rows: promptUtterances,
            speakerNames: speakerNames,
            modelIdentifier: modelIdentifier,
            expectedSpeakerIDs: promptUtterances.orderedSpeakerIDs
        )
    }

    static func buildMeetingPromptClassic(
        utterances: [UtteranceEstimate],
        speakerNames: [String: String],
        truncatedFromTotal: Int?,
        family: LLMModelFamily
    ) -> String {
        let speakers = utterances.orderedSpeakerIDs
        let speakerList = speakers.joined(separator: ", ")
        var lines: [String] = []
        lines.reserveCapacity(utterances.count + 14)
        lines.append("You are a meeting-minutes analyst summarizing a multi-speaker conversation.")
        lines.append("Ignore emotion and tone entirely. Focus only on WHAT was discussed and WHO said it.")
        lines.append("Read every utterance below and produce a JSON object with exactly three fields:")
        lines.append("  \"topic\" — one or two sentences giving an overall overview of what the meeting was about.")
        lines.append("  \"topics\" — array of the main subjects discussed. Each entry: { \"title\": <short topic title>, \"raisedBy\": <the speaker who first raised it, or null if unclear>, \"positions\": [ { \"speaker\": <speaker>, \"stance\": <that speaker's opinion / position on this topic> } ] }. Maximize coverage: capture every distinct topic, and for each one record the differing positions the various speakers took.")
        lines.append("  \"perSpeaker\" — array, exactly one entry per speaker id in this list: \(speakerList). Each entry: { \"speakerID\": <id>, \"talkingPoints\": [ <short string>, ... ] } listing that speaker's main talking points / contributions. Emit this field LAST.")
        lines.append("Refer to each speaker by the display name given in their `name=` field when present, otherwise by their speaker id.")
        lines.append("Each row below carries only a speaker label and the transcript text — no emotion data is provided, and none is wanted in the output.")
        lines.append("Return ONLY valid JSON, no prose before or after.")
        lines.append(SummarizerLocale.responseLanguageInstruction)
        // Qwen3's "thinking" mode emits a `<think>…</think>`
        // block before the answer; `/no_think` disables it for
        // this turn. Literal text for Llama, so family-gated.
        if family == .qwen {
            lines.append("/no_think")
        }
        if let total = truncatedFromTotal {
            lines.append("")
            lines.append("NOTE: This conversation has \(total) utterances total; the \(utterances.count) most distinctive utterances (chosen by session-relative TF-IDF, NOT the most recent) are shown below in chronological order. Treat them as a representative sample of the whole meeting — gaps between rows are expected.")
        }
        lines.append("")
        lines.append("Utterances:")
        for u in utterances {
            lines.append(meetingCompactLineClassic(for: u, speakerNames: speakerNames))
        }
        lines.append("")
        lines.append("---")
        lines.append("IMPORTANT: Follow the instructions above and produce exactly one valid JSON object with fields topic, topics, perSpeaker (in that order, perSpeaker LAST). The FIRST character of your output MUST be `{`. Do NOT echo the utterance list above; do NOT add any prose.")
        return lines.joined(separator: "\n")
    }

    /// Un-numbered text-only row for the classic meeting prompt —
    /// the pre-experiment format, kept byte-identical because the
    /// classic mode's value is "exactly the output the user
    /// validated".
    static func meetingCompactLineClassic(
        for u: UtteranceEstimate,
        speakerNames: [String: String]
    ) -> String {
        var fields: [String] = []
        fields.append("speaker=\(u.speakerID)")
        if let name = speakerNames[u.speakerID], !name.isEmpty {
            fields.append("name=\(name)")
        }
        fields.append("text=\"\(MLXLLMRendering.escapedTranscript(u.transcript))\"")
        return "- " + fields.joined(separator: " ")
    }

    // MARK: Meeting map-reduce

    /// Map-reduce for meetings past the single-pass cap: fixed
    /// chronological windows → per-window PARTIAL minutes (same
    /// schema, globally-numbered evidence rows) → one merge pass
    /// that dedups topics and unions positions. Every utterance is
    /// read; evidence numbers stay global across the whole session
    /// so the final parse validates them against the full row list.
    static func summarizeMeetingDeep(
        container: ModelContainer,
        modelIdentifier: String,
        utterances: [UtteranceEstimate],
        speakerNames: [String: String],
        glossaryTerms: [String],
        spec: any MLXLLMSpec
    ) async throws -> SessionSummary {
        let windowSize = spec.meetingDeepWindowSize
        let chunks = stride(from: 0, to: utterances.count, by: windowSize).map {
            offset -> (firstRowNumber: Int, rows: [UtteranceEstimate]) in
            let end = min(offset + windowSize, utterances.count)
            return (firstRowNumber: offset + 1, rows: Array(utterances[offset..<end]))
        }
        AppLog.app.info(
            "MLX meeting-deep (\(spec.family.rawValue, privacy: .public)): \(utterances.count, privacy: .public) utterances → \(chunks.count, privacy: .public) windows"
        )
        var intermediates: [MeetingWire] = []
        intermediates.reserveCapacity(chunks.count)
        for (idx, chunk) in chunks.enumerated() {
            if Task.isCancelled { throw CancellationError() }
            let prompt = buildMeetingPrompt(
                utterances: chunk.rows,
                firstRowNumber: chunk.firstRowNumber,
                speakerNames: speakerNames,
                glossaryTerms: glossaryTerms,
                windowContext: (index: idx, total: chunks.count),
                family: spec.family
            )
            let raw = try await runInference(
                container: container,
                prompt: prompt,
                maxTokens: spec.maxOutputTokens,
                repetitionPenalty: spec.repetitionPenalty,
                label: "MLX[\(spec.family.rawValue)] meeting-deep[\(idx + 1)/\(chunks.count)]"
            )
            if Task.isCancelled { throw CancellationError() }
            // Tolerant per-window parse: a window that fails to
            // parse contributes nothing rather than sinking the
            // whole pass (mirrors parseDeepWindow's resilience).
            if let wire = decodeMeetingWire(raw: raw) {
                intermediates.append(wire)
            } else {
                AppLog.app.warning(
                    "MLX meeting-deep[\(idx + 1, privacy: .public)/\(chunks.count, privacy: .public)] window parse failed; skipping window"
                )
            }
        }
        guard !intermediates.isEmpty else {
            throw SummarizerError.decodeFailed(
                reason: "meeting map-reduce: every window failed to parse"
            )
        }
        if Task.isCancelled { throw CancellationError() }
        let mergePrompt = buildMeetingMergePrompt(
            intermediates: intermediates,
            allUtterances: utterances,
            speakerNames: speakerNames,
            glossaryTerms: glossaryTerms,
            family: spec.family
        )
        let raw = try await runInference(
            container: container,
            prompt: mergePrompt,
            maxTokens: spec.maxOutputTokens,
            repetitionPenalty: spec.repetitionPenalty,
            label: "MLX[\(spec.family.rawValue)] meeting-merge"
        )
        if Task.isCancelled { throw CancellationError() }
        return try parseMeeting(
            raw: raw,
            rows: utterances,
            speakerNames: speakerNames,
            modelIdentifier: modelIdentifier,
            expectedSpeakerIDs: utterances.orderedSpeakerIDs
        )
    }

    /// Shared meeting prompt. Identical for both families (the
    /// rows are text-only, so there's nothing family-specific to
    /// vary except Qwen3's `/no_think` directive); per the spec
    /// note, meeting specs "differ only in model + compactLine",
    /// and the compact line here is the shared text-only one.
    /// Output language is steered by
    /// `SummarizerLocale.responseLanguageInstruction`.
    static func buildMeetingPrompt(
        utterances: [UtteranceEstimate],
        firstRowNumber: Int,
        speakerNames: [String: String],
        glossaryTerms: [String],
        windowContext: (index: Int, total: Int)?,
        family: LLMModelFamily
    ) -> String {
        let speakers = utterances.orderedSpeakerIDs
        let speakerList = speakers.joined(separator: ", ")
        let lastRowNumber = firstRowNumber + utterances.count - 1
        var lines: [String] = []
        lines.reserveCapacity(utterances.count + 18)
        lines.append("You are a meeting-minutes analyst summarizing a multi-speaker conversation.")
        lines.append("Ignore emotion and tone entirely. Focus only on WHAT was discussed and WHO said it.")
        if let w = windowContext {
            lines.append("This is window \(w.index + 1) of \(w.total) — rows [\(firstRowNumber)]–[\(lastRowNumber)] of a longer meeting. Produce PARTIAL minutes covering ONLY these rows; a later pass merges the windows. A topic may continue from a previous window — record what THIS window shows about it.")
        }
        lines.append("Read every utterance below and produce a JSON object with exactly three fields:")
        lines.append("  \"topic\" — one or two SHORT sentences giving an overall overview of what \(windowContext == nil ? "the meeting" : "this part of the meeting") was about.")
        lines.append("  \"topics\" — array of the main subjects discussed. Each entry: { \"title\": <short topic title>, \"raisedBy\": <the speaker who first raised it, or null if unclear>, \"positions\": [ { \"speaker\": <speaker>, \"stance\": <that speaker's opinion / position on this topic, in ONE short sentence>, \"evidence\": <the SINGLE row number that best supports this stance> } ] }. Capture every distinct topic and the differing positions speakers took — one short sentence per stance, exactly ONE evidence number per position.")
        lines.append("  \"perSpeaker\" — array, exactly one entry per speaker id in this list: \(speakerList). Each entry: { \"speakerID\": <id>, \"talkingPoints\": [ <short string>, ... ] } listing that speaker's main talking points / contributions (3-6 phrases, each under 10 words). Emit this field LAST.")
        lines.append("Each utterance row below is numbered like [\(firstRowNumber)]. Every \"evidence\" value is ONE integer — the row number that best supports the stance — and must be a number that actually appears below (\(firstRowNumber)–\(lastRowNumber)). NEVER invent or guess numbers. A claim you cannot tie to a specific row does not belong in the minutes.")
        lines.append("Refer to each speaker by the display name given in their `name=` field when present, otherwise by their speaker id.")
        lines.append("Each row below carries only a speaker label and the transcript text — no emotion data is provided, and none is wanted in the output.")
        if !glossaryTerms.isEmpty {
            lines.append("Domain terms curated by the user: \(glossaryTerms.joined(separator: ", ")). The transcript is automatic speech recognition output — when a transcript word is a plausible mis-hearing (same or similar reading) of one of these terms, treat it as that term and use the term's correct spelling in the minutes.")
        }
        lines.append("Return ONLY valid JSON, no prose before or after.")
        lines.append(SummarizerLocale.responseLanguageInstruction)
        // Qwen3's "thinking" mode emits a `<think>…</think>`
        // block before the answer; `/no_think` disables it for
        // this turn. Literal text for Llama, so family-gated.
        if family == .qwen {
            lines.append("/no_think")
        }
        lines.append("")
        lines.append("Utterances:")
        for (offset, u) in utterances.enumerated() {
            lines.append(meetingCompactLine(
                index: firstRowNumber + offset,
                for: u,
                speakerNames: speakerNames
            ))
        }
        lines.append("")
        lines.append("---")
        lines.append("IMPORTANT: Follow the instructions above and produce exactly one valid JSON object with fields topic, topics, perSpeaker (in that order, perSpeaker LAST). The FIRST character of your output MUST be `{`. Do NOT echo the utterance list above; do NOT add any prose.")
        return lines.joined(separator: "\n")
    }

    /// Merge prompt for the meeting map-reduce: the per-window
    /// PARTIAL minutes (as JSON) go in; one final meeting JSON in
    /// the same schema comes out. Evidence row numbers are global
    /// across the session, so the merge must carry them through
    /// verbatim — `parseMeeting` validates them against the full
    /// row list afterwards, so an invented number gets stripped
    /// rather than surfacing.
    static func buildMeetingMergePrompt(
        intermediates: [MeetingWire],
        allUtterances: [UtteranceEstimate],
        speakerNames: [String: String],
        glossaryTerms: [String],
        family: LLMModelFamily
    ) -> String {
        let speakers = allUtterances.orderedSpeakerIDs
        let speakerList = speakers.joined(separator: ", ")
        let encoder = JSONEncoder()
        let windowJSONs: [String] = intermediates.enumerated().map { idx, wire in
            let data = (try? encoder.encode(wire)) ?? Data()
            return "Window \(idx + 1): " + (String(data: data, encoding: .utf8) ?? "{}")
        }
        var lines: [String] = []
        lines.append("You are a meeting-minutes analyst. Below are PARTIAL meeting minutes for \(intermediates.count) consecutive windows of one meeting (\(allUtterances.count) numbered utterance rows total). Merge them into ONE final minutes JSON.")
        lines.append("Rules:")
        lines.append("- Topics that are the same subject across windows MERGE into one entry: combine their positions (one entry per speaker per topic; merge a speaker's stances across windows into ONE short sentence, keeping the single most representative evidence row number).")
        lines.append("- Keep genuinely distinct topics separate. Preserve first-raised order.")
        lines.append("- Evidence numbers reference the full session's rows (1–\(allUtterances.count)). Copy them EXACTLY as given; never renumber, never invent.")
        lines.append("- \"perSpeaker\": exactly one entry per speaker id in this list: \(speakerList). Merge each speaker's talking points across windows, dropping near-duplicates. Emit this field LAST.")
        lines.append("- \"topic\": one or two sentences covering the WHOLE meeting.")
        if !glossaryTerms.isEmpty {
            lines.append("- Domain terms (use their correct spelling): \(glossaryTerms.joined(separator: ", ")).")
        }
        lines.append("Return ONLY valid JSON with fields topic, topics, perSpeaker — same schema as the windows.")
        lines.append(SummarizerLocale.responseLanguageInstruction)
        if family == .qwen {
            lines.append("/no_think")
        }
        lines.append("")
        lines.append(contentsOf: windowJSONs)
        lines.append("")
        lines.append("---")
        lines.append("IMPORTANT: produce exactly one valid JSON object with fields topic, topics, perSpeaker (perSpeaker LAST). The FIRST character of your output MUST be `{`. No prose.")
        return lines.joined(separator: "\n")
    }

    /// Text-only per-utterance row for meeting mode: speaker id,
    /// optional display name, transcript. Deliberately omits the
    /// fused label / V/A/D / aP / tP block the affect modes'
    /// `MLXLLMSpec.compactLine` carries — meeting mode ignores
    /// emotion, and dropping it is what funds the larger
    /// `meetingMaxPromptUtterances` cap.
    static func meetingCompactLine(
        index: Int,
        for u: UtteranceEstimate,
        speakerNames: [String: String]
    ) -> String {
        var fields: [String] = []
        fields.append("speaker=\(u.speakerID)")
        if let name = speakerNames[u.speakerID], !name.isEmpty {
            fields.append("name=\(name)")
        }
        fields.append("text=\"\(MLXLLMRendering.escapedTranscript(u.transcript))\"")
        // The [n] prefix is the row's citation number — evidence
        // arrays in the output refer back to it.
        return "[\(index)] " + fields.joined(separator: " ")
    }

    // MARK: Deep (map-reduce)

    static func summarizeDeep(
        container: ModelContainer,
        modelIdentifier: String,
        utterances: [UtteranceEstimate],
        speakerNames: [String: String],
        spec: any MLXLLMSpec
    ) async throws -> SessionSummary {
        let chunks = stride(from: 0, to: utterances.count, by: spec.deepWindowSize).map {
            offset -> [UtteranceEstimate] in
            let end = min(offset + spec.deepWindowSize, utterances.count)
            return Array(utterances[offset..<end])
        }
        AppLog.app.info(
            "MLX deep (\(spec.family.rawValue, privacy: .public)): \(utterances.count, privacy: .public) utterances → \(chunks.count, privacy: .public) windows"
        )
        // Short-circuit single-window sessions back to the
        // single-pass path. Stamp as `.deep` to honor the
        // user's intent in the sheet footer even though we
        // selected trailing.
        if chunks.count <= 1 {
            return try await summarizeSinglePass(
                container: container,
                modelIdentifier: modelIdentifier,
                utterances: utterances,
                speakerNames: speakerNames,
                selection: .trailing,
                mode: .deep,
                boostedUtteranceIDs: [],
                spec: spec
            )
        }
        var intermediates: [MLXLLMDeepWindowIntermediate] = []
        intermediates.reserveCapacity(chunks.count)
        for (idx, chunk) in chunks.enumerated() {
            if Task.isCancelled { throw CancellationError() }
            let intermediate = try await runDeepWindow(
                container: container,
                chunk: chunk,
                speakerNames: speakerNames,
                windowIndex: idx,
                totalWindows: chunks.count,
                spec: spec
            )
            intermediates.append(intermediate)
            AppLog.app.info(
                "MLX deep[\(idx + 1, privacy: .public)/\(chunks.count, privacy: .public)] done (\(intermediate.perSpeaker.count, privacy: .public) speakers)"
            )
        }
        if Task.isCancelled { throw CancellationError() }
        return try await runDeepMerge(
            container: container,
            modelIdentifier: modelIdentifier,
            intermediates: intermediates,
            allUtterances: utterances,
            speakerNames: speakerNames,
            spec: spec
        )
    }

    static func runDeepWindow(
        container: ModelContainer,
        chunk: [UtteranceEstimate],
        speakerNames: [String: String],
        windowIndex: Int,
        totalWindows: Int,
        spec: any MLXLLMSpec
    ) async throws -> MLXLLMDeepWindowIntermediate {
        let prompt = spec.buildDeepWindowPrompt(
            utterances: chunk,
            speakerNames: speakerNames,
            windowIndex: windowIndex,
            totalWindows: totalWindows
        )
        let raw = try await runInference(
            container: container,
            prompt: prompt,
            maxTokens: spec.deepWindowOutputTokens,
            repetitionPenalty: spec.repetitionPenalty,
            label: "MLX[\(spec.family.rawValue)] deep[\(windowIndex + 1)/\(totalWindows)]"
        )
        if Task.isCancelled { throw CancellationError() }
        let windowPreview = String(raw.prefix(300))
        AppLog.app.info(
            "MLX deep[\(windowIndex + 1, privacy: .public)/\(totalWindows, privacy: .public)] raw: \(raw.count, privacy: .public) chars, preview: \(windowPreview, privacy: .public)"
        )
        return parseDeepWindow(raw: raw, windowIndex: windowIndex, chunk: chunk)
    }

    static func runDeepMerge(
        container: ModelContainer,
        modelIdentifier: String,
        intermediates: [MLXLLMDeepWindowIntermediate],
        allUtterances: [UtteranceEstimate],
        speakerNames: [String: String],
        spec: any MLXLLMSpec
    ) async throws -> SessionSummary {
        let prompt = spec.buildDeepMergePrompt(
            intermediates: intermediates,
            allUtterances: allUtterances,
            speakerNames: speakerNames
        )
        let raw = try await runInference(
            container: container,
            prompt: prompt,
            maxTokens: spec.maxOutputTokens,
            repetitionPenalty: spec.repetitionPenalty,
            label: "MLX[\(spec.family.rawValue)] deep[merge]"
        )
        if Task.isCancelled { throw CancellationError() }
        let mergePreview = String(raw.prefix(500))
        AppLog.app.info(
            "MLX deep merge raw output: \(raw.count, privacy: .public) chars, preview: \(mergePreview, privacy: .public)"
        )
        return try parse(
            raw: raw,
            speakerNames: speakerNames,
            modelIdentifier: modelIdentifier,
            mode: .deep,
            expectedSpeakerIDs: allUtterances.orderedSpeakerIDs
        )
    }

    // MARK: Inference helper

    /// Single inference call against the loaded container.
    /// Centralizes the `GenerateParameters` / timing-log /
    /// cancellation-hook dance so every prompt path looks
    /// identical at the call site. `label` is just for log
    /// readability.
    static func runInference(
        container: ModelContainer,
        prompt: String,
        maxTokens: Int,
        repetitionPenalty: Float?,
        label: String
    ) async throws -> String {
        do {
            return try await container.perform { context -> String in
                let userInput = UserInput(prompt: prompt)
                let lmInput = try await context.processor.prepare(input: userInput)
                let preparedMsg = String(
                    format: "%@ prepared input: %d prompt tokens",
                    label, lmInput.text.tokens.size
                )
                AppLog.app.info("\(preparedMsg, privacy: .public)")
                var parameters = GenerateParameters(
                    maxTokens: maxTokens,
                    // Deterministic for reproducibility — the
                    // summary is a derived artifact, not
                    // creative writing.
                    temperature: 0.2
                )
                if let penalty = repetitionPenalty {
                    parameters.repetitionPenalty = penalty
                }
                // Default prefill step is 512 tokens — packing
                // hundreds of tokens into a single Metal
                // command buffer was tripping the GPU watchdog
                // (`kIOGPUCommandBufferCallbackErrorTimeout` →
                // SIGABRT) on real iPad hardware. 128 sits
                // comfortably inside the watchdog window and
                // roughly halves prefill wall time on long
                // sessions. Drop back to 64 if the timeout
                // SIGABRT re-appears on any hardware revision.
                parameters.prefillStepSize = 128
                let startTime = Date()
                // Prefill OUTSIDE generate, with a cancellation
                // check between chunks — the library's own prefill
                // loop has none, so backgrounding mid-prefill kept
                // submitting GPU work after iOS revoked access and
                // aborted the process (see MLXCancellablePrefill).
                let (remaining, iterator) = try MLXCancellablePrefill.primedIterator(
                    input: lmInput,
                    context: context,
                    parameters: parameters
                )
                var firstTokenTime: Date? = nil
                let result = MLXLMCommon.generate(
                    input: remaining,
                    context: context,
                    iterator: iterator,
                    didGenerate: { (tokens: [Int]) -> GenerateDisposition in
                        if firstTokenTime == nil {
                            firstTokenTime = Date()
                            let prefillSec = firstTokenTime!
                                .timeIntervalSince(startTime)
                            let msg = String(
                                format: "%@ prefill done in %.2f s; starting decode",
                                label, prefillSec
                            )
                            AppLog.app.info("\(msg, privacy: .public)")
                        } else if tokens.count.isMultiple(of: 64) {
                            let elapsed = Date()
                                .timeIntervalSince(firstTokenTime!)
                            let tps = elapsed > 0
                                ? Double(tokens.count) / elapsed
                                : 0
                            let msg = String(
                                format: "%@ decode: %d tokens in %.1f s (%.1f t/s)",
                                label, tokens.count, elapsed, tps
                            )
                            AppLog.app.info("\(msg, privacy: .public)")
                        }
                        return Task.isCancelled ? .stop : .more
                    }
                )
                let totalSec = Date().timeIntervalSince(startTime)
                let finishMsg = String(
                    format: "%@ generate finished: %d tokens in %.1f s total",
                    label, result.tokens.count, totalSec
                )
                AppLog.app.info("\(finishMsg, privacy: .public)")
                if result.tokens.count >= maxTokens {
                    AppLog.app.warning(
                        "\(label, privacy: .public) output hit the \(maxTokens, privacy: .public)-token cap — JSON is likely truncated; parser will attempt recovery"
                    )
                }
                return result.output
            }
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            AppLog.app.error(
                "\(label, privacy: .public) generate failed: \(String(describing: error), privacy: .public)"
            )
            throw SummarizerError.inferenceFailed(
                reason: String(describing: error)
            )
        }
    }

    // MARK: Parser (final summary)

    /// Decode the LLM's JSON output into a `SessionSummary`.
    /// Tolerant of three failure modes the models exhibit in
    /// the field:
    /// 1. `<think>…</think>` reasoning blocks (Qwen3).
    /// 2. ```` ```json …``` ```` code fences.
    /// 3. Truncated output (hit `maxOutputTokens` mid-
    ///    `perSpeaker` array — `recoverTruncatedSummary`
    ///    salvages every complete entry).
    /// 4. `perSpeaker` emitted as a dict keyed by speakerID
    ///    instead of an array (common Llama variance —
    ///    `normalizePerSpeakerShape` rewrites the dict).
    static func parse(
        raw: String,
        speakerNames: [String: String],
        modelIdentifier: String,
        mode: SummarizeMode,
        expectedSpeakerIDs: [String]
    ) throws -> SessionSummary {
        let dethought = stripThinkBlocks(raw)
        let stripped = stripCodeFence(dethought)
        guard let braceStart = stripped.firstIndex(of: "{") else {
            throw SummarizerError.decodeFailed(reason: "no JSON object found")
        }
        struct Wire: Decodable {
            struct PerSpeaker: Decodable {
                let speakerID: String
                let summary: String
                let dominantMood: String
            }
            let setting: String?
            let topic: String
            let overallMood: String
            let perSpeaker: [PerSpeaker]
        }
        let strict: String? = {
            guard let braceEnd = stripped.lastIndex(of: "}") else { return nil }
            return String(stripped[braceStart...braceEnd])
        }()
        let decoded: Wire
        if let strict, let data = strict.data(using: .utf8),
           let ok = try? JSONDecoder().decode(Wire.self, from: data) {
            decoded = ok
        } else if let strict,
                  let data = strict.data(using: .utf8),
                  let normalized = normalizePerSpeakerShape(in: data),
                  let ok = try? JSONDecoder().decode(Wire.self, from: normalized) {
            AppLog.app.info(
                "MLX summarizer normalized perSpeaker dict → array (\(normalized.count, privacy: .public) bytes)"
            )
            decoded = ok
        } else {
            guard let recovered = recoverTruncatedSummary(
                stripped: String(stripped[braceStart...])
            ) else {
                throw SummarizerError.decodeFailed(reason: "no JSON object found")
            }
            guard let data = recovered.data(using: .utf8) else {
                throw SummarizerError.decodeFailed(reason: "non-utf8 output")
            }
            AppLog.app.info(
                "MLX summarizer recovered truncated JSON (\(data.count, privacy: .public) bytes)"
            )
            do {
                decoded = try JSONDecoder().decode(Wire.self, from: data)
            } catch {
                throw SummarizerError.decodeFailed(
                    reason: String(describing: error)
                )
            }
        }
        let perSpeaker = decoded.perSpeaker.map { entry in
            SessionSummary.SpeakerSummary(
                speakerID: entry.speakerID,
                speakerName: speakerNames[entry.speakerID],
                summary: entry.summary,
                dominantMood: entry.dominantMood
            )
        }
        // Post-hoc fill: Llama-Swallow (and occasionally Qwen
        // on edge cases) silently drops marginal-contribution
        // speakers from `perSpeaker`. Synthesize a placeholder
        // entry for any speaker in the input that's missing
        // from the output, so downstream UI sees a roster
        // that matches what it sent in.
        let filled = SessionSummary.fillMissingPerSpeaker(
            perSpeaker,
            expectedSpeakerIDs: expectedSpeakerIDs,
            speakerNames: speakerNames
        )
        if filled.count > perSpeaker.count {
            AppLog.app.info(
                "MLX summarizer filled \(filled.count - perSpeaker.count, privacy: .public) missing per-speaker entries (\(perSpeaker.count, privacy: .public) emitted, \(expectedSpeakerIDs.count, privacy: .public) expected)"
            )
        }
        return SessionSummary(
            inferredSetting: decoded.setting?
                .trimmingCharacters(in: .whitespacesAndNewlines),
            topic: decoded.topic,
            overallMood: decoded.overallMood,
            perSpeaker: filled,
            model: modelIdentifier,
            generatedAt: Date(),
            mode: mode
        )
    }

    // MARK: Parser (meeting)

    /// Decode the meeting-mode JSON into a `SessionSummary`. Same
    /// tolerance posture as `parse`: strip `<think>` blocks +
    /// code fences, strict-decode first, then fall back to
    /// `recoverTruncatedSummary` (the schema emits `perSpeaker`
    /// LAST, so the same forward-scan salvage that closes a
    /// truncated `perSpeaker` array works here too). All wire
    /// fields are optional so a partial object still decodes —
    /// missing topics / talking points become empty rather than a
    /// hard failure. The result drops affect entirely:
    /// `overallMood` and every `dominantMood` are "", `mode` is
    /// `.meeting`, and the structured topics + per-speaker talking
    /// points carry the content.
    /// Wire schema for the meeting JSON — shared by the single-pass
    /// output, the map-reduce window intermediates (re-encoded into
    /// the merge prompt, hence Codable not just Decodable), and the
    /// merge output. `evidence` arrays carry prompt ROW NUMBERS;
    /// `parseMeeting` validates and maps them to utterance IDs.
    internal struct MeetingWire: Codable, Sendable {
        /// `evidence` accepts either a single integer (the current
        /// prompt contract — a scalar is structurally spray-proof;
        /// array-valued evidence let Qwen enumerate 20+ rows per
        /// claim and blow the whole 4096-token output budget on
        /// numbers, observed on-device) or an array (window
        /// intermediates re-encoded into the merge prompt, and
        /// tolerance for loose model output). Encodes as an array.
        struct Evidence: Codable, Sendable {
            let numbers: [Int]
            init(numbers: [Int]) { self.numbers = numbers }
            init(from decoder: any Decoder) throws {
                let c = try decoder.singleValueContainer()
                if let single = try? c.decode(Int.self) {
                    numbers = [single]
                } else {
                    numbers = (try? c.decode([Int].self)) ?? []
                }
            }
            func encode(to encoder: any Encoder) throws {
                var c = encoder.singleValueContainer()
                try c.encode(numbers)
            }
        }
        struct Position: Codable, Sendable {
            let speaker: String?
            let stance: String?
            let evidence: Evidence?
        }
        struct Topic: Codable, Sendable {
            let title: String?
            let raisedBy: String?
            let evidence: Evidence?
            let positions: [Position]?
        }
        struct PerSpeaker: Codable, Sendable {
            let speakerID: String
            let talkingPoints: [String]?
        }
        let topic: String?
        let topics: [Topic]?
        let perSpeaker: [PerSpeaker]?
    }

    /// Tolerant MeetingWire decode: think-block strip, code-fence
    /// strip, strict pass, then truncation recovery. Nil on failure
    /// (map-reduce windows skip; single-pass/merge callers throw).
    static func decodeMeetingWire(raw: String) -> MeetingWire? {
        let dethought = stripThinkBlocks(raw)
        let stripped = stripCodeFence(dethought)
        guard let braceStart = stripped.firstIndex(of: "{") else { return nil }
        if let braceEnd = stripped.lastIndex(of: "}"),
           let data = String(stripped[braceStart...braceEnd]).data(using: .utf8),
           let ok = try? JSONDecoder().decode(MeetingWire.self, from: data) {
            return ok
        }
        if let recovered = recoverTruncatedSummary(
               stripped: String(stripped[braceStart...])
           ),
           let data = recovered.data(using: .utf8),
           let ok = try? JSONDecoder().decode(MeetingWire.self, from: data) {
            AppLog.app.info(
                "MLX meeting recovered truncated JSON (\(data.count, privacy: .public) bytes)"
            )
            return ok
        }
        // Output that hit the token cap mid-`topics` never emitted
        // the `perSpeaker` key, so the summary recovery above can't
        // help — salvage the complete topic entries instead
        // (observed on-device: 171-row meeting, 4096-token cap hit
        // while still in the first topics entries).
        if let recovered = recoverTruncatedMeeting(
               stripped: String(stripped[braceStart...])
           ),
           let data = recovered.data(using: .utf8),
           let ok = try? JSONDecoder().decode(MeetingWire.self, from: data) {
            AppLog.app.info(
                "MLX meeting salvaged truncated topics prefix (\(data.count, privacy: .public) bytes)"
            )
            return ok
        }
        return nil
    }

    /// Meeting variant of `recoverTruncatedSummary`: salvage the
    /// COMPLETE entries of the `topics` array from output truncated
    /// mid-array, drop the partial trailing entry, and close the
    /// object with an empty `perSpeaker` — `fillMissingPerSpeaker`
    /// re-adds every expected speaker as a placeholder, so the user
    /// gets the salvaged topics instead of a parse error. Returns
    /// nil when the topics array closed properly (truncation was
    /// later — the summary recovery's territory) or no complete
    /// entry exists.
    static func recoverTruncatedMeeting(stripped: String) -> String? {
        guard let topicsKey = stripped.range(of: "\"topics\"") else { return nil }
        guard let arrayOpen = stripped.range(
            of: "[",
            range: topicsKey.upperBound..<stripped.endIndex
        ) else { return nil }
        var depth = 0
        var inString = false
        var escape = false
        var lastCompleteEntryEnd: String.Index? = nil
        var i = arrayOpen.upperBound
        while i < stripped.endIndex {
            let ch = stripped[i]
            if escape {
                escape = false
            } else if inString {
                if ch == "\\" {
                    escape = true
                } else if ch == "\"" {
                    inString = false
                }
            } else {
                switch ch {
                case "\"":
                    inString = true
                case "{", "[":
                    depth += 1
                case "}":
                    depth -= 1
                    if depth == 0 {
                        lastCompleteEntryEnd = stripped.index(after: i)
                    }
                case "]":
                    depth -= 1
                    if depth < 0 {
                        // The topics array itself closed — output
                        // wasn't truncated here.
                        return nil
                    }
                default:
                    break
                }
            }
            i = stripped.index(after: i)
        }
        guard let end = lastCompleteEntryEnd else { return nil }
        return String(stripped[stripped.startIndex..<end]) + "],\"perSpeaker\":[]}"
    }

    static func parseMeeting(
        raw: String,
        rows: [UtteranceEstimate],
        speakerNames: [String: String],
        modelIdentifier: String,
        expectedSpeakerIDs: [String]
    ) throws -> SessionSummary {
        guard let decoded = decodeMeetingWire(raw: raw) else {
            throw SummarizerError.decodeFailed(reason: "meeting JSON parse failed")
        }
        // Rename: the model is told to refer to speakers by their
        // display name, but loose outputs may echo the raw id —
        // map id → display name where we can so the topic section
        // reads consistently with the per-speaker roster.
        func display(_ speaker: String?) -> String? {
            guard let speaker else { return nil }
            return speakerNames[speaker] ?? speaker
        }
        // Evidence: prompt row numbers are 1-based into `rows`.
        // Valid numbers map to utterance IDs; invalid ones (out of
        // range — the model invented or garbled a citation) are
        // STRIPPED, never guessed. The strip count is logged so
        // citation slippage is visible in the field. Claims keep
        // their text either way — validation polices the evidence,
        // not the content.
        var strippedEvidenceCount = 0
        func mapEvidence(_ evidence: MeetingWire.Evidence?) -> [UUID]? {
            MeetingEvidence.map(evidence?.numbers, rows: rows, stripped: &strippedEvidenceCount)
        }
        let topics: [SessionSummary.TopicSummary]? = decoded.topics?.map { t in
            SessionSummary.TopicSummary(
                title: t.title ?? "",
                raisedBy: display(t.raisedBy),
                positions: (t.positions ?? []).map { p in
                    SessionSummary.TopicSummary.Position(
                        speaker: display(p.speaker) ?? "",
                        stance: p.stance ?? "",
                        evidenceUtteranceIDs: mapEvidence(p.evidence)
                    )
                },
                evidenceUtteranceIDs: mapEvidence(t.evidence)
            )
        }
        if strippedEvidenceCount > 0 {
            AppLog.app.warning(
                "MLX meeting stripped \(strippedEvidenceCount, privacy: .public) invalid evidence citations (rows 1–\(rows.count, privacy: .public))"
            )
        }
        let perSpeaker = (decoded.perSpeaker ?? []).map { entry in
            SessionSummary.SpeakerSummary(
                speakerID: entry.speakerID,
                speakerName: speakerNames[entry.speakerID],
                // Meeting mode carries content in `talkingPoints`,
                // not the affect-oriented `summary` paragraph.
                summary: "",
                dominantMood: "",
                talkingPoints: entry.talkingPoints
            )
        }
        let filled = SessionSummary.fillMissingPerSpeaker(
            perSpeaker,
            expectedSpeakerIDs: expectedSpeakerIDs,
            speakerNames: speakerNames
        )
        if filled.count > perSpeaker.count {
            AppLog.app.info(
                "MLX meeting filled \(filled.count - perSpeaker.count, privacy: .public) missing per-speaker entries (\(perSpeaker.count, privacy: .public) emitted, \(expectedSpeakerIDs.count, privacy: .public) expected)"
            )
        }
        return SessionSummary(
            inferredSetting: nil,
            topic: decoded.topic ?? "",
            overallMood: "",
            perSpeaker: filled,
            topics: topics,
            model: modelIdentifier,
            generatedAt: Date(),
            mode: .meeting
        )
    }

    // MARK: Parser (window intermediate)

    /// Tolerant parser for the per-window intermediate.
    /// Strict pass first; recovery via
    /// `recoverTruncatedWindowIntermediate` for token-capped
    /// output; placeholder synthesized on any other failure
    /// so one bad window can't sink the whole deep pass.
    static func parseDeepWindow(
        raw: String,
        windowIndex: Int,
        chunk: [UtteranceEstimate]
    ) -> MLXLLMDeepWindowIntermediate {
        let dethought = stripThinkBlocks(raw)
        let stripped = stripCodeFence(dethought)
        let timeStart = chunk.first?.start ?? 0
        let timeEnd = chunk.last?.end ?? 0
        if let braceStart = stripped.firstIndex(of: "{"),
           let braceEnd = stripped.lastIndex(of: "}"),
           let data = String(stripped[braceStart...braceEnd]).data(using: .utf8),
           let decoded = try? JSONDecoder().decode(
               MLXLLMDeepWindowIntermediate.self,
               from: data
           ) {
            return decoded
        }
        if let braceStart = stripped.firstIndex(of: "{"),
           let recovered = recoverTruncatedWindowIntermediate(
               stripped: String(stripped[braceStart...])
           ),
           let data = recovered.data(using: .utf8),
           let decoded = try? JSONDecoder().decode(
               MLXLLMDeepWindowIntermediate.self,
               from: data
           ) {
            AppLog.app.info(
                "MLX deep[\(windowIndex + 1, privacy: .public)] recovered truncated intermediate (\(data.count, privacy: .public) bytes, \(decoded.perSpeaker.count, privacy: .public) speakers)"
            )
            return decoded
        }
        AppLog.app.info(
            "MLX deep[\(windowIndex + 1, privacy: .public)] failed to decode intermediate, synthesizing placeholder"
        )
        let speakerIDs = chunk.orderedSpeakerIDs
        return MLXLLMDeepWindowIntermediate(
            windowIndex: windowIndex,
            timeStart: timeStart,
            timeEnd: timeEnd,
            topicSnapshot: "(window summary unavailable)",
            moodSnapshot: "(window summary unavailable)",
            perSpeaker: speakerIDs.map {
                .init(
                    speakerID: $0,
                    notes: "(no notes captured for this window)",
                    dominantMood: ""
                )
            },
            modalityFlags: []
        )
    }

    // MARK: Recovery helpers

    /// Salvage a truncated `{"setting":..., "topic":...,
    /// "overallMood":..., "perSpeaker":[...]}` blob by
    /// scanning the `perSpeaker` array forward, tracking
    /// string-literal and brace-nesting state, and
    /// remembering the position right after the most recent
    /// top-level object that closed. Cut + append `]}` to
    /// close cleanly. Returns nil for shapes that don't match
    /// our prefix expectations.
    static func recoverTruncatedSummary(stripped: String) -> String? {
        guard let perSpeakerKey = stripped.range(of: "\"perSpeaker\"") else {
            return nil
        }
        guard let arrayOpenRange = stripped.range(
            of: "[",
            range: perSpeakerKey.upperBound..<stripped.endIndex
        ) else {
            return nil
        }
        var depth = 0
        var inString = false
        var escape = false
        var lastCompleteEntryEnd: String.Index? = nil
        var i = arrayOpenRange.upperBound
        while i < stripped.endIndex {
            let ch = stripped[i]
            if escape {
                escape = false
            } else if inString {
                if ch == "\\" {
                    escape = true
                } else if ch == "\"" {
                    inString = false
                }
            } else {
                switch ch {
                case "\"":
                    inString = true
                case "{":
                    depth += 1
                case "}":
                    depth -= 1
                    if depth == 0 {
                        lastCompleteEntryEnd = stripped.index(after: i)
                    }
                case "]" where depth == 0:
                    return nil
                default:
                    break
                }
            }
            i = stripped.index(after: i)
        }
        let cutEnd = lastCompleteEntryEnd ?? arrayOpenRange.upperBound
        var truncated = String(stripped[..<cutEnd])
        while let last = truncated.last,
              last == "," || last.isWhitespace {
            truncated.removeLast()
        }
        truncated.append("]}")
        return truncated
    }

    /// Same shape as `recoverTruncatedSummary` but for the
    /// deep-mode window intermediate. `modalityFlags` is
    /// optional on the Codable struct so its absence in the
    /// recovered prefix is fine.
    static func recoverTruncatedWindowIntermediate(
        stripped: String
    ) -> String? {
        guard let perSpeakerKey = stripped.range(of: "\"perSpeaker\"") else {
            return nil
        }
        guard let arrayOpenRange = stripped.range(
            of: "[",
            range: perSpeakerKey.upperBound..<stripped.endIndex
        ) else {
            return nil
        }
        var depth = 0
        var inString = false
        var escape = false
        var lastCompleteEntryEnd: String.Index? = nil
        var i = arrayOpenRange.upperBound
        while i < stripped.endIndex {
            let ch = stripped[i]
            if escape {
                escape = false
            } else if inString {
                if ch == "\\" {
                    escape = true
                } else if ch == "\"" {
                    inString = false
                }
            } else {
                switch ch {
                case "\"":
                    inString = true
                case "{":
                    depth += 1
                case "}":
                    depth -= 1
                    if depth == 0 {
                        lastCompleteEntryEnd = stripped.index(after: i)
                    }
                case "]" where depth == 0:
                    return nil
                default:
                    break
                }
            }
            i = stripped.index(after: i)
        }
        let cutEnd = lastCompleteEntryEnd ?? arrayOpenRange.upperBound
        var truncated = String(stripped[..<cutEnd])
        while let last = truncated.last,
              last == "," || last.isWhitespace {
            truncated.removeLast()
        }
        truncated.append("]}")
        return truncated
    }

    /// Rewrite a JSON blob where `perSpeaker` is a dict keyed
    /// by speakerID into one where `perSpeaker` is an array
    /// of `{speakerID, summary, dominantMood}` — the
    /// canonical shape `Wire` decodes. Returns nil when the
    /// blob's `perSpeaker` is already an array (or absent).
    /// Llama-Swallow has been observed emitting the dict
    /// form; Qwen always emits the array. This is the salvage
    /// path between strict and truncation-recovery.
    static func normalizePerSpeakerShape(in data: Data) -> Data? {
        guard var obj = try? JSONSerialization
            .jsonObject(with: data) as? [String: Any] else {
            return nil
        }
        guard let dict = obj["perSpeaker"] as? [String: Any] else {
            return nil
        }
        var arr: [[String: Any]] = []
        arr.reserveCapacity(dict.count)
        for (speakerID, value) in dict {
            guard var entry = value as? [String: Any] else { continue }
            if entry["speakerID"] == nil {
                entry["speakerID"] = speakerID
            }
            arr.append(entry)
        }
        obj["perSpeaker"] = arr
        return try? JSONSerialization.data(withJSONObject: obj)
    }

    // MARK: Output sanitizers

    /// Remove `<think>...</think>` blocks Qwen3 emits when
    /// reasoning mode is engaged. Greedy across newlines.
    /// Also drops a stray closing `</think>` if the model
    /// elided the opener (occasionally seen with
    /// `/no_think`).
    static func stripThinkBlocks(_ raw: String) -> String {
        var s = raw
        while let openRange = s.range(of: "<think>") {
            if let closeRange = s.range(
                of: "</think>",
                range: openRange.upperBound..<s.endIndex
            ) {
                s.removeSubrange(openRange.lowerBound..<closeRange.upperBound)
            } else {
                s.removeSubrange(openRange.lowerBound..<s.endIndex)
                break
            }
        }
        if let strayClose = s.range(of: "</think>") {
            s.removeSubrange(s.startIndex..<strayClose.upperBound)
        }
        return s
    }

    /// Strip ```` ```json … ``` ```` fences a chat-tuned
    /// model might emit even after being told "JSON only."
    /// Leaves bare JSON alone.
    static func stripCodeFence(_ raw: String) -> String {
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

// MARK: - Per-row rendering helpers (shared)

/// Common helpers for rendering numeric SER vectors in the
/// per-utterance line format. Shared because Qwen and Llama
/// would otherwise duplicate them verbatim.
internal enum MLXLLMRendering {
    /// Render the acoustic 9-class softmax in
    /// `Label.allCases` order so every row's vector lines up
    /// by position — easier for the LLM to compare rows
    /// column-wise. Missing classes fall back to 0.00 rather
    /// than being omitted, so the schema stays uniform.
    static func acoustic(_ score: CategoricalEmotion) -> String {
        CategoricalEmotion.Label.allCases.map { label in
            let p = score.probabilities[label] ?? 0
            return String(format: "%@=%.2f", label.rawValue, p)
        }.joined(separator: " ")
    }

    /// Render the text 8-class Plutchik intensity vector in
    /// `Label.allCases` order. Intensities, not a softmax
    /// (WRIME is multi-label), so they need not sum to 1.
    static func plutchik(_ score: PlutchikScore) -> String {
        PlutchikScore.Label.allCases.map { label in
            let p = score.probabilities[label] ?? 0
            return String(format: "%@=%.2f", label.rawValue, p)
        }.joined(separator: " ")
    }

    /// Escape backslashes and double quotes for embedding a
    /// transcript inside a `text="..."` field. Used by both
    /// families' `compactLine` so escaping is uniform.
    static func escapedTranscript(_ raw: String) -> String {
        raw
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
    }
}

// MARK: - Evidence mapping (shared across backends)

/// Maps model-cited prompt ROW NUMBERS (1-based into the numbered
/// row list the prompt showed) onto utterance IDs. Out-of-range
/// numbers — the model invented or garbled a citation — are
/// stripped and counted, never guessed; duplicates are deduped
/// preserving first-seen order. Shared by the MLX meeting parser,
/// the Apple FM meeting path, and the LM Studio meeting path so
/// "what counts as valid evidence" has exactly one definition.
internal enum MeetingEvidence {
    /// Defensive ceiling on stored citations per claim — the prompt
    /// asks for ONE; anything past this is model spray, not signal.
    static let maxPerClaim = 8

    static func map(
        _ numbers: [Int]?,
        rows: [UtteranceEstimate],
        stripped: inout Int
    ) -> [UUID]? {
        guard let numbers else { return nil }
        var seen = Set<Int>()
        var ids: [UUID] = []
        for n in numbers {
            guard n >= 1, n <= rows.count else {
                stripped += 1
                continue
            }
            guard seen.insert(n).inserted else { continue }
            ids.append(rows[n - 1].id)
            if ids.count >= maxPerClaim { break }
        }
        return ids
    }
}
