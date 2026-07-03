import Foundation
import Fusion
import XephonLogging

/// `SessionSummarizer` that off-loads inference to a user-configured
/// LM Studio server over the local network. Pairs with
/// `LMStudioTranscriptionReviewer` and slots alongside Apple FM
/// + MLX in `SummarizerCoordinator`'s backend dispatch.
///
/// **Per CLAUDE.md's local-first / remote-open posture:** the
/// `.lmStudio` backend is opt-in via the Settings card's Remote
/// LLM Server toggle. Audio never leaves the device — only the
/// post-ASR transcript text — but local-network transmission is
/// still off-device in spirit, so privacy.md and a visible Remote
/// badge on the toolbar back the opt-in up.
///
/// **Prompt reuse.** Builds the user message from `MLXQwenSpec`'s
/// `buildPrompt` so the structure stays in lock-step with the
/// MLX path. LM Studio applies the loaded model's chat template
/// server-side, just like MLX-LM does on-device, so a Qwen3
/// loaded in LM Studio receives the same effective prompt as the
/// in-process `MLXQwenSummarizer`. Llama-served servers tolerate
/// the `/no_think` directive (literal text, ignored by Llama);
/// if that ever proves noisy we can split into per-served-family
/// specs.
///
/// **Mode coverage.** Phase 1 supports `.trailing` and
/// `.heuristic` (single-pass selection mirrored from
/// `MLXLLMSummarizerCore.summarizeSinglePass`). `.deep` is best-
/// effort-honored by falling back to `.trailing` — the per-window
/// + merge orchestration is a larger change we can layer in
/// later when there's a concrete deep-mode use case for the
/// remote path.
public actor LMStudioSummarizer: SessionSummarizer {
    public let modelIdentifier: String
    private let client: LMStudioClient
    private let spec = MLXQwenSpec()
    /// When true, send an OpenAI-format `response_format` with
    /// the summary JSON schema attached. The coordinator passes
    /// `LMStudioSettings.useStructuredOutput` through here at
    /// construction time so a settings flip doesn't affect an
    /// in-flight inference.
    private let useStructuredOutput: Bool

    public init(
        modelIdentifier: String,
        client: LMStudioClient,
        useStructuredOutput: Bool = false
    ) {
        // `modelIdentifier` is what gets stamped into the
        // resulting `SessionSummary.model` for attribution; we
        // pass the LM Studio server's model id verbatim so the
        // produced summary reads e.g.
        // `model: "lmstudio:qwen3-8b-instruct"`. Empty model is
        // valid (LM Studio falls back to its loaded model) — we
        // still produce an attribution string in that case so
        // the JSON output isn't blank.
        if modelIdentifier.isEmpty {
            self.modelIdentifier = "lmstudio"
        } else {
            self.modelIdentifier = "lmstudio:\(modelIdentifier)"
        }
        self.client = client
        self.useStructuredOutput = useStructuredOutput
    }

    public var isReady: Bool {
        // We can't verify reachability without a round-trip, and
        // the picker UI gates the LM Studio option on the Settings
        // toggle separately. Returning `true` here keeps the
        // coordinator's `ready` check from blocking based on a
        // network probe that would fire on every body re-render.
        true
    }

    public func summarize(
        utterances: [UtteranceEstimate],
        speakerNames: [String: String],
        mode: SummarizeMode,
        boostedUtteranceIDs: Set<UUID>
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

        // Phase 1: deep falls back to trailing. Document the
        // demotion in the log so the user can correlate a
        // "Summary looks shorter than I expected on a long
        // session" question with the actual selection used.
        //
        // `.all` is the LM-Studio-only "no cap, no chunking,
        // full per-row metadata" mode — the server's context
        // window is assumed to fit, which the remote backend
        // is the only one we'd believe that of.
        let resolvedMode: SummarizeMode
        let selection: MLXLLMSelection
        let promptUtterances: [UtteranceEstimate]
        let truncatedFrom: Int?
        switch mode {
        case .meeting:
            // Content-focused minutes: text-only rows, topics +
            // talking points out, no affect. Own prompt / schema /
            // parser, so short-circuit the shared single-pass plumbing
            // below entirely.
            return try await summarizeMeeting(
                utterances: utterances,
                speakerNames: speakerNames,
                boostedUtteranceIDs: boostedUtteranceIDs
            )
        case .trailing:
            resolvedMode = .trailing
            selection = .trailing
            (promptUtterances, truncatedFrom) = selectUtterances(
                from: utterances,
                cap: spec.maxPromptUtterances,
                selection: selection,
                boostedUtteranceIDs: boostedUtteranceIDs
            )
        case .heuristic:
            resolvedMode = .heuristic
            selection = .heuristicTopN
            (promptUtterances, truncatedFrom) = selectUtterances(
                from: utterances,
                cap: spec.maxPromptUtterances,
                selection: selection,
                boostedUtteranceIDs: boostedUtteranceIDs
            )
        case .deep:
            AppLog.app.info("LMStudioSummarizer: .deep requested → demoted to .trailing (phase 1)")
            resolvedMode = .trailing
            selection = .trailing
            (promptUtterances, truncatedFrom) = selectUtterances(
                from: utterances,
                cap: spec.maxPromptUtterances,
                selection: selection,
                boostedUtteranceIDs: boostedUtteranceIDs
            )
        case .all:
            // No cap, no selection — pass every utterance
            // through `buildPrompt` with `truncatedFromTotal:
            // nil` so the prompt skips its "showing only the
            // most recent N of M" framing note. The model is
            // told it's seeing the whole conversation, which
            // is the truth here.
            AppLog.app.info(
                "LMStudioSummarizer: .all mode — sending \(utterances.count, privacy: .public) utterances uncapped"
            )
            resolvedMode = .all
            selection = .trailing
            promptUtterances = utterances
            truncatedFrom = nil
        }
        let prompt = spec.buildPrompt(
            utterances: promptUtterances,
            speakerNames: speakerNames,
            truncatedFromTotal: truncatedFrom,
            selection: selection
        )
        AppLog.app.info(
            "LMStudioSummarizer summarizing \(promptUtterances.count, privacy: .public) utterances (prompt \(prompt.count, privacy: .public) chars)"
        )
        let raw: String
        do {
            let responseFormatJSON = useStructuredOutput
                ? try? JSONEncoder().encode(LMStudioResponseFormat.jsonSchema(
                    name: "session_summary",
                    schema: LMStudioSchemas.summarySchema
                ))
                : nil
            raw = try await client.chat(
                userMessage: prompt,
                temperature: 0.2,
                maxTokens: spec.maxOutputTokens,
                responseFormatJSON: responseFormatJSON
            )
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            // Bubble up as SummarizerError so the coordinator's
            // existing error-surfacing path treats it uniformly
            // with the on-device backends' failures.
            throw SummarizerError.inferenceFailed(reason: String(describing: error))
        }
        if Task.isCancelled { throw CancellationError() }
        // Log a preview of the raw output (not just the count)
        // so a downstream parse failure ("no JSON object found",
        // etc.) is debuggable from Console without having to
        // rerun against a hand-inspected server. 400 chars is
        // enough to see the response's opening framing and any
        // refusal / preamble text that's keeping the parser
        // from finding the `{`.
        let preview = raw.count > 400
            ? String(raw.prefix(400)) + "…[truncated]"
            : raw
        AppLog.app.info(
            "LMStudioSummarizer raw output: \(raw.count, privacy: .public) chars, preview: \(preview, privacy: .public)"
        )
        return try MLXLLMSummarizerCore.parse(
            raw: raw,
            speakerNames: speakerNames,
            modelIdentifier: modelIdentifier,
            mode: resolvedMode,
            expectedSpeakerIDs: promptUtterances.orderedSpeakerIDs
        )
    }

    /// Mirrors the trailing / heuristic selection block inside
    /// `MLXLLMSummarizerCore.summarizeSinglePass` so the prompt
    /// the LM Studio server receives matches what the on-device
    /// MLX path would have sent for the same inputs. Kept inline
    /// (rather than refactoring the MLX path to share) because
    /// that helper takes a `ModelContainer` and the surgery to
    /// extract just the selection logic is wider than this
    /// adapter warrants.
    private func selectUtterances(
        from utterances: [UtteranceEstimate],
        cap: Int,
        selection: MLXLLMSelection,
        boostedUtteranceIDs: Set<UUID>
    ) -> (selected: [UtteranceEstimate], truncatedFromTotal: Int?) {
        guard utterances.count > cap else { return (utterances, nil) }
        switch selection {
        case .trailing:
            return (Array(utterances.suffix(cap)), utterances.count)
        case .heuristicTopN:
            let topIDs = Informativeness.topNBalancedBySpeaker(
                cap,
                utterances: utterances,
                boostedIDs: boostedUtteranceIDs
            )
            return (utterances.filter { topIDs.contains($0.id) }, utterances.count)
        }
    }

    // MARK: - Meeting mode

    /// Cap on transcript rows fed into a `.meeting` prompt.
    // ponytail: meeting rows are TEXT-ONLY (speaker + transcript, no
    // fused label / V/A/D / acoustic / Plutchik / demographics), so a
    // row costs ~3-4× less than the full-SER line `.all` sends. That
    // freed budget buys a much larger cap than the on-device 100. We
    // keep a hard cap (rather than borrowing `.all`'s uncapped pass)
    // so a pathologically long session can't blow even a remote
    // server's context window; 400 distinct lines is plenty of meeting
    // coverage and stays well inside a typical LM Studio context. Same
    // heuristic top-N-balanced-by-speaker selection as `.heuristic`.
    private static let meetingMaxPromptUtterances = 400

    /// `.meeting` path. Heuristic top-N selection (larger cap, text-
    /// only rows), a content-only prompt + schema, and a meeting
    /// parser that fills `SessionSummary.topics` +
    /// `SpeakerSummary.talkingPoints` while zeroing every affect field.
    private func summarizeMeeting(
        utterances: [UtteranceEstimate],
        speakerNames: [String: String],
        boostedUtteranceIDs: Set<UUID>
    ) async throws -> SessionSummary {
        let (promptUtterances, truncatedFrom) = selectUtterances(
            from: utterances,
            cap: Self.meetingMaxPromptUtterances,
            selection: .heuristicTopN,
            boostedUtteranceIDs: boostedUtteranceIDs
        )
        let prompt = buildMeetingPrompt(
            utterances: promptUtterances,
            speakerNames: speakerNames,
            truncatedFromTotal: truncatedFrom
        )
        AppLog.app.info(
            "LMStudioSummarizer meeting summarizing \(promptUtterances.count, privacy: .public) utterances (prompt \(prompt.count, privacy: .public) chars)"
        )
        let raw: String
        do {
            let responseFormatJSON = useStructuredOutput
                ? try? JSONEncoder().encode(LMStudioResponseFormat.jsonSchema(
                    name: "meeting_summary",
                    schema: LMStudioSchemas.meetingSchema
                ))
                : nil
            raw = try await client.chat(
                userMessage: prompt,
                temperature: 0.2,
                maxTokens: spec.maxOutputTokens,
                responseFormatJSON: responseFormatJSON
            )
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            throw SummarizerError.inferenceFailed(reason: String(describing: error))
        }
        if Task.isCancelled { throw CancellationError() }
        let preview = raw.count > 400
            ? String(raw.prefix(400)) + "…[truncated]"
            : raw
        AppLog.app.info(
            "LMStudioSummarizer meeting raw output: \(raw.count, privacy: .public) chars, preview: \(preview, privacy: .public)"
        )
        return try Self.parseMeeting(
            raw: raw,
            speakerNames: speakerNames,
            modelIdentifier: modelIdentifier,
            expectedSpeakerIDs: promptUtterances.orderedSpeakerIDs
        )
    }

    /// Build the meeting prompt. Rows are TEXT-ONLY — speaker label
    /// (id, plus the rename when present) + transcript, nothing else.
    /// The output schema is content-only: an overview, the main topics
    /// with attribution + per-speaker positions, and each speaker's
    /// talking points. Kept local to this file (not in
    /// `PromptCatalog` / `MLXQwenSpec`) since it's LM-Studio-specific.
    private func buildMeetingPrompt(
        utterances: [UtteranceEstimate],
        speakerNames: [String: String],
        truncatedFromTotal: Int?
    ) -> String {
        let speakers = utterances.orderedSpeakerIDs
        let speakerList = speakers.joined(separator: ", ")
        var lines: [String] = []
        lines.reserveCapacity(utterances.count + 16)
        lines.append("You are an analyst writing the minutes of a multi-speaker meeting.")
        lines.append("Read every line of the transcript below and produce a JSON object with three fields:")
        lines.append("  \"topic\" — one or two sentences giving an overall overview of what the meeting was about.")
        lines.append("  \"topics\" — array of the main subjects discussed. Each entry is { \"title\": <short topic name>, \"raisedBy\": <speaker id or name who first brought it up, or \"\" if unclear>, \"positions\": [ { \"speaker\": <id or name>, \"stance\": <that speaker's opinion / position on this topic> } ] }. Capture every distinct topic the meeting actually covered.")
        lines.append("  \"perSpeaker\" — array, one entry per speaker id in this list: \(speakerList). Each entry is { \"speakerID\": <id>, \"talkingPoints\": [ <short string>, … ] } listing that speaker's main points / contributions. Maximize coverage — list every substantive point the speaker made.")
        lines.append("Focus ONLY on content: who said what, and the positions taken. Do NOT comment on emotion, mood, tone, or affect.")
        lines.append("Return ONLY valid JSON, no prose before or after.")
        lines.append(SummarizerLocale.responseLanguageInstruction)
        // Disable Qwen3 thinking for a single turn (harmless literal
        // text for Llama-served models — see the type-level note).
        lines.append("/no_think")
        if let total = truncatedFromTotal {
            lines.append("")
            lines.append("NOTE: This meeting has \(total) lines total; the \(utterances.count) most distinctive lines (chosen by session-relative TF-IDF, NOT the most recent) are shown below in chronological order. Cover the whole meeting; gaps between lines are expected.")
        }
        lines.append("")
        lines.append("Transcript:")
        for u in utterances {
            lines.append(meetingLine(for: u, speakerNames: speakerNames))
        }
        lines.append("")
        lines.append("---")
        lines.append("IMPORTANT: Follow the instructions above and produce exactly one valid JSON object with fields topic, topics, perSpeaker. The FIRST character of your output MUST be `{`. Do NOT echo the transcript above; do NOT add any prose.")
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

    /// One TEXT-ONLY transcript row: `- <id> (<name>): <transcript>`.
    /// All emotion fields the SER-rich `compactLine` carries are
    /// deliberately dropped. The rename map is applied so the model
    /// can refer to speakers by their friendly name in `raisedBy` /
    /// `positions`.
    private func meetingLine(
        for u: UtteranceEstimate,
        speakerNames: [String: String]
    ) -> String {
        var label = u.speakerID
        if let name = speakerNames[u.speakerID], !name.isEmpty {
            label += " (\(name))"
        }
        return "- \(label): \(u.transcript)"
    }

    /// Parse the meeting JSON into a `SessionSummary`. Reuses the
    /// shared `<think>` / code-fence sanitizers and the lenient
    /// first-`{`-to-last-`}` slice the affect parser uses, then maps
    /// onto `topics` + per-speaker `talkingPoints` with every affect
    /// field zeroed (`overallMood` / `dominantMood` empty, no setting).
    /// `fillMissingPerSpeaker` guarantees a roster entry per input
    /// speaker even when the model drops one.
    static func parseMeeting(
        raw: String,
        speakerNames: [String: String],
        modelIdentifier: String,
        expectedSpeakerIDs: [String]
    ) throws -> SessionSummary {
        let dethought = MLXLLMSummarizerCore.stripThinkBlocks(raw)
        let stripped = MLXLLMSummarizerCore.stripCodeFence(dethought)
        guard let braceStart = stripped.firstIndex(of: "{"),
              let braceEnd = stripped.lastIndex(of: "}"),
              braceStart < braceEnd else {
            throw SummarizerError.decodeFailed(reason: "no JSON object found")
        }
        struct Wire: Decodable {
            struct Position: Decodable {
                let speaker: String?
                let stance: String?
            }
            struct Topic: Decodable {
                let title: String?
                let raisedBy: String?
                let positions: [Position]?
            }
            struct PerSpeaker: Decodable {
                let speakerID: String
                let talkingPoints: [String]?
            }
            let topic: String?
            let topics: [Topic]?
            let perSpeaker: [PerSpeaker]?
        }
        let jsonSlice = String(stripped[braceStart...braceEnd])
        guard let data = jsonSlice.data(using: .utf8),
              let decoded = try? JSONDecoder().decode(Wire.self, from: data) else {
            throw SummarizerError.decodeFailed(reason: "meeting JSON did not parse")
        }
        let topics: [SessionSummary.TopicSummary]? = decoded.topics.map { list in
            list.map { t in
                let by = t.raisedBy?.trimmingCharacters(in: .whitespacesAndNewlines)
                return SessionSummary.TopicSummary(
                    title: t.title ?? "",
                    raisedBy: (by?.isEmpty == false) ? by : nil,
                    positions: (t.positions ?? []).map { p in
                        .init(speaker: p.speaker ?? "", stance: p.stance ?? "")
                    }
                )
            }
        }
        let perSpeaker = (decoded.perSpeaker ?? []).map { entry in
            SessionSummary.SpeakerSummary(
                speakerID: entry.speakerID,
                speakerName: speakerNames[entry.speakerID],
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
}
