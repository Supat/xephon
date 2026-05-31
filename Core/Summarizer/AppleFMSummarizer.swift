import Foundation
import FoundationModels
import Fusion
import XephonLogging

/// Session summarizer backed by Apple Foundation Models (iPadOS
/// 26+, ~3B parameters, system-managed). Uses the same
/// `LanguageModelSession` + `@Generable` structured-output pattern
/// `FoundationModelsSER` already exercises for per-utterance
/// Plutchik scoring — fresh session per call so the 4096-token
/// context window doesn't accumulate across runs.
///
/// Lighter than the Qwen path: no 4 GB download, no MLX runtime,
/// no per-app memory orchestration required. The trade-off is
/// model size (3B vs 7B) and context (4k vs 32k) — long sessions
/// get aggressively truncated to fit, and multi-speaker reasoning
/// quality is meaningfully lower on subtle arcs.
public actor AppleFMSummarizer: SessionSummarizer {
    public let modelIdentifier = "apple-foundation-models"

    public var isReady: Bool {
        SystemLanguageModel.default.isAvailable
    }

    public init() {}

    /// Cap on prompt utterances for the fast path. Apple FM's
    /// 4096-token window is shared by instructions, the schema
    /// for constrained decoding, the utterance lines, AND the
    /// generated output — the reserved response budget eats into
    /// "what we can pass in" even though our prompt strictly
    /// looks smaller than 4096 tokens. At 30 utterances we were
    /// tripping `exceededContextWindowSize`; 15 leaves comfortable
    /// headroom for a ~500-token response and the schema. The
    /// trailing edge of a session has the most actionable arc
    /// anyway.
    private static let maxPromptUtterances = 15

    /// Per-window utterance count for `.deep` mode. Sized smaller
    /// than the Qwen/Llama 50 because Apple FM's 4k context is
    /// much tighter — even with the simpler intermediate schema,
    /// 15 lands at ~450 prompt tokens leaving plenty of room for
    /// the per-window Generable response. The trade-off vs. Qwen
    /// is more windows per session (a 600-utterance session
    /// becomes 40 windows here vs. 12 on Qwen) but each window
    /// runs much faster on FM's smaller model so the total wall
    /// time is comparable for short-to-medium sessions.
    private static let deepWindowSize = 15

    public func summarize(
        utterances: [UtteranceEstimate],
        speakerNames: [String: String],
        mode: SummarizeMode,
        boostedUtteranceIDs: Set<UUID>
    ) async throws -> SessionSummary {
        guard SystemLanguageModel.default.isAvailable else {
            throw SummarizerError.modelNotInstalled
        }
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
        case .fast:
            return try await summarizeSinglePass(
                utterances: utterances,
                speakerNames: speakerNames,
                selection: .trailing,
                mode: .fast,
                boostedUtteranceIDs: boostedUtteranceIDs
            )
        case .heuristic:
            return try await summarizeSinglePass(
                utterances: utterances,
                speakerNames: speakerNames,
                selection: .heuristicTopN,
                mode: .heuristic,
                boostedUtteranceIDs: boostedUtteranceIDs
            )
        case .deep:
            return try await summarizeDeep(
                utterances: utterances,
                speakerNames: speakerNames
            )
        }
    }

    /// Selection strategy for single-pass modes (`.fast` and
    /// `.heuristic`). Determines how the prompt window is
    /// filled when the session exceeds `maxPromptUtterances`.
    private enum Selection {
        case trailing
        case heuristicTopN
    }

    /// Single-pass summary over a `maxPromptUtterances`-sized
    /// window. Content of the window depends on `selection`:
    /// trailing N (`.fast`) or top-N by informativeness
    /// (`.heuristic`). Same inference call, same output schema
    /// — only the slice changes.
    private func summarizeSinglePass(
        utterances: [UtteranceEstimate],
        speakerNames: [String: String],
        selection: Selection,
        mode: SummarizeMode,
        boostedUtteranceIDs: Set<UUID>
    ) async throws -> SessionSummary {
        let promptUtterances: [UtteranceEstimate]
        let truncatedFrom: Int?
        if utterances.count > Self.maxPromptUtterances {
            switch selection {
            case .trailing:
                promptUtterances = Array(utterances.suffix(Self.maxPromptUtterances))
            case .heuristicTopN:
                // Speaker-balanced — each speaker (up to the
                // default `maxSpeakers = 10`) is guaranteed at
                // least one slot when the budget allows; the
                // remainder is distributed by total per-speaker
                // informativeness, with `boostedUtteranceIDs`
                // (caller-flagged keyword hits) getting a 4×
                // multiplicative boost. See
                // `Informativeness.topNBalancedBySpeaker`.
                let topIDs = Informativeness.topNBalancedBySpeaker(
                    Self.maxPromptUtterances,
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

        let speakers = promptUtterances.orderedSpeakerIDs
        let utteranceLines = promptUtterances
            .map { Self.compactLine(for: $0, speakerNames: speakerNames) }
            .joined(separator: "\n")
        let truncationNote: String
        if let total = truncatedFrom {
            switch selection {
            case .trailing:
                truncationNote = "\n\n(Showing the most recent \(promptUtterances.count) of \(total) utterances; frame overall mood as the trailing portion.)"
            case .heuristicTopN:
                truncationNote = "\n\n(Showing the \(promptUtterances.count) most distinctive of \(total) utterances by session-relative TF-IDF, in chronological order; frame overall mood as a representative sample, not a continuous trailing segment.)"
            }
        } else {
            truncationNote = ""
        }
        // Per-speaker demographic roster from the W2V2 age-gender
        // model. Built off the slice we actually feed the LLM so the
        // demographics reflect the same window as the per-modality
        // vectors do. Empty when no row carried age-gender output —
        // the roster line disappears cleanly in that case.
        let demographicsBlock = SpeakerDemographicsDigest
            .build(from: promptUtterances)
            .renderForPrompt(speakerIDs: speakers, speakerNames: speakerNames)
        let demographicsLine = demographicsBlock.isEmpty
            ? ""
            : "\n\n\(demographicsBlock)"
        // Pin output language to the user's iPadOS app-language
        // pick. Apple FM is less prone to language drift than Qwen
        // but the directive costs ~20 tokens and keeps both
        // backends behaviorally aligned.
        let languageDirective = SummarizerLocale.responseLanguageInstruction
        let userMessage = """
            \(languageDirective)

            Speakers present: \(speakers.joined(separator: ", ")).\(demographicsLine)

            Utterances:
            \(utteranceLines)\(truncationNote)
            """

        AppLog.app.info(
            "AppleFMSummarizer summarizing \(promptUtterances.count, privacy: .public) utterances"
        )
        let session = LanguageModelSession(instructions: Self.instructions)
        do {
            // `includeSchemaInPrompt: false` keeps the schema out
            // of the textual prompt — constrained decoding still
            // enforces the shape, but the nested
            // `perSpeaker: [GenerableSpeakerSummary]` schema is
            // bulky enough to push a 30-utterance prompt past the
            // 4096-token window. Worth ~300+ tokens back.
            let response = try await session.respond(
                to: userMessage,
                generating: GenerableSummary.self,
                includeSchemaInPrompt: false
            )
            let g = response.content
            let perSpeaker = g.perSpeaker.map { entry in
                SessionSummary.SpeakerSummary(
                    speakerID: entry.speakerID,
                    speakerName: speakerNames[entry.speakerID],
                    summary: entry.summary,
                    dominantMood: entry.dominantMood
                )
            }
            // Post-hoc fill — Apple FM is more disciplined than
            // Llama-Swallow about the per-speaker schema but
            // can still drop a one-utterance speaker on edge
            // cases; mirror the MLX core's safety net so the
            // roster always matches the input.
            let filled = SessionSummary.fillMissingPerSpeaker(
                perSpeaker,
                expectedSpeakerIDs: speakers,
                speakerNames: speakerNames
            )
            if filled.count > perSpeaker.count {
                AppLog.app.info(
                    "AppleFMSummarizer filled \(filled.count - perSpeaker.count, privacy: .public) missing per-speaker entries (\(perSpeaker.count, privacy: .public) emitted, \(speakers.count, privacy: .public) expected)"
                )
            }
            return SessionSummary(
                inferredSetting: g.setting,
                topic: g.topic,
                overallMood: g.overallMood,
                perSpeaker: filled,
                model: modelIdentifier,
                generatedAt: Date(),
                mode: mode
            )
        } catch let error as SummarizerError {
            throw error
        } catch {
            AppLog.app.error(
                "AppleFMSummarizer.respond failed: \(String(describing: error), privacy: .public)"
            )
            throw SummarizerError.inferenceFailed(
                reason: String(describing: error)
            )
        }
    }

    /// Map-reduce path: chunk every utterance into windows of
    /// `deepWindowSize`, summarize each window into a compact
    /// intermediate via constrained decoding on
    /// `GenerableWindowIntermediate`, then merge intermediates
    /// into the final `SessionSummary`. Same `LanguageModelSession`
    /// pattern as the fast path but iterated; wall time scales
    /// linearly with `numChunks`.
    private func summarizeDeep(
        utterances: [UtteranceEstimate],
        speakerNames: [String: String]
    ) async throws -> SessionSummary {
        let chunks = stride(from: 0, to: utterances.count, by: Self.deepWindowSize).map {
            offset -> [UtteranceEstimate] in
            let end = min(offset + Self.deepWindowSize, utterances.count)
            return Array(utterances[offset..<end])
        }
        AppLog.app.info(
            "AppleFMSummarizer deep mode: \(utterances.count, privacy: .public) utterances → \(chunks.count, privacy: .public) windows"
        )
        // Short-circuit when the session fits in one window — no
        // benefit to a merge pass over a single intermediate.
        // Stamp as `.deep` so the sheet footer reflects the
        // user's intent even though the actual selection was
        // trailing.
        if chunks.count <= 1 {
            return try await summarizeSinglePass(
                utterances: utterances,
                speakerNames: speakerNames,
                selection: .trailing,
                mode: .deep,
                boostedUtteranceIDs: []
            )
        }

        var intermediates: [DeepWindowIntermediate] = []
        intermediates.reserveCapacity(chunks.count)
        for (idx, chunk) in chunks.enumerated() {
            if Task.isCancelled { throw CancellationError() }
            let intermediate = try await runDeepWindow(
                chunk: chunk,
                speakerNames: speakerNames,
                windowIndex: idx,
                totalWindows: chunks.count
            )
            intermediates.append(intermediate)
            AppLog.app.info(
                "AppleFMSummarizer deep window \(idx + 1, privacy: .public)/\(chunks.count, privacy: .public) done (\(intermediate.perSpeaker.count, privacy: .public) speakers)"
            )
        }
        if Task.isCancelled { throw CancellationError() }
        return try await runDeepMerge(
            intermediates: intermediates,
            allUtterances: utterances,
            speakerNames: speakerNames
        )
    }

    /// Run one window pass — feed the chunk's utterances to a
    /// fresh `LanguageModelSession` with the window instructions,
    /// constrained to `GenerableWindowIntermediate`. On any error
    /// (context-window overrun, refusal, etc.) we synthesize a
    /// placeholder so one bad window doesn't sink the whole deep
    /// pass.
    private func runDeepWindow(
        chunk: [UtteranceEstimate],
        speakerNames: [String: String],
        windowIndex: Int,
        totalWindows: Int
    ) async throws -> DeepWindowIntermediate {
        let speakers = chunk.orderedSpeakerIDs
        let utteranceLines = chunk
            .map { Self.compactLine(for: $0, speakerNames: speakerNames) }
            .joined(separator: "\n")
        let timeStart = chunk.first?.start ?? 0
        let timeEnd = chunk.last?.end ?? 0
        let demographicsBlock = SpeakerDemographicsDigest
            .build(from: chunk)
            .renderForPrompt(speakerIDs: speakers, speakerNames: speakerNames)
        let demographicsLine = demographicsBlock.isEmpty
            ? ""
            : "\n\n\(demographicsBlock)"
        let languageDirective = SummarizerLocale.responseLanguageInstruction
        let userMessage = """
            \(languageDirective)

            This is window \(windowIndex + 1) of \(totalWindows), covering utterances from t=\(String(format: "%.1f", timeStart))s to t=\(String(format: "%.1f", timeEnd))s.
            Speakers in this window: \(speakers.joined(separator: ", ")).\(demographicsLine)

            Utterances:
            \(utteranceLines)
            """
        let session = LanguageModelSession(instructions: Self.windowInstructions)
        do {
            let response = try await session.respond(
                to: userMessage,
                generating: GenerableWindowIntermediate.self,
                includeSchemaInPrompt: false
            )
            let g = response.content
            return DeepWindowIntermediate(
                windowIndex: windowIndex,
                timeStart: timeStart,
                timeEnd: timeEnd,
                topicSnapshot: g.topicSnapshot,
                moodSnapshot: g.moodSnapshot,
                perSpeaker: g.perSpeaker.map { note in
                    DeepWindowIntermediate.PerSpeakerNote(
                        speakerID: note.speakerID,
                        notes: note.notes,
                        dominantMood: note.dominantMood
                    )
                }
            )
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            AppLog.app.error(
                "AppleFMSummarizer deep window \(windowIndex + 1, privacy: .public) failed: \(String(describing: error), privacy: .public); synthesizing placeholder"
            )
            return DeepWindowIntermediate(
                windowIndex: windowIndex,
                timeStart: timeStart,
                timeEnd: timeEnd,
                topicSnapshot: "(window summary unavailable)",
                moodSnapshot: "(window summary unavailable)",
                perSpeaker: speakers.map {
                    DeepWindowIntermediate.PerSpeakerNote(
                        speakerID: $0,
                        notes: "(no notes captured for this window)",
                        dominantMood: ""
                    )
                }
            )
        }
    }

    /// Run the final merge pass — flatten the window intermediates
    /// into a compact text representation and feed them to a fresh
    /// `LanguageModelSession` constrained to `GenerableSummary`.
    /// Same output schema as the fast path so downstream consumers
    /// don't care which mode produced the summary.
    private func runDeepMerge(
        intermediates: [DeepWindowIntermediate],
        allUtterances: [UtteranceEstimate],
        speakerNames: [String: String]
    ) async throws -> SessionSummary {
        let allSpeakers = allUtterances.orderedSpeakerIDs
        let demographicsBlock = SpeakerDemographicsDigest
            .build(from: allUtterances)
            .renderForPrompt(speakerIDs: allSpeakers, speakerNames: speakerNames)
        let demographicsLine = demographicsBlock.isEmpty
            ? ""
            : "\n\n\(demographicsBlock)"
        let languageDirective = SummarizerLocale.responseLanguageInstruction
        let intermediatesText = intermediates.map { window in
            var lines: [String] = []
            lines.append("Window \(window.windowIndex + 1) (t=\(String(format: "%.1f", window.timeStart))s–\(String(format: "%.1f", window.timeEnd))s):")
            lines.append("  topic: \(window.topicSnapshot)")
            lines.append("  mood: \(window.moodSnapshot)")
            for note in window.perSpeaker {
                lines.append("  \(note.speakerID) (\(note.dominantMood)): \(note.notes)")
            }
            return lines.joined(separator: "\n")
        }.joined(separator: "\n\n")
        let userMessage = """
            \(languageDirective)

            This conversation has \(allUtterances.count) utterances across \(intermediates.count) windows.
            Speakers present: \(allSpeakers.joined(separator: ", ")).\(demographicsLine)

            Window summaries (chronological):
            \(intermediatesText)
            """
        AppLog.app.info(
            "AppleFMSummarizer deep merge: \(intermediates.count, privacy: .public) windows"
        )
        let session = LanguageModelSession(instructions: Self.mergeInstructions)
        do {
            let response = try await session.respond(
                to: userMessage,
                generating: GenerableSummary.self,
                includeSchemaInPrompt: false
            )
            let g = response.content
            let perSpeaker = g.perSpeaker.map { entry in
                SessionSummary.SpeakerSummary(
                    speakerID: entry.speakerID,
                    speakerName: speakerNames[entry.speakerID],
                    summary: entry.summary,
                    dominantMood: entry.dominantMood
                )
            }
            let filled = SessionSummary.fillMissingPerSpeaker(
                perSpeaker,
                expectedSpeakerIDs: allSpeakers,
                speakerNames: speakerNames
            )
            if filled.count > perSpeaker.count {
                AppLog.app.info(
                    "AppleFMSummarizer deep merge filled \(filled.count - perSpeaker.count, privacy: .public) missing per-speaker entries (\(perSpeaker.count, privacy: .public) emitted, \(allSpeakers.count, privacy: .public) expected)"
                )
            }
            return SessionSummary(
                inferredSetting: g.setting,
                topic: g.topic,
                overallMood: g.overallMood,
                perSpeaker: filled,
                model: modelIdentifier,
                generatedAt: Date(),
                mode: .deep
            )
        } catch let error as SummarizerError {
            throw error
        } catch {
            AppLog.app.error(
                "AppleFMSummarizer deep merge failed: \(String(describing: error), privacy: .public)"
            )
            throw SummarizerError.inferenceFailed(
                reason: String(describing: error)
            )
        }
    }

    /// Per-window intermediate. In-memory only — never persisted,
    /// never crosses an actor boundary in user-facing API.
    private struct DeepWindowIntermediate {
        let windowIndex: Int
        let timeStart: Double
        let timeEnd: Double
        let topicSnapshot: String
        let moodSnapshot: String
        let perSpeaker: [PerSpeakerNote]

        struct PerSpeakerNote {
            let speakerID: String
            let notes: String
            let dominantMood: String
        }
    }

    // MARK: - Prompt + helpers

    private static let instructions = """
        Summarize a multi-speaker conversation. Each input line has
        speaker, time, fused emotion label, valence V (0..1, 0.5 = neutral),
        and arousal A (0..1, higher = stronger affect), then the transcript.
        First, infer the conversation's setting / situation / register in one
        short sentence (e.g. "casual phone catchup", "job interview",
        "classroom discussion"). Stay general — do not invent specific
        locations or institutions. Then produce a one-sentence topic, a
        one-paragraph overall mood consistent with that setting, and one
        per-speaker entry (short paragraph + dominant-mood phrase) for every
        speaker id in the input. Do not invent speakers.
        When the speaker demographics block lists a gender, use it as the
        canonical pronoun for that speaker throughout the summary — "she/her"
        for female, "he/him" for male, "they/them" for child or when no gender
        is listed. This directive is moot for languages that drop subject
        pronouns (Japanese, Korean, etc.).
        """

    /// Instructions for the per-window pass in `.deep` mode. The
    /// output is a compact intermediate (topic + mood snapshot +
    /// per-speaker notes), NOT a full `SessionSummary` — the merge
    /// pass synthesizes those into the final answer.
    private static let windowInstructions = """
        Summarize ONE WINDOW of a longer multi-speaker conversation.
        Each input line has speaker, time, fused emotion label,
        valence V (0..1, 0.5 = neutral), and arousal A (0..1, higher
        = stronger affect), then the transcript.
        Produce a compact intermediate — NOT a final summary, just a
        snapshot the merge pass will combine with other windows.
        Emit a short topic phrase for this window, a short mood
        phrase for this window, and one per-speaker entry (1-2
        sentences of notes + dominant-mood phrase) for every speaker
        id in this window. Do not invent speakers.
        """

    /// Instructions for the merge pass in `.deep` mode. Same output
    /// shape as the fast path's `instructions` (Generable schema is
    /// the same — `GenerableSummary`) so downstream consumers
    /// don't care which mode produced the summary, but the input is
    /// per-window intermediate text rather than raw utterances.
    private static let mergeInstructions = """
        Produce the FINAL summary of a multi-speaker conversation by
        synthesizing per-window intermediate summaries (provided in
        chronological order). Each speaker should be treated as one
        person across windows — do not split a speaker's arc into
        per-window sections.
        First, infer the conversation's setting / situation / register
        in one short sentence (e.g. "casual phone catchup", "job
        interview", "classroom discussion"). Stay general — do not
        invent specific locations or institutions. Then produce a
        one-sentence topic (factoring topic snapshots across all
        windows), a one-paragraph overall mood (describing the arc,
        not just the trailing window), and one per-speaker entry
        (short paragraph + dominant-mood phrase) for every speaker
        id in the input. Do not invent speakers.
        When the speaker demographics block lists a gender, use it as
        the canonical pronoun for that speaker throughout the summary
        — "she/her" for female, "he/him" for male, "they/them" for
        child or when no gender is listed. This directive is moot for
        languages that drop subject pronouns (Japanese, Korean, etc.).
        """

    /// Tight per-utterance line tuned for the 4k context.
    /// Drops dominance (least-used affect axis) and the
    /// optional `name=` field (we re-map names from the
    /// `speakerNames` dictionary onto the result anyway).
    /// Format: `S01 12.3s joy V0.42 A0.71 "text"`
    private static func compactLine(
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
}

@Generable
private struct GenerableSpeakerSummary {
    @Guide(description: "Canonical speaker id (e.g. S01, S02) — copy from the input")
    var speakerID: String
    @Guide(description: "One paragraph (3–5 sentences) on this speaker's emotional arc through the session")
    var summary: String
    @Guide(description: "Short phrase (1–6 words) capturing the speaker's dominant mood")
    var dominantMood: String
}

@Generable
private struct GenerableWindowSpeakerNote {
    @Guide(description: "Canonical speaker id (e.g. S01, S02) — copy from the input")
    var speakerID: String
    @Guide(description: "1-2 sentences on this speaker's contribution in this window")
    var notes: String
    @Guide(description: "Short phrase (1-6 words) for this speaker's dominant mood in this window")
    var dominantMood: String
}

@Generable
private struct GenerableWindowIntermediate {
    @Guide(description: "Short phrase capturing this window's topic")
    var topicSnapshot: String
    @Guide(description: "Short phrase capturing this window's emotional tone")
    var moodSnapshot: String
    @Guide(description: "Per-speaker notes for this window, one entry per distinct speaker id in this window's input")
    var perSpeaker: [GenerableWindowSpeakerNote]
}

@Generable
private struct GenerableSummary {
    // `setting` is intentionally the FIRST field so constrained
    // decoding commits to a frame (casual / clinical / classroom /
    // …) before generating topic / mood / per-speaker arcs.
    // Without this anchoring, the model's tone can drift mid-output
    // for ambiguous transcripts.
    @Guide(description: "One short sentence identifying the conversation's setting / situation / register (e.g. 'casual phone catchup between friends', 'job interview', 'classroom discussion'). Stay general — do not invent specific locations or institutions.")
    var setting: String
    @Guide(description: "One or two sentences on what the conversation is about")
    var topic: String
    @Guide(description: "One paragraph on the session's overall emotional tone, factoring V/A/D and labels, consistent with the setting above")
    var overallMood: String
    @Guide(description: "Per-speaker emotional arcs, one entry per distinct speaker id in the input")
    var perSpeaker: [GenerableSpeakerSummary]
}
