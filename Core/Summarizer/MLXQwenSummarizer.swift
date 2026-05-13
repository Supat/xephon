import Foundation
import Fusion
import SERAcoustic
import SERText
import XephonLogging
import MLX
import MLXLLM
import MLXLMCommon

/// MLX-backed summarizer using a Qwen2.5-Instruct 4-bit model
/// hydrated under `ModelStore`'s install directory. All inference
/// runs on-device — no network calls, no cloud fallback.
///
/// Lifecycle:
///   1. Construct with the resolved model directory + identifier.
///   2. Call `load()` (or let `summarize(...)` lazy-load on first
///      use) to bring the weights into memory. Loading 4-bit 7B is
///      ~5–10 s on M4 iPad Pro.
///   3. Call `summarize(utterances:speakerNames:)` to generate.
///   4. Call `unload()` to release the ~4 GB working set when the
///      summarize UI dismisses — keeping the model resident across
///      a recording session would push memory pressure too hard
///      next to W2V2 + emotion2vec + DeBERTa.
public actor MLXQwenSummarizer: SessionSummarizer {
    public let modelIdentifier: String
    private let modelDirectory: URL
    private var container: ModelContainer?

    /// Hard cap on tokens the LLM may emit per summary. Tuned to
    /// the realistic size of the structured JSON we ask for (topic
    /// + overall mood + ~5–8 per-speaker paragraphs) plus generous
    /// headroom. Critical not to leave this nil — Qwen will run to
    /// EOS otherwise, and the unbounded run can push the app past
    /// its memory budget alongside the 4.3 GB resident weights.
    private static let maxOutputTokens = 2048

    /// Cap on the number of utterances we feed into a single
    /// summary pass. Each row carries the full acoustic 9-class
    /// softmax + Plutchik 8-class intensity in addition to the
    /// fused label/V/A/D — ~2.3× the per-row token cost vs. a
    /// fused-only line. 100 utterances at this richer format ≈
    /// 10–14k input tokens, comfortably inside Qwen3's context
    /// window and inside the increased-memory-limit entitlement's
    /// per-app budget. Sessions longer than this take the most
    /// recent window — emotion arcs concentrate at the trailing
    /// edge of a conversation.
    private static let maxPromptUtterances = 100

    public init(modelIdentifier: String, modelDirectory: URL) {
        self.modelIdentifier = modelIdentifier
        self.modelDirectory = modelDirectory
    }

    public var isReady: Bool {
        container != nil
    }

    /// Bring the weights into memory. Idempotent. Surfaces
    /// `SummarizerError.modelLoadFailed` on any underlying MLX
    /// error (corrupted weights, format mismatch, etc.).
    public func load() async throws {
        if container != nil { return }
        AppLog.app.info(
            "MLXQwenSummarizer loading from \(self.modelDirectory.path, privacy: .public)"
        )
        // Cap MLX's buffer cache to keep the working set tight.
        // The default is bounded by Metal's recommendedMaxWorking-
        // SetSize, which on a 16 GB iPad sits high enough to push
        // the process over the Jetsam ceiling once Qwen weights
        // and the SER pipeline coexist. 32 MB is the value the
        // mlx-swift docs recommend for LLM evaluation on iOS.
        MLX.GPU.set(cacheLimit: 32 * 1024 * 1024)
        do {
            // MLXLMCommon resolves a directory containing
            // `config.json`, `tokenizer.json`, and the safetensors
            // shards into a `ModelContainer` that owns the loaded
            // weights + tokenizer for the duration of the actor.
            let configuration = ModelConfiguration(
                directory: modelDirectory
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
        AppLog.app.info("MLXQwenSummarizer unloaded")
    }

    public func summarize(
        utterances: [UtteranceEstimate],
        speakerNames: [String: String]
    ) async throws -> SessionSummary {
        try await load()
        guard let container else {
            throw SummarizerError.modelNotInstalled
        }
        guard !utterances.isEmpty else {
            return SessionSummary(
                topic: "",
                overallMood: "",
                perSpeaker: [],
                model: modelIdentifier,
                generatedAt: Date()
            )
        }

        // Truncate to the most recent window when the session
        // exceeds `maxPromptUtterances` — see the constant's doc
        // comment for the memory rationale.
        let promptUtterances: [UtteranceEstimate]
        let truncatedFrom: Int?
        if utterances.count > Self.maxPromptUtterances {
            promptUtterances = Array(utterances.suffix(Self.maxPromptUtterances))
            truncatedFrom = utterances.count
        } else {
            promptUtterances = utterances
            truncatedFrom = nil
        }
        let prompt = Self.buildPrompt(
            utterances: promptUtterances,
            speakerNames: speakerNames,
            truncatedFromTotal: truncatedFrom
        )
        if let truncatedFrom {
            AppLog.app.info(
                "MLXQwenSummarizer truncating \(truncatedFrom, privacy: .public) → \(promptUtterances.count, privacy: .public) utterances"
            )
        }
        AppLog.app.info(
            "MLXQwenSummarizer summarizing \(promptUtterances.count, privacy: .public) utterances (prompt \(prompt.count, privacy: .public) chars)"
        )
        let raw: String
        do {
            raw = try await container.perform { context -> String in
                let userInput = UserInput(prompt: prompt)
                let lmInput = try await context.processor.prepare(input: userInput)
                // `generate(...)` runs synchronously to completion
                // inside the actor and returns a `GenerateResult`.
                // We cap `maxTokens` because the default is nil
                // (unbounded), and Qwen will happily keep producing
                // tokens until EOS — combined with the 4.3 GB
                // resident weights, an unbounded run can push the
                // app over its memory budget and trip a SIGKILL.
                // 2048 tokens (~1500 words) is comfortable headroom
                // for a topic + overall mood + several per-speaker
                // paragraphs without risking OOM.
                // Two `generate(input:parameters:context:didGenerate:)`
                // overloads exist with different didGenerate arities
                // (`(Int) -> .` and `([Int]) -> .`). Pin the closure
                // parameter type so the compiler picks the `[Int]`
                // variant — the one whose return is `GenerateResult`
                // with `.output` already decoded for us.
                var parameters = GenerateParameters(
                    maxTokens: Self.maxOutputTokens,
                    // Deterministic for reproducibility — the
                    // summary is a derived artifact of the
                    // session, not creative writing.
                    temperature: 0.2
                )
                // Default prefill step is 512 tokens — packing
                // hundreds of tokens into a single Metal command
                // buffer was tripping the GPU watchdog
                // (`kIOGPUCommandBufferCallbackErrorTimeout` →
                // mlx::core::gpu::check_error SIGABRT) on real
                // iPad hardware. 64 splits prefill into many
                // smaller kernels well inside the timeout window.
                parameters.prefillStepSize = 64
                let result = try MLXLMCommon.generate(
                    input: lmInput,
                    parameters: parameters,
                    context: context,
                    // Cooperative cancellation: the user dismissing
                    // the summary sheet cancels the View-owned Task
                    // that's driving this call, which propagates
                    // here as `Task.isCancelled`. Returning `.stop`
                    // makes the generate loop bail at the next token
                    // boundary instead of running to EOS — without
                    // this hook a dismiss leaves the LLM grinding
                    // away in the background on a result no one will
                    // see.
                    didGenerate: { (_: [Int]) -> GenerateDisposition in
                        Task.isCancelled ? .stop : .more
                    }
                )
                return result.output
            }
        } catch let error as SummarizerError {
            throw error
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            AppLog.app.error(
                "MLXQwenSummarizer.generate failed: \(String(describing: error), privacy: .public)"
            )
            throw SummarizerError.inferenceFailed(
                reason: String(describing: error)
            )
        }
        // If generation was cancelled mid-stream the partial output
        // won't parse as the structured JSON we asked for — bail
        // before the parser has a chance to surface a misleading
        // decode error to the user.
        if Task.isCancelled { throw CancellationError() }
        AppLog.app.info(
            "MLXQwenSummarizer raw output: \(raw.count, privacy: .public) chars"
        )

        return try Self.parse(
            raw: raw,
            speakerNames: speakerNames,
            modelIdentifier: modelIdentifier
        )
    }

    // MARK: - Prompt + parser

    /// Order distinct speaker ids in their first-appearance order,
    /// matching the chip-bar ordering in the UI so the LLM's
    /// per-speaker arc reads in the same order the user is reading
    /// the rows in.
    private static func orderedSpeakerIDs(
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

    /// Build the chat-style prompt the model sees. Compact JSON-ish
    /// per-utterance lines keep the token budget bounded — every
    /// numerical score is preserved (the whole point of going
    /// beyond Apple FM was to keep these) but we elide the V/A/D
    /// detail when it's nil and round floats to 2 decimals.
    /// `truncatedFromTotal` is the original session size when the
    /// caller has narrowed `utterances` to a tail window for
    /// memory reasons — we inform the model so it knows the
    /// passage isn't the whole conversation and can frame its
    /// "overall mood" accordingly.
    private static func buildPrompt(
        utterances: [UtteranceEstimate],
        speakerNames: [String: String],
        truncatedFromTotal: Int?
    ) -> String {
        let speakers = orderedSpeakerIDs(from: utterances)
        var lines: [String] = []
        lines.reserveCapacity(utterances.count + 12)
        lines.append("You are an analyst summarizing a multi-speaker conversation.")
        lines.append("Read every utterance below and produce a JSON object with three fields:")
        lines.append("  \"topic\" — one or two sentences on what the conversation is about.")
        lines.append("  \"overallMood\" — one paragraph on the session's overall emotional tone.")
        lines.append("  \"perSpeaker\" — array, one entry per speaker id in this list: \(speakers.joined(separator: ", ")).")
        lines.append("Each perSpeaker entry has: { \"speakerID\": <id>, \"summary\": <one paragraph>, \"dominantMood\": <one short phrase> }.")
        lines.append("Each row carries: fused label and fused V/A/D (valence/arousal/dominance, 0–1), plus the raw per-modality probability vectors:")
        lines.append("  aP = acoustic 9-class softmax (angry, disgusted, fearful, happy, neutral, other, sad, surprised, unknown)")
        lines.append("  tP = text 8-class Plutchik intensity (joy, sadness, anticipation, surprise, anger, fear, disgust, trust)")
        lines.append("Use these to judge confidence and to flag modality disagreement — e.g. a row where aP says sad but tP says joy is worth calling out per-speaker; the fused label hides that signal.")
        lines.append("Return ONLY valid JSON, no prose before or after.")
        // Qwen3 ships with a "thinking" mode that emits a
        // `<think>...</think>` chain-of-thought block before the
        // actual response. That blows our 2048-token output cap and
        // confuses any "find the first '{'" parser since the
        // thinking text often contains braces. The `/no_think`
        // directive is Qwen3's documented switch to disable
        // reasoning for a single turn — it must appear in the user
        // prompt (system instructions are routed through Jinja
        // template logic that doesn't honor it the same way).
        lines.append("/no_think")
        if let total = truncatedFromTotal {
            lines.append("")
            lines.append("NOTE: This conversation has \(total) utterances total; only the most recent \(utterances.count) are shown below. Frame the overall mood as the trailing portion of the session, not the whole arc.")
        }
        lines.append("")
        lines.append("Utterances:")
        for u in utterances {
            lines.append(compactLine(for: u, speakerNames: speakerNames))
        }
        return lines.joined(separator: "\n")
    }

    /// One per-utterance line. Order stable so the model sees
    /// consistent positional cues across rows: speaker → time →
    /// fused → per-modality probability vectors → transcript.
    /// The transcript field is named `text` so to avoid collision
    /// with the text-SER's Plutchik vector we name that `tP`.
    private static func compactLine(
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
            fields.append("aP={\(renderAcoustic(acoustic))}")
        }
        if let plutchik = u.plutchik {
            fields.append("tP={\(renderPlutchik(plutchik))}")
        }
        let escaped = u.transcript
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
        fields.append("text=\"\(escaped)\"")
        return "- " + fields.joined(separator: " ")
    }

    /// Render the acoustic 9-class softmax in `Label.allCases`
    /// order so every row's vector lines up by position — easier
    /// for the LLM to compare rows column-wise. Missing classes
    /// fall back to 0.00 rather than being omitted, so the schema
    /// stays uniform across rows.
    private static func renderAcoustic(_ score: CategoricalEmotion) -> String {
        CategoricalEmotion.Label.allCases.map { label in
            let p = score.probabilities[label] ?? 0
            return String(format: "%@=%.2f", label.rawValue, p)
        }.joined(separator: " ")
    }

    /// Render the text 8-class Plutchik intensity vector in
    /// `Label.allCases` order. Same uniform-schema rationale as
    /// `renderAcoustic` — and note these are intensities, not a
    /// softmax (WRIME is multi-label), so they need not sum to 1.
    private static func renderPlutchik(_ score: PlutchikScore) -> String {
        PlutchikScore.Label.allCases.map { label in
            let p = score.probabilities[label] ?? 0
            return String(format: "%@=%.2f", label.rawValue, p)
        }.joined(separator: " ")
    }

    /// Decode the LLM's JSON output. Robust against the common
    /// failure modes of small-LLM output: leading / trailing prose
    /// outside the JSON block, fenced code blocks, or partial
    /// trailing fragments.
    private static func parse(
        raw: String,
        speakerNames: [String: String],
        modelIdentifier: String
    ) throws -> SessionSummary {
        // Belt-and-braces: even with `/no_think` in the prompt some
        // Qwen3 builds still emit an (often empty) `<think></think>`
        // pair, and any chain-of-thought inside can carry braces
        // that mislead "first '{'" scanning. Strip the block before
        // any other processing.
        let dethought = stripThinkBlocks(raw)
        let stripped = stripCodeFence(dethought)
        guard let braceStart = stripped.firstIndex(of: "{"),
              let braceEnd = stripped.lastIndex(of: "}") else {
            throw SummarizerError.decodeFailed(reason: "no JSON object found")
        }
        let jsonString = String(stripped[braceStart...braceEnd])
        guard let data = jsonString.data(using: .utf8) else {
            throw SummarizerError.decodeFailed(reason: "non-utf8 output")
        }
        struct Wire: Decodable {
            struct PerSpeaker: Decodable {
                let speakerID: String
                let summary: String
                let dominantMood: String
            }
            let topic: String
            let overallMood: String
            let perSpeaker: [PerSpeaker]
        }
        let decoded: Wire
        do {
            decoded = try JSONDecoder().decode(Wire.self, from: data)
        } catch {
            throw SummarizerError.decodeFailed(
                reason: String(describing: error)
            )
        }
        let perSpeaker = decoded.perSpeaker.map { entry in
            SessionSummary.SpeakerSummary(
                speakerID: entry.speakerID,
                speakerName: speakerNames[entry.speakerID],
                summary: entry.summary,
                dominantMood: entry.dominantMood
            )
        }
        return SessionSummary(
            topic: decoded.topic,
            overallMood: decoded.overallMood,
            perSpeaker: perSpeaker,
            model: modelIdentifier,
            generatedAt: Date()
        )
    }

    /// Remove any `<think>...</think>` reasoning blocks Qwen3 emits
    /// when its thinking mode is engaged. Greedy across newlines.
    /// Also drops a stray closing `</think>` if the model elided
    /// the opening tag (occasionally seen with `/no_think`).
    private static func stripThinkBlocks(_ raw: String) -> String {
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
    private static func stripCodeFence(_ raw: String) -> String {
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
