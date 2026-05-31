import Foundation
import Fusion
import SERAcoustic
import SERText
import XephonLogging
import MLX
import MLXLLM
import MLXLMCommon

/// MLX-backed summarizer hydrated under `ModelStore`'s install
/// directory. All inference runs on-device — no network calls,
/// no cloud fallback. Despite the `Qwen` in the class name (kept
/// for diff minimality during the multi-family rollout), this
/// class supports any LLM the MLX-LM factory can load — Qwen3,
/// Llama 3, etc. Family-specific prompt tokens (e.g. Qwen3's
/// `/no_think` directive) are gated on `family` so a Llama
/// backend doesn't get Qwen's control-token treated as literal
/// text.
///
/// Lifecycle:
///   1. Construct with the resolved model directory + identifier
///      + the model's family.
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
    private let family: LLMModelFamily
    private var container: ModelContainer?

    /// Hard cap on tokens the LLM may emit per summary. The output
    /// JSON is topic + overall mood + a per-speaker paragraph for
    /// every speaker in the input; a busy session with 4–6 speakers
    /// can want 3–4k tokens of output to fully populate. 2048 was
    /// landing the parser at "Unexpected end of file" on
    /// real-world conversations. 4096 fits the realistic max; the
    /// tolerant parser below salvages the prefix when the model
    /// still runs over.
    private static let maxOutputTokens = 4096

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

    /// Per-window utterance count for `.deep` mode's map-reduce
    /// pass. Smaller than `maxPromptUtterances` because deep mode
    /// runs many windows back-to-back — keeping each window
    /// tight bounds per-call prefill cost (which dominates wall
    /// time at this scale) and gives the merge pass uniformly-
    /// sized inputs to fuse. 50 lands at ~5–7k prompt tokens per
    /// window, leaves the merge pass comfortably under 12k
    /// tokens for sessions up to 1200 utterances (24 windows ×
    /// ~400-token intermediate each).
    private static let deepWindowSize = 50

    /// Output-token cap for per-window intermediate summaries —
    /// they're compact (topic + mood snapshot + per-speaker
    /// notes + modality flags), not full structured `SessionSummary`
    /// objects. Originally sized at 768, bumped to 1280 after a
    /// 50-utterance / 5-speaker window saturated 768 mid-perSpeaker
    /// array and forced the parser to fall through to the
    /// placeholder synth. 1280 fits a verbose window (8 speakers
    /// × ~120-token notes + topic/mood + a long modalityFlags list)
    /// with headroom; the truncation-recovery path below salvages
    /// anything that still spills.
    private static let deepWindowOutputTokens = 1280

    public init(
        modelIdentifier: String,
        modelDirectory: URL,
        family: LLMModelFamily = .qwen
    ) {
        self.modelIdentifier = modelIdentifier
        self.modelDirectory = modelDirectory
        self.family = family
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
        // and the SER pipeline coexist. The mlx-swift docs recommend
        // 32 MB for LLM eval on iOS, but the SER actors are torn
        // down before `summarize` runs so we have headroom — 128 MB
        // lets MLX keep more prefill/decode scratch buffers resident,
        // measurably improving generation throughput without
        // re-introducing Jetsam pressure.
        MLX.GPU.set(cacheLimit: 128 * 1024 * 1024)
        do {
            // MLXLMCommon resolves a directory containing
            // `config.json`, `tokenizer.json`, and the safetensors
            // shards into a `ModelContainer` that owns the loaded
            // weights + tokenizer for the duration of the actor.
            //
            // `extraEOSTokens` is family-specific — see
            // `LLMModelFamily.extraEOSTokens` for why Llama 3.1
            // in particular needs all three of its declared stop
            // tokens added explicitly (MLX-LM only honors a
            // single `eos_token` from tokenizer_config.json).
            let configuration = ModelConfiguration(
                directory: modelDirectory,
                extraEOSTokens: family.extraEOSTokens
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
        speakerNames: [String: String],
        mode: SummarizeMode
    ) async throws -> SessionSummary {
        try await load()
        guard container != nil else {
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
            return try await summarizeFast(
                utterances: utterances,
                speakerNames: speakerNames
            )
        case .deep:
            return try await summarizeDeep(
                utterances: utterances,
                speakerNames: speakerNames
            )
        }
    }

    /// Single-pass summary over the trailing
    /// `maxPromptUtterances` window. The historical path —
    /// fastest wall time, drops the prefix of long sessions.
    private func summarizeFast(
        utterances: [UtteranceEstimate],
        speakerNames: [String: String]
    ) async throws -> SessionSummary {
        guard let container else {
            throw SummarizerError.modelNotInstalled
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
            truncatedFromTotal: truncatedFrom,
            family: family
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
        // Capture family in a local so the closure isn't capturing
        // self (the closure runs on a non-isolated thread inside
        // container.perform).
        let family = self.family
        do {
            raw = try await container.perform { context -> String in
                let userInput = UserInput(prompt: prompt)
                let lmInput = try await context.processor.prepare(input: userInput)
                AppLog.app.info(
                    "MLXQwenSummarizer prepared input: \(lmInput.text.tokens.size, privacy: .public) prompt tokens"
                )
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
                // Defensive guard against repetition loops. Qwen3
                // doesn't need this (its instruction tuning is
                // strong enough that 0.2 temperature behaves), but
                // Llama-Swallow's Japanese fine-tune occasionally
                // gets stuck repeating a stanza when given English-
                // language JSON instructions, which on top of the
                // 4096 max-token cap can burn ~3 minutes of decode
                // time on tokens we'd never use. 1.05 is the
                // lightest touch that breaks loops without warping
                // legitimate repeated phrases (speaker names,
                // numeric scores).
                if family == .llama {
                    parameters.repetitionPenalty = 1.05
                }
                // Default prefill step is 512 tokens — packing
                // hundreds of tokens into a single Metal command
                // buffer was tripping the GPU watchdog
                // (`kIOGPUCommandBufferCallbackErrorTimeout` →
                // mlx::core::gpu::check_error SIGABRT) on real
                // iPad hardware. 128 sits comfortably inside the
                // watchdog window (32 sustained-prefill kernels for
                // a 4k-token prompt vs. 64 at the prior step size)
                // and roughly halves prefill wall time on long
                // sessions. Drop back to 64 if the timeout SIGABRT
                // re-appears on any hardware revision.
                parameters.prefillStepSize = 128
                // Timing instrumentation — without per-token log
                // visibility we can't tell whether a slow run is
                // stuck in prefill, running normally but emitting
                // way more tokens than expected, or churning at a
                // bad throughput. Capture the start, the first
                // token's arrival (= end of prefill), and a heartbeat
                // every 64 tokens during decode so the log tells us
                // the shape of the slowness on the field.
                let startTime = Date()
                var firstTokenTime: Date? = nil
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
                    didGenerate: { (tokens: [Int]) -> GenerateDisposition in
                        if firstTokenTime == nil {
                            firstTokenTime = Date()
                            let prefillSec = firstTokenTime!.timeIntervalSince(startTime)
                            let msg = String(
                                format: "MLXQwenSummarizer prefill done in %.2f s; starting decode",
                                prefillSec
                            )
                            AppLog.app.info("\(msg, privacy: .public)")
                        } else if tokens.count.isMultiple(of: 64) {
                            let elapsed = Date().timeIntervalSince(firstTokenTime!)
                            let tps = elapsed > 0 ? Double(tokens.count) / elapsed : 0
                            let msg = String(
                                format: "MLXQwenSummarizer decode: %d tokens in %.1f s (%.1f t/s)",
                                tokens.count, elapsed, tps
                            )
                            AppLog.app.info("\(msg, privacy: .public)")
                        }
                        return Task.isCancelled ? .stop : .more
                    }
                )
                let totalSec = Date().timeIntervalSince(startTime)
                let finishMsg = String(
                    format: "MLXQwenSummarizer generate finished: %d tokens in %.1f s total",
                    result.tokens.count, totalSec
                )
                AppLog.app.info("\(finishMsg, privacy: .public)")
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
        let outputPreview = String(raw.prefix(500))
        AppLog.app.info(
            "MLXQwenSummarizer raw output: \(raw.count, privacy: .public) chars, preview: \(outputPreview, privacy: .public)"
        )

        return try Self.parse(
            raw: raw,
            speakerNames: speakerNames,
            modelIdentifier: modelIdentifier
        )
    }

    // MARK: - Deep (map-reduce) summarization

    /// Map-reduce path: chunk every utterance into windows of
    /// `deepWindowSize`, summarize each window into a compact
    /// intermediate JSON, then merge intermediates into the final
    /// `SessionSummary`. Same loaded weights, same per-call
    /// memory ceiling as `summarizeFast` — wall time scales
    /// linearly with `numChunks`.
    private func summarizeDeep(
        utterances: [UtteranceEstimate],
        speakerNames: [String: String]
    ) async throws -> SessionSummary {
        guard container != nil else {
            throw SummarizerError.modelNotInstalled
        }
        let chunks = stride(from: 0, to: utterances.count, by: Self.deepWindowSize).map {
            offset -> [UtteranceEstimate] in
            let end = min(offset + Self.deepWindowSize, utterances.count)
            return Array(utterances[offset..<end])
        }
        AppLog.app.info(
            "MLXQwenSummarizer deep mode: \(utterances.count, privacy: .public) utterances → \(chunks.count, privacy: .public) windows"
        )
        // Short-circuit when the session fits in one window — no
        // benefit to a merge pass over a single intermediate; the
        // fast path's prompt is already designed for one shot.
        if chunks.count <= 1 {
            return try await summarizeFast(
                utterances: utterances,
                speakerNames: speakerNames
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
                "MLXQwenSummarizer deep window \(idx + 1, privacy: .public)/\(chunks.count, privacy: .public) done (\(intermediate.perSpeaker.count, privacy: .public) speakers)"
            )
        }
        if Task.isCancelled { throw CancellationError() }
        return try await runDeepMerge(
            intermediates: intermediates,
            allUtterances: utterances,
            speakerNames: speakerNames
        )
    }

    /// Run one window pass and parse its compact intermediate.
    /// Discards modality flags whose JSON couldn't decode rather
    /// than failing the whole deep summary — a single bad window
    /// shouldn't sink a 600-utterance session.
    private func runDeepWindow(
        chunk: [UtteranceEstimate],
        speakerNames: [String: String],
        windowIndex: Int,
        totalWindows: Int
    ) async throws -> DeepWindowIntermediate {
        guard let container else {
            throw SummarizerError.modelNotInstalled
        }
        let prompt = Self.buildDeepWindowPrompt(
            utterances: chunk,
            speakerNames: speakerNames,
            windowIndex: windowIndex,
            totalWindows: totalWindows,
            family: family
        )
        let raw = try await Self.runInference(
            container: container,
            prompt: prompt,
            maxTokens: Self.deepWindowOutputTokens,
            family: family,
            label: "MLXQwenSummarizer deep[\(windowIndex + 1)/\(totalWindows)]"
        )
        if Task.isCancelled { throw CancellationError() }
        let windowPreview = String(raw.prefix(300))
        AppLog.app.info(
            "MLXQwenSummarizer deep[\(windowIndex + 1, privacy: .public)/\(totalWindows, privacy: .public)] raw: \(raw.count, privacy: .public) chars, preview: \(windowPreview, privacy: .public)"
        )
        return Self.parseDeepWindow(raw: raw, windowIndex: windowIndex, chunk: chunk)
    }

    /// Run the final merge pass over the per-window intermediates
    /// and parse the result into a `SessionSummary` via the same
    /// tolerant parser the fast path uses.
    private func runDeepMerge(
        intermediates: [DeepWindowIntermediate],
        allUtterances: [UtteranceEstimate],
        speakerNames: [String: String]
    ) async throws -> SessionSummary {
        guard let container else {
            throw SummarizerError.modelNotInstalled
        }
        let prompt = Self.buildDeepMergePrompt(
            intermediates: intermediates,
            allUtterances: allUtterances,
            speakerNames: speakerNames,
            family: family
        )
        let raw = try await Self.runInference(
            container: container,
            prompt: prompt,
            maxTokens: Self.maxOutputTokens,
            family: family,
            label: "MLXQwenSummarizer deep[merge]"
        )
        if Task.isCancelled { throw CancellationError() }
        let mergePreview = String(raw.prefix(500))
        AppLog.app.info(
            "MLXQwenSummarizer deep merge raw output: \(raw.count, privacy: .public) chars, preview: \(mergePreview, privacy: .public)"
        )
        return try Self.parse(
            raw: raw,
            speakerNames: speakerNames,
            modelIdentifier: modelIdentifier
        )
    }

    /// Per-window intermediate. In-memory only — never persisted,
    /// never crosses an actor boundary in user-facing API.
    ///
    /// Several fields are optional even though the prompt asks
    /// for all of them — this lets the truncation-recovery path
    /// produce a decodable struct from a JSON blob that ran out
    /// mid-`perSpeaker` (where `modalityFlags` never got emitted).
    /// The merge prompt re-encodes the struct, so `nil` reads as
    /// "no signal for this field in this window" rather than as
    /// schema corruption.
    private struct DeepWindowIntermediate: Codable {
        let windowIndex: Int
        let timeStart: Double
        let timeEnd: Double
        let topicSnapshot: String
        let moodSnapshot: String
        let perSpeaker: [PerSpeakerNote]
        let modalityFlags: [String]?

        struct PerSpeakerNote: Codable {
            let speakerID: String
            let notes: String
            let dominantMood: String
        }
    }

    /// Inference helper shared by both window passes and merge.
    /// Centralizes the `container.perform`/`GenerateParameters`/
    /// timing-log dance that's identical at every site so the
    /// per-call wrappers stay readable. Static (and `family` /
    /// `label` are passed by value) so the closure capture stays
    /// pure-Sendable — keeps Swift 6 strict concurrency quiet.
    private static func runInference(
        container: ModelContainer,
        prompt: String,
        maxTokens: Int,
        family: LLMModelFamily,
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
                    temperature: 0.2
                )
                if family == .llama {
                    parameters.repetitionPenalty = 1.05
                }
                parameters.prefillStepSize = 128
                let startTime = Date()
                var firstTokenTime: Date? = nil
                let result = try MLXLMCommon.generate(
                    input: lmInput,
                    parameters: parameters,
                    context: context,
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

    /// Tolerant parser for the window intermediate. Strict pass
    /// first, then a recovery pass that cuts the JSON at the last
    /// complete `perSpeaker` entry and synthesizes the closing
    /// braces — same shape as the fast-path summarizer's
    /// `recoverTruncatedSummary`. If both fail (model refused,
    /// emitted prose only, hit token cap before even one perSpeaker
    /// entry closed), fall back to a chunk-derived placeholder so
    /// the merge pass still gets a row for this window — better
    /// to emit "this window: unable to summarize" than skip it
    /// silently or fail the whole deep pass on one bad chunk.
    private static func parseDeepWindow(
        raw: String,
        windowIndex: Int,
        chunk: [UtteranceEstimate]
    ) -> DeepWindowIntermediate {
        let dethought = stripThinkBlocks(raw)
        let stripped = stripCodeFence(dethought)
        let timeStart = chunk.first?.start ?? 0
        let timeEnd = chunk.last?.end ?? 0
        // Strict path
        if let braceStart = stripped.firstIndex(of: "{"),
           let braceEnd = stripped.lastIndex(of: "}"),
           let data = String(stripped[braceStart...braceEnd]).data(using: .utf8),
           let decoded = try? JSONDecoder().decode(
               DeepWindowIntermediate.self,
               from: data
           ) {
            return decoded
        }
        // Recovery path: model hit the token cap mid-perSpeaker.
        if let braceStart = stripped.firstIndex(of: "{"),
           let recovered = recoverTruncatedWindowIntermediate(
               stripped: String(stripped[braceStart...])
           ),
           let data = recovered.data(using: .utf8),
           let decoded = try? JSONDecoder().decode(
               DeepWindowIntermediate.self,
               from: data
           ) {
            AppLog.app.info(
                "MLXQwenSummarizer deep[\(windowIndex + 1, privacy: .public)] recovered truncated intermediate (\(data.count, privacy: .public) bytes, \(decoded.perSpeaker.count, privacy: .public) speakers)"
            )
            return decoded
        }
        AppLog.app.info(
            "MLXQwenSummarizer deep[\(windowIndex + 1, privacy: .public)] failed to decode intermediate, synthesizing placeholder"
        )
        let speakerIDs = chunk.orderedSpeakerIDs
        return DeepWindowIntermediate(
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

    /// Salvage a truncated window-intermediate blob the same way
    /// `recoverTruncatedSummary` salvages a fast-path summary:
    /// walk the `perSpeaker` array forward from its `[`, track
    /// string-literal and brace-nesting state, remember the
    /// position after the most recent top-level object that
    /// closed, then cut + append `]}` to close the array and
    /// outer object. `modalityFlags` is optional on the Codable
    /// struct, so its absence in the recovered prefix is fine.
    private static func recoverTruncatedWindowIntermediate(
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
                    // Array closed — the strict parser should have
                    // handled this; bail so the caller surfaces the
                    // original error path (likely a problem outside
                    // the perSpeaker array, e.g. corrupt fields
                    // before it).
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

    // MARK: - Prompt + parser

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
        truncatedFromTotal: Int?,
        family: LLMModelFamily
    ) -> String {
        let speakers = utterances.orderedSpeakerIDs
        let speakerList = speakers.joined(separator: ", ")
        var lines: [String] = []
        lines.reserveCapacity(utterances.count + 12)
        // Instruction prose is family-localized: Llama-Swallow is
        // a Japanese fine-tune of Llama 3.1 and responds more
        // concisely + reliably when prompted in Japanese; English
        // instructions caused it to drift into long apologetic
        // prose then EOS without emitting JSON. JSON field names
        // (setting/topic/overallMood/perSpeaker) and the
        // aP/tP class labels stay English in BOTH copies — they
        // are parsed back as English keys by `parse(...)` and the
        // class names match the data's enum cases.
        switch family {
        case .qwen:
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
        case .llama:
            lines.append("あなたは複数話者の会話を要約するアナリストです。")
            lines.append("以下のすべての発話を読み、次の4つのフィールドを持つJSONオブジェクトを1つ生成してください：")
            lines.append("  \"setting\" — 会話の状況・場面・レジスタを1文で示してください（例：「友人同士のカジュアルな電話」「就職面接」「教室での議論」）。具体的な場所や機関を創作せず、一般的に保ってください。最初に出力し、以降の内容と一貫させてください。")
            lines.append("  \"topic\" — 会話の主題を1〜2文で記述してください。")
            lines.append("  \"overallMood\" — 推定された状況と一貫した、セッション全体の感情的な雰囲気を1段落で記述してください。")
            lines.append("  \"perSpeaker\" — 配列。次の話者IDごとに1エントリ：\(speakerList)。")
            lines.append("各perSpeakerエントリの形式：{ \"speakerID\": <id>, \"summary\": <1段落>, \"dominantMood\": <短いフレーズ> }。")
            lines.append("各行には次の情報が含まれます：融合ラベル、融合V/A/D（valence/arousal/dominance、0〜1）、および各モダリティの生確率ベクトル：")
            lines.append("  aP = 音響9クラスsoftmax（angry, disgusted, fearful, happy, neutral, other, sad, surprised, unknown）")
            lines.append("  tP = テキスト8クラスPlutchik強度（joy, sadness, anticipation, surprise, anger, fear, disgust, trust）")
            lines.append("これらを使って信頼度を判断し、モダリティの不一致を話者ごとに指摘してください — 例：aPがsadだがtPがjoyである行は言及する価値があります。融合ラベルではこの情報が見えません。")
            lines.append("以下の話者人口統計ブロックに性別が記載されている場合、その話者の代名詞として全体で使用してください — 女性は「she/her」、男性は「he/him」、子供または性別が記載されていない場合は「they/them」。（日本語のように代名詞を省略する言語では無関係です。）")
            lines.append("有効なJSONのみを返し、前後に散文を含めないでください。")
        }
        // Pin the output language to whatever the user picked in
        // iPadOS Settings → Xephon → Language. Independent of the
        // prompt-language switch above — the user might pick
        // English-language output even on the Llama backend.
        lines.append(SummarizerLocale.responseLanguageInstruction)
        // Qwen3 ships with a "thinking" mode that emits a
        // `<think>...</think>` chain-of-thought block before the
        // actual response. The `/no_think` directive disables it
        // for a single turn. Gated on family: Llama 3 doesn't
        // recognize the token and would emit it verbatim.
        if family == .qwen {
            lines.append("/no_think")
        }
        if let total = truncatedFromTotal {
            lines.append("")
            switch family {
            case .qwen:
                lines.append("NOTE: This conversation has \(total) utterances total; only the most recent \(utterances.count) are shown below. Frame the overall mood as the trailing portion of the session, not the whole arc.")
            case .llama:
                lines.append("注意：この会話は全体で\(total)発話ありますが、以下には最新の\(utterances.count)発話のみが表示されています。overallMoodはセッション全体の弧ではなく、後半部分として位置付けてください。")
            }
        }
        // Per-speaker demographic roster, ordered by `speakers` so
        // the model reads it in the same order as the utterance
        // list. Empty string when no row carried age-gender output
        // (model not installed / clips too short to score) — the
        // join skips it cleanly.
        let demographics = SpeakerDemographicsDigest
            .build(from: utterances)
            .renderForPrompt(speakerIDs: speakers, speakerNames: speakerNames)
        if !demographics.isEmpty {
            lines.append("")
            lines.append(demographics)
        }
        lines.append("")
        switch family {
        case .qwen:  lines.append("Utterances:")
        case .llama: lines.append("発話：")
        }
        for u in utterances {
            lines.append(compactLine(for: u, speakerNames: speakerNames))
        }
        // Instruction sandwich — restate the directive right
        // before generation so the model's recent attention has
        // the "produce JSON" cue, not the last utterance's
        // `- speaker=…` line. Without this, smaller models on
        // 15k-token prompts (Llama-Swallow especially) treat the
        // prompt as text to continue and emit fabricated
        // utterance-format lines as "output," wasting the full
        // 4096-token budget on garbage.
        lines.append("")
        lines.append("---")
        switch family {
        case .qwen:
            lines.append("IMPORTANT: Follow the instructions above and produce exactly one valid JSON object with fields setting, topic, overallMood, perSpeaker. The FIRST character of your output MUST be `{`. Do NOT echo the utterance list above; do NOT add any prose.")
        case .llama:
            lines.append("重要：上記の指示に従い、setting、topic、overallMood、perSpeaker の4フィールドを持つ有効なJSONオブジェクトを1つだけ生成してください。出力の最初の文字は必ず `{` でなければなりません。上記の発話リストをエコーしないでください。散文も一切含めないでください。")
        }
        return lines.joined(separator: "\n")
    }

    /// Build the per-window prompt for `.deep` mode. Asks the
    /// model for a compact intermediate JSON object (NOT a full
    /// `SessionSummary`) — windowIndex, time bounds, topic +
    /// mood snapshot for this slice, per-speaker notes, and any
    /// notable modality-disagreement flags. The merge pass
    /// consumes these intermediates to produce the final
    /// structured summary.
    private static func buildDeepWindowPrompt(
        utterances: [UtteranceEstimate],
        speakerNames: [String: String],
        windowIndex: Int,
        totalWindows: Int,
        family: LLMModelFamily
    ) -> String {
        let speakers = utterances.orderedSpeakerIDs
        let speakerList = speakers.joined(separator: ", ")
        let timeStart = utterances.first?.start ?? 0
        let timeEnd = utterances.last?.end ?? 0
        let tStartStr = String(format: "%.1f", timeStart)
        let tEndStr = String(format: "%.1f", timeEnd)
        var lines: [String] = []
        lines.reserveCapacity(utterances.count + 18)
        switch family {
        case .qwen:
            lines.append("You are an analyst summarizing ONE WINDOW of a longer multi-speaker conversation.")
            lines.append("This is window \(windowIndex + 1) of \(totalWindows), covering utterances from t=\(tStartStr)s to t=\(tEndStr)s.")
            lines.append("Produce a JSON object — NOT the final summary, just a compact intermediate that a merge pass will combine with the other windows.")
            lines.append("Fields:")
            lines.append("  \"windowIndex\": \(windowIndex) (copy this number),")
            lines.append("  \"timeStart\": \(tStartStr),")
            lines.append("  \"timeEnd\": \(tEndStr),")
            lines.append("  \"topicSnapshot\": one short phrase on this window's topic,")
            lines.append("  \"moodSnapshot\": one short phrase on this window's emotional tone,")
            lines.append("  \"perSpeaker\": array of { \"speakerID\": <id>, \"notes\": one or two sentences on this speaker's contribution in this window, \"dominantMood\": short phrase }, one entry per speaker in: \(speakerList),")
            lines.append("  \"modalityFlags\": array of short strings, each flagging one row where the acoustic and text classifiers notably disagreed (e.g. \"S03 at 42.3s: acoustic=sad, text=joy\"). Empty array if no notable disagreement.")
            lines.append("Each row carries: fused label and fused V/A/D (valence/arousal/dominance, 0–1), plus the raw per-modality probability vectors:")
            lines.append("  aP = acoustic 9-class softmax (angry, disgusted, fearful, happy, neutral, other, sad, surprised, unknown)")
            lines.append("  tP = text 8-class Plutchik intensity (joy, sadness, anticipation, surprise, anger, fear, disgust, trust)")
            lines.append("Return ONLY valid JSON, no prose before or after.")
        case .llama:
            lines.append("あなたは長い複数話者の会話のうち1つのウィンドウを要約するアナリストです。")
            lines.append("これはウィンドウ\(windowIndex + 1)/\(totalWindows)で、t=\(tStartStr)秒からt=\(tEndStr)秒の発話を対象とします。")
            lines.append("JSONオブジェクトを1つ生成してください — 最終要約ではなく、マージパスが他のウィンドウと結合する簡潔な中間データです。")
            lines.append("フィールド：")
            lines.append("  \"windowIndex\": \(windowIndex)（この数字をコピー）、")
            lines.append("  \"timeStart\": \(tStartStr)、")
            lines.append("  \"timeEnd\": \(tEndStr)、")
            lines.append("  \"topicSnapshot\": このウィンドウの話題を短いフレーズで、")
            lines.append("  \"moodSnapshot\": このウィンドウの感情的な雰囲気を短いフレーズで、")
            lines.append("  \"perSpeaker\": { \"speakerID\": <id>, \"notes\": この話者の本ウィンドウでの貢献を1〜2文で, \"dominantMood\": 短いフレーズ } の配列。次の話者ごとに1エントリ：\(speakerList)、")
            lines.append("  \"modalityFlags\": 音響とテキストの分類器が顕著に不一致だった行を示す短い文字列の配列（例：「S03 at 42.3s: acoustic=sad, text=joy」）。顕著な不一致がない場合は空配列。")
            lines.append("各行には次の情報が含まれます：融合ラベル、融合V/A/D（valence/arousal/dominance、0〜1）、および各モダリティの生確率ベクトル：")
            lines.append("  aP = 音響9クラスsoftmax（angry, disgusted, fearful, happy, neutral, other, sad, surprised, unknown）")
            lines.append("  tP = テキスト8クラスPlutchik強度（joy, sadness, anticipation, surprise, anger, fear, disgust, trust）")
            lines.append("有効なJSONのみを返し、前後に散文を含めないでください。")
        }
        lines.append(SummarizerLocale.responseLanguageInstruction)
        if family == .qwen {
            lines.append("/no_think")
        }
        let demographics = SpeakerDemographicsDigest
            .build(from: utterances)
            .renderForPrompt(speakerIDs: speakers, speakerNames: speakerNames)
        if !demographics.isEmpty {
            lines.append("")
            lines.append(demographics)
        }
        lines.append("")
        switch family {
        case .qwen:  lines.append("Utterances:")
        case .llama: lines.append("発話：")
        }
        for u in utterances {
            lines.append(compactLine(for: u, speakerNames: speakerNames))
        }
        // Instruction sandwich — see `buildPrompt` for rationale.
        lines.append("")
        lines.append("---")
        switch family {
        case .qwen:
            lines.append("IMPORTANT: Follow the instructions above and produce exactly one window-intermediate JSON object with fields windowIndex, timeStart, timeEnd, topicSnapshot, moodSnapshot, perSpeaker, modalityFlags. The FIRST character of your output MUST be `{`. Do NOT echo the utterance list above; do NOT add any prose.")
        case .llama:
            lines.append("重要：上記の指示に従い、windowIndex、timeStart、timeEnd、topicSnapshot、moodSnapshot、perSpeaker、modalityFlags のフィールドを持つウィンドウ中間JSONオブジェクトを1つだけ生成してください。出力の最初の文字は必ず `{` でなければなりません。上記の発話リストをエコーしないでください。散文も一切含めないでください。")
        }
        return lines.joined(separator: "\n")
    }

    /// Build the merge prompt that fuses every window's
    /// intermediate into a single `SessionSummary`. The output
    /// schema MUST match the fast path's schema so the same
    /// `parse(raw:speakerNames:modelIdentifier:)` decoder works.
    /// Input here is small (N windows × ~400-token intermediate)
    /// even on very long sessions, so we can fit every window's
    /// notes alongside a roster of every speaker that ever spoke.
    private static func buildDeepMergePrompt(
        intermediates: [DeepWindowIntermediate],
        allUtterances: [UtteranceEstimate],
        speakerNames: [String: String],
        family: LLMModelFamily
    ) -> String {
        let allSpeakers = allUtterances.orderedSpeakerIDs
        let speakerList = allSpeakers.joined(separator: ", ")
        var lines: [String] = []
        lines.reserveCapacity(intermediates.count + 24)
        switch family {
        case .qwen:
            lines.append("You are an analyst producing the FINAL summary of a multi-speaker conversation.")
            lines.append("Below are JSON summaries of \(intermediates.count) consecutive windows of the conversation (in chronological order). The conversation has \(allUtterances.count) utterances total.")
            lines.append("Synthesize them into a single structured summary. Each speaker should be treated as one person across windows — do not split a speaker's arc into per-window sections.")
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
        case .llama:
            lines.append("あなたは複数話者の会話の最終要約を作成するアナリストです。")
            lines.append("以下は会話の\(intermediates.count)個の連続するウィンドウのJSON要約（時系列順）です。会話には合計\(allUtterances.count)発話があります。")
            lines.append("これらを1つの構造化された要約に統合してください。各話者はウィンドウを跨いで1人の人物として扱い、話者の弧をウィンドウごとに分割しないでください。")
            lines.append("")
            lines.append("次の4つのフィールドを持つJSONオブジェクトを1つ生成してください：")
            lines.append("  \"setting\" — 会話の状況・場面・レジスタを1文で示してください（例：「友人同士のカジュアルな電話」「就職面接」「教室での議論」）。具体的な場所や機関を創作せず、一般的に保ってください。最初に出力し、以降の内容と一貫させてください。")
            lines.append("  \"topic\" — 会話の主題を1〜2文で（すべてのウィンドウの話題スナップショットを考慮）。")
            lines.append("  \"overallMood\" — セッション全体の感情的な雰囲気を1段落で — 末尾のウィンドウだけでなく、弧全体を記述してください。")
            lines.append("  \"perSpeaker\" — 配列、次の話者IDごとに1エントリ：\(speakerList)。")
            lines.append("各perSpeakerエントリの形式：{ \"speakerID\": <id>, \"summary\": <セッション全体にわたる1段落>, \"dominantMood\": <短いフレーズ> }。")
            lines.append("ウィンドウ中間データには \"modalityFlags\" 配列が含まれています — 反復的または注目すべき音響↔テキストの不一致を話者ごとの記述で取り上げてください。すべてのフラグを列挙する必要はありません。")
            lines.append("以下の話者人口統計ブロックに性別が記載されている場合、その話者の代名詞として全体で使用してください — 女性は「she/her」、男性は「he/him」、子供または性別が記載されていない場合は「they/them」。（日本語のように代名詞を省略する言語では無関係です。）")
            lines.append("有効なJSONのみを返し、前後に散文を含めないでください。")
        }
        lines.append(SummarizerLocale.responseLanguageInstruction)
        if family == .qwen {
            lines.append("/no_think")
        }
        let demographics = SpeakerDemographicsDigest
            .build(from: allUtterances)
            .renderForPrompt(speakerIDs: allSpeakers, speakerNames: speakerNames)
        if !demographics.isEmpty {
            lines.append("")
            lines.append(demographics)
        }
        lines.append("")
        switch family {
        case .qwen:  lines.append("Window summaries (each is a JSON object):")
        case .llama: lines.append("ウィンドウ要約（各JSONオブジェクト）：")
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.withoutEscapingSlashes]
        for intermediate in intermediates {
            if let data = try? encoder.encode(intermediate),
               let json = String(data: data, encoding: .utf8) {
                lines.append(json)
            }
        }
        // Instruction sandwich — see `buildPrompt` for rationale.
        lines.append("")
        lines.append("---")
        switch family {
        case .qwen:
            lines.append("IMPORTANT: Follow the instructions above and produce exactly one final-summary JSON object with fields setting, topic, overallMood, perSpeaker. The FIRST character of your output MUST be `{`. Do NOT echo the window summaries above; do NOT add any prose.")
        case .llama:
            lines.append("重要：上記の指示に従い、setting、topic、overallMood、perSpeaker の4フィールドを持つ最終要約JSONオブジェクトを1つだけ生成してください。出力の最初の文字は必ず `{` でなければなりません。上記のウィンドウ要約をエコーしないでください。散文も一切含めないでください。")
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

    /// Decode the LLM's JSON output. Tolerates `<think>` blocks
    /// (defensive even though `/no_think` is in the prompt),
    /// ```fence``` wrappers, AND truncated output — when the model
    /// ran past the output-token cap mid-`perSpeaker` array we
    /// recover the per-speaker entries it managed to complete
    /// rather than throwing the whole summary away.
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
        guard let braceStart = stripped.firstIndex(of: "{") else {
            throw SummarizerError.decodeFailed(reason: "no JSON object found")
        }
        struct Wire: Decodable {
            struct PerSpeaker: Decodable {
                let speakerID: String
                let summary: String
                let dominantMood: String
            }
            // Optional so a model that legitimately couldn't infer a
            // setting (or an older prompt that didn't ask for one)
            // decodes cleanly instead of failing the whole summary.
            let setting: String?
            let topic: String
            let overallMood: String
            let perSpeaker: [PerSpeaker]
        }
        // Strict path: the substring between the first `{` and the
        // last `}` parses as our schema whenever the model closed
        // the JSON cleanly.
        let strict: String? = {
            guard let braceEnd = stripped.lastIndex(of: "}") else { return nil }
            return String(stripped[braceStart...braceEnd])
        }()
        let decoded: Wire
        if let strict, let data = strict.data(using: .utf8),
           let ok = try? JSONDecoder().decode(Wire.self, from: data) {
            decoded = ok
        } else {
            // Truncated output: walk the `perSpeaker` array forward
            // from its `[`, find the last balanced entry object, and
            // close the array + outer object manually. The in-flight
            // (broken) entry is discarded; every complete one — plus
            // `topic` and `overallMood` from earlier in the buffer —
            // is preserved.
            guard let recovered = recoverTruncatedSummary(
                stripped: String(stripped[braceStart...])
            ) else {
                throw SummarizerError.decodeFailed(reason: "no JSON object found")
            }
            guard let data = recovered.data(using: .utf8) else {
                throw SummarizerError.decodeFailed(reason: "non-utf8 output")
            }
            AppLog.app.info(
                "MLXQwenSummarizer recovered truncated JSON (\(data.count, privacy: .public) bytes)"
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
        return SessionSummary(
            inferredSetting: decoded.setting?
                .trimmingCharacters(in: .whitespacesAndNewlines),
            topic: decoded.topic,
            overallMood: decoded.overallMood,
            perSpeaker: perSpeaker,
            model: modelIdentifier,
            generatedAt: Date()
        )
    }

    /// Salvage a truncated `{"topic":..., "overallMood":...,
    /// "perSpeaker":[...]}` blob by scanning the `perSpeaker` array,
    /// tracking string-literal and brace-nesting state, and
    /// remembering the position immediately after the *most recent
    /// top-level object that closed* inside the array. When the scan
    /// hits end-of-input mid-entry, we cut at that remembered
    /// position and synthesize `]}` to close the array and outer
    /// object. `setting`, `topic`, and `overallMood` appear before
    /// `perSpeaker` in the prompt's schema description, so the
    /// model emits them first and they survive intact inside the
    /// prefix we keep.
    ///
    /// Returns nil when the input doesn't look like our expected
    /// shape — let the caller surface the original parse error in
    /// that case rather than silently returning an empty summary.
    private static func recoverTruncatedSummary(stripped: String) -> String? {
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
                    // Array closed normally — the strict parser
                    // should have handled this; bail so the caller
                    // surfaces the original error.
                    return nil
                default:
                    break
                }
            }
            i = stripped.index(after: i)
        }

        // Cap the prefix at the end of the last complete entry; if
        // no entry completed, cap at the array opener so we still
        // emit a valid empty-array summary that carries topic and
        // overallMood.
        let cutEnd = lastCompleteEntryEnd ?? arrayOpenRange.upperBound
        var truncated = String(stripped[..<cutEnd])
        while let last = truncated.last,
              last == "," || last.isWhitespace {
            truncated.removeLast()
        }
        truncated.append("]}")
        return truncated
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
