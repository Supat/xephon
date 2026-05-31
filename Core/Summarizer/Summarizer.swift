import Foundation
import Fusion

/// Errors a `SessionSummarizer` can raise.
public enum SummarizerError: Error, CustomStringConvertible {
    /// The model directory exists but couldn't be loaded (corrupted
    /// weights, missing config, format mismatch).
    case modelLoadFailed(reason: String)
    /// Inference failed mid-generation. `reason` carries the
    /// underlying error description for the logs / banner.
    case inferenceFailed(reason: String)
    /// The model emitted text that didn't parse as the structured
    /// `SessionSummary` schema. Rare with JSON-constrained
    /// generation but possible if the LLM refuses or truncates.
    case decodeFailed(reason: String)
    /// The model isn't installed on this device yet. The caller
    /// should surface a prompt asking the user to enable +
    /// download the summarizer model from Settings.
    case modelNotInstalled

    public var description: String {
        switch self {
        case .modelLoadFailed(let r):  return "Summarizer model load failed: \(r)"
        case .inferenceFailed(let r):  return "Summarizer inference failed: \(r)"
        case .decodeFailed(let r):     return "Summarizer output didn't match the expected schema: \(r)"
        case .modelNotInstalled:       return "Summarizer model isn't installed yet"
        }
    }
}

/// How a `SessionSummarizer` should weigh time vs. completeness.
/// Persisted via UserDefaults under `xephon.summarizeMode` so the
/// user's pick survives app relaunches.
public enum SummarizeMode: String, Sendable, Hashable, Codable, CaseIterable {
    /// Single pass over a trailing window. Implementations
    /// truncate to whatever fits comfortably in their context
    /// budget (100 for MLX, 15 for Apple FM). Wall time on the
    /// order of minutes; memory: just the model + one prompt's
    /// KV cache. Default — matches the historical behavior.
    case fast
    /// Map-reduce across every utterance. Implementations split
    /// the session into windows, summarize each into a compact
    /// intermediate, then merge intermediates into the final
    /// `SessionSummary`. Wall time scales with session length
    /// (roughly `numChunks × per-chunk inference + one merge
    /// pass`); peak memory matches `.fast` because only one
    /// chunk is in the KV cache at any time. Chosen when the
    /// user values completeness over latency — long sessions
    /// (200+ utterances) where the trailing-100 window would
    /// drop a meaningful prefix of the conversation.
    case deep
}

/// Abstract interface a session summarizer conforms to. Decouples
/// the consumer (`RecordingController` will eventually call
/// `summarize(_:)` from the "Summarize session" UI action) from the
/// concrete MLX-backed implementation, and lets tests inject a stub
/// that returns a fixed `SessionSummary` without booting an LLM.
///
/// The protocol is async + throwing because real implementations
/// can take tens of seconds and have many failure modes (model not
/// present, OOM, generation interrupted). UI callers must wrap in
/// `Task { … }` and surface progress / error states.
public protocol SessionSummarizer: Sendable {
    /// Identifier for the underlying model, e.g.
    /// `"qwen2.5-7b-instruct-4bit"`. Stamped into the produced
    /// `SessionSummary.model` so external consumers can attribute
    /// the output.
    var modelIdentifier: String { get async }

    /// True iff the model is loaded and ready to summarize. Used
    /// by the UI to decide between "Summarize" and "Download
    /// summarizer model first". Implementations that don't need
    /// hydration (mocks) return `true` unconditionally.
    var isReady: Bool { get async }

    /// Run the model over `utterances` and return a structured
    /// `SessionSummary`. The utterance list is the raw row data
    /// — the implementation is responsible for building its own
    /// compact prompt representation (per the on-device LLM's
    /// token budget) and for filling in the rename map from
    /// `speakerNames`.
    ///
    /// `speakerNames` maps `speakerID → display name` for rows
    /// whose speaker the user has renamed. Implementations should
    /// fold these into the prompt so the LLM uses the friendly
    /// name in its output, and stamp them into
    /// `SessionSummary.perSpeaker.speakerName` directly so the
    /// JSON carries the canonical id + friendly name pair.
    ///
    /// `mode` lets the caller trade wall time for completeness —
    /// see `SummarizeMode`. Backends that don't support a deep
    /// path may treat `.deep` as `.fast` (the protocol contract
    /// is "best-effort honor"); the MLX-backed implementation is
    /// the load-bearing one for long-session deep summaries.
    func summarize(
        utterances: [UtteranceEstimate],
        speakerNames: [String: String],
        mode: SummarizeMode
    ) async throws -> SessionSummary
}

extension SessionSummarizer {
    /// Convenience overload defaulting to `.fast`. Keeps existing
    /// call sites (tests, the reviewer's parallel `review(...)`
    /// pathway, etc.) compiling unchanged; new callers that care
    /// about the mode opt in explicitly.
    public func summarize(
        utterances: [UtteranceEstimate],
        speakerNames: [String: String]
    ) async throws -> SessionSummary {
        try await summarize(
            utterances: utterances,
            speakerNames: speakerNames,
            mode: .fast
        )
    }
}

/// Which on-device backend powers the session summarizer. The
/// user picks one from Settings when the summarizer is enabled.
/// Persisted via UserDefaults under `xephon.summarizerBackend`.
public enum SummarizerBackend: String, Sendable, Hashable, Codable, CaseIterable {
    /// Apple Foundation Models (~3B, iPadOS 26+). Lightweight and
    /// system-managed — already in-memory courtesy of the text
    /// SER's FoundationModelsSER. 4096-token context window, so
    /// long sessions need aggressive truncation. Default backend
    /// because it Just Works once the feature is enabled.
    case appleFM
    /// Qwen3-8B-Instruct (4-bit MLX, ~4.6 GB on disk). Multilingual
    /// generalist with strong JSON adherence; 32k context. Opt-in
    /// download; release the analysis pipeline before invoking to
    /// fit under iOS's per-app memory ceiling.
    case qwen
    /// Llama-3.1-Swallow-8B-Instruct (4-bit MLX, ~4.6 GB on disk).
    /// Tokyo Tech's Japanese fine-tune of Llama 3.1; aimed at
    /// stronger conversational JP at the cost of slightly weaker
    /// JSON discipline vs Qwen3. Same opt-in download + memory
    /// orchestration as `qwen`. Falls under Llama 3 Community
    /// License + tokyotech-llm terms — both permissive for
    /// research use.
    case llamaSwallow
}

/// Which model architecture a `SessionSummarizer` / reviewer is
/// pointed at. Used by the MLX-backed implementations to toggle a
/// few prompt-token differences (e.g. Qwen3's `/no_think` line is
/// literal text for Llama and would corrupt its output). The
/// architectural model loading itself is family-agnostic —
/// `LLMModelFactory` introspects `config.json`'s `model_type` and
/// picks the right MLX module.
public enum LLMModelFamily: String, Sendable, Hashable, Codable, CaseIterable {
    case qwen
    case llama

    /// Additional stop tokens to pass into MLX-LM's
    /// `ModelConfiguration.extraEOSTokens`. MLX-LM only honors a
    /// single `eosTokenId` from `tokenizer_config.json#eos_token`,
    /// but Llama 3.1's `config.json` declares three valid stop
    /// tokens (`<|end_of_text|>`, `<|eom_id|>`, `<|eot_id|>`) and
    /// the Swallow Japanese fine-tune has been observed to emit
    /// the base-model `<|end_of_text|>` to end its turn instead
    /// of the chat-template `<|eot_id|>` — without this set MLX
    /// runs the model all the way to `maxOutputTokens` because
    /// the stop signal it does emit isn't on the recognized list.
    /// Qwen3 stops cleanly on `<|im_end|>` (its tokenizer's
    /// `eos_token`) but we add `<|endoftext|>` defensively in
    /// case a future quant drops the chat template.
    public var extraEOSTokens: Set<String> {
        switch self {
        case .qwen:
            return ["<|endoftext|>"]
        case .llama:
            return ["<|end_of_text|>", "<|eom_id|>", "<|eot_id|>"]
        }
    }
}
