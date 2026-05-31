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

/// How a `SessionSummarizer` should pick utterances and run
/// inference. Persisted via UserDefaults under
/// `xephon.summarizerMode` so the user's pick survives
/// app relaunches.
public enum SummarizeMode: String, Sendable, Hashable, Codable, CaseIterable {
    /// Single pass over a TRAILING window. Implementations
    /// truncate to whatever fits comfortably in their context
    /// budget (100 for MLX, 15 for Apple FM), keeping the most
    /// recent utterances. Wall time on the order of minutes;
    /// memory: just the model + one prompt's KV cache. Default
    /// — matches the historical behavior. Best when the
    /// trailing edge of the conversation carries the most
    /// actionable arc (typical).
    ///
    /// New persisted value is `"trailing"` (matching the case
    /// name). The custom `init?(rawValue:)` below also accepts
    /// the legacy `"fast"` string so existing UserDefaults
    /// preferences, `.xph` bundles, and JSON exports decode
    /// cleanly; new encodes write `"trailing"`.
    case trailing
    /// Single pass over a HEURISTICALLY-SELECTED window —
    /// same context budget as `.trailing`, but the selection is
    /// `Informativeness.topN` instead of `suffix(...)`. Picks
    /// the N most distinctive utterances by session-relative
    /// TF-IDF, with a Japanese backchannel/filler penalty
    /// down-weighting "うん"/"そう"/"えーと"-heavy rows. Same
    /// wall time as `.trailing`; better content coverage on long
    /// sessions where the trailing window would drop a
    /// meaningful prefix. Tradeoff vs `.deep`: same speed as
    /// `.trailing` (one inference pass), but the LLM only ever
    /// sees a curated subset, not every utterance.
    case heuristic
    /// Map-reduce across EVERY utterance. Implementations
    /// split the session into windows, summarize each into a
    /// compact intermediate, then merge intermediates into the
    /// final `SessionSummary`. Wall time scales with session
    /// length (roughly `numChunks × per-chunk inference + one
    /// merge pass`); peak memory matches `.trailing` because only
    /// one chunk is in the KV cache at any time. Chosen when
    /// the user values complete coverage over latency.
    case deep

    /// Custom rawValue initializer for backward compatibility.
    /// Accepts the legacy `"fast"` string (used before the case
    /// was renamed from `.fast` to `.trailing`) and maps it to
    /// `.trailing`. New writes go through the default rawValue
    /// path and emit `"trailing"`, so over time persisted state
    /// migrates forward without an explicit migration step.
    public init?(rawValue: String) {
        switch rawValue {
        case "trailing":  self = .trailing
        case "heuristic": self = .heuristic
        case "deep":      self = .deep
        case "fast":      self = .trailing   // legacy
        default:          return nil
        }
    }
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
    /// path may treat `.deep` as `.trailing` (the protocol contract
    /// is "best-effort honor"); the MLX-backed implementation is
    /// the load-bearing one for long-session deep summaries.
    ///
    /// `boostedUtteranceIDs` — IDs the caller has flagged as
    /// containing user-curated keywords (matched per the
    /// caller's preferred normalizer). Only consulted by the
    /// `.heuristic` mode's `Informativeness` ranker, which
    /// applies a heavy multiplicative boost so these rows
    /// reliably land in the heuristic-selected prompt window.
    /// `.trailing` and `.deep` ignore the set: `.trailing` always picks
    /// the trailing window regardless of content, and `.deep`
    /// processes every utterance so no selection bias applies.
    /// Default empty (no boost).
    func summarize(
        utterances: [UtteranceEstimate],
        speakerNames: [String: String],
        mode: SummarizeMode,
        boostedUtteranceIDs: Set<UUID>
    ) async throws -> SessionSummary
}

extension SessionSummarizer {
    /// Convenience overload defaulting to `.trailing` with no
    /// keyword boost. Keeps existing call sites (tests, the
    /// reviewer's parallel `review(...)` pathway, etc.)
    /// compiling unchanged; new callers that care about the
    /// mode / keyword boost opt in explicitly.
    public func summarize(
        utterances: [UtteranceEstimate],
        speakerNames: [String: String]
    ) async throws -> SessionSummary {
        try await summarize(
            utterances: utterances,
            speakerNames: speakerNames,
            mode: .trailing,
            boostedUtteranceIDs: []
        )
    }

    /// Convenience overload that omits the keyword-boost set
    /// (callers without curated keywords). Forwards an empty
    /// set so backends always see the canonical four-arg
    /// signature.
    public func summarize(
        utterances: [UtteranceEstimate],
        speakerNames: [String: String],
        mode: SummarizeMode
    ) async throws -> SessionSummary {
        try await summarize(
            utterances: utterances,
            speakerNames: speakerNames,
            mode: mode,
            boostedUtteranceIDs: []
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
}
