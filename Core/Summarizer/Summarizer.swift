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
    func summarize(
        utterances: [UtteranceEstimate],
        speakerNames: [String: String]
    ) async throws -> SessionSummary
}
