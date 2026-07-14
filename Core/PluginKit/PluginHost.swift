import Foundation
import Fusion

/// Capability services the host hands to a plugin. Each capability
/// is its own protocol so tests can stub them individually and a
/// future scripted tier can gate them per plugin.
@MainActor
public protocol PluginHost: AnyObject {
    /// Read-only session access.
    var session: any SessionReading { get }
    /// Schema-constrained LLM generation routed through the user's
    /// configured summarizer backend.
    var inference: any InferenceService { get }
    /// Per-plugin persistent storage. Pass the plugin's own type
    /// (`Self.self`) — the host stamps writes with that plugin's
    /// `payloadVersion` and namespaces by its `id`.
    func storage(for plugin: any XephonPlugin.Type) -> any PluginStorage
}

// MARK: - Session reading

/// Read-only, snapshot-shaped view of the current session. No live
/// controller types cross this boundary — a snapshot is value data,
/// taken at call time, immutable afterwards.
@MainActor
public protocol SessionReading: AnyObject {
    func snapshot() -> SessionSnapshot
}

/// Value snapshot of the session state plugins may read.
public struct SessionSnapshot: Sendable {
    /// Changes when a new session replaces the current one — the
    /// same invalidation signal the app's own views key on.
    public let sessionToken: UUID
    /// User-supplied session title ("" when unnamed).
    public let title: String
    /// The per-utterance analysis stream, in chronological order.
    public let utterances: [UtteranceEstimate]
    /// Monotonic mutation counter for `utterances` — matches the
    /// `version` carried by `SessionEvent.utterancesChanged`.
    public let utterancesVersion: Int
    /// User-assigned display names keyed by speaker id ("S01" → …).
    public let speakerNames: [String: String]

    public init(
        sessionToken: UUID,
        title: String,
        utterances: [UtteranceEstimate],
        utterancesVersion: Int,
        speakerNames: [String: String]
    ) {
        self.sessionToken = sessionToken
        self.title = title
        self.utterances = utterances
        self.utterancesVersion = utterancesVersion
        self.speakerNames = speakerNames
    }
}

// MARK: - Inference

/// Schema-constrained text generation. Plugins never link an ML
/// runtime or open a network connection themselves — routing every
/// generation through the host is what makes the app's privacy
/// rules (on-device default, cloud only behind explicit consent)
/// structural rather than reviewed-for.
public protocol InferenceService: Sendable {
    /// Whether generation is currently possible (backend configured,
    /// model installed). Check before offering LLM-driven features.
    @MainActor var availability: InferenceAvailability { get }

    /// Generate raw model output for `prompt`. When `schemaJSON`
    /// (a JSON Schema document) is provided, the host applies the
    /// strongest enforcement the active backend supports —
    /// guaranteed constrained decoding where available, prompt-
    /// contract elsewhere; callers must still parse defensively.
    func generate(
        prompt: String,
        schemaJSON: String?,
        maxOutputTokens: Int
    ) async throws -> String
}

public enum InferenceAvailability: Sendable, Equatable {
    case available
    case unavailable(reason: String)
}

public enum PluginInferenceError: Error, Sendable, CustomStringConvertible {
    case unavailable(reason: String)
    case generationFailed(reason: String)

    public var description: String {
        switch self {
        case .unavailable(let r):      return "Inference unavailable: \(r)"
        case .generationFailed(let r): return "Generation failed: \(r)"
        }
    }
}

// MARK: - Storage

/// Per-plugin persistence. The session payload rides inside the
/// `.xph` bundle (namespaced by plugin id, stamped with the
/// plugin's `payloadVersion`) and is restored on session load;
/// payloads written by plugins not installed in the current build
/// are preserved verbatim through load → save.
@MainActor
public protocol PluginStorage: AnyObject {
    /// This plugin's payload for the CURRENT session. Setting nil
    /// removes it. Cleared by the host when the session resets.
    var sessionPayloadData: Data? { get set }
    /// `payloadVersion` the currently-stored payload was written
    /// under — read it on `sessionLoaded` to migrate old payloads.
    /// Nil when no payload is stored.
    var sessionPayloadVersion: Int? { get }
}
