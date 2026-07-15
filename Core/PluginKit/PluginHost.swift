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
    /// File export through the app's root picker.
    var export: any ExportPresenting { get }
    /// File import through the app's root picker.
    var imports: any ImportPresenting { get }
    /// The narrow session-write surface (keyword seeding, section
    /// proposals).
    var annotations: any SessionAnnotating { get }
    /// Per-plugin persistent storage. Pass the plugin's own type
    /// (`Self.self`) — the host stamps writes with that plugin's
    /// `payloadVersion` and namespaces by its `id`.
    func storage(for plugin: any XephonPlugin.Type) -> any PluginStorage
    /// Toggle audio playback of the utterance with `id` — same
    /// behavior as the transcript row's play button (tap to play,
    /// tap again to stop; playing another row stops the first).
    /// No-op when the id doesn't resolve or the session has no
    /// playable audio (mic-mode sessions don't).
    func requestPlayback(utteranceID: UUID)
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

// MARK: - Annotation

/// The narrow write surface plugins get into the session. Every
/// write is user-visible, user-editable state — plugins propose,
/// the user owns.
@MainActor
public protocol SessionAnnotating: AnyObject {
    /// Seed keyword texts into the user's keyword bank under a
    /// group named `groupName` (created if absent). Texts already
    /// present anywhere in the bank are skipped (case-insensitive
    /// on the trimmed form) — the bank is user-owned and seeding
    /// must never duplicate or overwrite. Idempotent, so calling
    /// on every activation is fine.
    func contributeKeywords(_ seeds: [PluginKeywordSeed], groupName: String)

    /// Propose named utterance-range sections (the Sections page).
    /// Proposals whose title already exists in the session are
    /// skipped — sections are user-owned once created, and a
    /// re-detection run must not duplicate or clobber edits.
    /// Returns the number actually added.
    @discardableResult
    func proposeSections(_ proposals: [PluginSectionProposal]) -> Int
}

/// One proposed section. Utterance ids come from the session
/// snapshot; the host validates both ends still exist at proposal
/// time and drops the proposal otherwise.
public struct PluginSectionProposal: Sendable, Hashable {
    public let title: String
    public let startUtteranceID: UUID
    public let endUtteranceID: UUID

    public init(title: String, startUtteranceID: UUID, endUtteranceID: UUID) {
        self.title = title
        self.startUtteranceID = startUtteranceID
        self.endUtteranceID = endUtteranceID
    }
}

/// One keyword a plugin wants in the bank. A neutral type — the
/// app's own `Keyword` (ids, tag colors, selection) stays behind
/// the boundary.
public struct PluginKeywordSeed: Sendable, Hashable {
    public let text: String

    public init(_ text: String) {
        self.text = text
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

    /// Run `body` as one inference batch: hosts that pay a model
    /// load/unload cycle per `generate` keep the model resident for
    /// the whole batch instead (observed on-device: a six-item fill
    /// spent most of its wall clock reloading weights between
    /// calls). Defaults to a plain passthrough for hosts without
    /// lifecycle costs.
    func withBatch<T: Sendable>(
        _ body: @Sendable () async throws -> T
    ) async rethrows -> T
}

extension InferenceService {
    public func withBatch<T: Sendable>(
        _ body: @Sendable () async throws -> T
    ) async rethrows -> T {
        try await body()
    }
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

    /// Cross-session per-plugin storage (imported template packs,
    /// plugin settings). Namespaced by plugin id; survives app
    /// restarts and session resets. Small payloads only — this is
    /// defaults-backed, not a file store.
    func persistentData(forKey key: String) -> Data?
    func setPersistentData(_ data: Data?, forKey key: String)
}
