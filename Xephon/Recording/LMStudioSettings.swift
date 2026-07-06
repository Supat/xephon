import Foundation
import Observation

/// User-configured settings for the LM Studio remote-LLM backend.
///
/// LM Studio runs an OpenAI-compatible HTTP server on the local
/// network (default port 1234). When enabled, the summarizer +
/// transcription-reviewer pick the `.lmStudio` backend route
/// (post-ASR transcript only — no audio) instead of running an
/// MLX model on-device. Per CLAUDE.md's local-first / remote-open
/// posture, the toggle defaults to `false` and the picker option
/// is gated on it.
///
/// Persisted under the `xephon.lmStudio.*` keyspace so the user's
/// host/port/model survives relaunches without having to re-enter
/// every time. The `@Observable` machinery lets the Settings card
/// bind directly to the fields without an intermediate forwarder.
@MainActor
@Observable
final class LMStudioSettings {
    /// Master switch. When `false`, the backend picker hides the
    /// LM Studio option and the coordinator refuses to dispatch
    /// to the adapters even if the persisted backend is
    /// `.lmStudio` (falls back to whatever else is configured).
    var enabled: Bool {
        didSet { schedulePersist(\.enabled, key: Self.enabledKey) }
    }

    /// Hostname or IP address of the LM Studio server. Empty
    /// disables the connect. Defaults to `localhost` because
    /// the most common setup is LM Studio on the same Mac the
    /// user sideloads from; iPad users override. Whitespace
    /// trimming is applied at use time inside `baseURL` rather
    /// than on the setter to avoid mutating the property
    /// inside its own `didSet` (which would re-enter the
    /// `@Observable` `_modify` accessor's exclusive-access
    /// region and trip Swift's runtime exclusivity check).
    var host: String {
        didSet { schedulePersist(\.host, key: Self.hostKey) }
    }

    /// TCP port LM Studio is bound to. 1234 is the documented
    /// default. Stored as Int rather than UInt16 so the SwiftUI
    /// TextField bind doesn't have to round-trip through a
    /// formatter — the validation happens at request time.
    var port: Int {
        didSet { schedulePersist(\.port, key: Self.portKey) }
    }

    /// Model identifier the LM Studio server should route the
    /// request to. Matches what the server returns from
    /// `GET /v1/models` (e.g. `"qwen3-8b-instruct"`). Empty
    /// means "let the server pick its default loaded model" —
    /// LM Studio accepts an empty `model` field and falls back
    /// to whatever is loaded. Whitespace trimming, same as
    /// `host`, lives in the consumer (`LMStudioClient` reads
    /// the value directly) rather than in didSet.
    var modelID: String {
        didSet { schedulePersist(\.modelID, key: Self.modelIDKey) }
    }

    /// Per-request timeout in seconds. Defaults to 300 (5 min)
    /// — `URLRequest.timeoutInterval` measures "time until
    /// additional data arrives", and a non-streaming LM Studio
    /// chat completion holds the connection silent for the
    /// whole generation, so the effective ceiling here is the
    /// total Summary / Reviewer pass duration. Long sessions
    /// on Deep mode against a busy server can take several
    /// minutes; the user can bump the slider up to 30 min for
    /// pathological cases. Test connection uses a separate
    /// 15 s implicit timeout since GET /v1/models is a quick
    /// metadata read.
    var requestTimeoutSeconds: Double {
        didSet { schedulePersist(\.requestTimeoutSeconds, key: Self.timeoutKey) }
    }

    /// When true, requests include OpenAI-format
    /// `response_format: { type: "json_schema", … }` describing
    /// the expected Summary / Reviewer JSON shape. Some LM Studio
    /// backends honor this with real constrained decoding;
    /// others ignore it as a hint. The tolerant parser already
    /// handles freeform output, so this is an opt-in robustness
    /// boost — defaults to off because a server that rejects
    /// unknown response_format types would fail the request
    /// outright.
    var useStructuredOutput: Bool {
        didSet { schedulePersist(\.useStructuredOutput, key: Self.structuredKey) }
    }

    /// Resolved base URL `http://<host>:<port>` for the LM Studio
    /// server, or nil when host is empty or invalid. The client
    /// uses this to build `/v1/chat/completions` and `/v1/models`.
    var baseURL: URL? {
        let trimmed = host.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, port > 0, port < 65_536 else { return nil }
        var components = URLComponents()
        components.scheme = "http"
        components.host = trimmed
        components.port = port
        return components.url
    }

    private static let enabledKey    = "xephon.lmStudio.enabled"
    private static let hostKey       = "xephon.lmStudio.host"
    private static let portKey       = "xephon.lmStudio.port"
    private static let modelIDKey    = "xephon.lmStudio.modelID"
    private static let timeoutKey    = "xephon.lmStudio.timeoutSeconds"
    private static let structuredKey = "xephon.lmStudio.useStructuredOutput"

    init(defaults: UserDefaults = .standard) {
        self.enabled = defaults.bool(forKey: Self.enabledKey)
        self.host = defaults.string(forKey: Self.hostKey) ?? "localhost"
        let storedPort = defaults.integer(forKey: Self.portKey)
        self.port = storedPort > 0 ? storedPort : 1234
        self.modelID = defaults.string(forKey: Self.modelIDKey) ?? ""
        let storedTimeout = defaults.double(forKey: Self.timeoutKey)
        // Existing installs with the old 120 s default get
        // migrated up to 300 s on next launch — 120 s was too
        // tight for real Summary passes (the user-visible bug
        // report behind this change). New installs land at the
        // new default directly.
        if storedTimeout >= 120, storedTimeout <= 120.5 {
            self.requestTimeoutSeconds = 300
        } else if storedTimeout > 0 {
            self.requestTimeoutSeconds = storedTimeout
        } else {
            self.requestTimeoutSeconds = 300
        }
        self.useStructuredOutput = defaults.bool(forKey: Self.structuredKey)
    }

    /// Hop to the next MainActor tick before reading the
    /// property and writing it to UserDefaults. Reading the
    /// property directly from inside `didSet` would re-enter
    /// the `@Observable` `_modify` accessor's exclusive-access
    /// region — same hazard `GlossaryStore.didMutate` documents
    /// at length. The deferred Task runs once the setter has
    /// fully returned and the modify region is released.
    private func schedulePersist<T>(_ keyPath: KeyPath<LMStudioSettings, T>, key: String) {
        Task { @MainActor [weak self] in
            guard let self else { return }
            UserDefaults.standard.set(self[keyPath: keyPath], forKey: key)
            self.onChange?()
        }
    }

    /// Fired on the MainActor (next tick, alongside the persist)
    /// after any setting changes. RecordingController wires this to
    /// the summarizer's auto re-run when LM Studio is the active
    /// backend.
    var onChange: (@MainActor () -> Void)?
}
