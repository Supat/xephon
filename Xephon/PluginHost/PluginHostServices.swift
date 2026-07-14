import Foundation
import Export
import XephonPluginKit

/// The app-side implementation of `PluginHost` — the ONLY place
/// that knows both the plugin API and `RecordingController`. Keep
/// it snapshot-shaped (docs/plugin_architecture.md §8, "facade
/// drift"): resist adding one-off accessors per plugin request.
@MainActor
final class PluginHostServices: PluginHost {
    /// Unowned: XephonApp owns both the controller and this object,
    /// and the controller never outlives the process.
    private unowned let recorder: RecordingController
    private let inferenceAdapter = PluginInferencePlaceholder()

    init(recorder: RecordingController) {
        self.recorder = recorder
    }

    var session: any SessionReading { self }
    var inference: any InferenceService { inferenceAdapter }

    func storage(for plugin: any XephonPlugin.Type) -> any PluginStorage {
        PluginStorageAdapter(
            store: recorder.pluginPayloads,
            id: plugin.id,
            writeVersion: plugin.payloadVersion
        )
    }
}

extension PluginHostServices: SessionReading {
    func snapshot() -> SessionSnapshot {
        SessionSnapshot(
            sessionToken: recorder.sessionToken,
            title: recorder.sessionTitle,
            utterances: recorder.utterances,
            utterancesVersion: recorder.utterancesVersion,
            speakerNames: recorder.speakerNameOverrides
        )
    }
}

/// Phase 0 stand-in for the inference carve-out. The real adapter
/// (schema-in/JSON-out over the configured summarizer backend)
/// lands with its first consumer, EvalFormPlugin — carving
/// SummarizerCoordinator's mode-coupled dispatch without a caller
/// to shape it would be speculative. Honest surface until then:
/// reports unavailable, throws typed.
struct PluginInferencePlaceholder: InferenceService {
    private static let reason =
        "Plugin inference lands with the EvalForm plugin (Phase 2)."

    @MainActor var availability: InferenceAvailability {
        .unavailable(reason: Self.reason)
    }

    func generate(
        prompt: String,
        schemaJSON: String?,
        maxOutputTokens: Int
    ) async throws -> String {
        throw PluginInferenceError.unavailable(reason: Self.reason)
    }
}

// MARK: - Payload storage

/// In-memory session-payload table, owned by RecordingController and
/// bridged into `SessionDocument.pluginPayloads` on save/load. Holds
/// EVERY payload from the loaded bundle — including ids no installed
/// plugin claims — so unknown plugins' data survives load → save
/// verbatim (the contract documented on the SessionDocument field).
@MainActor
final class PluginPayloadStore {
    private(set) var payloads: [String: PluginPayload] = [:]

    /// Adopt a loaded bundle's payload table wholesale.
    func replaceAll(_ new: [String: PluginPayload]) {
        payloads = new
    }

    /// New session — every stored payload referenced the session
    /// that's being discarded.
    func clear() {
        payloads.removeAll()
    }

    /// Empty round-trips as nil so plugin-free sessions stay
    /// byte-identical to pre-plugin bundles.
    var forExport: [String: PluginPayload]? {
        payloads.isEmpty ? nil : payloads
    }

    subscript(id: String) -> PluginPayload? {
        get { payloads[id] }
        set { payloads[id] = newValue }
    }
}

/// Per-plugin view over the shared payload store. Writes are
/// stamped with the owning plugin's `payloadVersion`; reads expose
/// the stored version so the plugin can migrate old payloads.
@MainActor
final class PluginStorageAdapter: PluginStorage {
    private let store: PluginPayloadStore
    private let id: PluginID
    private let writeVersion: Int

    init(store: PluginPayloadStore, id: PluginID, writeVersion: Int) {
        self.store = store
        self.id = id
        self.writeVersion = writeVersion
    }

    var sessionPayloadData: Data? {
        get { store[id.rawValue]?.data }
        set {
            if let newValue {
                store[id.rawValue] = PluginPayload(
                    version: writeVersion,
                    data: newValue
                )
            } else {
                store[id.rawValue] = nil
            }
        }
    }

    var sessionPayloadVersion: Int? {
        store[id.rawValue]?.version
    }
}
