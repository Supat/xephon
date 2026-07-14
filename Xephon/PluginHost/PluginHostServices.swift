import Foundation
import UniformTypeIdentifiers
import Export
import XephonLogging
import XephonPluginKit

/// The app-side implementation of `PluginHost` — the ONLY place
/// that knows both the plugin API and `RecordingController`. Keep
/// it snapshot-shaped (docs/plugin_architecture.md §8, "facade
/// drift"): resist adding one-off accessors per plugin request.
@MainActor
final class PluginHostServices: PluginHost {
    /// Unowned: the process-wide controller (owned by XephonApp)
    /// never outlives this object.
    private unowned let recorder: RecordingController
    /// Strong: the coordinator is a plain state object with no
    /// back-references, and plugin exports must keep working even
    /// while ContentView is mid-reconstruction.
    private let filePicker: FilePickerCoordinator
    private let inferenceAdapter = PluginInferencePlaceholder()

    init(recorder: RecordingController, filePicker: FilePickerCoordinator) {
        self.recorder = recorder
        self.filePicker = filePicker
    }

    var session: any SessionReading { self }
    var inference: any InferenceService { inferenceAdapter }
    var export: any ExportPresenting { self }

    func storage(for plugin: any XephonPlugin.Type) -> any PluginStorage {
        PluginStorageAdapter(
            store: recorder.pluginPayloads,
            id: plugin.id,
            writeVersion: plugin.payloadVersion
        )
    }

    func requestPlayback(utteranceID: UUID) {
        guard let utterance = recorder.utterances.first(where: { $0.id == utteranceID }) else {
            AppLog.app.warning(
                "plugin requestPlayback: unknown utterance \(utteranceID, privacy: .public)"
            )
            return
        }
        recorder.togglePlayback(for: utterance)
    }
}

extension PluginHostServices: ExportPresenting {
    func presentExport(
        data: Data,
        contentType: UTType,
        suggestedFilename: String,
        completion: @escaping @MainActor (PluginExportOutcome) -> Void
    ) {
        filePicker.presentExport(
            data: data,
            contentType: contentType,
            defaultFilename: suggestedFilename
        ) { result in
            switch result {
            case .success:
                completion(.saved)
            case .failure(let error):
                // The root exporter reports user cancellation as a
                // failure; distinguish the one signal Cocoa gives us
                // and treat everything else as a real failure.
                if let cocoa = error as? CocoaError, cocoa.code == .userCancelled {
                    completion(.cancelled)
                } else {
                    completion(.failed(reason: String(describing: error)))
                }
            }
        }
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
