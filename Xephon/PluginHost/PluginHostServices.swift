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
    private let inferenceAdapter: PluginInferenceAdapter

    init(recorder: RecordingController, filePicker: FilePickerCoordinator) {
        self.recorder = recorder
        self.filePicker = filePicker
        self.inferenceAdapter = PluginInferenceAdapter(recorder: recorder)
    }

    var session: any SessionReading { self }
    var inference: any InferenceService { inferenceAdapter }
    var export: any ExportPresenting { self }
    var imports: any ImportPresenting { self }
    var annotations: any SessionAnnotating { self }

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

extension PluginHostServices: ImportPresenting {
    func presentImport(
        contentTypes: [UTType],
        completion: @escaping @MainActor (PluginImportOutcome) -> Void
    ) {
        filePicker.presentImport(allowedTypes: contentTypes) { result in
            switch result {
            case .success(let url):
                // Scope + read stay host-side: plugins get bytes.
                let scoped = url.startAccessingSecurityScopedResource()
                defer { if scoped { url.stopAccessingSecurityScopedResource() } }
                do {
                    completion(.loaded(try Data(contentsOf: url)))
                } catch {
                    completion(.failed(reason: String(describing: error)))
                }
            case .failure(let error):
                if let cocoa = error as? CocoaError, cocoa.code == .userCancelled {
                    completion(.cancelled)
                } else {
                    completion(.failed(reason: String(describing: error)))
                }
            }
        }
    }
}

extension PluginHostServices: SessionAnnotating {
    func proposeSections(_ proposals: [PluginSectionProposal]) -> Int {
        let existingTitles = Set(recorder.sections.sections.map(\.title))
        let liveIDs = Set(recorder.utterances.map(\.id))
        var added = 0
        for proposal in proposals {
            guard !existingTitles.contains(proposal.title),
                  liveIDs.contains(proposal.startUtteranceID),
                  liveIDs.contains(proposal.endUtteranceID)
            else { continue }
            recorder.sections.add(ConversationSection(
                id: UUID(),
                title: proposal.title,
                startUtteranceID: proposal.startUtteranceID,
                endUtteranceID: proposal.endUtteranceID
            ))
            added += 1
        }
        if added > 0 {
            AppLog.app.info(
                "plugin section proposals: +\(added, privacy: .public) of \(proposals.count, privacy: .public)"
            )
        }
        return added
    }

    func contributeKeywords(_ seeds: [PluginKeywordSeed], groupName: String) {
        let store = recorder.keywords
        let existing = Set(store.keywords.map {
            $0.text.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        })
        let fresh = seeds.filter {
            let t = $0.text.trimmingCharacters(in: .whitespacesAndNewlines)
            return !t.isEmpty && !existing.contains(t.lowercased())
        }
        guard !fresh.isEmpty else { return }
        // Reuse the group when a prior seeding (or the user)
        // already created it; only make a new one when absent.
        let groupID = store.groups.first(where: { $0.name == groupName })?.id
            ?? store.addGroup(name: groupName)
        for seed in fresh {
            store.add(seed.text, groupID: groupID)
        }
        AppLog.app.info(
            "plugin keyword seeding: +\(fresh.count, privacy: .public) into \(groupName, privacy: .public)"
        )
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

/// The Phase 2 inference carve-out: routes plugin generation
/// through `SummarizerCoordinator.pluginGenerate`, which shares
/// the built-in runs' inference gate and memory lifecycle
/// (pipeline release → generate → unload + rewarm). Maps the
/// coordinator's typed refusals onto `PluginInferenceError`.
@MainActor
final class PluginInferenceAdapter: InferenceService {
    private unowned let recorder: RecordingController

    init(recorder: RecordingController) {
        self.recorder = recorder
    }

    var availability: InferenceAvailability {
        guard recorder.summarizer.enabled else {
            return .unavailable(reason: "Summarizer is disabled in Settings.")
        }
        guard recorder.summarizer.ready else {
            return .unavailable(
                reason: "The selected summarizer backend isn't ready (model not installed / unavailable)."
            )
        }
        return .available
    }

    func generate(
        prompt: String,
        schemaJSON: String?,
        maxOutputTokens: Int
    ) async throws -> String {
        do {
            return try await recorder.summarizer.pluginGenerate(
                prompt: prompt,
                schemaJSON: schemaJSON,
                maxOutputTokens: maxOutputTokens
            )
        } catch let refusal as SummarizerCoordinator.PluginGenerateError {
            throw PluginInferenceError.unavailable(reason: refusal.description)
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            throw PluginInferenceError.generationFailed(
                reason: String(describing: error)
            )
        }
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

    // Cross-session tier: defaults-backed, namespaced per plugin.
    // Small payloads only (template packs, settings) — documented
    // on the protocol.

    private func persistentKey(_ key: String) -> String {
        "plugin.data.\(id.rawValue).\(key)"
    }

    func persistentData(forKey key: String) -> Data? {
        UserDefaults.standard.data(forKey: persistentKey(key))
    }

    func setPersistentData(_ data: Data?, forKey key: String) {
        if let data {
            UserDefaults.standard.set(data, forKey: persistentKey(key))
        } else {
            UserDefaults.standard.removeObject(forKey: persistentKey(key))
        }
    }
}
