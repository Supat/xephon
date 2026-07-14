import Foundation
import SwiftUI
import Testing
import UniformTypeIdentifiers
@testable import Xephon
import XephonPluginKit
import EvalFormPlugin
import Export

/// Phase 0 exit tests for the plugin architecture
/// (docs/plugin_architecture.md §7): a test-only HelloPlugin
/// activates through the registry, receives session events, and
/// persists a payload through the `.xph` document round-trip —
/// including verbatim preservation of payloads whose plugin isn't
/// installed. The remaining Phase 0 exit criterion (a plugin target
/// cannot import the app target) is enforced by the build graph:
/// XephonPluginKit is an SPM target with no path to the app module.
@Suite("Plugin architecture Phase 0")
@MainActor
struct PluginKitTests {

    // MARK: - Test plugin

    /// Collects what the plugin observed. A @MainActor class so the
    /// Sendable plugin struct can carry a reference to it.
    @MainActor
    final class HelloPluginLog {
        var events: [SessionEvent] = []
        var activations = 0
    }

    struct HelloPlugin: XephonPlugin {
        static let id = PluginID("test.hello")
        static let displayName = "Hello"
        static let payloadVersion = 3

        let log: HelloPluginLog

        func activate(host: any PluginHost) -> PluginHandle {
            log.activations += 1
            // Write a payload immediately so the storage path is
            // exercised by plain activation.
            host.storage(for: Self.self).sessionPayloadData = Data("hello".utf8)
            return PluginHandle(
                onSessionEvent: { [log] event in
                    log.events.append(event)
                },
                pages: [
                    PluginPageDescriptor(
                        id: "test.hello.page",
                        title: "Hello",
                        systemImage: "hand.wave"
                    ) { AnyView(EmptyView()) }
                ],
                menuCommands: [
                    PluginMenuCommand(id: "test.hello.cmd", title: "Hello") {}
                ]
            )
        }
    }

    /// Minimal host: real storage adapters over a real payload
    /// store, canned snapshot, placeholder inference, recording
    /// stubs for export + playback.
    @MainActor
    final class StubPluginHost: PluginHost, SessionReading, ExportPresenting,
                                SessionAnnotating {
        let store: PluginPayloadStore
        private(set) var playbackRequests: [UUID] = []
        private(set) var exportedData: [Data] = []

        init(store: PluginPayloadStore) {
            self.store = store
        }

        var session: any SessionReading { self }
        var inference: any InferenceService { StubInference() }
        var export: any ExportPresenting { self }
        var annotations: any SessionAnnotating { self }
        private(set) var seededKeywords: [String] = []

        func storage(for plugin: any XephonPlugin.Type) -> any PluginStorage {
            PluginStorageAdapter(
                store: store,
                id: plugin.id,
                writeVersion: plugin.payloadVersion
            )
        }

        func requestPlayback(utteranceID: UUID) {
            playbackRequests.append(utteranceID)
        }

        func contributeKeywords(_ seeds: [PluginKeywordSeed], groupName: String) {
            seededKeywords.append(contentsOf: seeds.map(\.text))
        }

        func presentExport(
            data: Data,
            contentType: UTType,
            suggestedFilename: String,
            completion: @escaping @MainActor (PluginExportOutcome) -> Void
        ) {
            exportedData.append(data)
            completion(.saved)
        }

        func snapshot() -> SessionSnapshot {
            SessionSnapshot(
                sessionToken: UUID(),
                title: "",
                utterances: [],
                utterancesVersion: 0,
                speakerNames: [:]
            )
        }
    }

    struct StubInference: InferenceService {
        @MainActor var availability: InferenceAvailability {
            .unavailable(reason: "test")
        }

        func generate(
            prompt: String,
            schemaJSON: String?,
            maxOutputTokens: Int
        ) async throws -> String {
            throw PluginInferenceError.unavailable(reason: "test")
        }
    }

    private func freshDefaults(_ name: String) -> UserDefaults {
        let suite = "PluginKitTests.\(name)"
        let defaults = UserDefaults(suiteName: suite)!
        defaults.removePersistentDomain(forName: suite)
        return defaults
    }

    // MARK: - Registry

    @Test func activationDeliversEventsAndStampsPayload() {
        let store = PluginPayloadStore()
        let registry = PluginRegistry(defaults: freshDefaults("activate"))
        let log = HelloPluginLog()

        registry.install([HelloPlugin(log: log)], host: StubPluginHost(store: store))
        #expect(log.activations == 1)

        registry.broadcast(.sessionLoaded)
        registry.broadcast(.utterancesChanged(version: 7))
        #expect(log.events == [.sessionLoaded, .utterancesChanged(version: 7)])

        // The activation wrote a payload; the adapter stamped it
        // with the plugin's declared payloadVersion.
        #expect(store["test.hello"] == PluginPayload(version: 3, data: Data("hello".utf8)))
    }

    @Test func disabledPluginDoesNotActivateUntilToggled() {
        let defaults = freshDefaults("toggle")
        defaults.set(false, forKey: "plugin.enabled.test.hello")

        let registry = PluginRegistry(defaults: defaults)
        let log = HelloPluginLog()
        registry.install(
            [HelloPlugin(log: log)],
            host: StubPluginHost(store: PluginPayloadStore())
        )
        #expect(log.activations == 0)

        registry.broadcast(.sessionLoaded)
        #expect(log.events.isEmpty)

        registry.setEnabled(true, id: HelloPlugin.id)
        #expect(log.activations == 1)
        registry.broadcast(.sessionCleared)
        #expect(log.events == [.sessionCleared])

        // Disabling drops the handle: no further delivery, and the
        // preference persists.
        registry.setEnabled(false, id: HelloPlugin.id)
        registry.broadcast(.sessionLoaded)
        #expect(log.events == [.sessionCleared])
        #expect(defaults.bool(forKey: "plugin.enabled.test.hello") == false)
    }

    // MARK: - Phase 1: UI contributions

    @Test func uiContributionsTrackActivationState() {
        let registry = PluginRegistry(defaults: freshDefaults("ui"))
        let log = HelloPluginLog()
        registry.install(
            [HelloPlugin(log: log)],
            host: StubPluginHost(store: PluginPayloadStore())
        )

        #expect(registry.activePages.map(\.id) == ["test.hello.page"])
        #expect(registry.activeMenuCommands.map(\.id) == ["test.hello.cmd"])

        // Disabling removes the contributions live; re-enabling
        // restores them (via a fresh activation).
        registry.setEnabled(false, id: HelloPlugin.id)
        #expect(registry.activePages.isEmpty)
        #expect(registry.activeMenuCommands.isEmpty)

        registry.setEnabled(true, id: HelloPlugin.id)
        #expect(registry.activePages.map(\.id) == ["test.hello.page"])
        #expect(log.activations == 2)
    }

    @Test func hostStubsRecordExportAndPlayback() {
        let host = StubPluginHost(store: PluginPayloadStore())
        let rowID = UUID()
        host.requestPlayback(utteranceID: rowID)
        #expect(host.playbackRequests == [rowID])

        var outcome: PluginExportOutcome?
        host.presentExport(
            data: Data("x".utf8),
            contentType: .plainText,
            suggestedFilename: "x.txt"
        ) { outcome = $0 }
        #expect(outcome == .saved)
        #expect(host.exportedData == [Data("x".utf8)])
    }

    // MARK: - Phase 2: EvalFormPlugin integration

    @Test func evalFormPluginActivatesWithContributions() {
        let host = StubPluginHost(store: PluginPayloadStore())
        let handle = EvalFormPlugin().activate(host: host)
        #expect(handle.pages.count == 1)
        #expect(handle.menuCommands.count == 1)
        // Activation seeds the sheet vocabulary into the bank.
        #expect(host.seededKeywords.contains("ヒョコヒョコ"))
        #expect(host.seededKeywords.contains("ハーシュネス"))
    }

    // MARK: - Payload round-trip

    @Test func payloadsRoundTripThroughDocumentIncludingUnknownIDs() throws {
        let store = PluginPayloadStore()
        store["test.hello"] = PluginPayload(version: 3, data: Data("hello".utf8))
        // A payload written by a plugin THIS build doesn't have —
        // must come back byte-identical after encode → decode.
        store["future.unknown"] = PluginPayload(version: 9, data: Data([0x01, 0x02, 0x03]))

        let document = SessionDocument(
            sourceKind: .microphone,
            audioFilename: nil,
            audio: nil,
            utterances: [],
            pluginPayloads: store.forExport
        )
        let decoded = try SessionBundle.decode(try SessionBundle.encode(document))

        let restored = PluginPayloadStore()
        restored.replaceAll(decoded.pluginPayloads ?? [:])
        #expect(restored["test.hello"] == PluginPayload(version: 3, data: Data("hello".utf8)))
        #expect(restored["future.unknown"] == PluginPayload(version: 9, data: Data([0x01, 0x02, 0x03])))
        #expect(restored.payloads.count == 2)

        // The stored version rides through for migration checks.
        let adapter = PluginStorageAdapter(
            store: restored,
            id: PluginID("test.hello"),
            writeVersion: 4
        )
        #expect(adapter.sessionPayloadVersion == 3)
    }

    @Test func emptyPayloadTableRoundTripsAsNil() throws {
        // Plugin-free sessions must stay byte-clean: an empty store
        // exports nil, and a document without the field decodes nil.
        #expect(PluginPayloadStore().forExport == nil)

        let document = SessionDocument(
            sourceKind: .microphone,
            audioFilename: nil,
            audio: nil,
            utterances: []
        )
        let decoded = try SessionBundle.decode(try SessionBundle.encode(document))
        #expect(decoded.pluginPayloads == nil)
    }
}
