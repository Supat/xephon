import Foundation
import Testing
@testable import Xephon
import XephonPluginKit
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
            return PluginHandle { [log] event in
                log.events.append(event)
            }
        }
    }

    /// Minimal host: real storage adapters over a real payload
    /// store, canned snapshot, placeholder inference.
    @MainActor
    final class StubPluginHost: PluginHost, SessionReading {
        let store: PluginPayloadStore

        init(store: PluginPayloadStore) {
            self.store = store
        }

        var session: any SessionReading { self }
        var inference: any InferenceService { PluginInferencePlaceholder() }

        func storage(for plugin: any XephonPlugin.Type) -> any PluginStorage {
            PluginStorageAdapter(
                store: store,
                id: plugin.id,
                writeVersion: plugin.payloadVersion
            )
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
