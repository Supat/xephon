import Foundation
import XephonPluginKit
import XephonLogging

/// Compile-time plugin registry — the app-side owner of installed
/// plugins and their activation state. See docs/plugin_architecture.md
/// §2: registration is a hardcoded array in `XephonApp`; there is no
/// runtime discovery on purpose.
///
/// Enable/disable is persisted per plugin id in UserDefaults and
/// takes effect on toggle (activate/deactivate immediately, not at
/// next launch). A disabled plugin's `.xph` payloads still round-
/// trip opaquely — the payload store is the controller's, not the
/// registry's.
@MainActor
@Observable
final class PluginRegistry {
    /// One installed plugin plus its live handle (nil while
    /// disabled). `Identifiable` for the future settings list.
    struct Entry: Identifiable {
        let plugin: any XephonPlugin
        var handle: PluginHandle?

        var pluginType: any XephonPlugin.Type { type(of: plugin) }
        var id: String { pluginType.id.rawValue }
        var displayName: String { pluginType.displayName }
        var properName: String { pluginType.properName }
        var isActive: Bool { handle != nil }
    }

    private(set) var entries: [Entry] = []

    @ObservationIgnored private let defaults: UserDefaults
    /// Strong on purpose: the registry is the host object's owner
    /// (XephonApp retains only the registry; PluginHostServices
    /// references nothing back, so no cycle).
    @ObservationIgnored private var host: (any PluginHost)?

    /// `defaults` injectable so tests don't touch the app domain.
    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
    }

    var isEmpty: Bool { entries.isEmpty }

    /// Install `plugins` and activate the enabled ones. Call once
    /// at startup; installing twice is a programmer error (the
    /// hardcoded install list makes duplicates impossible in
    /// practice, so this just logs and ignores repeats by id).
    func install(_ plugins: [any XephonPlugin], host: any PluginHost) {
        self.host = host
        for plugin in plugins {
            let id = type(of: plugin).id
            guard !entries.contains(where: { $0.id == id.rawValue }) else {
                AppLog.app.warning(
                    "PluginRegistry: duplicate install ignored for \(id.rawValue, privacy: .public)"
                )
                continue
            }
            var entry = Entry(plugin: plugin, handle: nil)
            if isEnabled(id) {
                entry.handle = plugin.activate(host: host)
                AppLog.app.info(
                    "PluginRegistry: activated \(id.rawValue, privacy: .public)"
                )
            }
            entries.append(entry)
        }
    }

    /// Whether `id` is enabled (default: enabled — installing a
    /// plugin is the opt-in; the toggle exists to opt back out).
    func isEnabled(_ id: PluginID) -> Bool {
        defaults.object(forKey: Self.enabledKey(id)) as? Bool ?? true
    }

    /// Persist the toggle and activate/deactivate immediately.
    /// Deactivation just drops the handle — the plugin's stored
    /// payloads and settings are untouched.
    func setEnabled(_ enabled: Bool, id: PluginID) {
        defaults.set(enabled, forKey: Self.enabledKey(id))
        guard let idx = entries.firstIndex(where: { $0.id == id.rawValue }) else {
            return
        }
        if enabled, entries[idx].handle == nil, let host {
            entries[idx].handle = entries[idx].plugin.activate(host: host)
            AppLog.app.info(
                "PluginRegistry: activated \(id.rawValue, privacy: .public)"
            )
        } else if !enabled, entries[idx].handle != nil {
            entries[idx].handle = nil
            AppLog.app.info(
                "PluginRegistry: deactivated \(id.rawValue, privacy: .public)"
            )
        }
    }

    /// Deliver `event` to every active plugin. Called by
    /// RecordingController's event sink; a no-op with no active
    /// plugins.
    func broadcast(_ event: SessionEvent) {
        for entry in entries {
            entry.handle?.onSessionEvent?(event)
        }
    }

    /// Control-pane pages of every ACTIVE plugin, in install order.
    /// Disabled plugins contribute nothing (their handle is nil),
    /// so toggling a plugin adds/removes its pages live.
    var activePages: [PluginPageDescriptor] {
        entries.compactMap(\.handle).flatMap(\.pages)
    }

    /// Menu items of every ACTIVE plugin, in install order.
    var activeMenuCommands: [PluginMenuCommand] {
        entries.compactMap(\.handle).flatMap(\.menuCommands)
    }

    private static func enabledKey(_ id: PluginID) -> String {
        "plugin.enabled.\(id.rawValue)"
    }
}
