import Foundation

/// Stable identifier for a plugin. Reverse-DNS-ish by convention
/// (`"xephon.evalform"`), used as the key for `.xph` payload
/// namespacing, settings namespacing, and log categories — so it
/// must never change once a plugin has shipped payloads.
public struct PluginID: RawRepresentable, Hashable, Sendable, Codable,
                        CustomStringConvertible {
    public let rawValue: String

    public init(rawValue: String) {
        self.rawValue = rawValue
    }

    public init(_ rawValue: String) {
        self.init(rawValue: rawValue)
    }

    public var description: String { rawValue }
}

/// A Xephon plugin — a compiled module registered with the app's
/// `PluginRegistry` at startup. See docs/plugin_architecture.md for
/// the architecture this implements (tier T1) and the rules plugins
/// must follow.
///
/// Plugins depend on this module (and Core value-type modules)
/// ONLY — never on the app target. Everything a plugin reads or
/// writes goes through the `PluginHost` services handed to
/// `activate(host:)`.
public protocol XephonPlugin: Sendable {
    /// Stable identity. Payloads in `.xph` bundles are keyed by
    /// this — changing it orphans previously-saved plugin data.
    static var id: PluginID { get }

    /// Localized, user-facing name (settings list, page titles).
    static var displayName: String { get }

    /// Version stamped onto every `.xph` payload this plugin
    /// writes. Bump on incompatible payload changes; the previously
    /// stored version is readable back via
    /// `PluginStorage.sessionPayloadVersion` so the plugin can
    /// migrate on load.
    static var payloadVersion: Int { get }

    /// Called once by the registry when the plugin is (or becomes)
    /// enabled. The returned handle carries the plugin's event
    /// callback and is retained by the registry until deactivation.
    @MainActor func activate(host: any PluginHost) -> PluginHandle
}

/// What `activate(host:)` hands back: the plugin's live surface as
/// seen by the host — the session-event callback plus declarative
/// UI contributions. All fixed at activation; a plugin whose page
/// content varies drives that variation through its own observable
/// state, not by re-registering.
@MainActor
public final class PluginHandle {
    /// Invoked on the MainActor for every session-lifecycle event.
    /// Keep it cheap — kick real work into a Task.
    public let onSessionEvent: ((SessionEvent) -> Void)?
    /// Control-pane pages, appended after the built-in pages while
    /// the plugin is active.
    public let pages: [PluginPageDescriptor]
    /// Hardware-keyboard menu items, listed under the host-owned
    /// Plugins menu while the plugin is active.
    public let menuCommands: [PluginMenuCommand]

    public init(
        onSessionEvent: ((SessionEvent) -> Void)? = nil,
        pages: [PluginPageDescriptor] = [],
        menuCommands: [PluginMenuCommand] = []
    ) {
        self.onSessionEvent = onSessionEvent
        self.pages = pages
        self.menuCommands = menuCommands
    }
}

/// Session-lifecycle events delivered to active plugins.
public enum SessionEvent: Sendable, Equatable {
    /// A saved session finished loading (`.xph` import). A fresh
    /// `SessionReading.snapshot()` reflects the loaded state, and
    /// the plugin's persisted payload (if any) is readable.
    case sessionLoaded
    /// Session state was cleared for a new recording/analysis.
    /// Plugin session payloads were cleared with it.
    case sessionCleared
    /// The utterance list changed (append, edit, re-evaluation,
    /// speaker reassignment, …). `version` is the host's monotonic
    /// mutation counter — equal versions mean nothing changed.
    case utterancesChanged(version: Int)
}
