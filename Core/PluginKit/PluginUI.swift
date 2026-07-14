import SwiftUI
import UniformTypeIdentifiers

/// One control-pane page contributed by a plugin. The host appends
/// plugin pages after the built-in ones in the left column's
/// swipeable TabView; the plugin supplies card-level content and
/// the host owns the page chrome (scroll container, paddings,
/// page-indexing) so every page keeps the app's look.
public struct PluginPageDescriptor: Identifiable {
    /// Unique across the app — convention: `"<pluginID>.<pageKey>"`.
    /// Also the page's stable identity across TabView rebuilds.
    public let id: String
    /// Page title (used for accessibility and future page pickers).
    public let title: String
    /// SF Symbol shown alongside the title where applicable.
    public let systemImage: String
    /// Card content builder. Called on the MainActor during body
    /// evaluation — keep it render-only; state lives in the
    /// plugin's own @Observable models.
    public let content: @MainActor () -> AnyView

    public init(
        id: String,
        title: String,
        systemImage: String,
        content: @escaping @MainActor () -> AnyView
    ) {
        self.id = id
        self.title = title
        self.systemImage = systemImage
        self.content = content
    }
}

/// One hardware-keyboard menu item contributed by a plugin. The
/// host renders all plugin commands under a single Plugins menu —
/// plugins never touch `CommandGroup`/`CommandMenu` directly (the
/// iPadOS 26 traps live in one host-owned place).
public struct PluginMenuCommand: Identifiable {
    public let id: String
    public let title: String
    public let systemImage: String?
    /// Invoked on the MainActor when the item is selected. Gating
    /// (enabled/disabled) is intentionally not modeled yet — a
    /// command that can't run should present its own explanation.
    public let action: @MainActor () -> Void

    public init(
        id: String,
        title: String,
        systemImage: String? = nil,
        action: @escaping @MainActor () -> Void
    ) {
        self.id = id
        self.title = title
        self.systemImage = systemImage
        self.action = action
    }
}

/// Host-presented file export. Implementations route through the
/// app's single `.fileExporter` at the navigation root — plugins
/// MUST NOT attach their own file-picker modifiers (they silently
/// collide on iPadOS 26; see the FilePickerCoordinator discipline
/// in CLAUDE.md).
@MainActor
public protocol ExportPresenting: AnyObject {
    /// Present the system save dialog over `data`. `contentType`
    /// must be in the app's compile-time writable whitelist —
    /// plugins introducing a new type add it there (a compile-time
    /// concern by design). `completion` reports the user's outcome;
    /// cancellation is not an error.
    func presentExport(
        data: Data,
        contentType: UTType,
        suggestedFilename: String,
        completion: @escaping @MainActor (PluginExportOutcome) -> Void
    )
}

public enum PluginExportOutcome: Sendable, Equatable {
    case saved
    case cancelled
    case failed(reason: String)
}

/// Host-presented file import. Same root-picker discipline as
/// `ExportPresenting`; the host owns the security-scope dance and
/// hands plugins bytes, never URLs.
@MainActor
public protocol ImportPresenting: AnyObject {
    /// Present the system open dialog restricted to `contentTypes`
    /// and read the picked file. `contentTypes` must be within the
    /// app's compile-time readable whitelist.
    func presentImport(
        contentTypes: [UTType],
        completion: @escaping @MainActor (PluginImportOutcome) -> Void
    )
}

public enum PluginImportOutcome: Sendable, Equatable {
    case loaded(Data)
    case cancelled
    case failed(reason: String)
}
