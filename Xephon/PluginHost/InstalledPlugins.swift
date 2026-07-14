import Foundation
import XephonPluginKit

/// THE compile-time plugin install list (docs/plugin_architecture.md
/// §2 — no runtime discovery on purpose; adding a plugin is a
/// one-line diff here plus its SPM target).
///
/// Debug builds carry `DebugSamplePlugin` so the Phase 1 exit
/// criteria (a plugin page rendering live session data, an export
/// through the root picker, a menu item) stay verifiable on device.
/// The Run action builds Release (see project.yml), so normal
/// sideloads don't show it.
@MainActor
func xephonInstalledPlugins() -> [any XephonPlugin] {
    #if DEBUG
    return [DebugSamplePlugin()]
    #else
    return []
    #endif
}
