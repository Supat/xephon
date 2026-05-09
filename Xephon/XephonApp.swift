import SwiftUI
import Observation
import XephonLogging

@main
struct XephonApp: App {
    /// Shared bus the App's `.commands` block writes to and ContentView
    /// observes. Plain SwiftUI `@FocusedValue` commands are fragile on
    /// iPadOS 26 — the focus engine won't propagate the focused-scene
    /// value to a command unless an actual UIView in the scene is
    /// focused, so the menu item silently disables. An @Observable
    /// shared model bypasses focus entirely; both sides bind to the
    /// same reference and `.onChange` fires reliably.
    @State private var menuCommands = MenuCommands()

    init() {
        AppLog.app.info("Xephon launching")
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(menuCommands)
        }
        // Hardware-keyboard menu integration. iPadOS 26's menu strip
        // (cmd-hold) and macOS / Catalyst menu bar both honor these.
        .commands {
            CommandGroup(replacing: .newItem) {
                Button(String(localized: "menu.openAudioFile")) {
                    menuCommands.openAudioFileToken = UUID()
                }
                .keyboardShortcut("o", modifiers: .command)
                Divider()
                Button(String(localized: "menu.exportJSON")) {
                    menuCommands.exportJSONToken = UUID()
                }
                .keyboardShortcut("s", modifiers: .command)
            }
        }
    }
}

/// Tokens for command → view dispatch. Each command bumps a UUID; the
/// observing view's `.onChange` fires on every change (even repeated
/// triggers of the same command), since the new UUID is always
/// different from the prior one.
@MainActor
@Observable
final class MenuCommands {
    /// Bumped by the File → Open… menu item. ContentView watches this
    /// and raises the file picker.
    var openAudioFileToken: UUID = UUID()
    /// Bumped by the File → Export to JSON menu item. ContentView
    /// watches this and runs the same exporter the toolbar button uses.
    var exportJSONToken: UUID = UUID()
}
