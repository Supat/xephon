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
                Button(String(localized: "menu.importSession")) {
                    menuCommands.importSessionToken = UUID()
                }
                .keyboardShortcut("o", modifiers: [.command, .shift])
                Divider()
                Button(String(localized: "menu.saveSession")) {
                    menuCommands.saveSessionToken = UUID()
                }
                .keyboardShortcut("s", modifiers: .command)
                Button(String(localized: "menu.exportJSON")) {
                    menuCommands.exportJSONToken = UUID()
                }
                .keyboardShortcut("s", modifiers: [.command, .shift])
            }
            // ⌘F focuses the utterance search field. Lives in the Edit
            // menu's pasteboard region (which is where Find traditionally
            // sits on Apple platforms). Same UUID-token bridge as the
            // File commands above.
            CommandGroup(after: .pasteboard) {
                Button(String(localized: "menu.findInUtterances")) {
                    menuCommands.findToken = UUID()
                }
                .keyboardShortcut("f", modifiers: .command)
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
    /// Bumped by File → Save Session… (⌘S). Writes the current
    /// analysis (utterances + audio when file-mode) to a `.xph` file.
    var saveSessionToken: UUID = UUID()
    /// Bumped by File → Import Session… (⇧⌘O). Replaces the
    /// in-memory analysis with the contents of a `.xph` file.
    var importSessionToken: UUID = UUID()
    /// Bumped by the Edit → Find menu item (⌘F). ContentView watches
    /// this and moves keyboard focus into the utterance search field.
    var findToken: UUID = UUID()
}
