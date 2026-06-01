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
            // Two adjacent `CommandGroup`s instead of one with an
            // inline `Divider()` — the latter doesn't render a
            // visible separator on iPadOS 26's menu strip (SwiftUI
            // bug). Splitting on the `.newItem` / `.saveItem`
            // placements gets us a system-drawn divider between
            // Open*/Save* and matches the conventional File menu
            // grouping users expect.
            CommandGroup(replacing: .newItem) {
                Button(String(localized: "menu.openAudioFile")) {
                    menuCommands.openAudioFileToken = UUID()
                }
                .keyboardShortcut("o", modifiers: .command)
                Button(String(localized: "menu.importSession")) {
                    menuCommands.importSessionToken = UUID()
                }
                .keyboardShortcut("o", modifiers: [.command, .shift])
            }
            CommandGroup(replacing: .saveItem) {
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
            // View → <page> items, one per ControlPaneView TabView
            // page. ⌘1–⌘6 mirror the browser-tab convention so a
            // hardware keyboard can switch pages without reaching
            // for the trackpad. Same UUID-token bus as the File /
            // Edit commands — ControlPaneView's `.onChange` writes
            // to `selectedTab` on each fire.
            CommandMenu(String(localized: "menu.view")) {
                Button(String(localized: "menu.view.settings")) {
                    menuCommands.viewSettingsToken = UUID()
                }
                .keyboardShortcut("1", modifiers: .command)
                Button(String(localized: "menu.view.affect")) {
                    menuCommands.viewAffectToken = UUID()
                }
                .keyboardShortcut("2", modifiers: .command)
                Button(String(localized: "menu.view.speakers")) {
                    menuCommands.viewSpeakersToken = UUID()
                }
                .keyboardShortcut("3", modifiers: .command)
                Button(String(localized: "menu.view.sections")) {
                    menuCommands.viewSectionsToken = UUID()
                }
                .keyboardShortcut("4", modifiers: .command)
                Button(String(localized: "menu.view.keywords")) {
                    menuCommands.viewKeywordsToken = UUID()
                }
                .keyboardShortcut("5", modifiers: .command)
                Button(String(localized: "menu.view.summarizer")) {
                    menuCommands.viewSummarizerToken = UUID()
                }
                .keyboardShortcut("6", modifiers: .command)
                // System-drawn separator between page-switching
                // items and the sheet-presentation items below.
                Divider()
                // Sheet presentations. Mirror the chrome
                // toolbar's Summarize / Review / Search &
                // Replace buttons so users who navigate by menu
                // (hardware keyboard, VoiceOver, etc.) have the
                // same access. No keyboard shortcuts — File
                // already claims ⌘S / ⇧⌘S / ⌘F, and adding
                // modifier combinations here without clear
                // convention would just be noise.
                Button(String(localized: "menu.view.summary")) {
                    menuCommands.presentSummaryToken = UUID()
                }
                .disabled(!menuCommands.canPresentSummary)
                Button(String(localized: "menu.view.review")) {
                    menuCommands.presentReviewToken = UUID()
                }
                .disabled(!menuCommands.canPresentReview)
                Button(String(localized: "menu.view.searchReplace")) {
                    menuCommands.presentSearchReplaceToken = UUID()
                }
                .disabled(!menuCommands.canPresentSearchReplace)
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
    /// Bumped by the View → <page> menu items (⌘1–⌘6).
    /// `ControlPaneView` watches each and flips `selectedTab` to the
    /// corresponding page in its swipeable TabView. One UUID per
    /// destination so a re-tap of the same item re-fires the
    /// `.onChange` (different UUID each time) — useful when the user
    /// has scrolled away from a card and wants to jump back to its
    /// page header.
    var viewSettingsToken: UUID = UUID()
    var viewAffectToken: UUID = UUID()
    var viewSpeakersToken: UUID = UUID()
    var viewSectionsToken: UUID = UUID()
    var viewKeywordsToken: UUID = UUID()
    var viewSummarizerToken: UUID = UUID()
    /// Bumped by the View → Summary / Review / Find & Replace
    /// menu items. ContentView watches each and forwards to
    /// the matching `llmCoord.present...(recorder:)` call so
    /// the chrome's toolbar buttons aren't the only access
    /// path. No tab-switching here — these open modal sheets
    /// over whatever page is currently visible.
    var presentSummaryToken: UUID = UUID()
    var presentReviewToken: UUID = UUID()
    var presentSearchReplaceToken: UUID = UUID()

    /// Mirrors of the chrome-toolbar enable gates for the three
    /// sheet-presentation menu items. ContentView writes these
    /// from its `.onChange` watchers on the underlying recorder
    /// state so the menu's `.disabled(...)` reflects the same
    /// conditions as the toolbar buttons. Defaults to `false`
    /// (everything disabled) at app launch — ContentView's
    /// first `.task` brings them up to the live values before
    /// the user reaches the menu.
    var canPresentSummary: Bool = false
    var canPresentReview: Bool = false
    var canPresentSearchReplace: Bool = false
}
