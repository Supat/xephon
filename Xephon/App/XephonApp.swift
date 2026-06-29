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
                Button {
                    menuCommands.openAudioFileToken = UUID()
                } label: {
                    Label(
                        String(localized: "menu.openAudioFile"),
                        systemImage: "waveform.badge.plus"
                    )
                }
                .keyboardShortcut("o", modifiers: .command)
                Button {
                    menuCommands.importSessionToken = UUID()
                } label: {
                    Label(
                        String(localized: "menu.importSession"),
                        systemImage: "folder"
                    )
                }
                .keyboardShortcut("o", modifiers: [.command, .shift])
            }
            CommandGroup(replacing: .saveItem) {
                Button {
                    menuCommands.saveSessionToken = UUID()
                } label: {
                    Label(
                        String(localized: "menu.saveSession"),
                        systemImage: "square.and.arrow.down"
                    )
                }
                .keyboardShortcut("s", modifiers: .command)
                .disabled(!menuCommands.canSaveSession)
                Button {
                    menuCommands.exportJSONToken = UUID()
                } label: {
                    Label(
                        String(localized: "menu.exportJSON"),
                        systemImage: "square.and.arrow.up"
                    )
                }
                .keyboardShortcut("s", modifiers: [.command, .shift])
                .disabled(!menuCommands.canExportJSON)
            }
            // Edit > Undo / Edit > Redo. Backed by
            // `RecordingController.undoManager`; both menu items are
            // gated on `canUndo` / `canRedo` mirrored into
            // `MenuCommands` (refreshed via the
            // NSUndoManagerCheckpoint observer set up in
            // `EventBridgeModifier`). Replaces the placement so the
            // standard Edit > Undo slot routes through our stack
            // instead of the system's nil default. UIKit's per-
            // keystroke text-edit undo continues to win when a
            // TextField holds first responder — those Cmd-Z presses
            // route to the field, not to our menu item.
            CommandGroup(replacing: .undoRedo) {
                Button {
                    menuCommands.undoToken = UUID()
                } label: {
                    Label(
                        String(localized: "menu.undo"),
                        systemImage: "arrow.uturn.backward"
                    )
                }
                .keyboardShortcut("z", modifiers: .command)
                .disabled(!menuCommands.canUndo)
                Button {
                    menuCommands.redoToken = UUID()
                } label: {
                    Label(
                        String(localized: "menu.redo"),
                        systemImage: "arrow.uturn.forward"
                    )
                }
                .keyboardShortcut("z", modifiers: [.command, .shift])
                .disabled(!menuCommands.canRedo)
            }
            // ⌘F focuses the utterance search field. Lives in the Edit
            // menu's pasteboard region (which is where Find traditionally
            // sits on Apple platforms). Same UUID-token bridge as the
            // File commands above.
            CommandGroup(after: .pasteboard) {
                Button {
                    menuCommands.findToken = UUID()
                } label: {
                    Label(
                        String(localized: "menu.findInUtterances"),
                        systemImage: "magnifyingglass"
                    )
                }
                .keyboardShortcut("f", modifiers: .command)
            }
            // View menu additions. Anchored via
            // `CommandGroup(after: .sidebar)` so the items extend
            // the system-provided View menu rather than creating
            // a parallel `CommandMenu("View")` — the latter
            // showed up as TWO "View" entries in the macOS menu
            // bar (Designed-for-iPad on Apple Silicon Mac) and
            // emitted "duplicate identifier" warnings in
            // `UIMenuBuilder` because SwiftUI's iPadOS 26 bridge
            // double-registers `CommandMenu` titles.
            //
            // ⌘1–⌘6 are page switchers — mirror the browser-tab
            // convention so a hardware keyboard can move through
            // the left pane without the trackpad. Sheet items
            // below the divider mirror the chrome toolbar's
            // Summarize / Review / Search-and-Replace buttons.
            CommandGroup(after: .sidebar) {
                Button {
                    menuCommands.viewSettingsToken = UUID()
                } label: {
                    Label(
                        String(localized: "menu.view.settings"),
                        systemImage: "slider.horizontal.3"
                    )
                }
                .keyboardShortcut("1", modifiers: .command)
                Button {
                    menuCommands.viewAffectToken = UUID()
                } label: {
                    Label(
                        String(localized: "menu.view.affect"),
                        systemImage: "chart.bar"
                    )
                }
                .keyboardShortcut("2", modifiers: .command)
                Button {
                    menuCommands.viewSpeakersToken = UUID()
                } label: {
                    Label(
                        String(localized: "menu.view.speakers"),
                        systemImage: "person.2"
                    )
                }
                .keyboardShortcut("3", modifiers: .command)
                Button {
                    menuCommands.viewSectionsToken = UUID()
                } label: {
                    Label(
                        String(localized: "menu.view.sections"),
                        systemImage: "bookmark"
                    )
                }
                .keyboardShortcut("4", modifiers: .command)
                Button {
                    menuCommands.viewKeywordsToken = UUID()
                } label: {
                    Label(
                        String(localized: "menu.view.keywords"),
                        systemImage: "tag"
                    )
                }
                .keyboardShortcut("5", modifiers: .command)
                Button {
                    menuCommands.viewSummarizerToken = UUID()
                } label: {
                    Label(
                        String(localized: "menu.view.summarizer"),
                        systemImage: "wand.and.rays"
                    )
                }
                .keyboardShortcut("6", modifiers: .command)
                // System-drawn separator between page-switching
                // items and the sheet-presentation items below.
                Divider()
                // Sheet presentations. Glyphs match the chrome
                // toolbar's buttons so the menu reads as the
                // same actions.
                Button {
                    menuCommands.presentSummaryToken = UUID()
                } label: {
                    Label(
                        String(localized: "menu.view.summary"),
                        systemImage: "text.book.closed"
                    )
                }
                .disabled(!menuCommands.canPresentSummary)
                Button {
                    menuCommands.presentReviewToken = UUID()
                } label: {
                    Label(
                        String(localized: "menu.view.review"),
                        systemImage: "exclamationmark.bubble"
                    )
                }
                .disabled(!menuCommands.canPresentReview)
                Button {
                    menuCommands.presentSearchReplaceToken = UUID()
                } label: {
                    Label(
                        String(localized: "menu.view.searchReplace"),
                        systemImage: "magnifyingglass.circle"
                    )
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
    /// Bumped by Edit → Undo (⌘Z). ContentView watches this and
    /// invokes `recorder.undoManager.undo()`. Routed through the
    /// token bus rather than calling the UndoManager directly from
    /// the CommandGroup so the gate-disabled state stays in sync
    /// across MainActor isolation boundaries.
    var undoToken: UUID = UUID()
    /// Bumped by Edit → Redo (⌘⇧Z). Same pattern as `undoToken`.
    var redoToken: UUID = UUID()
    /// Mirror of `recorder.undoManager.canUndo`. Refreshed by
    /// EventBridgeModifier's NSUndoManagerCheckpoint observer.
    var canUndo: Bool = false
    /// Mirror of `recorder.undoManager.canRedo`. Same refresh path.
    var canRedo: Bool = false
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
    /// Mirror of the chrome-toolbar Export button gate for the
    /// File → Save Session and File → Export to JSON menu items.
    /// Both share the same precondition
    /// (`recorder.isIdleWithTranscript`) so one flag suffices —
    /// kept as separate properties so a future divergence is a
    /// one-line edit rather than a refactor. ContentView's
    /// `syncMenuItemGates()` writes both when the recorder's
    /// idle / utterance state changes.
    var canSaveSession: Bool = false
    var canExportJSON: Bool = false
}
