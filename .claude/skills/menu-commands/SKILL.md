---
name: menu-commands
description: Use when adding, editing, or reviewing Xephon's hardware-keyboard menu items (macOS menu bar, iPadOS 26 cmd-hold menu strip) — new File/Edit/View entries, keyboard shortcuts, dividers, `.disabled` gates, or anything touching `XephonApp.commands`, `CommandGroup`, `CommandMenu`, `MenuCommands`, or `@FocusedValue` for menus. The bus pattern in this app has several iPadOS-26-specific traps; this skill captures them.
---

# Menu commands in Xephon

The app reaches the macOS menu bar (Designed-for-iPad on Apple
Silicon Mac) and the iPadOS 26 cmd-hold menu strip through one
`.commands` builder on `XephonApp`. Menu items dispatch to
`ContentView` through `MenuCommands`, a shared `@Observable` bus
held in `XephonApp`'s `@State`.

## Why a bus instead of `@FocusedValue`

`@FocusedValue` is the textbook way to wire a SwiftUI menu item to
focused-scene work — but on iPadOS 26 the focus engine only
propagates focused values when an actual `UIView` in the scene
holds focus. Menu items silently disable whenever the focus isn't
on a focusable view, which in this app is most of the time. The
`@Observable` bus bypasses focus entirely: both sides bind to the
same reference and `.onChange` fires reliably.

## Hard rules

1. **Anchor menu items with `CommandGroup`, never `CommandMenu`.**
   `CommandMenu("View")` double-registers on the Designed-for-iPad
   bridge and renders two "View" entries with a "duplicate
   identifier" warning from `UIMenuBuilder`. Extend system menus
   with `CommandGroup(after: .sidebar)`, `CommandGroup(replacing:
   .newItem)`, etc.
2. **No inline `Divider()` inside a `CommandGroup`** — silently
   doesn't render on iPadOS 26's menu strip. Split into two
   adjacent `CommandGroup`s at different `.newItem` / `.saveItem`
   placements; the system draws a divider between groups for free.
3. **Token dispatch for action items**: each command is a
   `var fooToken: UUID = UUID()` on `MenuCommands`. The menu's
   button writes `menuCommands.fooToken = UUID()` so re-tapping
   the same item re-fires the consumer's `.onChange` (different
   UUID every time). Plain `Bool` toggles deduplicate and miss
   the second press.
4. **Gated items mirror the chrome's gate**. `MenuCommands` holds
   a `canFoo: Bool` for every `.disabled(!menuCommands.canFoo)` in
   `XephonApp`. The consumer view (`ContentView.syncMenuItemGates()`)
   writes those mirrors from `.onChange` watchers on the underlying
   recorder state. The menu builder lives up in `XephonApp` and
   shouldn't grow a recorder reference just to read a gate.

## Pattern — adding a new action menu item

1. Add to `MenuCommands` in `Xephon/App/XephonApp.swift`:
   ```swift
   /// Bumped by the <where> menu item. <ConsumerView> watches this
   /// and <does the work>.
   var fooToken: UUID = UUID()
   ```

2. Add to `XephonApp.body`'s `.commands` builder, in the appropriate
   `CommandGroup`:
   ```swift
   Button {
       menuCommands.fooToken = UUID()
   } label: {
       Label(String(localized: "menu.foo"), systemImage: "…")
   }
   .keyboardShortcut("…", modifiers: …)   // optional
   ```

3. Watch it in `ContentView` (or whichever view holds the
   environment access to `menuCommands`):
   ```swift
   .onChange(of: menuCommands.fooToken) { _, _ in
       // do the work — usually call into recorder / coord
   }
   ```

4. Add the `String(localized: "menu.foo")` key to the project's
   `Localizable.xcstrings`.

## Pattern — adding a new GATED menu item

Steps 1-4 above, plus:

5. Add a `canFoo: Bool = false` to `MenuCommands` next to the token.

6. Gate the button: `.disabled(!menuCommands.canFoo)`.

7. Update `syncMenuItemGates()` in `ContentView.swift` to write
   `menuCommands.canFoo` from the relevant recorder predicate.

8. Add an `.onChange(of: recorder.somePredicate) { _, _ in syncMenuItemGates() }`
   watcher so the gate updates live when the predicate flips.
   `.onAppear { syncMenuItemGates() }` already seeds the initial
   value at view appear.

## System anchors used in this app

- `CommandGroup(replacing: .newItem)` — File → Open… / Import Session…
- `CommandGroup(replacing: .saveItem)` — File → Save Session / Export to JSON
- `CommandGroup(after: .pasteboard)` — Edit → Find
- `CommandGroup(after: .sidebar)` — View → page-switchers + sheet items

Add new items to the existing groups when they belong there. Create
a new `CommandGroup` only when none of the system anchors fit (e.g.
a wholly new top-level concept).

## Files to know

- `Xephon/App/XephonApp.swift` — `@main App`, the `.commands`
  builder, and the `MenuCommands` `@Observable` class.
- `Xephon/App/ContentView.swift` — token watchers (search for
  `.onChange(of: menuCommands.…)`) and `syncMenuItemGates()`.

## Don't

- Don't add `@FocusedValue` for menu wiring. Use the bus.
- Don't use `CommandMenu("View")` to add View-menu items. Use
  `CommandGroup(after: .sidebar)`.
- Don't put `Divider()` inline inside a `CommandGroup`. Split groups.
- Don't gate menu items by reading recorder state directly from the
  `.commands` builder. Mirror through `MenuCommands.can*` flags.
- Don't reuse a single `Bool` for a "fire" command. Use a UUID token
  so repeat-presses register.
