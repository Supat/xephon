import Foundation

/// User-managed list of `ConversationSection`s for the
/// current session. Lives in memory only — sections reference
/// per-session utterance IDs and don't survive a session
/// swap; `pruneDangling(validIDs:)` lets the controller drop
/// stale entries when the utterance set turns over.
///
/// `@Observable` + `@MainActor` so SwiftUI re-renders the
/// `SectionsCard` automatically on every mutation.
@Observable
@MainActor
public final class SectionStore {
    public private(set) var sections: [ConversationSection] = []

    public init() {}

    public func add(_ section: ConversationSection) {
        sections.append(section)
    }

    public func update(_ section: ConversationSection) {
        guard let idx = sections.firstIndex(where: { $0.id == section.id }) else {
            return
        }
        sections[idx] = section
    }

    public func remove(id: UUID) {
        sections.removeAll { $0.id == id }
    }

    public func clear() {
        sections.removeAll()
    }

    /// Replace the whole list in one shot. Used by the
    /// `.xph` load path to restore the sections that were
    /// saved alongside the utterances. Single assignment so
    /// `@Observable` fires once for the bulk update instead
    /// of N times via `add`.
    public func replaceAll(_ newSections: [ConversationSection]) {
        sections = newSections
    }

    /// Drop sections whose start or end references an
    /// utterance no longer in the session. Called when the
    /// controller's utterance list churns (new recording,
    /// new file analysis, imported bundle) so the user
    /// doesn't see a card full of "(missing)" labels after
    /// switching sessions. Nil bounds are ignored — an
    /// incomplete section with one set bound only counts as
    /// dangling if THAT bound is invalid.
    public func pruneDangling(validIDs: Set<UUID>) {
        sections.removeAll { sec in
            if let s = sec.startUtteranceID, !validIDs.contains(s) {
                return true
            }
            if let e = sec.endUtteranceID, !validIDs.contains(e) {
                return true
            }
            return false
        }
    }
}
