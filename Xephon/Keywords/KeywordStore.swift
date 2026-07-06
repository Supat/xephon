import Foundation
import XephonLogging

/// One user-managed keyword. The wrapping `id` lets SwiftUI track
/// list rows across edits / reorders without falling back to
/// `\.self`-on-String (which breaks the moment two keywords share
/// text, even transiently while typing).
///
/// `groupID` is optional — nil means the keyword lives in the
/// implicit "Ungrouped" section. Stays optional rather than being
/// folded into a sentinel "ungrouped" group so the document
/// format doesn't need a synthetic group id reserved at the JSON
/// layer.
public struct Keyword: Sendable, Hashable, Codable, Identifiable {
    public let id: UUID
    public var text: String
    public var groupID: UUID?
    /// Optional user-assigned color tag. nil = no color (the
    /// default). Pure visual metadata — no filter or prompt logic
    /// reads it, which is also why mutating it deliberately skips
    /// the store's `onChange` hook (see `setTagColor`).
    public var tagColor: KeywordTagColor?

    public init(
        id: UUID = UUID(),
        text: String,
        groupID: UUID? = nil,
        tagColor: KeywordTagColor? = nil
    ) {
        self.id = id
        self.text = text
        self.groupID = groupID
        self.tagColor = tagColor
    }

    // Custom Codable so JSON files saved before the grouping
    // feature (no `groupID` key on each keyword) still decode as
    // ungrouped instead of failing — and likewise files saved
    // before the color-tag feature decode as untagged. Mirrors the
    // `KeywordDocument.groups` `decodeIfPresent` pattern below.
    private enum CodingKeys: String, CodingKey {
        case id, text, groupID, tagColor
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.id = try c.decode(UUID.self, forKey: .id)
        self.text = try c.decode(String.self, forKey: .text)
        self.groupID = try c.decodeIfPresent(UUID.self, forKey: .groupID)
        // Unknown raw values (palette renames in a future version)
        // degrade to untagged rather than failing the whole file.
        self.tagColor = try? c.decodeIfPresent(KeywordTagColor.self, forKey: .tagColor)
    }
}

/// The 24-color tag palette. Raw-value Codable so persisted JSON
/// stays human-readable and stable across reorderings of this
/// enum. Case order IS the palette's display order in the picker
/// grid: warm hues → cool hues → earth/neutral tones, six per
/// row over four rows. The SwiftUI `Color` mapping lives with the
/// picker UI (KeywordsCard) — this type stays presentation-free
/// so the store module doesn't import SwiftUI.
public enum KeywordTagColor: String, Sendable, Hashable, Codable, CaseIterable {
    case red, orange, amber, yellow, lime, green
    case emerald, teal, cyan, sky, blue, indigo
    case violet, purple, fuchsia, pink, rose, maroon
    case brown, olive, navy, slate, gray, stone
}

/// Named bucket the user can assign keywords to. Persisted in the
/// document and rendered as a section header on the Keywords
/// page. Ungrouped keywords (those with `groupID == nil`) appear
/// in an implicit section that isn't represented here.
public struct KeywordGroup: Sendable, Hashable, Codable, Identifiable {
    public let id: UUID
    public var name: String

    public init(id: UUID = UUID(), name: String) {
        self.id = id
        self.name = name
    }
}

/// User-managed keyword list, persisted to Application Support as
/// a single JSON file and observable by SwiftUI for live UI
/// updates. Same lifecycle shape as `GlossaryStore`: the store is
/// the single source of truth, mutations write through to disk on
/// the next MainActor tick, and an optional `onChange` hook fires
/// after persistence completes for downstream consumers.
///
/// Keywords can optionally belong to a named `KeywordGroup`; the
/// Keywords page renders one section per group plus an implicit
/// Ungrouped section. Group membership is persisted; the selection
/// state below is not.
@Observable
@MainActor
public final class KeywordStore {
    /// Ordered list of keywords. Surfaced in input order (newest at
    /// the bottom within their group). Reorder via drag isn't
    /// supported today — the list is short enough that delete +
    /// re-add covers it.
    public var keywords: [Keyword] {
        didSet { didMutate() }
    }

    /// Ordered list of named groups. Append-only via the UI (new
    /// groups land at the end); user can rename or delete any.
    public var groups: [KeywordGroup] {
        didSet { didMutate() }
    }

    /// Called after every persisted mutation. Hook for consumers
    /// that want to react to keyword-list changes without polling
    /// the store. Optional because no in-tree consumer wires it
    /// yet — the card mutates the store directly.
    public var onChange: (@MainActor () -> Void)?

    /// Set of selected keyword ids. Session-only — deliberately
    /// NOT routed through `didMutate` so it doesn't persist across
    /// launches and doesn't fire the `onChange` hook. The
    /// transcript filter reads the text of every keyword whose id
    /// is in this set and treats the union as a single OR filter
    /// layered on top of the search field.
    ///
    /// Toggling an individual keyword adds / removes its id;
    /// toggling a group header flips every keyword in that group
    /// at once (see `toggleGroupSelection`). A tap on an already-
    /// selected keyword removes it from the set; clearing the
    /// last one releases the filter entirely.
    public var selectedKeywordIDs: Set<UUID> = []

    /// Convenience read for downstream consumers — the live
    /// `Keyword` entries currently selected, in `keywords` order.
    /// Computed fresh on every access so it always tracks the
    /// store's current state.
    public var selectedKeywords: [Keyword] {
        keywords.filter { selectedKeywordIDs.contains($0.id) }
    }

    /// Compatibility shim for any single-selection caller (e.g.
    /// the find-and-replace sheet's pre-fill path, which only
    /// auto-fills when exactly one keyword is selected). Returns
    /// nil for zero, the entry for one, nil for many — so callers
    /// can pattern-match a single-selection intent without first
    /// counting.
    public var singleSelectedKeyword: Keyword? {
        let sel = selectedKeywords
        return sel.count == 1 ? sel.first : nil
    }

    private let fileURL: URL

    /// Initialize from the on-disk JSON. A missing / corrupt file
    /// degrades to an empty list — the user can re-import or
    /// hand-build from there. Corrupt-file recovery doesn't
    /// quarantine the bad file on the assumption that the user
    /// would rather have a working keywords panel than a
    /// forensics trail.
    public init() {
        self.fileURL = Self.defaultFileURL()
        let loaded = Self.read(from: fileURL)
        self.keywords = loaded?.keywords ?? []
        self.groups = loaded?.groups ?? []
    }

    /// Test / preview initializer: injected file URL + initial
    /// document, no disk read.
    public init(fileURL: URL, document: KeywordDocument) {
        self.fileURL = fileURL
        self.keywords = document.keywords
        self.groups = document.groups
    }

    // MARK: - Keyword mutations

    /// Append a keyword. Trims whitespace and drops empty input so
    /// the list can't accumulate blank rows from accidental
    /// submits. No dedup — the user might want duplicates (e.g.
    /// the same surface form in two different intended contexts).
    /// `groupID` is the destination group; nil = ungrouped.
    public func add(_ text: String, groupID: UUID? = nil) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        keywords.append(Keyword(text: trimmed, groupID: groupID))
    }

    public func remove(at offsets: IndexSet) {
        let removedIDs = Set(offsets.map { keywords[$0].id })
        keywords.remove(atOffsets: offsets)
        selectedKeywordIDs.subtract(removedIDs)
    }

    public func remove(id: UUID) {
        keywords.removeAll { $0.id == id }
        selectedKeywordIDs.remove(id)
    }

    public func removeAll() {
        keywords.removeAll()
        selectedKeywordIDs.removeAll()
    }

    /// Move a keyword to a different group (or to ungrouped when
    /// `groupID` is nil). No-op when the keyword isn't in the
    /// store or the assignment is already correct, so the UI can
    /// fire this idempotently from a Menu without guard-checks.
    public func assignKeyword(_ keywordID: UUID, toGroup groupID: UUID?) {
        guard let idx = keywords.firstIndex(where: { $0.id == keywordID }),
              keywords[idx].groupID != groupID else { return }
        keywords[idx].groupID = groupID
    }

    /// Reorder so `keywordID` lands directly before
    /// `targetKeywordID` in the flat `keywords` array, AND inherit
    /// the target's group assignment. Used by the drag-to-reorder
    /// affordance on the card; dropping a row from group A onto a
    /// row in group B both reorders AND reassigns. No-op when
    /// either id isn't in the store or the drop is a self-move.
    public func move(_ keywordID: UUID, beforeKeywordWithID targetKeywordID: UUID) {
        guard keywordID != targetKeywordID,
              let sourceIdx = keywords.firstIndex(where: { $0.id == keywordID })
        else { return }
        var moved = keywords.remove(at: sourceIdx)
        // Re-find the target after the removal — its index may
        // have shifted down by one if it was past `sourceIdx`.
        guard let targetIdx = keywords.firstIndex(where: { $0.id == targetKeywordID }) else {
            // Defensive restore: target vanished between the start
            // and end of this call (shouldn't normally happen).
            keywords.insert(moved, at: min(sourceIdx, keywords.count))
            return
        }
        moved.groupID = keywords[targetIdx].groupID
        keywords.insert(moved, at: targetIdx)
    }

    /// Move `keywordID` to the END of the supplied group (nil =
    /// Ungrouped). Used when the user drops onto a group header
    /// rather than a specific keyword row — natural for filling
    /// an empty group, or for appending past the last existing
    /// row of a populated one. No-op when `keywordID` isn't in
    /// the store.
    public func move(_ keywordID: UUID, toEndOfGroup groupID: UUID?) {
        guard let sourceIdx = keywords.firstIndex(where: { $0.id == keywordID }) else { return }
        var moved = keywords.remove(at: sourceIdx)
        moved.groupID = groupID
        if let lastIdx = keywords.lastIndex(where: { $0.groupID == groupID }) {
            keywords.insert(moved, at: lastIdx + 1)
        } else {
            // No siblings in this group — append to the very end
            // of the flat array. The section render filters by
            // groupID so position past other groups is fine.
            keywords.append(moved)
        }
    }

    // MARK: - Group mutations

    /// Add a new group with `name` (trimmed). Returns the new
    /// group's id so callers can immediately focus it or use it
    /// as the target of a follow-up keyword add. Drops empty
    /// names — the UI sheet should enforce non-empty input, but
    /// guard here too so a typoed Done doesn't create a blank
    /// header.
    @discardableResult
    public func addGroup(name: String) -> UUID? {
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        let group = KeywordGroup(name: trimmed)
        groups.append(group)
        return group.id
    }

    /// Rename an existing group. No-op when the id isn't in the
    /// store or the trimmed name is empty.
    public func renameGroup(_ groupID: UUID, to name: String) {
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty,
              let idx = groups.firstIndex(where: { $0.id == groupID }) else {
            return
        }
        groups[idx].name = trimmed
    }

    /// Delete a group. Every keyword that referenced it moves to
    /// the implicit Ungrouped section so deleting a group never
    /// silently drops keyword data. Selection survives — a
    /// selected keyword stays selected after its group is gone.
    public func removeGroup(_ groupID: UUID) {
        for idx in keywords.indices where keywords[idx].groupID == groupID {
            keywords[idx].groupID = nil
        }
        groups.removeAll { $0.id == groupID }
    }

    // MARK: - Selection

    /// Toggle the selection of a single keyword. The selection
    /// set holds every currently-selected keyword id; this method
    /// flips one of them on or off. The transcript filter ORs
    /// across the texts of every id in the set.
    public func toggleSelection(_ keyword: Keyword) {
        if selectedKeywordIDs.contains(keyword.id) {
            selectedKeywordIDs.remove(keyword.id)
        } else {
            selectedKeywordIDs.insert(keyword.id)
        }
    }

    /// Toggle the selection of every keyword in the supplied
    /// group id (nil = the Ungrouped section). If all of the
    /// group's keywords are already selected, clear them all from
    /// the selection; otherwise add them all. Lets a group-header
    /// tap "select everything below me" in one gesture.
    public func toggleGroupSelection(groupID: UUID?) {
        let groupIDs = Set(
            keywords.filter { $0.groupID == groupID }.map(\.id)
        )
        guard !groupIDs.isEmpty else { return }
        if groupIDs.isSubset(of: selectedKeywordIDs) {
            selectedKeywordIDs.subtract(groupIDs)
        } else {
            selectedKeywordIDs.formUnion(groupIDs)
        }
    }

    /// True when every keyword in the supplied group id is in
    /// the current selection set. The group header uses this to
    /// render an accent-tinted "selected" state.
    public func isGroupFullySelected(_ groupID: UUID?) -> Bool {
        let groupIDs = keywords.filter { $0.groupID == groupID }.map(\.id)
        guard !groupIDs.isEmpty else { return false }
        return groupIDs.allSatisfy { selectedKeywordIDs.contains($0) }
    }

    // MARK: - Import / Export

    /// Replace the entire list (keywords + groups) with the
    /// contents of a decoded `KeywordDocument`. Used by the
    /// import flow. Re-keys incoming entries with fresh UUIDs so
    /// two sessions importing the same source file don't collide
    /// when one of them later exports and re-imports — keyword
    /// `groupID` references are rewritten through an id remap so
    /// group memberships stay intact across the rekey.
    public func replaceContents(with document: KeywordDocument) {
        var groupIDMap: [UUID: UUID] = [:]
        let newGroups = document.groups.map { old -> KeywordGroup in
            let fresh = KeywordGroup(name: old.name)
            groupIDMap[old.id] = fresh.id
            return fresh
        }
        let newKeywords = document.keywords.map { old -> Keyword in
            let mappedGroup = old.groupID.flatMap { groupIDMap[$0] }
            return Keyword(text: old.text, groupID: mappedGroup)
        }
        groups = newGroups
        keywords = newKeywords
        selectedKeywordIDs.removeAll()
    }

    public func exportDocument() -> KeywordDocument {
        KeywordDocument(
            version: KeywordDocument.currentVersion,
            keywords: keywords,
            groups: groups
        )
    }

    public func exportJSONData() throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return try encoder.encode(exportDocument())
    }

    /// Parse a user-supplied JSON file and replace state. Throws
    /// only on decode failure; the caller surfaces the error.
    public func importJSONData(_ data: Data) throws {
        let decoder = JSONDecoder()
        let doc = try decoder.decode(KeywordDocument.self, from: data)
        replaceContents(with: doc)
    }

    // MARK: - Internal

    /// Set (or clear, with nil) a keyword's color tag. Persists
    /// like any other mutation but SUPPRESSES the `onChange` hook:
    /// the hook's consumer is the summarizer's auto re-run
    /// (keyword text/selection feed the heuristic/meeting prompts)
    /// and a color change alters nothing any prompt reads —
    /// re-running a multi-minute LLM pass over a swatch tap would
    /// be pure waste.
    public func setTagColor(_ color: KeywordTagColor?, for keywordID: UUID) {
        guard let idx = keywords.firstIndex(where: { $0.id == keywordID }),
              keywords[idx].tagColor != color else { return }
        suppressOnChangeForNextMutation = true
        keywords[idx].tagColor = color
    }

    /// One-shot latch read by `didMutate` — set by mutations that
    /// must persist without notifying the `onChange` consumer
    /// (currently only `setTagColor`).
    private var suppressOnChangeForNextMutation = false

    private func didMutate() {
        // Same deferral trick `GlossaryStore.didMutate` uses — see
        // its doc-comment for the simultaneous-access rationale.
        // Persist + observer notification run on the next
        // MainActor tick, after the `_modify` exclusive-access
        // region for the property being set has released.
        // The suppress latch is read SYNCHRONOUSLY (before the
        // deferral) so it can't leak onto an unrelated later
        // mutation.
        let notify = !suppressOnChangeForNextMutation
        suppressOnChangeForNextMutation = false
        Task { @MainActor [weak self] in
            guard let self else { return }
            do {
                try self.persist()
            } catch {
                AppLog.app.warning(
                    "KeywordStore persist failed: \(String(describing: error), privacy: .public)"
                )
            }
            if notify { self.onChange?() }
        }
    }

    private func persist() throws {
        let data = try exportJSONData()
        let dir = fileURL.deletingLastPathComponent()
        try FileManager.default.createDirectory(
            at: dir, withIntermediateDirectories: true
        )
        try data.write(to: fileURL, options: .atomic)
    }

    private static func read(from url: URL) -> KeywordDocument? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(KeywordDocument.self, from: data)
    }

    private static func defaultFileURL() -> URL {
        let base = (try? FileManager.default.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )) ?? FileManager.default.temporaryDirectory
        return base
            .appendingPathComponent("Keywords", isDirectory: true)
            .appendingPathComponent("keywords.json", isDirectory: false)
    }
}

/// JSON shape for on-disk storage AND user-facing import/export.
/// Versioned because the entry layout may evolve. The current
/// shape carries both `keywords` and `groups`; older files that
/// pre-date grouping decode with `groups = []` and every keyword
/// untagged (its synthesized Codable handles missing `groupID`
/// the same way).
public struct KeywordDocument: Sendable, Hashable, Codable {
    public static let currentVersion = 1

    public let version: Int
    public let keywords: [Keyword]
    public let groups: [KeywordGroup]

    public init(
        version: Int = KeywordDocument.currentVersion,
        keywords: [Keyword],
        groups: [KeywordGroup] = []
    ) {
        self.version = version
        self.keywords = keywords
        self.groups = groups
    }

    private enum CodingKeys: String, CodingKey {
        case version, keywords, groups
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.version = try c.decode(Int.self, forKey: .version)
        self.keywords = try c.decode([Keyword].self, forKey: .keywords)
        self.groups = try c.decodeIfPresent([KeywordGroup].self, forKey: .groups) ?? []
    }
}
