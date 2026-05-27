import Foundation
import XephonLogging

/// One user-managed keyword. The wrapping `id` lets SwiftUI track
/// list rows across edits / reorders without falling back to
/// `\.self`-on-String (which breaks the moment two keywords share
/// text, even transiently while typing).
public struct Keyword: Sendable, Hashable, Codable, Identifiable {
    public let id: UUID
    public var text: String

    public init(id: UUID = UUID(), text: String) {
        self.id = id
        self.text = text
    }
}

/// User-managed keyword list, persisted to Application Support as
/// a single JSON file and observable by SwiftUI for live UI
/// updates. Same lifecycle shape as `GlossaryStore`: the store is
/// the single source of truth, mutations write through to disk on
/// the next MainActor tick, and an optional `onChange` hook fires
/// after persistence completes for downstream consumers.
///
/// The list itself carries no semantics — it's a free-form keyword
/// bank the user can curate and reuse from features that opt into
/// it. Persistence policy matches the glossary's write-through-on-
/// every-mutation rule for the same reason: the file is small,
/// and "I tapped × on a keyword" needs to survive an app restart.
@Observable
@MainActor
public final class KeywordStore {
    /// Ordered list of keywords. Surfaced in input order (newest at
    /// the bottom). Reorder via drag isn't supported today — the
    /// list is short enough that delete + re-add covers it.
    public var keywords: [Keyword] {
        didSet { didMutate() }
    }

    /// Called after every persisted mutation. Hook for consumers
    /// that want to react to keyword-list changes without polling
    /// the store. Optional because no in-tree consumer wires it
    /// yet — the card mutates the store directly.
    public var onChange: (@MainActor () -> Void)?

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
    }

    /// Test / preview initializer: injected file URL + initial
    /// document, no disk read.
    public init(fileURL: URL, document: KeywordDocument) {
        self.fileURL = fileURL
        self.keywords = document.keywords
    }

    // MARK: - Mutations

    /// Append a keyword. Trims whitespace and drops empty input so
    /// the list can't accumulate blank rows from accidental
    /// submits. No dedup — the user might want duplicates (e.g.
    /// the same surface form in two different intended contexts).
    public func add(_ text: String) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        keywords.append(Keyword(text: trimmed))
    }

    public func remove(at offsets: IndexSet) {
        keywords.remove(atOffsets: offsets)
    }

    public func remove(id: UUID) {
        keywords.removeAll { $0.id == id }
    }

    public func removeAll() {
        keywords.removeAll()
    }

    // MARK: - Import / Export

    /// Replace the entire list with the contents of a decoded
    /// `KeywordDocument`. Used by the import flow. Re-keys
    /// incoming entries with fresh UUIDs so two sessions importing
    /// the same source file don't collide when one of them later
    /// exports and re-imports.
    public func replaceContents(with document: KeywordDocument) {
        keywords = document.keywords.map { Keyword(text: $0.text) }
    }

    public func exportDocument() -> KeywordDocument {
        KeywordDocument(
            version: KeywordDocument.currentVersion,
            keywords: keywords
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

    private func didMutate() {
        // Same deferral trick `GlossaryStore.didMutate` uses — see
        // its doc-comment for the simultaneous-access rationale.
        // Persist + observer notification run on the next
        // MainActor tick, after the `_modify` exclusive-access
        // region for the property being set has released.
        Task { @MainActor [weak self] in
            guard let self else { return }
            do {
                try self.persist()
            } catch {
                AppLog.app.warning(
                    "KeywordStore persist failed: \(String(describing: error), privacy: .public)"
                )
            }
            self.onChange?()
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
/// Versioned because the entry layout may evolve (tagging,
/// per-keyword enable, ordering hints) and we want old files to
/// still load.
public struct KeywordDocument: Sendable, Hashable, Codable {
    public static let currentVersion = 1

    public let version: Int
    public let keywords: [Keyword]

    public init(
        version: Int = KeywordDocument.currentVersion,
        keywords: [Keyword]
    ) {
        self.version = version
        self.keywords = keywords
    }
}
