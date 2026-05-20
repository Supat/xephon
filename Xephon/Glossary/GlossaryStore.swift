import Foundation
import SERText
import XephonLogging

/// User-editable custom glossary, persisted to Application
/// Support as a single JSON file and observable by SwiftUI for
/// live UI updates.
///
/// The store is the single source of truth: the sheet binds
/// directly to its properties, and the `RecordingController`
/// pushes a fresh `LexiconBias` snapshot to the pipeline via
/// `onChange` whenever entries / `isEnabled` mutate. Snapshots
/// are value-typed and Sendable, so they flow across the
/// pipeline's actor boundary without sharing mutable state.
///
/// Persistence policy: write-through on every mutation. The
/// glossary is tiny (a few hundred entries at most, each ~30 B
/// JSON) so the cost is negligible compared to keeping
/// in-memory and on-disk state in lockstep, which matters when
/// the user expects "I tapped the toggle off" to survive an
/// app restart.
@Observable
@MainActor
public final class GlossaryStore {
    /// Global enable. When false, `currentLexicon` returns an
    /// empty bias snapshot regardless of `entries` — letting the
    /// user park a populated glossary without removing it.
    public var isEnabled: Bool {
        didSet { didMutate() }
    }

    /// Master toggle for the ASR-hint pathway. When on, the
    /// streaming pipeline forwards terms (those with
    /// `useAsASRHint == true`) to `SFSpeechRecognizer.contextualStrings`
    /// on the re-evaluate / transcribe-range path. Off by
    /// default — the user has to explicitly opt in because
    /// SFSpeechRecognizer's general transcription quality is
    /// lower than SpeechTranscriber's; the trade is bias toward
    /// hinted terms vs. general accuracy.
    public var isASRHintEnabled: Bool {
        didSet { didMutate() }
    }

    /// Ordered glossary entries. UI surfaces them in input order
    /// (newest at the bottom); reorder via drag isn't supported
    /// today — the list is short enough that delete + re-add
    /// covers it.
    public var entries: [LexiconBiasEntry] {
        didSet { didMutate() }
    }

    /// Pushed by the controller; fires after any mutation that
    /// changes the lexicon snapshot the pipeline should see.
    /// `onChange` is called *after* persistence so the on-disk
    /// state is already consistent if the controller looks at
    /// it.
    public var onChange: (@MainActor () -> Void)?

    /// Snapshot suitable for `SwitchingTextSER.setLexicon`.
    /// When `isEnabled` is false, hand back an empty bias so
    /// the pipeline can treat "off" and "no entries" uniformly.
    public var currentLexicon: LexiconBias {
        isEnabled ? LexiconBias(entries: entries) : LexiconBias(entries: [])
    }

    /// Deduplicated list of `term` strings to feed
    /// `SFSpeechRecognizer.contextualStrings` on the next
    /// hinted ASR call. Empty when `isASRHintEnabled` is off OR
    /// no entry has `useAsASRHint == true` — caller treats both
    /// as "no hints" identically. Whitespace-only entries are
    /// dropped to keep SFSpeechRecognizer from receiving empty
    /// strings.
    public var currentASRHints: [String] {
        guard isASRHintEnabled else { return [] }
        var seen = Set<String>()
        var out: [String] = []
        for entry in entries where entry.useAsASRHint {
            let term = entry.term.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !term.isEmpty else { continue }
            if seen.insert(term).inserted { out.append(term) }
        }
        return out
    }

    private let fileURL: URL

    /// Initialize from the on-disk JSON. A missing / corrupt
    /// file degrades to an empty disabled glossary — the user
    /// can re-import or hand-build from there. Corrupt-file
    /// recovery doesn't quarantine the bad file on the
    /// assumption that the user would rather have a working
    /// glossary panel than a forensics trail.
    public init() {
        self.fileURL = Self.defaultFileURL()
        let loaded = Self.read(from: fileURL)
        self.entries = loaded?.entries ?? []
        self.isEnabled = loaded?.isEnabled ?? false
        self.isASRHintEnabled = loaded?.isASRHintEnabled ?? false
    }

    /// Initializer for tests / previews: use an injected file
    /// URL + initial document, no disk read.
    public init(fileURL: URL, document: GlossaryDocument) {
        self.fileURL = fileURL
        self.entries = document.entries
        self.isEnabled = document.isEnabled
        self.isASRHintEnabled = document.isASRHintEnabled
    }

    // MARK: - Mutations

    public func add(_ entry: LexiconBiasEntry) {
        entries.append(entry)
    }

    public func remove(at offsets: IndexSet) {
        entries.remove(atOffsets: offsets)
    }

    public func update(_ entry: LexiconBiasEntry) {
        guard let idx = entries.firstIndex(where: { $0.id == entry.id }) else {
            return
        }
        entries[idx] = entry
    }

    public func removeAll() {
        entries.removeAll()
    }

    // MARK: - Import / Export

    /// Replace the entire glossary with the contents of a
    /// decoded `GlossaryDocument`. Used by the import flow.
    /// `isEnabled` is taken from the document so a user can
    /// share a "glossary already armed" file or "glossary
    /// off — flip the switch yourself" file at their discretion.
    public func replaceContents(with document: GlossaryDocument) {
        // Re-key incoming entries with fresh UUIDs so two
        // sessions importing the same source don't collide
        // when one of them later exports and re-imports.
        entries = document.entries.map {
            LexiconBiasEntry(
                term: $0.term,
                label: $0.label,
                weight: $0.weight,
                useAsASRHint: $0.useAsASRHint
            )
        }
        isEnabled = document.isEnabled
        isASRHintEnabled = document.isASRHintEnabled
    }

    /// Encode the current state for `Transferable` / file
    /// export. Versioned so future schema changes can be
    /// detected at import time.
    public func exportDocument() -> GlossaryDocument {
        GlossaryDocument(
            version: GlossaryDocument.currentVersion,
            isEnabled: isEnabled,
            isASRHintEnabled: isASRHintEnabled,
            entries: entries
        )
    }

    public func exportJSONData() throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return try encoder.encode(exportDocument())
    }

    /// Parse a user-supplied JSON file and replace state. Throws
    /// only on decode failure; the caller surfaces the error to
    /// the user.
    public func importJSONData(_ data: Data) throws {
        let decoder = JSONDecoder()
        let doc = try decoder.decode(GlossaryDocument.self, from: data)
        replaceContents(with: doc)
    }

    // MARK: - Internal

    private func didMutate() {
        // Defer past the current modify-access scope. `@Observable`
        // synthesizes a `_modify` accessor pattern where `didSet`
        // runs *inside* the exclusive-access region for the
        // property being set — so reading `entries` / `isEnabled`
        // again from within `didSet` (which we do via
        // `exportDocument()` in `persist`, and again via
        // `currentLexicon` inside the controller's `onChange`
        // hook) trips Swift's runtime exclusivity check and
        // crashes with "Simultaneous accesses ... modification
        // requires exclusive access." Hopping to the next
        // MainActor tick lets the set complete (modify access
        // released) before persist + observer notification run.
        Task { @MainActor [weak self] in
            guard let self else { return }
            do {
                try self.persist()
            } catch {
                // Persist failures shouldn't block UI mutations —
                // log + carry on. A retry on the next mutation
                // will usually succeed; if not, the user sees
                // nothing broken until app restart.
                AppLog.app.warning(
                    "GlossaryStore persist failed: \(String(describing: error), privacy: .public)"
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

    private static func read(from url: URL) -> GlossaryDocument? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(GlossaryDocument.self, from: data)
    }

    private static func defaultFileURL() -> URL {
        let base = (try? FileManager.default.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )) ?? FileManager.default.temporaryDirectory
        return base
            .appendingPathComponent("Glossary", isDirectory: true)
            .appendingPathComponent("glossary.json", isDirectory: false)
    }
}

/// JSON shape for on-disk storage AND for user-facing
/// import/export. Versioned because the entry layout will
/// probably evolve (regex matching, per-entry enable toggle,
/// language tag) and we want old files to still load.
public struct GlossaryDocument: Sendable, Hashable, Codable {
    public static let currentVersion = 1

    public let version: Int
    public let isEnabled: Bool
    public let isASRHintEnabled: Bool
    public let entries: [LexiconBiasEntry]

    public init(
        version: Int = GlossaryDocument.currentVersion,
        isEnabled: Bool,
        isASRHintEnabled: Bool = false,
        entries: [LexiconBiasEntry]
    ) {
        self.version = version
        self.isEnabled = isEnabled
        self.isASRHintEnabled = isASRHintEnabled
        self.entries = entries
    }

    // Custom Codable so old glossary JSON files (no
    // `isASRHintEnabled` key) decode as `false` rather than
    // failing. Mirrors the per-entry `useAsASRHint` decode path.
    private enum CodingKeys: String, CodingKey {
        case version, isEnabled, isASRHintEnabled, entries
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.version = try c.decode(Int.self, forKey: .version)
        self.isEnabled = try c.decode(Bool.self, forKey: .isEnabled)
        self.isASRHintEnabled = try c.decodeIfPresent(Bool.self, forKey: .isASRHintEnabled) ?? false
        self.entries = try c.decode([LexiconBiasEntry].self, forKey: .entries)
    }
}
