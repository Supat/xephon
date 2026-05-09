import Foundation
import CryptoKit
import Observation
import XephonLogging

/// Resolves the on-disk URL for each ML model the app needs.
///
/// Resolution order, per file:
///   1. Already installed under `Application Support/Models/<tag>/`
///      with the right SHA-256 → use it.
///   2. Present in `Bundle.main` (development build that retained the
///      `Models/` tree as bundle resources) → use it. Bundle copies
///      are trusted without re-hashing — they were embedded at build
///      time from the same Models/ directory the developer hashed.
///   3. Otherwise → download from the pinned GitHub Release, verify
///      hash, install atomically.
///
/// The actor isolates download/install state. Per-file progress is
/// mirrored to a `@MainActor @Observable ModelDownloadState` for the
/// SwiftUI setup view to bind to.
actor ModelStore {
    private let manifest: [ModelEntry]
    private let installRoot: URL
    private let urlSession: URLSession
    /// Cached resolutions from a successful `ensureModels()` call. Looked
    /// up by `installPath` (e.g. "wrime-roberta/model.onnx").
    private var resolved: [String: URL] = [:]

    let state: ModelDownloadState

    init(
        state: ModelDownloadState,
        manifest: [ModelEntry] = ModelManifest.entries,
        installRoot: URL? = nil,
        urlSession: URLSession? = nil
    ) {
        self.manifest = manifest
        self.installRoot = installRoot ?? Self.defaultInstallRoot()
        // Default URLSession with a generous resource timeout — model
        // downloads are 100s of MB and the default 7 days for resource
        // is fine, but we want to detect total stalls within a few
        // minutes rather than hours.
        if let urlSession {
            self.urlSession = urlSession
        } else {
            let config = URLSessionConfiguration.default
            config.timeoutIntervalForRequest = 60
            config.timeoutIntervalForResource = 30 * 60
            config.waitsForConnectivity = true
            self.urlSession = URLSession(configuration: config)
        }
        self.state = state
    }

    /// `Application Support/Models/<tag>/`. Excluded from iCloud backup
    /// because re-downloading is cheaper than syncing 800 MB.
    private static func defaultInstallRoot() -> URL {
        let fm = FileManager.default
        let base = (try? fm.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )) ?? fm.temporaryDirectory
        let url = base.appendingPathComponent(ModelManifest.installSubdirectory, isDirectory: true)
        try? fm.createDirectory(at: url, withIntermediateDirectories: true)
        // Best-effort exclude-from-backup; ignore failures (sandbox
        // edge cases, simulator quirks).
        var resourced = url
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        try? resourced.setResourceValues(values)
        return url
    }

    // MARK: - Public API

    /// Idempotent: ensures every manifest file is locally available.
    /// Safe to call repeatedly — second call is a no-op once resolved.
    /// Surfaces progress on `state` (MainActor) throughout.
    func ensureModels() async throws {
        await state.begin(totalEntries: manifest.count)
        defer {
            // Always finalize the UI state, even on throw — caller decides
            // whether to retry; we don't want a stuck spinner.
            Task { @MainActor in state.markIdleIfRunning() }
        }

        for (idx, entry) in manifest.enumerated() {
            await state.startEntry(index: idx, displayName: entry.displayName)
            for file in entry.files {
                try Task.checkCancellation()
                let url = try await resolve(file: file, in: entry)
                resolved[file.installPath] = url
            }
            await state.completeEntry(index: idx)
        }
        await state.markCompleted()
    }

    /// Force a fresh download next call by clearing the install dir +
    /// memo. Used by the "Re-download" affordance in setup view.
    func resetForRedownload() throws {
        let fm = FileManager.default
        if fm.fileExists(atPath: installRoot.path) {
            try fm.removeItem(at: installRoot)
        }
        try fm.createDirectory(at: installRoot, withIntermediateDirectories: true)
        resolved.removeAll()
    }

    /// Lookup helpers for the SER constructors. `ensureModels()` must
    /// have completed first; otherwise these throw.
    func resolvedURL(for installPath: String) throws -> URL {
        guard let url = resolved[installPath] else {
            throw ModelStoreError.notResolved(installPath)
        }
        return url
    }

    /// The directory containing a given file. Convenience for the wrime
    /// tokenizer init which needs the parent dir.
    func resolvedDirectory(for installPath: String) throws -> URL {
        try resolvedURL(for: installPath).deletingLastPathComponent()
    }

    // MARK: - Per-file resolution

    private func resolve(file: ModelFile, in entry: ModelEntry) async throws -> URL {
        let installURL = installRoot.appendingPathComponent(file.installPath)
        try FileManager.default.createDirectory(
            at: installURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )

        // Path 1: already installed and valid.
        if FileManager.default.fileExists(atPath: installURL.path) {
            if try await Self.sha256(of: installURL) == file.sha256 {
                AppLog.app.info("ModelStore: \(file.installPath, privacy: .public) installed (hash OK)")
                await state.fileSatisfied(name: file.assetName, source: "installed")
                return installURL
            }
            AppLog.app.warning("ModelStore: \(file.installPath, privacy: .public) hash mismatch; will re-download")
            try? FileManager.default.removeItem(at: installURL)
        }

        // Path 2: present in app bundle (dev build).
        if let bundleURL = file.bundleResource.locate() {
            AppLog.app.info("ModelStore: \(file.installPath, privacy: .public) ← bundle (dev shortcut)")
            await state.fileSatisfied(name: file.assetName, source: "bundle")
            return bundleURL
        }

        // Path 3: download from GitHub Release.
        let remoteURL = ModelManifest.releaseAssetBaseURL.appendingPathComponent(file.assetName)
        try await download(file: file, from: remoteURL, to: installURL)
        return installURL
    }

    // MARK: - Download

    private func download(file: ModelFile, from remote: URL, to install: URL) async throws {
        AppLog.app.info("ModelStore: downloading \(file.assetName, privacy: .public) (≈\(file.approximateBytes / 1024 / 1024) MB)")
        await state.startFile(
            name: file.assetName,
            expectedBytes: file.approximateBytes
        )

        let (tempURL, response) = try await urlSession.download(from: remote)
        defer { try? FileManager.default.removeItem(at: tempURL) }

        if let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
            throw ModelStoreError.httpError(asset: file.assetName, status: http.statusCode)
        }

        // Verify hash before installing — corrupted asset shouldn't
        // overwrite a working file (in case the user is re-downloading).
        let actual = try await Self.sha256(of: tempURL)
        guard actual.caseInsensitiveCompare(file.sha256) == .orderedSame else {
            throw ModelStoreError.hashMismatch(
                asset: file.assetName,
                expected: file.sha256,
                actual: actual
            )
        }

        // Atomic install: rename within the same volume.
        if FileManager.default.fileExists(atPath: install.path) {
            try FileManager.default.removeItem(at: install)
        }
        try FileManager.default.moveItem(at: tempURL, to: install)

        let bytes = (try? install.resourceValues(forKeys: [.fileSizeKey]).fileSize).map(Int64.init) ?? 0
        await state.completeFile(name: file.assetName, bytes: bytes)
        AppLog.app.info("ModelStore: installed \(file.installPath, privacy: .public) (\(bytes / 1024 / 1024) MB)")
    }

    private static func sha256(of url: URL) async throws -> String {
        try await Task.detached(priority: .userInitiated) {
            let handle = try FileHandle(forReadingFrom: url)
            defer { try? handle.close() }
            var hasher = SHA256()
            // 4 MiB chunks — small enough to keep peak memory bounded
            // when hashing a 300 MB file, large enough to amortize
            // syscall overhead.
            let chunkSize = 4 * 1024 * 1024
            while true {
                let data = try handle.read(upToCount: chunkSize) ?? Data()
                if data.isEmpty { break }
                hasher.update(data: data)
            }
            return hasher.finalize().map { String(format: "%02x", $0) }.joined()
        }.value
    }
}

// MARK: - Observable progress

/// MainActor-confined view model the SwiftUI setup view reads.
/// `ModelStore` mutates it via Task hops; UI just observes.
@MainActor
@Observable
final class ModelDownloadState {
    enum Phase: Sendable {
        case idle
        case running
        case completed
        case failed(String)
    }

    private(set) var phase: Phase = .idle
    private(set) var totalEntries: Int = 0
    private(set) var completedEntries: Int = 0
    /// Active file being downloaded (assetName).
    private(set) var currentFile: String?
    /// Active entry's user-facing label.
    private(set) var currentEntry: String?

    /// Per-file rolling totals. Keyed by assetName so repeat resolves
    /// (after a Retry) overwrite cleanly rather than appending.
    private(set) var fileBytes: [String: Int64] = [:]
    private(set) var fileExpected: [String: Int64] = [:]
    private(set) var fileStatus: [String: FileStatus] = [:]

    enum FileStatus: Sendable, Equatable {
        case pending
        case satisfied(source: String)  // "installed" or "bundle"
        case downloading
        case completed
        case failed(String)
    }

    var totalBytes: Int64 { fileExpected.values.reduce(0, +) }
    var downloadedBytes: Int64 { fileBytes.values.reduce(0, +) }
    var fractionComplete: Double {
        let total = max(1, totalEntries)
        return Double(completedEntries) / Double(total)
    }

    func begin(totalEntries: Int) {
        self.phase = .running
        self.totalEntries = totalEntries
        self.completedEntries = 0
        self.currentFile = nil
        self.currentEntry = nil
    }

    func startEntry(index: Int, displayName: String) {
        currentEntry = displayName
    }

    func completeEntry(index: Int) {
        completedEntries = index + 1
        currentFile = nil
    }

    func fileSatisfied(name: String, source: String) {
        fileStatus[name] = .satisfied(source: source)
    }

    func startFile(name: String, expectedBytes: Int64) {
        currentFile = name
        fileExpected[name] = expectedBytes
        fileBytes[name] = 0
        fileStatus[name] = .downloading
    }

    func completeFile(name: String, bytes: Int64) {
        fileBytes[name] = bytes
        fileExpected[name] = bytes
        fileStatus[name] = .completed
    }

    func markCompleted() {
        phase = .completed
        currentFile = nil
        currentEntry = nil
    }

    func markFailed(_ message: String) {
        phase = .failed(message)
    }

    /// Defensive cleanup invoked from `ModelStore.ensureModels()`'s defer
    /// path — only flips `running` → `idle` so a successful `completed`
    /// state isn't clobbered.
    func markIdleIfRunning() {
        if case .running = phase { phase = .idle }
    }

    func reset() {
        phase = .idle
        totalEntries = 0
        completedEntries = 0
        currentFile = nil
        currentEntry = nil
        fileBytes.removeAll()
        fileExpected.removeAll()
        fileStatus.removeAll()
    }
}

enum ModelStoreError: Error, CustomStringConvertible {
    case httpError(asset: String, status: Int)
    case hashMismatch(asset: String, expected: String, actual: String)
    case notResolved(String)

    var description: String {
        switch self {
        case .httpError(let asset, let status):
            return "Download failed for \(asset): HTTP \(status)"
        case .hashMismatch(let asset, let expected, let actual):
            return "Hash mismatch for \(asset): expected \(expected), got \(actual)"
        case .notResolved(let path):
            return "Model not resolved: \(path) (call ensureModels() first)"
        }
    }
}
