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
    /// Persisted hash cache. Keyed by `<manifest version>::<URL.path>`
    /// in UserDefaults so each cold launch can skip rehashing the same
    /// 800 MB of weights when nothing has changed. Bumping
    /// `ModelManifest.releaseTag` invalidates the keyspace automatically.
    private let userDefaults: UserDefaults
    /// In-flight hydration. Tracked so `resetForRedownload()` can
    /// cancel + await it before wiping the install dir, instead of
    /// pulling files out from under an active download.
    private var hydrationTask: Task<Void, any Error>?

    let state: ModelDownloadState

    init(
        state: ModelDownloadState,
        manifest: [ModelEntry] = ModelManifest.entries,
        installRoot: URL? = nil,
        urlSession: URLSession? = nil,
        userDefaults: UserDefaults = .standard
    ) throws {
        self.manifest = manifest
        self.userDefaults = userDefaults
        self.installRoot = try installRoot ?? Self.defaultInstallRoot()
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
    ///
    /// Throws if Application Support can't be reached — the previous
    /// fallback to `temporaryDirectory` was a footgun: tmp is purgeable,
    /// so a transient sandbox issue at first launch would silently install
    /// 800 MB into a directory iOS could nuke at any time. Better to
    /// surface the failure and let the user retry / restart.
    private static func defaultInstallRoot() throws -> URL {
        let fm = FileManager.default
        let base = try fm.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )
        let url = base.appendingPathComponent(ModelManifest.installSubdirectory, isDirectory: true)
        try fm.createDirectory(at: url, withIntermediateDirectories: true)
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
    /// If a previous call is in flight, awaits it instead of starting a
    /// concurrent pass (the file system + state mutations aren't safe
    /// to interleave).
    /// Surfaces progress on `state` (MainActor) throughout.
    func ensureModels() async throws {
        if let existing = hydrationTask {
            try await existing.value
            return
        }
        let task = Task { try await self.runEnsureModels() }
        hydrationTask = task
        defer { hydrationTask = nil }
        try await task.value
    }

    private func runEnsureModels() async throws {
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
    /// Cancels any in-flight `ensureModels()` first so we don't pull
    /// files out from under an active download (which would yield
    /// half-installed state and a confusing failure mode).
    func resetForRedownload() async throws {
        if let inflight = hydrationTask {
            inflight.cancel()
            // Drain rather than throw — we expect a CancellationError
            // here and that's not an error from the caller's view.
            _ = try? await inflight.value
        }
        let fm = FileManager.default
        if fm.fileExists(atPath: installRoot.path) {
            try fm.removeItem(at: installRoot)
        }
        try fm.createDirectory(at: installRoot, withIntermediateDirectories: true)
        resolved.removeAll()
    }

    /// True iff every file declared by the given optional entry is
    /// present on disk. Doesn't re-hash — a tampered copy will be
    /// caught at `ensureOptional` time (or at the inferencer's own
    /// load step). Cheap enough to call from a UI getter that
    /// renders the "Summarize session" button state.
    func isOptionalInstalled(id: String) -> Bool {
        guard let entry = ModelManifest.optionalEntries.first(where: { $0.id == id }) else {
            return false
        }
        for file in entry.files {
            let url = installRoot.appendingPathComponent(file.installPath)
            if !FileManager.default.fileExists(atPath: url.path) {
                return false
            }
        }
        return true
    }

    /// Directory the optional entry's files live in. Returns nil
    /// when the entry isn't installed yet. Used by the summarizer
    /// actor as the `modelDirectory` it passes to MLXLMCommon's
    /// `ModelConfiguration(directory:)`.
    func optionalDirectory(id: String) -> URL? {
        guard isOptionalInstalled(id: id),
              let entry = ModelManifest.optionalEntries.first(where: { $0.id == id }),
              let firstFile = entry.files.first
        else { return nil }
        let firstURL = installRoot.appendingPathComponent(firstFile.installPath)
        return firstURL.deletingLastPathComponent()
    }

    /// Download (or verify-already-installed) one optional entry on
    /// demand. Surfaces progress through the same `state` the
    /// first-launch hydration uses, so the SetupView-style progress
    /// chrome can render the summarizer download inline in the
    /// Settings card. No-op when the entry is already installed
    /// and its hashes verify.
    func ensureOptional(id: String) async throws {
        guard let entry = ModelManifest.optionalEntries.first(where: { $0.id == id }) else {
            throw ModelStoreError.notResolved(id)
        }
        await state.begin(totalEntries: 1)
        defer { Task { @MainActor in state.markIdleIfRunning() } }
        await state.startEntry(index: 0, displayName: entry.displayName)
        for file in entry.files {
            try Task.checkCancellation()
            let url = try await resolve(file: file, in: entry)
            resolved[file.installPath] = url
        }
        await state.completeEntry(index: 0)
        await state.markCompleted()
    }

    /// Remove all files for an optional entry. Surfaces as a
    /// "Remove summarizer model" action in Settings so the user
    /// can reclaim ~4 GB without resetting the entire manifest.
    func removeOptional(id: String) throws {
        guard let entry = ModelManifest.optionalEntries.first(where: { $0.id == id }) else {
            return
        }
        let fm = FileManager.default
        for file in entry.files {
            let url = installRoot.appendingPathComponent(file.installPath)
            try? fm.removeItem(at: url)
            resolved.removeValue(forKey: file.installPath)
        }
        // Drop the parent directory too if it's now empty (and not
        // shared with anything else under the install root).
        if let firstFile = entry.files.first {
            let parent = installRoot
                .appendingPathComponent(firstFile.installPath)
                .deletingLastPathComponent()
            if let contents = try? fm.contentsOfDirectory(atPath: parent.path), contents.isEmpty {
                try? fm.removeItem(at: parent)
            }
        }
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

        // Path 1: already installed and valid (memoized hash).
        if FileManager.default.fileExists(atPath: installURL.path) {
            if try await verifiedHash(at: installURL, expected: file.sha256) {
                AppLog.app.info("ModelStore: \(file.installPath, privacy: .public) installed (hash OK)")
                await state.fileSatisfied(name: file.assetName, source: "installed")
                return installURL
            }
            AppLog.app.warning("ModelStore: \(file.installPath, privacy: .public) hash mismatch; will re-download")
            try? FileManager.default.removeItem(at: installURL)
            invalidateCachedHash(at: installURL)
        }

        // Path 2: present in app bundle. Verify hash too — a stale bundle
        // (e.g. Models/ on disk diverged from the manifest between an
        // FP16 conversion and a manifest SHA bump) used to silently load
        // a wrong model and only show up as a runtime SER failure with
        // no clear root cause.
        if let bundleURL = file.bundleResource.locate() {
            if try await verifiedHash(at: bundleURL, expected: file.sha256) {
                AppLog.app.info("ModelStore: \(file.installPath, privacy: .public) ← bundle (hash OK)")
                await state.fileSatisfied(name: file.assetName, source: "bundle")
                return bundleURL
            }
            AppLog.app.warning("ModelStore: \(file.installPath, privacy: .public) bundle hash mismatch; falling through to download")
            invalidateCachedHash(at: bundleURL)
        }

        // Path 3: download. Prefer the file's explicit
        // `directRemoteURL` when set (Qwen MLX files come from
        // huggingface.co/mlx-community/… because they exceed GitHub
        // Releases' 2 GB Free-tier asset limit); otherwise compose
        // the URL from `releaseAssetBaseURL + assetName`. Either
        // path verifies SHA-256 before atomic install.
        let remoteURL = file.directRemoteURL
            ?? ModelManifest.releaseAssetBaseURL.appendingPathComponent(file.assetName)
        try await download(file: file, from: remoteURL, to: installURL)
        return installURL
    }

    // MARK: - Cached hashing
    //
    // SHA-256 of an 800 MB tree takes ~5 seconds on first cold launch
    // even with the file system cache warm. Memoize per-path-per-version
    // in UserDefaults so subsequent launches skip the work entirely while
    // still catching corruption (file size changes invalidate the cache).
    // Keyed by the manifest version so a `releaseTag` bump auto-flushes.

    private static let hashCacheKey = "ModelStore.hashCache.v1"

    private func cachedHashKey(for url: URL) -> String {
        "\(ModelManifest.releaseTag)::\(url.path)"
    }

    private func cachedHashEntry(at url: URL) -> (size: Int64, hash: String)? {
        guard let dict = userDefaults.dictionary(forKey: Self.hashCacheKey),
              let entry = dict[cachedHashKey(for: url)] as? [String: Any],
              let size = entry["size"] as? NSNumber,
              let hash = entry["hash"] as? String
        else { return nil }
        return (size.int64Value, hash)
    }

    private func storeHash(_ hash: String, size: Int64, at url: URL) {
        var dict = userDefaults.dictionary(forKey: Self.hashCacheKey) ?? [:]
        dict[cachedHashKey(for: url)] = ["size": size, "hash": hash]
        userDefaults.set(dict, forKey: Self.hashCacheKey)
    }

    private func invalidateCachedHash(at url: URL) {
        var dict = userDefaults.dictionary(forKey: Self.hashCacheKey) ?? [:]
        dict.removeValue(forKey: cachedHashKey(for: url))
        userDefaults.set(dict, forKey: Self.hashCacheKey)
    }

    /// Returns true if the file at `url` hashes to `expected`. Uses the
    /// per-version cache to skip rehashing when both the cached size and
    /// hash match. Side-effect: stores a fresh entry on cache miss.
    private func verifiedHash(at url: URL, expected: String) async throws -> Bool {
        let currentSize = (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize).map(Int64.init) ?? -1
        if let cached = cachedHashEntry(at: url),
           cached.size == currentSize,
           cached.hash.caseInsensitiveCompare(expected) == .orderedSame {
            return true
        }
        let actual = try await Self.sha256(of: url)
        let matches = actual.caseInsensitiveCompare(expected) == .orderedSame
        if matches {
            storeHash(actual, size: currentSize, at: url)
        } else {
            invalidateCachedHash(at: url)
        }
        return matches
    }

    // MARK: - Download

    private func download(file: ModelFile, from remote: URL, to install: URL) async throws {
        AppLog.app.info("ModelStore: downloading \(file.assetName, privacy: .public) (≈\(file.approximateBytes / 1024 / 1024) MB)")
        await state.startFile(
            name: file.assetName,
            expectedBytes: file.approximateBytes
        )

        // Per-task delegate that forwards
        // `urlSession(_:downloadTask:didWriteData:…)` callbacks
        // into the shared `state` so the Settings card's circular
        // progress fills smoothly during the big safetensors
        // download instead of staying at 0% until the file lands.
        // Captures `state` actor + asset name via the closure;
        // each task gets its own tracker so concurrent downloads
        // (none today, but safe by construction) don't interleave.
        let state = state
        let assetName = file.assetName
        let tracker = DownloadProgressTracker { totalWritten, totalExpected in
            Task { @MainActor in
                state.updateFileProgress(
                    name: assetName,
                    bytesWritten: totalWritten,
                    totalExpected: totalExpected
                )
            }
        }
        let (tempURL, response) = try await urlSession.download(
            from: remote,
            delegate: tracker
        )
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
        // Seed the hash cache with the freshly verified value so the next
        // launch skips the rehash for this file.
        storeHash(actual, size: bytes, at: install)
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

    /// Called from `ModelStore.download`'s URLSessionDownload-
    /// Delegate every time the system reports more bytes
    /// received. Updates the per-file tally so the Settings card's
    /// circular progress can render a smooth fill across the long
    /// safetensors download. `totalExpected` from the URL session
    /// is authoritative once the response lands — overwrites the
    /// best-effort `approximateBytes` from the manifest.
    func updateFileProgress(name: String, bytesWritten: Int64, totalExpected: Int64) {
        fileBytes[name] = bytesWritten
        if totalExpected > 0 {
            fileExpected[name] = totalExpected
        }
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

/// Per-download delegate that surfaces `URLSessionDownloadTask`'s
/// progress callbacks as `(bytesWritten, totalExpected)` pairs.
/// Passed as the `delegate:` argument to
/// `URLSession.download(from:delegate:)` so each download gets its
/// own progress channel without touching the shared session's
/// delegate. Inherits from `URLSessionDownloadDelegate` to receive
/// the `didWriteData` callback (URLSessionTaskDelegate alone
/// doesn't surface byte-level progress for download tasks).
///
/// `@unchecked Sendable` is required because `NSObject` is not
/// Sendable. Safe in this case: every stored property is `let`
/// (only `onProgress`, itself `@Sendable`), so there's no mutable
/// shared state to race over. The delegate callbacks are simple
/// pass-throughs that forward to the immutable closure.
private final class DownloadProgressTracker: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    let onProgress: @Sendable (Int64, Int64) -> Void

    init(onProgress: @escaping @Sendable (Int64, Int64) -> Void) {
        self.onProgress = onProgress
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64,
        totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        onProgress(totalBytesWritten, totalBytesExpectedToWrite)
    }

    /// Required by `URLSessionDownloadDelegate` but a no-op here —
    /// `URLSession.download(from:delegate:)` returns the temp file
    /// URL directly via the async return value, so we don't need
    /// to relocate it from the delegate callback.
    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didFinishDownloadingTo location: URL
    ) {}
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
