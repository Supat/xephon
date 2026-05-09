import Foundation
import OnnxRuntimeBindings

/// Process-wide ORT runtime singletons. ONNX Runtime documentation
/// states that `ORTEnv` should exist as a single instance per process
/// — multiple instances duplicate native logging state and threadpool
/// configuration without any benefit. Before this module each ORT
/// session built (W2V2, emotion2vec, DeBERTa, plus every CoreML-EP →
/// CPU rebuild on the failure path) constructed its own `ORTEnv`,
/// silently leaking native state on each rebuild.
///
/// The cached `ORTEnv` is created lazily on first access and lives
/// until the process exits. `fatalError` on init failure is
/// intentional: the SER subsystem cannot function without the env, so
/// surfacing the failure as a fatal makes the misconfiguration
/// explicit rather than letting every subsequent session's
/// constructor fail with the same root cause.
public enum ORTRuntime {
    /// `ORTEnv` is an Objective-C class with its own internal thread
    /// safety; it isn't formally Sendable, so Swift 6's strict
    /// concurrency rejects a plain `static let`. `nonisolated(unsafe)`
    /// asserts the access is safe under existing synchronization
    /// (ORT's own native locking) — accurate here, since ORTEnv is
    /// designed to be a singleton consumed concurrently.
    public nonisolated(unsafe) static let sharedEnv: ORTEnv = {
        do {
            return try ORTEnv(loggingLevel: .warning)
        } catch {
            fatalError("ORTRuntime: ORTEnv init failed: \(error)")
        }
    }()
}
