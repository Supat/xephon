import Foundation
import Audio

/// Streaming variant of `Transcriber`. The caller pushes audio incrementally
/// via `feed(_:)` and consumes finalized segments from the AsyncStream returned
/// by `start()`. Only segments outside the analyzer's volatile range — i.e.
/// committed sentence boundaries — are emitted.
public protocol StreamingTranscriber: Actor {
    var locale: Locale { get }
    /// Begin a streaming session. The returned stream yields `ASRSegment`s as
    /// they finalize. Closes when `finish()` is called.
    func start() async throws -> AsyncStream<ASRSegment>
    /// Push 16 kHz mono Float32 samples into the analyzer. Non-blocking.
    func feed(_ buffer: AudioChunk) async
    /// Flush any remaining audio, emit any pending finals, then close the stream.
    func finish() async
    /// In-flight transcript that hasn't been finalized yet (i.e., text within
    /// the analyzer's volatile range). Empty when nothing is being revised.
    /// Useful for "live preview" UI.
    var volatileText: String { get async }
}

public extension StreamingTranscriber {
    var volatileText: String { get async { "" } }
}
