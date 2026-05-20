import Foundation
@preconcurrency import AVFoundation
@preconcurrency import Speech
import Audio
import XephonLogging

/// `SFSpeechRecognizer`-backed transcriber, exposed specifically
/// so the user's custom glossary terms can be forwarded as
/// `contextualStrings` — the only Apple on-device ASR API that
/// accepts a vocabulary hint. We don't put this on the streaming
/// hot path: `SFSpeechRecognizer`'s general transcription quality
/// is lower than `SpeechTranscriber`'s, so it earns its keep only
/// when the user explicitly wants vocabulary bias and is willing
/// to trade some general accuracy for it.
///
/// Used by `RecordingController.transcribeRange(...)` when the
/// glossary's `isASRHintEnabled` is on. The streaming pipeline
/// still uses `SpeechAnalyzerTranscriber`.
public actor SFSpeechRecognizerTranscriber {
    /// Output bundle. Confidence is the mean of per-segment
    /// confidences SFSpeechRecognizer reports (`0...1`); `nil`
    /// when the recognizer emitted no scored segments (rare,
    /// effectively means "silence").
    public struct Result: Sendable {
        public let text: String
        public let confidence: Float?
    }

    public let locale: Locale
    private let recognizer: SFSpeechRecognizer

    public init?(locale: Locale) {
        guard let r = SFSpeechRecognizer(locale: locale) else {
            return nil
        }
        self.locale = locale
        self.recognizer = r
        r.defaultTaskHint = .dictation
    }

    /// Whether the recognizer is currently allowed to run.
    /// SFSpeechRecognizer's per-locale availability flips false
    /// when the OS-level dictation model for that locale isn't
    /// resident; we honor the flag rather than letting requests
    /// fail mid-call.
    public var isAvailable: Bool {
        recognizer.isAvailable
    }

    /// Request user authorization once. Idempotent — subsequent
    /// calls return the cached status without re-prompting. The
    /// app must declare `NSSpeechRecognitionUsageDescription` in
    /// its Info.plist (Xephon already does for `SpeechAnalyzer`).
    public static func requestAuthorization() async -> SFSpeechRecognizerAuthorizationStatus {
        await withCheckedContinuation { (continuation: CheckedContinuation<SFSpeechRecognizerAuthorizationStatus, Never>) in
            SFSpeechRecognizer.requestAuthorization { status in
                continuation.resume(returning: status)
            }
        }
    }

    /// Transcribe `audio` with `hints` forwarded to
    /// `request.contextualStrings`. Returns the final hypothesis
    /// only — partial results are skipped (`shouldReportPartialResults
    /// = false`) since callers want a single deterministic answer.
    ///
    /// `requiresOnDeviceRecognition` is forced `true` to honor
    /// the project's strict on-device policy; the call throws
    /// `ASRError.modelUnavailable` if the locale doesn't ship a
    /// resident model on the user's device.
    public func transcribe(
        _ audio: AudioChunk,
        hints: [String]
    ) async throws -> Result {
        guard !audio.samples.isEmpty else {
            return Result(text: "", confidence: nil)
        }
        guard recognizer.isAvailable else {
            throw ASRError.modelUnavailable(
                reason: "SFSpeechRecognizer not available for \(locale.identifier)"
            )
        }
        let request = SFSpeechAudioBufferRecognitionRequest()
        request.shouldReportPartialResults = false
        request.requiresOnDeviceRecognition = true
        request.taskHint = .dictation
        if !hints.isEmpty {
            request.contextualStrings = hints
        }
        let buffer = try Self.makePCMBuffer(from: audio)
        request.append(buffer)
        request.endAudio()

        let recognizer = self.recognizer
        return try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Result, any Error>) in
            // `recognitionTask` callback may fire multiple times
            // (partial → final) even with shouldReportPartialResults
            // = false; guard against double-resume via a one-shot
            // latch. We hold the task only to keep it alive for
            // the duration of the call.
            let resumed = OneShotLatch()
            var task: SFSpeechRecognitionTask?
            task = recognizer.recognitionTask(with: request) { result, error in
                if let error {
                    if resumed.tryClose() {
                        continuation.resume(throwing: ASRError.underlying(error))
                    }
                    return
                }
                guard let result, result.isFinal else { return }
                let transcription = result.bestTranscription
                let segments = transcription.segments
                let avgConfidence: Float? = segments.isEmpty
                    ? nil
                    : segments.map(\.confidence).reduce(0, +) / Float(segments.count)
                if resumed.tryClose() {
                    continuation.resume(returning: Result(
                        text: transcription.formattedString,
                        confidence: avgConfidence
                    ))
                }
                _ = task // keep alive until callback fires
            }
        }
    }

    /// Convert our `AudioChunk` (16 kHz mono Float32, non-
    /// interleaved) into an `AVAudioPCMBuffer` the
    /// `SFSpeechAudioBufferRecognitionRequest` can consume.
    /// AVAudioPCMBuffer allocation + sample copy; throws if the
    /// format can't be constructed for the chunk's sample rate
    /// (shouldn't happen — the pipeline canonicalizes to 16 kHz
    /// everywhere).
    private static func makePCMBuffer(from audio: AudioChunk) throws -> AVAudioPCMBuffer {
        guard let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: audio.sampleRate,
            channels: 1,
            interleaved: false
        ) else {
            throw ASRError.audioFormatMismatch(
                expected: "float32 mono",
                got: "sampleRate=\(audio.sampleRate)"
            )
        }
        let frameCount = AVAudioFrameCount(audio.samples.count)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
            throw ASRError.audioFormatMismatch(
                expected: "AVAudioPCMBuffer alloc",
                got: "failed for \(frameCount) frames"
            )
        }
        buffer.frameLength = frameCount
        if let channel = buffer.floatChannelData?[0] {
            audio.samples.withUnsafeBufferPointer { src in
                channel.update(from: src.baseAddress!, count: Int(frameCount))
            }
        }
        return buffer
    }
}

/// One-shot latch for the SFSpeechRecognizer callback bridge.
/// `recognitionTask`'s closure can in principle fire more than
/// once (error then nil-result, or final then a duplicate final);
/// resuming a `CheckedContinuation` twice traps. The latch lets
/// the first writer through and silently no-ops every later one.
/// Class semantics (reference type) so all callback invocations
/// share state without an `@escaping inout` ceremony.
private final class OneShotLatch: @unchecked Sendable {
    private let lock = NSLock()
    private var closed = false

    func tryClose() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard !closed else { return false }
        closed = true
        return true
    }
}
