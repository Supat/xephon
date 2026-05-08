import Foundation
@preconcurrency import WhisperKit
import Audio
import XephonLogging

// Fallback ASR: WhisperKit + a Core ML Whisper variant.
// Default uses Apple-accelerated `large-v3-turbo` from argmaxinc/whisperkit-coreml,
// which is multilingual and strong on Japanese; swap `model` for a Kotoba-Whisper
// Core ML conversion when one is available.
public actor WhisperKitTranscriber: Transcriber {
    public let locale: Locale
    private let modelName: String
    // WhisperKit isn't Sendable; serialize via this actor.
    private nonisolated(unsafe) var pipeline: WhisperKit?

    public init(
        locale: Locale = Locale(identifier: "ja_JP"),
        modelName: String = "openai_whisper-large-v3-turbo"
    ) {
        self.locale = locale
        self.modelName = modelName
    }

    public func transcribe(_ buffer: Audio.AudioChunk) async throws -> [ASRSegment] {
        let kit = try await ensurePipeline()

        let languageCode = locale.language.languageCode?.identifier // "ja"
        let options = DecodingOptions(
            verbose: false,
            task: .transcribe,
            language: languageCode,
            usePrefillPrompt: true,
            withoutTimestamps: false,
            wordTimestamps: false
        )

        do {
            let results: [TranscriptionResult] = try await kit.transcribe(
                audioArray: buffer.samples,
                decodeOptions: options,
                callback: nil,
                segmentCallback: nil
            )
            let segments = results.flatMap { $0.segments }
            AppLog.asr.info(
                "WhisperKit produced \(segments.count, privacy: .public) segments (model=\(self.modelName, privacy: .public))"
            )
            return segments.map { seg in
                ASRSegment(
                    text: seg.text.trimmingCharacters(in: .whitespacesAndNewlines),
                    start: TimeInterval(seg.start) + buffer.timestamp,
                    end: TimeInterval(seg.end) + buffer.timestamp,
                    // avgLogprob is in log space; exp() yields a 0…1 token-geo-mean confidence proxy.
                    confidence: Float(min(1.0, max(0.0, exp(Double(seg.avgLogprob)))))
                )
            }
        } catch {
            throw ASRError.underlying(error)
        }
    }

    private func ensurePipeline() async throws -> WhisperKit {
        if let pipeline { return pipeline }
        AppLog.asr.info("Loading WhisperKit model: \(self.modelName, privacy: .public)")
        let config = WhisperKitConfig(
            model: modelName,
            verbose: false,
            logLevel: .info,
            prewarm: true,
            load: true,
            download: true
        )
        do {
            let kit = try await WhisperKit(config)
            self.pipeline = kit
            return kit
        } catch {
            throw ASRError.modelUnavailable(reason: String(describing: error))
        }
    }
}
