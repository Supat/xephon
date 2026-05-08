import Foundation
import FluidAudio
import Audio
import XephonLogging

// Fallback ASR: FluidAudio Qwen3-ASR (Core ML, iOS 18+).
// Strong Japanese/Chinese/Korean/Vietnamese performance; benchmark against
// Kotoba-Whisper-v2.0 on your own conversational data before committing.
//
// Note: Qwen3-ASR returns one transcript blob per audio buffer (no per-segment
// timestamps), so we emit a single ASRSegment spanning the whole input. Use
// SpeechAnalyzer or WhisperKit when word-level timing matters.
public actor Qwen3ASRTranscriber: Transcriber {
    public let locale: Locale
    private let manager: Qwen3AsrManager
    private var loaded = false

    public init(locale: Locale = Locale(identifier: "ja_JP")) {
        self.locale = locale
        self.manager = Qwen3AsrManager()
    }

    public func transcribe(_ buffer: AudioChunk) async throws -> [ASRSegment] {
        if !loaded {
            try await loadModels()
        }
        let durationSeconds = Double(buffer.samples.count) / buffer.sampleRate
        let languageHint = Self.languageHint(from: locale)
        do {
            let text = try await manager.transcribe(
                audioSamples: buffer.samples,
                language: languageHint
            )
            AppLog.asr.info(
                "Qwen3-ASR transcribed \(durationSeconds, privacy: .public)s (lang=\(languageHint ?? "auto", privacy: .public))"
            )
            return [
                ASRSegment(
                    text: text,
                    start: buffer.timestamp,
                    end: buffer.timestamp + durationSeconds,
                    confidence: nil
                )
            ]
        } catch {
            throw ASRError.underlying(error)
        }
    }

    private func loadModels() async throws {
        AppLog.asr.info("Downloading Qwen3-ASR Core ML models (first run)…")
        do {
            let dir = try await Qwen3AsrModels.download()
            try await manager.loadModels(from: dir)
            loaded = true
        } catch {
            throw ASRError.modelUnavailable(reason: String(describing: error))
        }
    }

    private static func languageHint(from locale: Locale) -> String? {
        // Qwen3 accepts ISO codes ("en", "ja", "zh", "ko", "vi"); nil = auto.
        switch locale.language.languageCode?.identifier {
        case "ja": return "ja"
        case "en": return "en"
        case "zh": return "zh"
        case "ko": return "ko"
        case "vi": return "vi"
        default: return nil
        }
    }
}
