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
        // Resolve to the typed `Qwen3AsrConfig.Language` enum
        // and pass it through the enum-typed overload. Functionally
        // equivalent to the String overload (the typed one just
        // forwards `englishName` to it internally) but the explicit
        // enum makes mis-mapping bugs surface at the call site
        // rather than as a silent "language not recognized → auto-
        // detect" warning buried in FluidAudio's log. The resolved
        // value is logged so the user can confirm post-hoc that the
        // expected language hint actually reached the model — a
        // common failure when wrong-language output appears.
        let resolvedLanguage = Self.languageEnum(from: locale)
        do {
            let rawText = try await manager.transcribe(
                audioSamples: buffer.samples,
                language: resolvedLanguage
            )
            // Qwen3 honors the language hint via task tokens in
            // its prompt, but on ambiguous audio it can still
            // ignore the hint and emit wrong-script output
            // (Korean Hangul or pure Chinese when the session is
            // Japanese, etc.). Strip out-of-script characters
            // post-hoc so the picker's language is enforced,
            // not merely suggested. See `filterToLanguage`.
            let filteredText = Self.filterToLanguage(rawText, resolvedLanguage)
            let droppedCount = rawText.unicodeScalars.count - filteredText.unicodeScalars.count
            if droppedCount > 0 {
                AppLog.asr.info(
                    "Qwen3-ASR dropped \(droppedCount, privacy: .public) out-of-script scalar(s) (lang=\(resolvedLanguage?.rawValue ?? "auto", privacy: .public))"
                )
            }
            AppLog.asr.info(
                "Qwen3-ASR transcribed \(durationSeconds, privacy: .public)s (lang=\(resolvedLanguage?.rawValue ?? "auto", privacy: .public))"
            )
            return [
                ASRSegment(
                    text: filteredText,
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

    /// Strip Unicode scalars that don't belong to the target
    /// language's script. Qwen3-ASR can drift to its training-
    /// dominant language (Chinese in particular) on ambiguous
    /// audio even with the language hint set; this enforces the
    /// picker's pick by dropping out-of-script output so the
    /// downstream pipeline never sees Korean Hangul on a
    /// Japanese session, etc.
    ///
    /// Per-language allow list is conservative — only strips
    /// scalars from clearly-incompatible scripts. Latin / ASCII
    /// punctuation / digits / whitespace always pass through,
    /// as does CJK Han wherever it's reasonable (Japanese
    /// kanji, Korean Hanja, Chinese hanzi all share that block).
    ///
    /// Returns the filtered text. Empty when the entire output
    /// was wrong-script; the caller emits the segment anyway so
    /// downstream consumers can see the empty cell rather than
    /// silently dropping it.
    private static func filterToLanguage(
        _ text: String,
        _ language: Qwen3AsrConfig.Language?
    ) -> String {
        guard let language else { return text }
        let disallowed = Self.disallowedScripts(for: language)
        var output = String.UnicodeScalarView()
        output.reserveCapacity(text.unicodeScalars.count)
        for scalar in text.unicodeScalars {
            if !disallowed.contains(scalar) {
                output.append(scalar)
            }
        }
        return String(output).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// CharacterSet of Unicode scalars that should NOT appear
    /// in a transcript for the given target language. Built once
    /// per call (cheap; CharacterSet unions on small ranges are
    /// fast and Swift's CoW kicks in).
    private static func disallowedScripts(
        for language: Qwen3AsrConfig.Language
    ) -> CharacterSet {
        // Block ranges (start, inclusive end). Defined here
        // rather than as top-level lets so the compiler can
        // inline the literals.
        func range(_ lo: UInt32, _ hi: UInt32) -> CharacterSet {
            CharacterSet(charactersIn: Unicode.Scalar(lo)!...Unicode.Scalar(hi)!)
        }
        let hangulJamo       = range(0x1100, 0x11FF)
        let hangulCompatJamo = range(0x3130, 0x318F)
        let hangulSyllables  = range(0xAC00, 0xD7AF)
        let hangul = hangulJamo.union(hangulCompatJamo).union(hangulSyllables)

        let hiragana       = range(0x3040, 0x309F)
        let katakana       = range(0x30A0, 0x30FF)
        let katakanaPhExt  = range(0x31F0, 0x31FF)
        let halfwidthKana  = range(0xFF65, 0xFF9F)
        let kana = hiragana.union(katakana).union(katakanaPhExt).union(halfwidthKana)

        // CJK Unified Ideographs + extensions. Excluded for
        // non-CJK targets (English, Vietnamese).
        let han = range(0x4E00, 0x9FFF)
            .union(range(0x3400, 0x4DBF))
            .union(range(0xF900, 0xFAFF))

        let cyrillic = range(0x0400, 0x04FF)
        let arabic   = range(0x0600, 0x06FF)
        let thai     = range(0x0E00, 0x0E7F)
        let devanagari = range(0x0900, 0x097F)

        // Anything-but-target unrelated scripts. Common to
        // every CJK target.
        let cjkUnrelated = cyrillic.union(arabic).union(thai).union(devanagari)

        switch language {
        case .japanese:
            // Keep hiragana / katakana / kanji / Latin. Drop
            // Korean Hangul and unrelated scripts.
            return hangul.union(cjkUnrelated)
        case .korean:
            // Keep Hangul + Hanja (han) + Latin. Drop Japanese
            // kana and unrelated scripts.
            return kana.union(cjkUnrelated)
        case .chinese, .cantonese:
            // Keep Han + Latin. Drop Japanese kana, Korean
            // Hangul, and unrelated scripts.
            return kana.union(hangul).union(cjkUnrelated)
        default:
            // Non-CJK targets: drop ALL CJK + Hangul +
            // unrelated scripts.
            return kana.union(hangul).union(han).union(cjkUnrelated)
        }
    }

    /// Resolve the active locale to FluidAudio's
    /// `Qwen3AsrConfig.Language` enum so the transcribe call can
    /// use the typed overload. Falls back to parsing the locale
    /// identifier prefix when `language.languageCode` is nil
    /// (defensive — region-tagged locales like "ja_JP" should
    /// resolve cleanly on iOS 17+, but historical Locale APIs
    /// have returned nil for the language code in some
    /// configurations, which would silently degrade to auto-
    /// detect and let the model pick a wrong language).
    private static func languageEnum(from locale: Locale) -> Qwen3AsrConfig.Language? {
        let code = locale.language.languageCode?.identifier
            ?? locale.identifier.split(separator: "_").first.map(String.init)
            ?? locale.identifier
        switch code.lowercased() {
        case "ja": return .japanese
        case "en": return .english
        case "zh": return .chinese
        case "ko": return .korean
        case "vi": return .vietnamese
        default:   return nil
        }
    }
}
