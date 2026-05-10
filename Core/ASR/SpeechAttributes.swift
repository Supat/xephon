import Foundation
import Speech
import CoreMedia

/// Helpers for extracting per-result data from `SpeechTranscriber.Result`'s
/// `AttributedString` text. Apple delivers `transcriptionConfidence` as a
/// per-token attribute (`Double` in `[0, 1]`) on every run, not as a
/// scalar on the result itself, so callers that want a per-utterance
/// number have to reduce across runs.
enum SpeechAttributes {
    /// Mean of the per-run `transcriptionConfidence` values inside `text`,
    /// returned as `Float` for ASRSegment / fusion. Returns `nil` when the
    /// transcriber wasn't configured with `attributeOptions: [.transcriptionConfidence]`,
    /// or when the result genuinely has no confidence-bearing runs.
    /// Length-weighting (by token count) was tried; in practice the simple
    /// arithmetic mean tracks "how confident is the model overall?" closely
    /// enough for fusion's text-weight scaling.
    static func averageConfidence(in text: AttributedString) -> Float? {
        var total: Double = 0
        var count: Int = 0
        for run in text.runs {
            if let conf = run[AttributeScopes.SpeechAttributes.ConfidenceAttribute.self] {
                total += conf
                count += 1
            }
        }
        guard count > 0 else { return nil }
        return Float(total / Double(count))
    }

    /// Per-run audio-time tokens for downstream speaker-change splitting.
    /// Each run carries its own `audioTimeRange` (`CMTimeRange`) attribute
    /// when the transcriber was configured with
    /// `attributeOptions: [.audioTimeRange]`. Returns an empty array when
    /// the option wasn't set or no runs carried timing.
    /// Skips empty-text runs so callers can safely concatenate token
    /// texts without re-introducing the leading whitespace SpeechAnalyzer
    /// sometimes emits between tokens.
    static func tokens(in text: AttributedString) -> [ASRSegment.Token] {
        var out: [ASRSegment.Token] = []
        for run in text.runs {
            guard let range = run[AttributeScopes.SpeechAttributes.TimeRangeAttribute.self] else { continue }
            let runText = String(text[run.range].characters)
            guard !runText.isEmpty else { continue }
            let start = range.start.seconds
            let end = (range.start + range.duration).seconds
            out.append(ASRSegment.Token(text: runText, start: start, end: end))
        }
        return out
    }
}
