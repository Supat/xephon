import Foundation
import Speech

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
}
