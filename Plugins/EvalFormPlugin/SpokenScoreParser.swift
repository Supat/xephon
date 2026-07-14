import Foundation

/// Deterministic capture of scores the evaluator SPOKE — the
/// highest-trust tier of the extraction pipeline
/// (docs/eval_form_autofill_research.md §2 option D): a stated
/// score is regex-matched, never generated, so this path cannot
/// hallucinate. Anything the parser doesn't match is left to the
/// LLM tier, which must mark its output as inferred.
public enum SpokenScoreParser {

    /// One captured relative-strength score.
    public struct StatedStrength: Equatable, Sendable {
        /// Signed, quantized score (e.g. -0.25).
        public let value: Double
        /// UTF-16 range in the (NFKC-folded) transcript — for
        /// highlight/debug, not identity.
        public let rangeUTF16: NSRange
    }

    /// Signed strength scores in `transcript`, quantized to
    /// `allowedValues`. Grammar: an explicit sign marker
    /// (マイナス/プラス/+/-/−/±) followed by a decimal within the
    /// scale. ASR emits spoken numbers as digits (full-width
    /// folded here), so "マイナス0.25ぐらい" matches. An UNSIGNED
    /// number is deliberately NOT a score — bare numbers appear in
    /// speeds ("60キロ"), road names, and times far too often.
    /// ±0 / プラマイゼロ map to 0.
    public static func statedStrengths(
        in transcript: String,
        allowedValues: [Double]
    ) -> [StatedStrength] {
        let folded = transcript.precomposedStringWithCompatibilityMapping
        var results: [StatedStrength] = []

        // プラマイゼロ / ±0 / プラスマイナスゼロ → 0. Checked first
        // so the generic sign+number pattern below doesn't eat the
        // "マイナスゼロ" suffix of プラスマイナスゼロ as −0.
        let zeroPattern = "(プラマイ|プラスマイナス|±)\\s*(ゼロ|0(\\.0+)?)"
        // Sign marker + decimal. The sign group is required;
        // "0.25" alone never matches. `ー` (long-vowel mark) is a
        // frequent ASR rendering of a spoken minus before a digit,
        // so it's accepted as a minus marker only when directly
        // followed by a number.
        let signedPattern = "(マイナス|ﾏｲﾅｽ|プラス|ﾌﾟﾗｽ|[-−+ー])\\s*([01](\\.[0-9]+)?|\\.[0-9]+)"

        guard let zeroRegex = try? NSRegularExpression(pattern: zeroPattern),
              let signedRegex = try? NSRegularExpression(pattern: signedPattern)
        else { return [] }

        let ns = folded as NSString
        let full = NSRange(location: 0, length: ns.length)

        // Ranges claimed by the ±0 family — the signed pattern
        // must not re-match inside them.
        var claimed: [NSRange] = []
        for match in zeroRegex.matches(in: folded, range: full) {
            claimed.append(match.range)
            results.append(StatedStrength(value: 0, rangeUTF16: match.range))
        }

        for match in signedRegex.matches(in: folded, range: full) {
            guard !claimed.contains(where: { NSIntersectionRange($0, match.range).length > 0 })
            else { continue }
            let signText = ns.substring(with: match.range(at: 1))
            let numberText = ns.substring(with: match.range(at: 2))
            guard let magnitude = Double(numberText) else { continue }
            let negative = ["マイナス", "ﾏｲﾅｽ", "-", "−", "ー"].contains(signText)
            let value = negative ? -magnitude : magnitude
            // Quantization gate: only the scale's exact values
            // count as scores. "マイナス0.3" is conversation, not a
            // sheet entry — leave it to the (flagged) LLM tier.
            guard allowedValues.contains(where: { abs($0 - value) < 0.0005 })
            else { continue }
            results.append(StatedStrength(value: value, rangeUTF16: match.range))
        }
        return results.sorted { $0.rangeUTF16.location < $1.rangeUTF16.location }
    }

    /// A stated 1–9 preference (好き/嫌い) rating: requires BOTH a
    /// digit in range followed by 点 AND a preference word in the
    /// same utterance — conservative on purpose; bare digits are
    /// everywhere in drive talk.
    public static func statedPreference(
        in transcript: String,
        minimum: Int,
        maximum: Int
    ) -> Int? {
        let folded = transcript.precomposedStringWithCompatibilityMapping
        guard folded.contains("好き") || folded.contains("嫌い") else { return nil }
        let pattern = "([1-9])\\s*点"
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return nil }
        let ns = folded as NSString
        let full = NSRange(location: 0, length: ns.length)
        guard let match = regex.matches(in: folded, range: full).first,
              let value = Int(ns.substring(with: match.range(at: 1))),
              (minimum...maximum).contains(value)
        else { return nil }
        return value
    }
}
