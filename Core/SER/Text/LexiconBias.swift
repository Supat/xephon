import Foundation

/// One row of the user's custom glossary: a term that, when found
/// in an utterance's transcript, biases the text-SER Plutchik
/// distribution toward `label` by `weight`.
///
/// `weight` is the UI-facing 0.0–1.0 knob. Bias math scales it to
/// an additive logit (see `LexiconBias.apply`) so weight 1.0
/// roughly multiplies the matched class's odds-ratio by `e^logitScale`
/// — strong enough to dominate a wavering distribution without
/// flat-out forcing the label.
public struct LexiconBiasEntry: Sendable, Hashable, Codable, Identifiable {
    public let id: UUID
    public var term: String
    public var label: PlutchikScore.Label
    public var weight: Float

    public init(
        id: UUID = UUID(),
        term: String,
        label: PlutchikScore.Label,
        weight: Float
    ) {
        self.id = id
        self.term = term
        self.label = label
        self.weight = weight
    }
}

/// Snapshot of the user's glossary, in the shape consumed by
/// `SwitchingTextSER.classifyBiased`. Sendable + value-typed so
/// it can flow across actor boundaries without sharing mutable
/// state — the controller pushes a fresh snapshot whenever the
/// `GlossaryStore` mutates.
public struct LexiconBias: Sendable, Hashable, Codable {
    public let entries: [LexiconBiasEntry]
    /// Additive logit applied per unit `weight`. Tuning headroom:
    /// at 2.0 a weight of 1.0 yields `+2.0` logit ≈ 7.4× odds
    /// boost; weight 0.5 ≈ 2.7×. Soft enough that the model still
    /// dictates the *shape* of the distribution; user weights tilt
    /// the leaderboard.
    public static let logitScale: Float = 2.0
    /// Logit assigned to labels whose backend probability came
    /// back as 0. We can't take `log(0) = -inf`, but skipping
    /// the label entirely (a) drops it from the post-bias
    /// dictionary so the per-utterance UI shows a missing bar
    /// instead of a 0% bar, and (b) makes glossary weights
    /// powerless to rescue a label the backend zeroed out — even
    /// when the user explicitly tagged a matching term for it.
    /// Picking a finite floor (about p ≈ 1.8 %) lets a
    /// max-weight glossary hit lift such a label to a meaningful
    /// share of the renormalized distribution while leaving
    /// unbiased zero-prob labels at roughly that 1–2 % baseline,
    /// which is invisible next to any label the model gave real
    /// mass to but keeps the chart consistent.
    public static let zeroLogitFloor: Float = -4.0

    public init(entries: [LexiconBiasEntry] = []) {
        self.entries = entries
    }

    /// Apply the glossary to `score` given the transcript `text`.
    /// For each entry whose term appears in `text`
    /// (case-insensitive substring; Japanese has no word
    /// boundaries to honor), add `weight * logitScale` to that
    /// label's logit, then re-softmax. Returns the biased score
    /// and the deduplicated list of matched term strings so the
    /// UI can badge the row.
    ///
    /// Empty entries / no matches / a `.deberta`-or-FM score that
    /// only fills a subset of the 8 labels all degrade to a
    /// passthrough — the original score comes back unchanged with
    /// `matched == []`.
    public func apply(
        _ score: PlutchikScore,
        to text: String
    ) -> (biased: PlutchikScore, matched: [String]) {
        guard !entries.isEmpty, !text.isEmpty else {
            return (score, [])
        }
        let haystack = text.lowercased()
        var perLabelBias: [PlutchikScore.Label: Float] = [:]
        var matched: [String] = []
        var seen = Set<String>()
        for entry in entries {
            let needle = entry.term
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .lowercased()
            guard !needle.isEmpty else { continue }
            guard haystack.contains(needle) else { continue }
            perLabelBias[entry.label, default: 0]
                += entry.weight * Self.logitScale
            if seen.insert(needle).inserted {
                matched.append(entry.term)
            }
        }
        guard !matched.isEmpty else { return (score, []) }
        // Logit-space bias + softmax. Convert each probability to
        // a logit (ln p), add the per-label bias, exponentiate,
        // and renormalize. Labels the backend gave probability 0
        // get the `zeroLogitFloor` (≈ ln 0.018) instead of -inf
        // so that (a) they survive into the post-bias dictionary
        // — preserving the 8-label chart shape on the UI side —
        // and (b) a glossary entry tagged for one of them can
        // still lift it to a meaningful share of the
        // distribution. Without the floor, a "this term means
        // joy" glossary entry against a row the backend zeroed
        // out for joy would silently drop joy from the output
        // entirely.
        var logits: [PlutchikScore.Label: Float] = [:]
        for label in PlutchikScore.Label.allCases {
            let p = score.probabilities[label] ?? 0
            let baseLogit: Float = p > 0
                ? Float(Foundation.log(Double(p)))
                : Self.zeroLogitFloor
            logits[label] = baseLogit + (perLabelBias[label] ?? 0)
        }
        guard !logits.isEmpty else { return (score, matched) }
        let maxLogit = logits.values.max() ?? 0
        var expSum: Float = 0
        var exps: [PlutchikScore.Label: Float] = [:]
        for (label, l) in logits {
            let e = Float(Foundation.exp(Double(l - maxLogit)))
            exps[label] = e
            expSum += e
        }
        guard expSum > 0 else { return (score, matched) }
        var biasedProbs: [PlutchikScore.Label: Float] = [:]
        for (label, e) in exps {
            biasedProbs[label] = e / expSum
        }
        return (PlutchikScore(probabilities: biasedProbs), matched)
    }
}
