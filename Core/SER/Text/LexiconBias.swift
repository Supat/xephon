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
        // and renormalize. Probabilities at 0 stay at 0 (their
        // logit is -inf; no finite bias can rescue them) — that's
        // intentional: the backend said "this class is impossible
        // here" and we don't want a glossary entry to wholesale
        // override that judgment.
        var logits: [PlutchikScore.Label: Float] = [:]
        for label in PlutchikScore.Label.allCases {
            let p = score.probabilities[label] ?? 0
            guard p > 0 else { continue }
            logits[label] = Float(Foundation.log(Double(p)))
                + (perLabelBias[label] ?? 0)
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
