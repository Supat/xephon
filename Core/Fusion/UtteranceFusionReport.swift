import Foundation

/// Pre-computed fusion-attribution bundle for one `UtteranceEstimate`,
/// rendered by the per-row inspector on `UtteranceRow`. Pulls the
/// six derived properties that used to live as computed vars on the
/// View out into a stateless value type — same arithmetic, but
/// callable from tests and independently of SwiftUI.
///
/// All formatting that doesn't need localization stays here (the
/// inspector strings have always been English-only research-tool
/// copy). The presentation-state badges (`TextBackendBadge`,
/// `ModalityBadge`) stay view-side because they carry SwiftUI
/// `Color` values.
public struct UtteranceFusionReport {
    /// Inputs to the V/A three-pull mini-scatter. Nil when there
    /// isn't a meaningful geometry to render — gated on BOTH
    /// modalities contributing, because single-modality rows have
    /// a fused point coincident with the source and the scatter
    /// would draw one dot on top of another with no arrows.
    public let scatterInputs: ScatterInputs?

    /// Top 3 normalized fused-label candidates so the user can see
    /// not just *which* label won but *how confidently* — a 0.42 vs
    /// 0.39 runner-up reads very differently from 0.85 vs 0.05.
    /// Nil when neither modality contributed score (no top label
    /// to attribute anyway).
    public let topFusedLabels: [LabelScore]?

    /// Compact "0.62 · 0.48 · 0.55" rendering of the fused V/A/D
    /// numbers. Nil only when none of the three components fused
    /// — in which case the row's fusion summary is also nil and
    /// there's nothing to attribute.
    public let fusedVADSummary: String?

    /// True when the top acoustic class (mapped through the
    /// shared Plutchik→acoustic table for comparability) and the
    /// top Plutchik class name different acoustic-label buckets.
    /// Used to flag rows where the two modalities openly disagree
    /// — the fused label hides this signal; the inspector should
    /// not.
    ///
    /// Excludes acoustic "other"/"unknown" buckets (sink classes
    /// — disagreement there is uninformative) and only counts
    /// text classes that have a mapping (trust / anticipation
    /// route to "other" too, see `LateFusion`'s mapping doc).
    public let disagreesAcrossModalities: Bool

    /// Compact summary of how V/A fusion weighted the two sides
    /// for this utterance. Nil when neither modality contributed
    /// (the fused V/A would be nil too, so there's nothing to
    /// attribute). When only one modality was present, reports
    /// that side as 100% so the user can see which side carried
    /// the result.
    public let fusionWeightSummary: String?

    /// Compact summary of each modality's overall influence on the
    /// fused-label argmax. Nil when there's no top label or when
    /// neither modality contributed any score (defensive — a
    /// well-formed estimate should always have at least one side
    /// of input).
    public let labelFusionSummary: String?

    public struct ScatterInputs {
        public let acoustic: (v: Float, a: Float)?
        public let text: (v: Float, a: Float)?
        public let fused: (v: Float, a: Float)?
    }

    public struct LabelScore {
        public let label: String
        public let score: Float
    }

    public init(
        utterance: UtteranceEstimate,
        acousticWeight: Float,
        textWeightFloor: Float
    ) {
        self.scatterInputs = Self.scatterInputs(for: utterance)
        self.topFusedLabels = Self.topFusedLabels(
            for: utterance,
            acousticWeight: acousticWeight,
            textWeightFloor: textWeightFloor
        )
        self.fusedVADSummary = Self.fusedVADSummary(for: utterance)
        self.disagreesAcrossModalities = Self.disagreesAcrossModalities(for: utterance)
        self.fusionWeightSummary = Self.fusionWeightSummary(
            for: utterance,
            acousticWeight: acousticWeight,
            textWeightFloor: textWeightFloor
        )
        self.labelFusionSummary = Self.labelFusionSummary(
            for: utterance,
            acousticWeight: acousticWeight,
            textWeightFloor: textWeightFloor
        )
    }

    // MARK: - Components

    private static func scatterInputs(
        for utterance: UtteranceEstimate
    ) -> ScatterInputs? {
        guard let dim = utterance.dimensional,
              let plutchik = utterance.plutchik else { return nil }
        guard let v = utterance.fusedValence,
              let a = utterance.fusedArousal else { return nil }
        let textV = LateFusion.plutchikToValence(plutchik)
        let textA = LateFusion.plutchikToArousal(plutchik)
        return ScatterInputs(
            acoustic: (v: dim.valence, a: dim.arousal),
            text: (v: textV, a: textA),
            fused: (v: v, a: a)
        )
    }

    private static func topFusedLabels(
        for utterance: UtteranceEstimate,
        acousticWeight: Float,
        textWeightFloor: Float
    ) -> [LabelScore]? {
        guard let scored = LateFusion.labelFusionScores(
            acoustic: utterance.acousticCategorical,
            plutchik: utterance.plutchik,
            asrConfidence: utterance.asrConfidence ?? 0.5,
            acousticWeight: acousticWeight,
            textWeightFloor: textWeightFloor
        ), !scored.isEmpty else { return nil }
        return scored.prefix(3).map { LabelScore(label: $0.label, score: $0.score) }
    }

    private static func fusedVADSummary(
        for utterance: UtteranceEstimate
    ) -> String? {
        let v = utterance.fusedValence
        let a = utterance.fusedArousal
        let d = utterance.fusedDominance
        guard v != nil || a != nil || d != nil else { return nil }
        func fmt(_ x: Float?) -> String {
            x.map { String(format: "%.2f", $0) } ?? "—"
        }
        return "\(fmt(v)) · \(fmt(a)) · \(fmt(d))"
    }

    private static func disagreesAcrossModalities(
        for utterance: UtteranceEstimate
    ) -> Bool {
        guard let acoustic = utterance.acousticCategorical?.probabilities,
              let plutchik = utterance.plutchik?.probabilities else {
            return false
        }
        let validAcoustic = acoustic
            .filter { $0.key != .unknown && $0.key != .other }
        guard let topAcoustic = validAcoustic.max(by: { $0.value < $1.value })?.key else {
            return false
        }
        let mappedText: [(label: String, value: Float)] = plutchik.compactMap { entry in
            guard let mapped = LateFusion.plutchikToAcousticLabelMapping[entry.key],
                  mapped != "other" else { return nil }
            return (label: mapped, value: entry.value)
        }
        guard let topText = mappedText.max(by: { $0.value < $1.value })?.label else {
            return false
        }
        return topText != topAcoustic.rawValue
    }

    private static func fusionWeightSummary(
        for utterance: UtteranceEstimate,
        acousticWeight: Float,
        textWeightFloor: Float
    ) -> String? {
        let hasAcoustic = utterance.dimensional != nil
        let hasText = utterance.plutchik != nil
        switch (hasAcoustic, hasText) {
        case (false, false):
            return nil
        case (true, false):
            return "Acoustic 100% (no text)"
        case (false, true):
            return "Text 100% (no acoustic)"
        case (true, true):
            let share = LateFusion.vaFusionShare(
                asrConfidence: utterance.asrConfidence ?? 0.5,
                acousticWeight: acousticWeight,
                textWeightFloor: textWeightFloor
            )
            return String(
                format: "Acoustic %.0f%% · Text %.0f%%",
                share.acoustic * 100,
                share.text * 100
            )
        }
    }

    private static func labelFusionSummary(
        for utterance: UtteranceEstimate,
        acousticWeight: Float,
        textWeightFloor: Float
    ) -> String? {
        guard utterance.fusedTopLabel != nil else { return nil }
        guard let share = LateFusion.labelFusionShare(
            acoustic: utterance.acousticCategorical,
            plutchik: utterance.plutchik,
            asrConfidence: utterance.asrConfidence ?? 0.5,
            acousticWeight: acousticWeight,
            textWeightFloor: textWeightFloor
        ) else { return nil }
        return String(
            format: "Acoustic %.0f%% · Text %.0f%%",
            share.acoustic * 100,
            share.text * 100
        )
    }
}
