import Foundation
import Fusion
import SERText

/// Resolver that maps an `UtteranceEstimate`'s text-backend tag +
/// per-utterance modality-disagreement score onto the two view-model
/// badges drawn at the top-right of each row. Pure mapping with no
/// view dependency — same arithmetic, but factored out of
/// `UtteranceRow` so the view body stays a renderer.
///
/// The badge view-models themselves (`TextBackendBadge`,
/// `ModalityBadge`) live in `DisplayFormatting.swift` because they
/// carry SwiftUI `Color` values.
struct UtteranceBadges {
    let textBackend: TextBackendBadge?
    let modality: ModalityBadge?
    let lexiconBias: LexiconBiasBadge?
    /// Display name for the text-SER backend, used to label the
    /// detail-panel "Text SER (…)" section header. Falls back to
    /// "Plutchik" when the backend is unknown or absent.
    let textBackendName: String

    init(utterance: UtteranceEstimate) {
        self.textBackend = Self.resolveTextBackend(utterance)
        self.modality = Self.resolveModality(utterance)
        self.lexiconBias = Self.resolveLexiconBias(utterance)
        self.textBackendName = Self.resolveTextBackendName(utterance)
    }

    private static func resolveLexiconBias(
        _ utterance: UtteranceEstimate
    ) -> LexiconBiasBadge? {
        guard let matched = utterance.lexiconBiasMatched, !matched.isEmpty else {
            return nil
        }
        return LexiconBiasBadge(matched: matched)
    }

    private static func resolveTextBackend(
        _ utterance: UtteranceEstimate
    ) -> TextBackendBadge? {
        guard let raw = utterance.textBackend else { return nil }
        if raw == SwitchingTextSER.foundationModelsGuardrailBackend {
            return TextBackendBadge(
                label: String(localized: "textSER.appleFMViolation"),
                isGuardrail: true
            )
        }
        guard let backend = SwitchingTextSER.Backend(rawValue: raw) else { return nil }
        return TextBackendBadge(label: backend.badgeLabel, isGuardrail: false)
    }

    private static func resolveModality(
        _ utterance: UtteranceEstimate
    ) -> ModalityBadge? {
        guard let score = ModalityDisagreement.score(
            acoustic: utterance.acousticCategorical,
            plutchik: utterance.plutchik
        ), score.tvd >= ModalityDisagreement.flagThreshold else { return nil }
        if score.topsAreOpposites {
            return ModalityBadge(
                label: String(localized: "modality.opposite"),
                tint: .red,
                accessibility: String(localized: "modality.opposite.a11y")
            )
        }
        return ModalityBadge(
            label: String(localized: "modality.split"),
            tint: .orange,
            accessibility: String(localized: "modality.split.a11y")
        )
    }

    private static func resolveTextBackendName(
        _ utterance: UtteranceEstimate
    ) -> String {
        guard let raw = utterance.textBackend else { return "Plutchik" }
        if raw == SwitchingTextSER.foundationModelsGuardrailBackend {
            return String(localized: "textSER.appleFMViolation")
        }
        guard let backend = SwitchingTextSER.Backend(rawValue: raw) else {
            return "Plutchik"
        }
        return backend.badgeLabel
    }
}
