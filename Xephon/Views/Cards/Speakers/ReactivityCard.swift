import SwiftUI
import Fusion
import XephonUtilities

/// Surfaces backlog items #12–#14 — how each speaker reacts to the
/// session around them. Three sections stacked vertically:
///
///  12. **Role mix** — per-speaker count of response vs. initiative
///      turns, with the response share rendered as a thin bar so the
///      eye can compare "reactive" vs. "agenda-setting" speakers at
///      a glance.
///  13. **Recovery time** — per-speaker median seconds from a
///      sub-threshold V turn back to (or above) the speaker's
///      session baseline.
///  14. **Reaction to interruption** — per-speaker mean V change
///      from an interrupted turn to the speaker's next turn.
///      Negative = the speaker's affect drops after being talked
///      over.
///
/// Backed by `ReactivityDynamics`. Sits below
/// `AccommodationCohesionCard` on the speaker-analytics page.
struct ReactivityCard: View {
    let utterances: [UtteranceEstimate]
    /// Keys the `.task(id:)` recompute — the parent passes
    /// `recorder.utterancesVersion` so the analytics below run once
    /// per utterance-list change instead of on every body re-eval
    /// (scroll/focus churn re-evals visible cards constantly).
    let utterancesVersion: Int

    /// See TurnTakingCard.Computed — same off-render-path pattern.
    private struct Computed {
        var ratios = ReactivityDynamics.roleRatios(utterances: [])
        var recoveries = ReactivityDynamics.recoveryTimes(utterances: [])
        var reactions = ReactivityDynamics.interruptionReactions(utterances: [])
        init(utterances: [UtteranceEstimate]) {
            ratios = ReactivityDynamics.roleRatios(utterances: utterances)
            recoveries = ReactivityDynamics.recoveryTimes(utterances: utterances)
            reactions = ReactivityDynamics.interruptionReactions(utterances: utterances)
        }
    }

    /// nil until the first `.task` fires (one frame); body falls
    /// back to empty-input products so the empty-state copy renders
    /// rather than stale or missing sections.
    @State private var computed: Computed?

    private static let speakerLabelWidth: CGFloat = 44

    var body: some View {
        let c = computed ?? Computed(utterances: [])
        let ratios = c.ratios
        let recoveries = c.recoveries
        let reactions = c.reactions
        VStack(alignment: .leading, spacing: 14) {
            header(speakerCount: ratios.count)
            if utterances.isEmpty {
                Text(String(localized: "reactivity.empty"))
                    .font(.caption)
                    .foregroundStyle(.tertiary)
                    .frame(maxWidth: .infinity, alignment: .leading)
            } else {
                roleSection(ratios: ratios)
                Divider().opacity(0.4)
                recoverySection(tallies: recoveries)
                Divider().opacity(0.4)
                reactionSection(reactions: reactions)
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .glassEffect(in: RoundedRectangle(cornerRadius: 12, style: .continuous))
        .task(id: utterancesVersion) {
            computed = Computed(utterances: utterances)
        }
    }

    @ViewBuilder
    private func header(speakerCount: Int) -> some View {
        HStack(spacing: 6) {
            Text(String(localized: "reactivity.header"))
                .font(.caption.bold())
                .foregroundStyle(.secondary)
            Spacer(minLength: 4)
            if speakerCount > 0 {
                Text("\(speakerCount)")
                    .font(.caption2.monospacedDigit())
                    .foregroundStyle(.tertiary)
            }
        }
    }

    // MARK: - #12 Role mix

    @ViewBuilder
    private func roleSection(
        ratios: [ReactivityDynamics.RoleRatio]
    ) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            sectionTitle(String(localized: "reactivity.section.role"))
            if ratios.isEmpty {
                Text(String(localized: "reactivity.role.none"))
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            } else {
                ForEach(ratios, id: \.speakerID) { ratio in
                    HStack(spacing: 8) {
                        Text(ratio.speakerID)
                            .font(.caption2.bold())
                            .foregroundStyle(speakerTint(for: ratio.speakerID))
                            .frame(width: Self.speakerLabelWidth, alignment: .leading)
                        responseBar(share: ratio.responseShare, tint: speakerTint(for: ratio.speakerID))
                        Text(String(format: "R %d · I %d", ratio.responses, ratio.initiatives))
                            .font(.caption2.monospacedDigit())
                            .foregroundStyle(.secondary)
                            .frame(width: 72, alignment: .trailing)
                    }
                }
            }
            Text(String(localized: "reactivity.footnote.role"))
                .font(.caption2)
                .foregroundStyle(.tertiary)
        }
    }

    @ViewBuilder
    private func responseBar(share: Double, tint: Color) -> some View {
        GeometryReader { geo in
            ZStack(alignment: .leading) {
                Capsule()
                    .fill(.secondary.opacity(0.18))
                Capsule()
                    .fill(tint)
                    .frame(width: max(2, geo.size.width * CGFloat(share)))
            }
        }
        .frame(height: 6)
    }

    // MARK: - #13 Recovery time

    @ViewBuilder
    private func recoverySection(
        tallies: [ReactivityDynamics.RecoveryTally]
    ) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            sectionTitle(String(localized: "reactivity.section.recovery"))
            if tallies.isEmpty {
                Text(String(localized: "reactivity.recovery.none"))
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            } else {
                SpeakerTallyTable(
                    rows: tallies,
                    rowID: \.speakerID,
                    speakerID: { $0.speakerID },
                    columnHeaders: [
                        String(localized: "reactivity.col.median"),
                        String(localized: "reactivity.col.recovered"),
                        String(localized: "reactivity.col.unresolved"),
                    ],
                    cells: { tally in
                        Text(tally.medianRecoverySec.map { formatClock($0) } ?? "—")
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(.primary)
                            .frame(maxWidth: .infinity, alignment: .center)
                        Text("\(tally.recoveredCount)")
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(.primary)
                            .frame(maxWidth: .infinity, alignment: .center)
                        Text("\(tally.unresolvedCount)")
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(
                                tally.unresolvedCount > 0
                                    ? AnyShapeStyle(Color.orange)
                                    : AnyShapeStyle(HierarchicalShapeStyle.tertiary)
                            )
                            .frame(maxWidth: .infinity, alignment: .center)
                    },
                    labelWidth: Self.speakerLabelWidth
                )
            }
            Text(String(localized: "reactivity.footnote.recovery"))
                .font(.caption2)
                .foregroundStyle(.tertiary)
        }
    }

    // MARK: - #14 Interruption reaction

    @ViewBuilder
    private func reactionSection(
        reactions: [ReactivityDynamics.InterruptionReaction]
    ) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            sectionTitle(String(localized: "reactivity.section.reaction"))
            if reactions.isEmpty {
                Text(String(localized: "reactivity.reaction.none"))
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            } else {
                ForEach(reactions, id: \.speakerID) { reaction in
                    HStack(spacing: 8) {
                        Text(reaction.speakerID)
                            .font(.caption2.bold())
                            .foregroundStyle(speakerTint(for: reaction.speakerID))
                            .frame(width: Self.speakerLabelWidth, alignment: .leading)
                        deltaBar(delta: reaction.meanValenceDelta)
                        Text(String(format: "%+.2f · ×%d",
                            reaction.meanValenceDelta,
                            reaction.sampleCount
                        ))
                            .font(.caption2.monospacedDigit())
                            .foregroundStyle(.secondary)
                            .frame(width: 72, alignment: .trailing)
                    }
                }
            }
            Text(String(localized: "reactivity.footnote.reaction"))
                .font(.caption2)
                .foregroundStyle(.tertiary)
        }
    }

    @ViewBuilder
    private func deltaBar(delta: Double) -> some View {
        GeometryReader { geo in
            let half = geo.size.width * 0.5
            // Clamp display to ±0.30 so a normal range fills most
            // of the bar without saturating on a single outlier.
            let clamped = delta.clamped(to: -0.30...0.30)
            let magnitude = abs(clamped) / 0.30
            let barWidth = half * CGFloat(magnitude)
            let tint: Color = delta < 0 ? .red : .green
            ZStack(alignment: .center) {
                // Zero baseline.
                Capsule()
                    .fill(.secondary.opacity(0.18))
                    .frame(height: 6)
                if delta < 0 {
                    Rectangle()
                        .fill(tint)
                        .frame(width: barWidth, height: 6)
                        .offset(x: -barWidth / 2)
                } else if delta > 0 {
                    Rectangle()
                        .fill(tint)
                        .frame(width: barWidth, height: 6)
                        .offset(x: barWidth / 2)
                }
                Rectangle()
                    .fill(.secondary)
                    .frame(width: 1, height: 8)
            }
        }
        .frame(height: 8)
    }

    // MARK: - Shared

    @ViewBuilder
    private func sectionTitle(_ text: String) -> some View {
        Text(text)
            .font(.caption2.weight(.semibold))
            .foregroundStyle(.secondary)
            .textCase(.uppercase)
    }
}
