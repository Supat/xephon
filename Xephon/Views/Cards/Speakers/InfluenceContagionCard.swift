import SwiftUI
import Fusion

/// Surfaces the three directed-influence + emotional-contagion
/// analyses from `docs/social_dynamics_backlog.md` #5–#7:
///
///   5. **Multi-lag leadership matrix.** Directed-pair heatmap, rows
///      = leader, columns = follower. Cell tint = signed valence
///      leadership averaged across lags 1–3, sample-weighted.
///   6. **Mood rescue.** Per-speaker count of "rescue" turns — V
///      rose above a recent partner's low-V turn — alongside the
///      converse "downed" count.
///   7. **Contagion windows.** List of seed turns whose strong
///      valence pulled ≥2 other speakers into alignment within a
///      60 s window. Reads as "who drove the room's mood here."
///
/// Sits on the speaker-analytics page below `AffectiveSynchronyCard`.
/// All three reads recompute per render — backed by pure functions
/// over the utterance list (and #5 piggybacks on the existing
/// `AffectiveSynchrony.compute` pass) so caching adds more
/// bookkeeping than the work it saves.
struct InfluenceContagionCard: View {
    let utterances: [UtteranceEstimate]
    /// Keys the `.task(id:)` recompute — the parent passes
    /// `recorder.utterancesVersion` so the analytics below run once
    /// per utterance-list change instead of on every body re-eval
    /// (scroll/focus churn re-evals visible cards constantly).
    let utterancesVersion: Int

    /// See TurnTakingCard.Computed — same off-render-path pattern.
    /// This card is the heaviest of the seven (six analytics passes
    /// including cross-correlation) — exactly the body-compute the
    /// pattern exists to keep off the scroll path.
    private struct Computed {
        var synchrony = AffectiveSynchrony.compute(utterances: [])
        var leadership = InfluenceDynamics.multiLagLeadership(
            from: AffectiveSynchrony.compute(utterances: [])
        )
        var rescues = InfluenceDynamics.moodRescues(utterances: [])
        var rescueTallies = InfluenceDynamics.moodRescueTallies(
            from: InfluenceDynamics.moodRescues(utterances: []), in: []
        )
        var windows = InfluenceDynamics.contagionWindows(utterances: [])
        var modalityTallies = ModalityDisagreement.tallies(utterances: [])
        var speakers: [String] = []
        init(utterances: [UtteranceEstimate]) {
            synchrony = AffectiveSynchrony.compute(utterances: utterances)
            leadership = InfluenceDynamics.multiLagLeadership(from: synchrony)
            rescues = InfluenceDynamics.moodRescues(utterances: utterances)
            rescueTallies = InfluenceDynamics.moodRescueTallies(
                from: rescues, in: utterances
            )
            windows = InfluenceDynamics.contagionWindows(utterances: utterances)
            modalityTallies = ModalityDisagreement.tallies(utterances: utterances)
            speakers = utterances
                .sorted(by: { $0.start < $1.start })
                .orderedSpeakerIDs
        }
    }

    /// nil until the first `.task` fires (one frame); body falls
    /// back to empty-input products so the empty-state copy renders
    /// rather than stale or missing sections.
    @State private var computed: Computed?

    private static let cellSize: CGFloat = 36
    private static let cellSpacing: CGFloat = 2
    private static let speakerLabelWidth: CGFloat = 44

    var body: some View {
        let c = computed ?? Computed(utterances: [])
        let leadership = c.leadership
        // (c.synchrony / c.rescues are intermediates consumed inside
        // Computed's init — leadership and rescueTallies are their
        // derived products; no local bindings needed.)
        let rescueTallies = c.rescueTallies
        let windows = c.windows
        let modalityTallies = c.modalityTallies
        let speakers = c.speakers

        VStack(alignment: .leading, spacing: 14) {
            header(speakerCount: speakers.count)
            if utterances.isEmpty || speakers.isEmpty {
                Text(String(localized: "influence.empty"))
                    .font(.caption)
                    .foregroundStyle(.tertiary)
                    .frame(maxWidth: .infinity, alignment: .leading)
            } else {
                leadershipMatrix(pairs: leadership, speakers: speakers)
                Divider().opacity(0.4)
                rescueSection(tallies: rescueTallies)
                Divider().opacity(0.4)
                contagionSection(windows: windows)
                Divider().opacity(0.4)
                modalitySection(tallies: modalityTallies)
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
            Text(String(localized: "influence.header"))
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

    // MARK: - #5 Leadership matrix

    @ViewBuilder
    private func leadershipMatrix(
        pairs: [InfluenceDynamics.DirectedLeadership],
        speakers: [String]
    ) -> some View {
        let map = Dictionary(
            uniqueKeysWithValues: pairs.map {
                (SpeakerPairKey($0.leader, $0.follower), $0)
            }
        )
        VStack(alignment: .leading, spacing: 4) {
            sectionTitle(String(localized: "influence.section.leadership"))
            SpeakerMatrixGrid(
                speakers: speakers,
                cellSize: Self.cellSize,
                cellSpacing: Self.cellSpacing,
                labelWidth: Self.speakerLabelWidth
            ) { leader, follower in
                leadershipCell(
                    leader: leader,
                    follower: follower,
                    map: map
                )
            }
            Text(String(localized: "influence.footnote.leadership"))
                .font(.caption2)
                .foregroundStyle(.tertiary)
        }
    }

    @ViewBuilder
    private func leadershipCell(
        leader: String,
        follower: String,
        map: [SpeakerPairKey: InfluenceDynamics.DirectedLeadership]
    ) -> some View {
        if leader == follower {
            Text("—")
                .font(.caption.monospacedDigit())
                .foregroundStyle(.tertiary)
                .frame(width: Self.cellSize, height: Self.cellSize)
        } else if let entry = map[SpeakerPairKey(leader, follower)],
                  let v = entry.valenceLeadership {
            // Bipolar tint: green = follower echoes leader (V > 0),
            // red = follower diverges (V < 0). Magnitude → opacity.
            let mag = min(1.0, abs(v))
            let tint: Color = v >= 0
                ? Color(red: 0.20, green: 0.65, blue: 0.30)
                : Color(red: 0.85, green: 0.25, blue: 0.25)
            ZStack {
                RoundedRectangle(cornerRadius: 4, style: .continuous)
                    .fill(tint.opacity(0.15 + 0.55 * mag))
                Text(String(format: "%+.2f", v))
                    .font(.caption2.monospacedDigit())
                    .foregroundStyle(.primary)
            }
            .frame(width: Self.cellSize, height: Self.cellSize)
        } else {
            ZStack {
                RoundedRectangle(cornerRadius: 4, style: .continuous)
                    .fill(Color.secondary.opacity(0.04))
                Text("")
            }
            .frame(width: Self.cellSize, height: Self.cellSize)
        }
    }

    // MARK: - #6 Mood rescue

    @ViewBuilder
    private func rescueSection(
        tallies: [InfluenceDynamics.MoodRescueTally]
    ) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            sectionTitle(String(localized: "influence.section.rescue"))
            if tallies.isEmpty {
                Text(String(localized: "influence.rescue.none"))
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            } else {
                SpeakerTallyTable(
                    rows: tallies,
                    rowID: \.speakerID,
                    speakerID: { $0.speakerID },
                    columnHeaders: [
                        String(localized: "influence.col.rescuer"),
                        String(localized: "influence.col.downed"),
                    ],
                    cells: { tally in
                        tallyCount(tally.asRescuer)
                        tallyCount(tally.asDowned)
                    },
                    labelWidth: Self.speakerLabelWidth
                )
            }
            Text(String(localized: "influence.footnote.rescue"))
                .font(.caption2)
                .foregroundStyle(.tertiary)
        }
    }

    /// Centered cell: monospaced int or em-dash for zero, with the
    /// dash rendered in tertiary so the eye reads "nothing here"
    /// without re-checking the number.
    @ViewBuilder
    private func tallyCount(_ value: Int) -> some View {
        Text(value > 0 ? "\(value)" : "—")
            .font(.caption.monospacedDigit())
            .foregroundStyle(value > 0 ? .primary : .tertiary)
            .frame(maxWidth: .infinity, alignment: .center)
    }

    // MARK: - #7 Contagion windows

    @ViewBuilder
    private func contagionSection(
        windows: [InfluenceDynamics.ContagionWindow]
    ) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            sectionTitle(String(localized: "influence.section.contagion"))
            if windows.isEmpty {
                Text(String(localized: "influence.contagion.none"))
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            } else {
                ForEach(Array(windows.enumerated()), id: \.offset) { _, window in
                    contagionRow(window)
                }
            }
            Text(String(localized: "influence.footnote.contagion"))
                .font(.caption2)
                .foregroundStyle(.tertiary)
        }
    }

    @ViewBuilder
    private func contagionRow(
        _ window: InfluenceDynamics.ContagionWindow
    ) -> some View {
        HStack(alignment: .center, spacing: 8) {
            // Direction glyph + tint.
            let isPositive = window.direction == .positive
            Image(systemName: isPositive
                ? "arrow.up.right.circle.fill"
                : "arrow.down.right.circle.fill")
                .font(.callout)
                .foregroundStyle(isPositive ? .green : .red)
            // Seed speaker chip.
            Text(window.seedSpeaker)
                .font(.caption.bold())
                .foregroundStyle(speakerTint(for: window.seedSpeaker))
            // Time stamp.
            Text(formatClock(window.seedTime))
                .font(.caption2.monospacedDigit())
                .foregroundStyle(.secondary)
            // Followers as chained chips.
            HStack(spacing: 3) {
                Text(String(format: "→"))
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                ForEach(window.followers, id: \.self) { f in
                    Text(f)
                        .font(.caption2.bold())
                        .foregroundStyle(speakerTint(for: f))
                }
            }
            Spacer(minLength: 4)
            // Seed valence value.
            Text(String(format: "V %.2f", window.seedValence))
                .font(.caption2.monospacedDigit())
                .foregroundStyle(.tertiary)
        }
        .padding(.vertical, 2)
    }

    // MARK: - #11 Modality disagreement

    @ViewBuilder
    private func modalitySection(
        tallies: [ModalityDisagreement.SpeakerTally]
    ) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            sectionTitle(String(localized: "influence.section.modality"))
            if tallies.isEmpty {
                Text(String(localized: "influence.modality.none"))
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            } else {
                SpeakerTallyTable(
                    rows: tallies,
                    rowID: \.speakerID,
                    speakerID: { $0.speakerID },
                    columnHeaders: [
                        String(localized: "influence.col.flagged"),
                        String(localized: "influence.col.opposite"),
                        String(localized: "influence.col.meanTVD"),
                    ],
                    cells: { tally in
                        tallyCount(tally.flaggedCount)
                        // Opposite-modality count is tinted red
                        // even at zero — the count of contradictory
                        // top labels is the alarming signal here,
                        // not a neutral metric.
                        Text(tally.oppositeCount > 0 ? "\(tally.oppositeCount)" : "—")
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(
                                tally.oppositeCount > 0
                                    ? AnyShapeStyle(Color.red)
                                    : AnyShapeStyle(HierarchicalShapeStyle.tertiary)
                            )
                            .frame(maxWidth: .infinity, alignment: .center)
                        Text(tally.meanTVD.map { String(format: "%.2f", $0) } ?? "—")
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(.primary)
                            .frame(maxWidth: .infinity, alignment: .center)
                    },
                    labelWidth: Self.speakerLabelWidth
                )
            }
            Text(String(localized: "influence.footnote.modality"))
                .font(.caption2)
                .foregroundStyle(.tertiary)
        }
    }

    @ViewBuilder
    private func sectionTitle(_ text: String) -> some View {
        Text(text)
            .font(.caption2.weight(.semibold))
            .foregroundStyle(.secondary)
            .textCase(.uppercase)
    }
}
