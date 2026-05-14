import SwiftUI
import Fusion
import SERText

/// Pairwise affective-synchrony view stacked into three sections:
///
///   1. **Pair rows** — directed leader → follower with the lag-0
///      correlation as the headline number and a small lag-profile
///      sparkline showing how the correlation evolves from lag 0
///      out to `result.maxLag` (one intervening turn, two, …).
///   2. **Leadership** — per-speaker average lag-1 correlation,
///      weighted by sample count. High = others tend to echo this
///      speaker's emotion one turn later.
///   3. **Plutchik dyads** — across-speaker tally of adjacent
///      Plutchik primary-dyad pairings (joy↔trust = love,
///      anger↔disgust = contempt, …) over consecutive turn-pairs.
///
/// Lives on the cluster-diagnostics page below `SpeakerHeatmapCard`.
/// Recomputes on every render — the underlying passes are O(turns)
/// each and the result is small enough that caching adds more
/// bookkeeping than the work it saves.
struct AffectiveSynchronyCard: View {
    let utterances: [UtteranceEstimate]

    /// Which axis the pair rows render — fused valence or fused
    /// arousal. Drives both the headline correlation column and the
    /// sparkline data.
    @State private var axis: Axis = .valence

    enum Axis: Hashable {
        case valence, arousal
    }

    var body: some View {
        let result = AffectiveSynchrony.compute(utterances: utterances)
        let leadership = AffectiveSynchrony.leadershipScores(from: result, atLag: 1)
        let dyads = AffectiveSynchrony.plutchikDyadTallies(utterances: utterances)
        VStack(alignment: .leading, spacing: 8) {
            header(pairCount: result.pairs.count)
            if result.pairs.isEmpty {
                Text(String(localized: "synchrony.empty"))
                    .font(.caption)
                    .foregroundStyle(.tertiary)
                    .frame(maxWidth: .infinity, alignment: .leading)
            } else {
                pairRowsSection(result: result)
                if !leadership.isEmpty {
                    Divider().padding(.vertical, 4)
                    leadershipSection(leadership)
                }
                if !dyads.isEmpty {
                    Divider().padding(.vertical, 4)
                    dyadSection(dyads)
                }
                Text(String(localized: "synchrony.footnote"))
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                    .padding(.top, 4)
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .glassEffect(in: RoundedRectangle(cornerRadius: 12, style: .continuous))
    }

    // MARK: - Header

    @ViewBuilder
    private func header(pairCount: Int) -> some View {
        HStack {
            Text(String(localized: "synchrony.header"))
                .font(.caption.bold())
                .foregroundStyle(.secondary)
            Spacer()
            if pairCount > 0 {
                Button {
                    axis = (axis == .valence) ? .arousal : .valence
                } label: {
                    Label(
                        axisLabel,
                        systemImage: "slider.horizontal.3"
                    )
                    .font(.caption2)
                    .labelStyle(.titleAndIcon)
                }
                .buttonStyle(.plain)
                .foregroundStyle(
                    AnyShapeStyle(HierarchicalShapeStyle.secondary)
                )
                Text("\(pairCount)")
                    .font(.caption2.monospacedDigit())
                    .foregroundStyle(.tertiary)
            }
        }
    }

    private var axisLabel: String {
        switch axis {
        case .valence: return String(localized: "synchrony.metric.valence")
        case .arousal: return String(localized: "synchrony.metric.arousal")
        }
    }

    // MARK: - Pair rows

    @ViewBuilder
    private func pairRowsSection(
        result: AffectiveSynchrony.Result
    ) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            ForEach(rankedPairs(from: result), id: \.pair) { entry in
                pairRow(for: entry, maxLag: result.maxLag)
            }
        }
    }

    private func rankedPairs(
        from result: AffectiveSynchrony.Result
    ) -> [AffectiveSynchrony.PairResult] {
        result.pairs.sorted { lhs, rhs in
            let lhsValue = axisValue(lhs)
            let rhsValue = axisValue(rhs)
            switch (lhsValue, rhsValue) {
            case let (l?, r?): return abs(l) > abs(r)
            case (.some, .none): return true
            case (.none, .some): return false
            case (.none, .none): return false
            }
        }
    }

    private func axisValue(
        _ entry: AffectiveSynchrony.PairResult
    ) -> Double? {
        switch axis {
        case .valence: return entry.valenceCorrelation
        case .arousal: return entry.arousalCorrelation
        }
    }

    private func axisProfile(
        _ entry: AffectiveSynchrony.PairResult
    ) -> [Double?] {
        switch axis {
        case .valence: return entry.valenceProfile
        case .arousal: return entry.arousalProfile
        }
    }

    @ViewBuilder
    private func pairRow(
        for entry: AffectiveSynchrony.PairResult,
        maxLag: Int
    ) -> some View {
        let value = axisValue(entry)
        HStack(spacing: 8) {
            speakerChip(entry.pair.leader)
            Image(systemName: "arrow.right")
                .font(.caption2)
                .foregroundStyle(.tertiary)
            speakerChip(entry.pair.follower)
            Spacer(minLength: 4)
            // Lag-profile sparkline (omitted when only lag 0 carries
            // data — visually noisy and uninformative otherwise).
            if maxLag > 0,
               axisProfile(entry).dropFirst().contains(where: { $0 != nil }) {
                lagSparkline(axisProfile(entry))
            }
            correlationBar(for: value)
            Text(formattedCorrelation(value))
                .font(.caption2.monospacedDigit())
                .foregroundStyle(.primary)
                .frame(width: 44, alignment: .trailing)
            Text("n=\(entry.sampleCount)")
                .font(.caption2.monospacedDigit())
                .foregroundStyle(.tertiary)
                .frame(width: 40, alignment: .trailing)
        }
        .padding(.horizontal, 4)
        .padding(.vertical, 2)
    }

    @ViewBuilder
    private func speakerChip(_ id: String) -> some View {
        Text(id)
            .font(.caption.monospaced())
            .foregroundStyle(speakerTint(for: id))
            .padding(.horizontal, 6)
            .padding(.vertical, 1)
            .background(
                speakerTint(for: id).opacity(0.15),
                in: Capsule()
            )
    }

    @ViewBuilder
    private func correlationBar(for value: Double?) -> some View {
        let height: CGFloat = 6
        let halfWidth: CGFloat = 32
        ZStack(alignment: .center) {
            Capsule()
                .fill(Color(uiColor: .tertiarySystemFill))
                .frame(width: halfWidth * 2, height: height)
            Rectangle()
                .fill(Color(uiColor: .separator))
                .frame(width: 1, height: height + 2)
            if let v = value {
                let clamped = max(-1.0, min(1.0, v))
                let magnitude = CGFloat(abs(clamped)) * halfWidth
                Capsule()
                    .fill(clamped >= 0 ? Color.green : Color.red)
                    .frame(width: magnitude, height: height)
                    .offset(
                        x: clamped >= 0
                            ? magnitude / 2
                            : -magnitude / 2
                    )
            }
        }
        .frame(width: halfWidth * 2, height: height + 2)
    }

    /// Small Canvas sparkline showing the correlation evolution over
    /// lags 0…maxLag. Y-axis is fixed at ±1 with a midline at 0;
    /// each lag is a dot connected by a polyline, dot color tinted
    /// green / red by sign. Width is bounded so it slots into the
    /// row inline without forcing layout to grow.
    @ViewBuilder
    private func lagSparkline(_ profile: [Double?]) -> some View {
        Canvas { ctx, size in
            let height = size.height
            let width = size.width
            let mid = height / 2
            // Midline
            var midline = Path()
            midline.move(to: CGPoint(x: 0, y: mid))
            midline.addLine(to: CGPoint(x: width, y: mid))
            ctx.stroke(
                midline,
                with: .color(Color(uiColor: .separator)),
                lineWidth: 0.5
            )
            // Points
            let count = max(profile.count, 1)
            var path = Path()
            var hasLastPoint = false
            for (i, value) in profile.enumerated() {
                let x = count > 1
                    ? width * CGFloat(i) / CGFloat(count - 1)
                    : width / 2
                guard let v = value else {
                    hasLastPoint = false
                    continue
                }
                let clamped = max(-1.0, min(1.0, v))
                let y = mid - CGFloat(clamped) * (height / 2 - 1)
                let pt = CGPoint(x: x, y: y)
                if hasLastPoint {
                    path.addLine(to: pt)
                } else {
                    path.move(to: pt)
                }
                hasLastPoint = true
                let dotR: CGFloat = 1.5
                let dotRect = CGRect(
                    x: pt.x - dotR, y: pt.y - dotR,
                    width: 2 * dotR, height: 2 * dotR
                )
                ctx.fill(
                    Path(ellipseIn: dotRect),
                    with: .color(clamped >= 0 ? Color.green : Color.red)
                )
            }
            ctx.stroke(
                path,
                with: .color(Color.secondary.opacity(0.6)),
                lineWidth: 0.8
            )
        }
        .frame(width: 36, height: 14)
        .accessibilityHidden(true)
    }

    // MARK: - Leadership

    @ViewBuilder
    private func leadershipSection(
        _ scores: [AffectiveSynchrony.LeadershipScore]
    ) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 6) {
                Text(String(localized: "synchrony.leadership.header"))
                    .font(.caption2.bold())
                    .foregroundStyle(.secondary)
                    .textCase(.uppercase)
                Text(String(localized: "synchrony.leadership.lagNote"))
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
            ForEach(scores, id: \.speakerID) { score in
                let value: Double? = (axis == .valence)
                    ? score.valenceLeadership
                    : score.arousalLeadership
                HStack(spacing: 8) {
                    speakerChip(score.speakerID)
                    Spacer(minLength: 4)
                    correlationBar(for: value)
                    Text(formattedCorrelation(value))
                        .font(.caption2.monospacedDigit())
                        .foregroundStyle(.primary)
                        .frame(width: 44, alignment: .trailing)
                    Text("n=\(score.sampleCount)")
                        .font(.caption2.monospacedDigit())
                        .foregroundStyle(.tertiary)
                        .frame(width: 40, alignment: .trailing)
                }
                .padding(.horizontal, 4)
                .padding(.vertical, 2)
            }
        }
    }

    // MARK: - Plutchik dyads

    @ViewBuilder
    private func dyadSection(
        _ tallies: [AffectiveSynchrony.DyadTally]
    ) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(String(localized: "synchrony.dyad.header"))
                .font(.caption2.bold())
                .foregroundStyle(.secondary)
                .textCase(.uppercase)
            // Horizontal flow with wrapping. Each chip is a small
            // capsule: "love · 12". Width-bounded VStack of HStacks
            // would be cleaner with FlowLayout, but a single HStack
            // with `.lineLimit(1)` per chip + lazy width handling
            // covers the typical N≤8 dyads case fine.
            HStack(spacing: 6) {
                ForEach(tallies, id: \.dyad) { tally in
                    dyadChip(tally)
                }
                Spacer(minLength: 0)
            }
        }
    }

    @ViewBuilder
    private func dyadChip(_ tally: AffectiveSynchrony.DyadTally) -> some View {
        let tint = dyadTint(tally.dyad)
        HStack(spacing: 4) {
            Text(localizedDyadName(tally.dyad))
                .font(.caption2.bold())
            Text("\(tally.count)")
                .font(.caption2.monospacedDigit())
                .foregroundStyle(.secondary)
        }
        .padding(.horizontal, 6)
        .padding(.vertical, 2)
        .foregroundStyle(tint)
        .background(tint.opacity(0.18), in: Capsule())
    }

    private func localizedDyadName(
        _ dyad: AffectiveSynchrony.PlutchikDyad
    ) -> String {
        switch dyad {
        case .love:        return String(localized: "synchrony.dyad.love")
        case .submission:  return String(localized: "synchrony.dyad.submission")
        case .awe:         return String(localized: "synchrony.dyad.awe")
        case .disapproval: return String(localized: "synchrony.dyad.disapproval")
        case .remorse:     return String(localized: "synchrony.dyad.remorse")
        case .contempt:    return String(localized: "synchrony.dyad.contempt")
        case .aggression:  return String(localized: "synchrony.dyad.aggression")
        case .optimism:    return String(localized: "synchrony.dyad.optimism")
        }
    }

    /// Distinct tint per dyad on the V/A plane — positive-valence
    /// dyads (love, optimism, awe) lean warm; negative-valence
    /// (contempt, remorse, aggression, disapproval) lean cool;
    /// submission is neutral. Distinct enough to read in a row of
    /// 4–6 chips without checking the label.
    private func dyadTint(_ dyad: AffectiveSynchrony.PlutchikDyad) -> Color {
        switch dyad {
        case .love:        return Color(red: 0.93, green: 0.30, blue: 0.55)
        case .optimism:    return Color(red: 0.95, green: 0.65, blue: 0.20)
        case .awe:         return Color(red: 0.20, green: 0.70, blue: 0.85)
        case .submission:  return Color(red: 0.55, green: 0.45, blue: 0.75)
        case .disapproval: return Color(red: 0.40, green: 0.55, blue: 0.75)
        case .remorse:     return Color(red: 0.45, green: 0.45, blue: 0.50)
        case .contempt:    return Color(red: 0.75, green: 0.25, blue: 0.30)
        case .aggression:  return Color(red: 0.85, green: 0.40, blue: 0.20)
        }
    }

    private func formattedCorrelation(_ value: Double?) -> String {
        guard let value else { return "—" }
        return String(format: "%+.2f", value)
    }
}
