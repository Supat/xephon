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
    /// Pair whose inspector popover is currently open, if any.
    /// Tapping a pair row toggles it; the popover surfaces the
    /// full lag profile for both V and A — content that used to
    /// sit inline as a sparkline but overflowed the iPad portrait
    /// row budget. Mirrors `SpeakerBehaviorCard`'s tap-popover
    /// pattern so the affordance is consistent across cluster
    /// diagnostics cards.
    @State private var inspectedPair: AffectiveSynchrony.DirectedPair?

    enum Axis: Hashable {
        case valence, arousal
    }

    /// Trailing value-column width (correlation + sample count).
    /// Sized for the widest plausible "−0.99 · 999" rendering at
    /// caption2 monospaced; pinning both pair rows and leadership
    /// rows to this width keeps the bars + numbers aligned across
    /// sections.
    private static let valueColumnWidth: CGFloat = 64

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
        HStack(spacing: 6) {
            Text(String(localized: "synchrony.header"))
                .font(.caption.bold())
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .minimumScaleFactor(0.8)
            Spacer(minLength: 4)
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
                    .lineLimit(1)
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

    @ViewBuilder
    private func pairRow(
        for entry: AffectiveSynchrony.PairResult,
        maxLag: Int
    ) -> some View {
        // Layout sized for the iPad portrait left-pane budget
        // (~280pt). Inline width is tight on purpose — anything
        // richer (lag-profile sparkline, per-lag sample counts)
        // lives in the tap-popover.
        let value = axisValue(entry)
        Button {
            inspectedPair = (inspectedPair == entry.pair)
                ? nil : entry.pair
        } label: {
            HStack(spacing: 6) {
                speakerChip(entry.pair.leader)
                Image(systemName: "arrow.right")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                speakerChip(entry.pair.follower)
                Spacer(minLength: 4)
                correlationBar(for: value)
                // Value + sample count combined into one monospaced
                // column ("+0.52 · 12"). Fixed-width frame so the
                // bar's right edge lines up across pair rows AND
                // leadership rows — without it the bar sat at a
                // floating X position because the value text's
                // intrinsic width swings ~20pt between "+0.5 · 5"
                // and "−0.99 · 100". Per-lag sample counts live in
                // the tap-popover.
                Text(formattedValueWithCount(value, entry.sampleCount))
                    .font(.caption2.monospacedDigit())
                    .foregroundStyle(.primary)
                    .lineLimit(1)
                    .frame(width: Self.valueColumnWidth, alignment: .trailing)
            }
            .padding(.horizontal, 4)
            .padding(.vertical, 2)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .popover(
            isPresented: Binding(
                get: { inspectedPair == entry.pair },
                set: { isPresented in
                    if !isPresented { inspectedPair = nil }
                }
            ),
            arrowEdge: .top
        ) {
            pairInspectorPopover(entry, maxLag: maxLag)
                .presentationCompactAdaptation(.popover)
        }
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
        // Half-width 24 (total 48pt) keeps the inline row inside the
        // iPad portrait left-pane budget while staying readable —
        // the bar's job is sign + rough magnitude, not precise
        // value. Numeric value sits next to it for the exact read.
        let halfWidth: CGFloat = 24
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

    // MARK: - Pair inspector popover

    /// Tap-revealed inspector showing both V and A lag profiles +
    /// a per-lag numeric table for the pair. Wider sparklines than
    /// would fit inline, and the full numeric breakdown for users
    /// who want to read precise correlations rather than rough
    /// magnitude off the headline bar.
    @ViewBuilder
    private func pairInspectorPopover(
        _ entry: AffectiveSynchrony.PairResult,
        maxLag: Int
    ) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 6) {
                speakerChip(entry.pair.leader)
                Image(systemName: "arrow.right")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                speakerChip(entry.pair.follower)
            }
            Divider()
            sparklineGroup(
                title: String(localized: "synchrony.metric.valence"),
                profile: entry.valenceProfile
            )
            sparklineGroup(
                title: String(localized: "synchrony.metric.arousal"),
                profile: entry.arousalProfile
            )
            Divider()
            lagTable(entry, maxLag: maxLag)
        }
        .padding(12)
        .frame(minWidth: 260)
    }

    @ViewBuilder
    private func sparklineGroup(
        title: String,
        profile: [Double?]
    ) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.caption2.weight(.semibold))
                .foregroundStyle(.tertiary)
                .textCase(.uppercase)
            lagSparkline(profile)
                .frame(height: 18)
        }
    }

    /// Small Canvas plot of correlation evolution over lags
    /// 0…maxLag. Y-axis fixed at ±1 with a midline at 0; each
    /// lag is a dot connected by a polyline; dot color tinted
    /// green / red by sign. Nil values (sample-count below the
    /// threshold) break the line so the user can see which lags
    /// actually carried data.
    @ViewBuilder
    private func lagSparkline(_ profile: [Double?]) -> some View {
        Canvas { ctx, size in
            let height = size.height
            let width = size.width
            let mid = height / 2
            var midline = Path()
            midline.move(to: CGPoint(x: 0, y: mid))
            midline.addLine(to: CGPoint(x: width, y: mid))
            ctx.stroke(
                midline,
                with: .color(Color(uiColor: .separator)),
                lineWidth: 0.5
            )
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
                let y = mid - CGFloat(clamped) * (height / 2 - 2)
                let pt = CGPoint(x: x, y: y)
                if hasLastPoint {
                    path.addLine(to: pt)
                } else {
                    path.move(to: pt)
                }
                hasLastPoint = true
                let dotR: CGFloat = 2
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
                lineWidth: 0.9
            )
        }
        .accessibilityHidden(true)
    }

    /// 4-column table (Lag · V · A · n) listing the correlation at
    /// each lag with its supporting sample count. "—" for lags
    /// where the sample count fell below the threshold.
    @ViewBuilder
    private func lagTable(
        _ entry: AffectiveSynchrony.PairResult,
        maxLag: Int
    ) -> some View {
        VStack(spacing: 2) {
            HStack {
                Text(String(localized: "synchrony.popover.lag"))
                    .frame(width: 36, alignment: .leading)
                Text(String(localized: "synchrony.popover.valenceShort"))
                    .frame(width: 52, alignment: .trailing)
                Text(String(localized: "synchrony.popover.arousalShort"))
                    .frame(width: 52, alignment: .trailing)
                Text(String(localized: "synchrony.popover.sampleCount"))
                    .frame(width: 36, alignment: .trailing)
            }
            .font(.caption2.weight(.semibold))
            .foregroundStyle(.tertiary)
            ForEach(0...maxLag, id: \.self) { lag in
                HStack {
                    Text("\(lag)")
                        .frame(width: 36, alignment: .leading)
                    Text(formattedCorrelation(
                        lag < entry.valenceProfile.count
                            ? entry.valenceProfile[lag] : nil
                    ))
                    .frame(width: 52, alignment: .trailing)
                    Text(formattedCorrelation(
                        lag < entry.arousalProfile.count
                            ? entry.arousalProfile[lag] : nil
                    ))
                    .frame(width: 52, alignment: .trailing)
                    Text("\(lag < entry.sampleCountsByLag.count ? entry.sampleCountsByLag[lag] : 0)")
                        .frame(width: 36, alignment: .trailing)
                }
                .font(.caption2.monospacedDigit())
                .foregroundStyle(.primary)
            }
        }
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
                HStack(spacing: 6) {
                    speakerChip(score.speakerID)
                    Spacer(minLength: 4)
                    correlationBar(for: value)
                    Text(formattedValueWithCount(value, score.sampleCount))
                        .font(.caption2.monospacedDigit())
                        .foregroundStyle(.primary)
                        .lineLimit(1)
                        .frame(width: Self.valueColumnWidth, alignment: .trailing)
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
            // Horizontal scroll lets the chip row exceed the card's
            // width without forcing the card to grow — typical N≤8
            // dyads usually fit, but a wide ja/zh chip text can
            // push past the iPad portrait left-pane budget.
            // `showsIndicators: false` keeps the scroll affordance
            // invisible at rest; the user discovers it on swipe.
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 6) {
                    ForEach(tallies, id: \.dyad) { tally in
                        dyadChip(tally)
                    }
                }
                .padding(.horizontal, 2)
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

    /// Single-column readout combining correlation + sample count
    /// for the narrow inline row, e.g. "+0.52 · 12". When the
    /// correlation is missing, falls back to "— · n".
    private func formattedValueWithCount(
        _ value: Double?,
        _ sampleCount: Int
    ) -> String {
        let corr = formattedCorrelation(value)
        return "\(corr) · \(sampleCount)"
    }
}
