import SwiftUI
import Fusion

/// Surfaces the four turn-taking dynamics analyses from
/// `docs/social_dynamics_backlog.md` #1–#4. Three visual blocks:
///
///   1. Per-speaker row — median floor-holding run length and
///      backchannel rate, the two purely-per-speaker metrics.
///   2. Interruption matrix — directed pair counts, rows =
///      interrupter, columns = victim.
///   3. Response-latency matrix — directed pair medians (seconds),
///      rows = responder, columns = partner.
///
/// All speakers participating in the session appear in both
/// matrices, so an empty cell reads as "no overlap / no response"
/// rather than "speaker missing." The diagonal is a `—` since
/// self-interruptions and self-responses aren't meaningful here.
struct TurnTakingCard: View {
    let utterances: [UtteranceEstimate]

    private static let cellSize: CGFloat = 36
    private static let cellSpacing: CGFloat = 2
    private static let speakerLabelWidth: CGFloat = 44

    var body: some View {
        let profile = TurnTakingDynamics.compute(utterances: utterances)
        let speakers = orderedSpeakerIDs(from: utterances)
        VStack(alignment: .leading, spacing: 14) {
            header(speakerCount: speakers.count)
            if utterances.isEmpty || speakers.isEmpty {
                emptyState
            } else {
                perSpeakerSection(
                    floorHolding: profile.floorHolding,
                    backchannels: profile.backchannels,
                    speakers: speakers
                )
                Divider().opacity(0.4)
                interruptionMatrix(
                    pairs: profile.interruptions,
                    speakers: speakers
                )
                Divider().opacity(0.4)
                latencyMatrix(
                    pairs: profile.responseLatencies,
                    speakers: speakers
                )
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .glassEffect(in: RoundedRectangle(cornerRadius: 12, style: .continuous))
    }

    @ViewBuilder
    private func header(speakerCount: Int) -> some View {
        HStack(spacing: 6) {
            Text(String(localized: "turntaking.header"))
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

    @ViewBuilder
    private var emptyState: some View {
        Text(String(localized: "turntaking.empty"))
            .font(.caption)
            .foregroundStyle(.tertiary)
            .frame(maxWidth: .infinity, alignment: .leading)
    }

    @ViewBuilder
    private func perSpeakerSection(
        floorHolding: [TurnTakingDynamics.FloorHolding],
        backchannels: [TurnTakingDynamics.Backchannel],
        speakers: [String]
    ) -> some View {
        let floorMap = Dictionary(
            uniqueKeysWithValues: floorHolding.map { ($0.speakerID, $0) }
        )
        let bcMap = Dictionary(
            uniqueKeysWithValues: backchannels.map { ($0.speakerID, $0) }
        )
        VStack(alignment: .leading, spacing: 6) {
            sectionTitle(String(localized: "turntaking.section.perSpeaker"))
            HStack(spacing: 0) {
                Color.clear.frame(width: Self.speakerLabelWidth)
                Text(String(localized: "turntaking.col.floor"))
                    .font(.caption2.bold())
                    .foregroundStyle(.tertiary)
                    .frame(maxWidth: .infinity, alignment: .center)
                Text(String(localized: "turntaking.col.backchannel"))
                    .font(.caption2.bold())
                    .foregroundStyle(.tertiary)
                    .frame(maxWidth: .infinity, alignment: .center)
            }
            ForEach(speakers, id: \.self) { spk in
                HStack(spacing: 0) {
                    speakerChip(spk)
                        .frame(width: Self.speakerLabelWidth, alignment: .leading)
                    Text(floorMap[spk].map { formatClock($0.medianSeconds) } ?? "—")
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.primary)
                        .frame(maxWidth: .infinity, alignment: .center)
                    Text(bcMap[spk].map { String(format: "%.0f%%", $0.rate * 100) } ?? "—")
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.primary)
                        .frame(maxWidth: .infinity, alignment: .center)
                }
            }
        }
    }

    @ViewBuilder
    private func interruptionMatrix(
        pairs: [TurnTakingDynamics.InterruptionPair],
        speakers: [String]
    ) -> some View {
        let map = Dictionary(
            uniqueKeysWithValues: pairs.map {
                (PairKey($0.interrupter, $0.victim), $0.count)
            }
        )
        let maxCount = pairs.map(\.count).max() ?? 0
        VStack(alignment: .leading, spacing: 4) {
            sectionTitle(String(localized: "turntaking.section.interruptions"))
            matrixGrid(speakers: speakers) { interrupter, victim in
                interruptionCell(
                    interrupter: interrupter,
                    victim: victim,
                    map: map,
                    maxCount: maxCount
                )
            }
            Text(String(localized: "turntaking.footnote.interruptions"))
                .font(.caption2)
                .foregroundStyle(.tertiary)
        }
    }

    @ViewBuilder
    private func interruptionCell(
        interrupter: String,
        victim: String,
        map: [PairKey: Int],
        maxCount: Int
    ) -> some View {
        if interrupter == victim {
            Text("—")
                .font(.caption.monospacedDigit())
                .foregroundStyle(.tertiary)
                .frame(width: Self.cellSize, height: Self.cellSize)
        } else {
            let count = map[PairKey(interrupter, victim)] ?? 0
            let intensity = maxCount > 0 ? Double(count) / Double(maxCount) : 0
            ZStack {
                RoundedRectangle(cornerRadius: 4, style: .continuous)
                    .fill(
                        speakerTint(for: interrupter)
                            .opacity(count == 0 ? 0.04 : 0.15 + 0.45 * intensity)
                    )
                Text(count == 0 ? "" : "\(count)")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(count == 0 ? .tertiary : .primary)
            }
            .frame(width: Self.cellSize, height: Self.cellSize)
        }
    }

    @ViewBuilder
    private func latencyMatrix(
        pairs: [TurnTakingDynamics.PairLatency],
        speakers: [String]
    ) -> some View {
        let map = Dictionary(
            uniqueKeysWithValues: pairs.map {
                (PairKey($0.responder, $0.partner), $0)
            }
        )
        let validLatencies = pairs.map(\.medianSeconds).filter { $0.isFinite }
        let minLatency = validLatencies.min() ?? 0
        let maxLatency = validLatencies.max() ?? 0
        VStack(alignment: .leading, spacing: 4) {
            sectionTitle(String(localized: "turntaking.section.responseLatency"))
            matrixGrid(speakers: speakers) { responder, partner in
                latencyCell(
                    responder: responder,
                    partner: partner,
                    map: map,
                    minLatency: minLatency,
                    maxLatency: maxLatency
                )
            }
            Text(String(localized: "turntaking.footnote.responseLatency"))
                .font(.caption2)
                .foregroundStyle(.tertiary)
        }
    }

    @ViewBuilder
    private func latencyCell(
        responder: String,
        partner: String,
        map: [PairKey: TurnTakingDynamics.PairLatency],
        minLatency: Double,
        maxLatency: Double
    ) -> some View {
        if responder == partner {
            Text("—")
                .font(.caption.monospacedDigit())
                .foregroundStyle(.tertiary)
                .frame(width: Self.cellSize, height: Self.cellSize)
        } else if let entry = map[PairKey(responder, partner)] {
            // Faster response = greener (lower latency = warmer
            // social proximity). Span the actual session's range
            // so the gradient is meaningful even when all latencies
            // are short. Falls back to gray on a degenerate span.
            let span = maxLatency - minLatency
            let normalized: Double = span > 0
                ? (entry.medianSeconds - minLatency) / span
                : 0.5
            let tint: Color = Color(
                red: 0.20 + 0.60 * normalized,
                green: 0.70 - 0.40 * normalized,
                blue: 0.30
            )
            ZStack {
                RoundedRectangle(cornerRadius: 4, style: .continuous)
                    .fill(tint.opacity(0.30))
                Text(String(format: "%.1f", entry.medianSeconds))
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

    /// Shared grid scaffold: column header row of speaker chips +
    /// per-row leading speaker label + cells from the supplied
    /// builder. Horizontally scrollable when a session has more
    /// speakers than fit on screen.
    @ViewBuilder
    private func matrixGrid<Cell: View>(
        speakers: [String],
        @ViewBuilder cell: @escaping (_ row: String, _ col: String) -> Cell
    ) -> some View {
        ScrollView(.horizontal, showsIndicators: false) {
            VStack(alignment: .leading, spacing: Self.cellSpacing) {
                HStack(spacing: Self.cellSpacing) {
                    Color.clear.frame(width: Self.speakerLabelWidth)
                    ForEach(speakers, id: \.self) { col in
                        speakerChip(col)
                            .frame(width: Self.cellSize, alignment: .center)
                    }
                }
                ForEach(speakers, id: \.self) { row in
                    HStack(spacing: Self.cellSpacing) {
                        speakerChip(row)
                            .frame(width: Self.speakerLabelWidth, alignment: .leading)
                        ForEach(speakers, id: \.self) { col in
                            cell(row, col)
                        }
                    }
                }
            }
        }
    }

    @ViewBuilder
    private func sectionTitle(_ text: String) -> some View {
        Text(text)
            .font(.caption2.weight(.semibold))
            .foregroundStyle(.secondary)
            .textCase(.uppercase)
    }

    @ViewBuilder
    private func speakerChip(_ id: String) -> some View {
        Text(id)
            .font(.caption2.bold())
            .foregroundStyle(speakerTint(for: id))
            .lineLimit(1)
            .minimumScaleFactor(0.7)
    }

    /// First-appearance ordering matches the chip-bar so the matrices
    /// read in the same order the user is already scanning above.
    private func orderedSpeakerIDs(from utts: [UtteranceEstimate]) -> [String] {
        var seen: Set<String> = []
        var ordered: [String] = []
        for u in utts.sorted(by: { $0.start < $1.start })
            where !seen.contains(u.speakerID) {
            seen.insert(u.speakerID)
            ordered.append(u.speakerID)
        }
        return ordered
    }

    private struct PairKey: Hashable {
        let row: String
        let col: String
        init(_ row: String, _ col: String) {
            self.row = row
            self.col = col
        }
    }
}
