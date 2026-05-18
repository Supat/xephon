import SwiftUI
import Diarization
import XephonUtilities

/// Pairwise speaker-similarity heatmap, rendered as a small N×N grid
/// of cosine distances between every pair of speakers' averaged
/// embeddings. Lives below `StatisticsCard` alongside
/// `SpeakerClusterCard` (the 2D PCA scatter) as a diagnostic view
/// of the diarizer's internal cluster.
///
/// Lower distance = warmer color (red) — short distance between two
/// distinct speakers is the alarming case, meaning the diarizer
/// might be conflating them. Long distance (green) = clean
/// separation. The diagonal is always zero by definition and
/// renders as the warmest tone.
///
/// N here is bounded by FluidAudio's diarizer caps (≤ 4 for
/// Sortformer, ≤ 10 for LS-EEND), so the O(N²) distance loop and
/// the corresponding view tree are trivially cheap to recompute on
/// every snapshot refresh.
struct SpeakerHeatmapCard: View {
    let cluster: SpeakerClusterSnapshot
    /// Speaker id of the currently-focused utterance, or nil. Used
    /// to outline the matching row + column in the grid so the
    /// user can see "this is the speaker whose row I'm reading"
    /// across the diagnostics page.
    var highlightedSpeakerID: String?
    /// Speaker ids referenced by at least one utterance. Drives
    /// the "Linked only" toggle: when on, rows + columns for
    /// speakers absent from this set are dropped from the grid
    /// so the matrix focuses on the conversation's active cast.
    /// Nil disables the toggle.
    var linkedSpeakerIDs: Set<String>?

    /// Header toggle: hide speakers whose id isn't in
    /// `linkedSpeakerIDs`. Same caption2 link-icon button the
    /// cluster card uses.
    @State private var hideUnreferencedSpeakers: Bool = false

    /// Identifies the cell whose popover is currently shown. Carries
    /// the row/column speaker ids + the cosine distance so the
    /// popover body is self-contained (the cell view itself doesn't
    /// own the popover; the card hosts a single anchor-driven
    /// popover keyed by this state).
    @State private var selectedCell: HeatCellSelection?

    struct HeatCellSelection: Identifiable, Equatable {
        let rowSpeaker: String
        let columnSpeaker: String
        let distance: Float
        // Pair the two ids in a stable order so `.popover(item:)`
        // dismisses + re-presents when a different cell is tapped
        // (the modifier keys re-presentation off `id` equality).
        var id: String { "\(rowSpeaker)|\(columnSpeaker)" }
    }

    /// Preferred side length of each grid cell when there's room.
    /// The grid auto-shrinks each cell below this when the available
    /// width can't fit N at full size, with a hard floor so cells
    /// remain visually distinguishable.
    private static let maxCellSize: CGFloat = 22
    /// Floor for the per-cell side length. Below ~10 pt the numeric
    /// label drops out and the cells become pure color samples; at
    /// 6 pt they're still readable as a heatmap on iPad.
    private static let minCellSize: CGFloat = 6
    /// Threshold below which the numeric "0.85" label is dropped
    /// because the glyph won't fit. The grid stays informative as
    /// a pure color matrix.
    private static let labelDropCellSize: CGFloat = 14
    /// Gap between cells so each one reads as a separate sample;
    /// without this the grid looks like a continuous heat surface
    /// and the per-pair quantization is lost.
    private static let cellSpacing: CGFloat = 2
    /// Width of the leftmost column (speaker-id labels). Wide enough
    /// for "S10".
    private static let rowLabelWidth: CGFloat = 28

    var body: some View {
        let speakers = visibleSpeakers
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 6) {
                Text(String(localized: "cluster.heatmap.header"))
                    .font(.caption.bold())
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .minimumScaleFactor(0.8)
                Spacer(minLength: 4)
                // Only surface the toggle when at least one
                // cluster speaker isn't in the utterance-linked
                // set — otherwise it'd be a no-op control.
                if let linked = linkedSpeakerIDs,
                   cluster.speakers.contains(where: { !linked.contains($0.id) }) {
                    Button {
                        hideUnreferencedSpeakers.toggle()
                    } label: {
                        Label(
                            String(localized: "cluster.scatter.linkedOnly"),
                            systemImage: hideUnreferencedSpeakers
                                ? "link.circle.fill"
                                : "link.circle"
                        )
                        .font(.caption2)
                        .labelStyle(.titleAndIcon)
                        .lineLimit(1)
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(
                        hideUnreferencedSpeakers
                            ? AnyShapeStyle(Color.accentColor)
                            : AnyShapeStyle(HierarchicalShapeStyle.secondary)
                    )
                }
                if !speakers.isEmpty {
                    Text("\(speakers.count)")
                        .font(.caption2.monospacedDigit())
                        .foregroundStyle(.tertiary)
                }
            }
            if speakers.isEmpty {
                Text(String(localized: "cluster.empty"))
                    .font(.caption)
                    .foregroundStyle(.tertiary)
                    .frame(maxWidth: .infinity, alignment: .leading)
            } else {
                grid(for: speakers)
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .glassEffect(in: RoundedRectangle(cornerRadius: 12, style: .continuous))
    }

    /// Speaker list the grid actually renders. Applies the
    /// "Linked only" filter when the toggle is on and a
    /// referenced-id set is available; otherwise the full
    /// `cluster.speakers` array.
    private var visibleSpeakers: [SpeakerClusterSnapshot.Speaker] {
        guard hideUnreferencedSpeakers,
              let linked = linkedSpeakerIDs else { return cluster.speakers }
        return cluster.speakers.filter { linked.contains($0.id) }
    }

    @ViewBuilder
    private func grid(for speakers: [SpeakerClusterSnapshot.Speaker]) -> some View {
        // GeometryReader gives us the card's interior width post-
        // padding, which is the only signal we have for how big a
        // square N×N grid can be without overflowing. Wrapping the
        // grid in a fixed-height frame matched to the computed cell
        // side keeps the layout deterministic (GeometryReader's own
        // sizing is greedy by default, which would blow the card
        // out vertically).
        let n = CGFloat(max(speakers.count, 1))
        GeometryReader { proxy in
            let cellSize = Self.cellSide(available: proxy.size.width, n: n)
            let showLabels = cellSize >= Self.labelDropCellSize
            grid(speakers: speakers, cellSize: cellSize, showLabels: showLabels)
        }
        .frame(height: Self.gridHeight(forSpeakerCount: speakers.count))
    }

    @ViewBuilder
    private func grid(
        speakers: [SpeakerClusterSnapshot.Speaker],
        cellSize: CGFloat,
        showLabels: Bool
    ) -> some View {
        // The grid's intrinsic width is `rowLabelWidth + n × cellSize
        // + (n - 1) × cellSpacing`, which is usually narrower than
        // the card's interior — cells cap at `preferredCellSide`
        // (~22pt). The outer card uses `alignment: .leading`, so
        // without an explicit centering frame the grid hugs the
        // left edge and leaves an asymmetric gutter on the right.
        // `.frame(maxWidth: .infinity, alignment: .center)` keeps
        // the grid at its intrinsic width but centers it inside
        // the available row.
        VStack(alignment: .leading, spacing: Self.cellSpacing) {
            // Top header row: blank corner + speaker ids across.
            // Dropped entirely when cells are too small to hold a
            // legible speaker-id; the row labels on the left edge
            // still convey column order because the matrix is
            // symmetric (col k = row k along the diagonal).
            if showLabels {
                HStack(spacing: Self.cellSpacing) {
                    Color.clear
                        .frame(width: Self.rowLabelWidth, height: cellSize)
                    ForEach(speakers, id: \.id) { spk in
                        Text(spk.id)
                            .font(.caption2.monospaced())
                            .foregroundStyle(speakerTint(for: spk.id))
                            .frame(width: cellSize, height: cellSize)
                            .lineLimit(1)
                            .minimumScaleFactor(0.5)
                    }
                }
            }
            ForEach(speakers, id: \.id) { rowSpk in
                HStack(spacing: Self.cellSpacing) {
                    Text(rowSpk.id)
                        .font(.caption2.monospaced())
                        .foregroundStyle(speakerTint(for: rowSpk.id))
                        .frame(width: Self.rowLabelWidth, height: cellSize, alignment: .leading)
                        .lineLimit(1)
                    ForEach(speakers, id: \.id) { colSpk in
                        let d = Self.cosineDistance(rowSpk.centroid, colSpk.centroid)
                        cell(
                            rowSpk: rowSpk,
                            colSpk: colSpk,
                            distance: d,
                            cellSize: cellSize,
                            showLabel: showLabels,
                            highlighted: highlightedSpeakerID == rowSpk.id
                                || highlightedSpeakerID == colSpk.id
                        )
                    }
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .center)
    }

    /// One heat cell: filled rounded rectangle + (optionally) the
    /// numeric distance overlaid. Hosts the popover anchor for that
    /// cell — tapping anywhere on the cell selects it and opens the
    /// popover with the pair's ids + distance.
    @ViewBuilder
    private func cell(
        rowSpk: SpeakerClusterSnapshot.Speaker,
        colSpk: SpeakerClusterSnapshot.Speaker,
        distance d: Float,
        cellSize: CGFloat,
        showLabel: Bool,
        highlighted: Bool = false
    ) -> some View {
        let selection = HeatCellSelection(
            rowSpeaker: rowSpk.id,
            columnSpeaker: colSpk.id,
            distance: d
        )
        ZStack {
            RoundedRectangle(cornerRadius: 3, style: .continuous)
                .fill(Self.heatColor(distance: d))
            if showLabel {
                Text(String(format: "%.2f", d))
                    .font(.system(size: 8, weight: .semibold))
                    .foregroundStyle(.white.opacity(0.95))
            }
            if highlighted {
                // Outline cells in the highlighted speaker's row
                // or column so the cross of cells stands out
                // against the surrounding grid. White stroke at
                // 90% reads on every heat color we render.
                RoundedRectangle(cornerRadius: 3, style: .continuous)
                    .stroke(
                        Color.white.opacity(0.9),
                        lineWidth: 1.5
                    )
            }
        }
        .frame(width: cellSize, height: cellSize)
        // `.contentShape` ensures the whole cell area takes the tap
        // even when the inner Text doesn't fill it (tiny cells).
        .contentShape(RoundedRectangle(cornerRadius: 3, style: .continuous))
        .onTapGesture { selectedCell = selection }
        .popover(
            item: Binding(
                get: { selectedCell?.id == selection.id ? selectedCell : nil },
                set: { selectedCell = $0 }
            ),
            attachmentAnchor: .point(.center),
            arrowEdge: .top
        ) { sel in
            cellPopover(sel)
        }
    }

    /// Popover body for a tapped heatmap cell. Shows the two speaker
    /// ids in their assigned tint colors so the user can read which
    /// pair this distance refers to without cross-referencing the
    /// grid axes, plus the cosine distance to three decimals (the
    /// in-cell label rounds to two; the popover gives the precision
    /// users came for).
    @ViewBuilder
    private func cellPopover(_ sel: HeatCellSelection) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                speakerChip(sel.rowSpeaker)
                Image(systemName: "arrow.left.and.right")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                speakerChip(sel.columnSpeaker)
            }
            HStack(spacing: 6) {
                RoundedRectangle(cornerRadius: 3, style: .continuous)
                    .fill(Self.heatColor(distance: sel.distance))
                    .frame(width: 14, height: 14)
                Text(String(format: "distance: %.3f", sel.distance))
                    .font(.callout.monospacedDigit())
                    .foregroundStyle(.primary)
            }
            Text(distanceInterpretation(sel.distance))
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        .padding(12)
        .presentationCompactAdaptation(.popover)
    }

    @ViewBuilder
    private func speakerChip(_ id: String) -> some View {
        Text(id)
            .font(.caption.monospaced().bold())
            .foregroundStyle(speakerTint(for: id))
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(speakerTint(for: id).opacity(0.15), in: Capsule())
    }

    /// One-liner reading of where the distance falls on the heat
    /// scale, mirroring the same thresholds used by `heatColor`.
    /// Helps users back-project the number to "is this concerning?"
    /// without having to remember the gradient stops.
    private func distanceInterpretation(_ d: Float) -> String {
        if d < 0.4 {
            return String(localized: "cluster.heatmap.tip.close")
        } else if d < 0.7 {
            return String(localized: "cluster.heatmap.tip.mid")
        } else {
            return String(localized: "cluster.heatmap.tip.far")
        }
    }

    /// Per-cell side length that fits N cells + the row-label
    /// gutter inside `available` width. Clamped to `[minCellSize,
    /// maxCellSize]` so small sessions don't waste space and huge
    /// sessions don't disappear into single-pixel cells.
    private static func cellSide(available: CGFloat, n: CGFloat) -> CGFloat {
        guard n > 0, available > 0 else { return maxCellSize }
        let usable = available - rowLabelWidth - cellSpacing * (n + 1)
        let raw = usable / n
        return raw.clamped(to: minCellSize...maxCellSize)
    }

    /// Total grid height for `count` speakers at the size the grid
    /// will pick after measuring. Used to give the GeometryReader a
    /// concrete height so it doesn't expand to fill the parent.
    /// Computed assuming the worst case (`maxCellSize`) since that's
    /// always an upper bound — the actual rendered grid will be at
    /// most this tall.
    private static func gridHeight(forSpeakerCount count: Int) -> CGFloat {
        // count + 1 row (header + N data rows), spaced.
        let rows = CGFloat(count + 1)
        return rows * maxCellSize + (rows - 1) * cellSpacing
    }

    /// Cosine distance for two L2-normalized vectors (the FluidAudio
    /// extractor's invariant). For unit-norm `a` and `b`,
    /// `cos_sim = a·b` and `cos_dist = 1 − cos_sim`. We clamp the
    /// result into `[0, 1]` even though the theoretical range is
    /// `[0, 2]` — FluidAudio's embeddings cluster tightly enough that
    /// negative similarities don't show up in practice and the
    /// extended range would only waste color budget on cells we
    /// never see.
    static func cosineDistance(_ a: [Float], _ b: [Float]) -> Float {
        let n = min(a.count, b.count)
        guard n > 0 else { return 1 }
        var dot: Float = 0
        for i in 0..<n { dot += a[i] * b[i] }
        return (1 - dot).clamped(to: 0...1)
    }

    /// Three-stop gradient: short distance (potential collision) →
    /// warm red, mid → orange/yellow, long distance (clean
    /// separation) → green. Matches the mental model "red is bad"
    /// — a red cell off the diagonal means the diarizer might be
    /// merging two real speakers.
    static func heatColor(distance: Float) -> Color {
        let clamped = Double(distance.clamped(to: 0...1))
        if clamped < 0.4 {
            let t = clamped / 0.4
            return Color(
                red: 0.86,
                green: 0.20 + 0.50 * t,
                blue: 0.22
            )
        } else if clamped < 0.7 {
            let t = (clamped - 0.4) / 0.3
            return Color(
                red: 0.86 - 0.50 * t,
                green: 0.70,
                blue: 0.22 + 0.30 * t
            )
        } else {
            return Color(red: 0.36, green: 0.70, blue: 0.52)
        }
    }
}
