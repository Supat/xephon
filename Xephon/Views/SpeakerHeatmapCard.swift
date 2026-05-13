import SwiftUI
import Diarization

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

    /// Side length of each grid cell. Tuned to fit 10 speakers + the
    /// row-label gutter inside the 1/3-width control pane on iPad
    /// portrait without overflow.
    private static let cellSize: CGFloat = 22
    /// Gap between cells so each one reads as a separate sample;
    /// without this the grid looks like a continuous heat surface
    /// and the per-pair quantization is lost.
    private static let cellSpacing: CGFloat = 2
    /// Width of the leftmost column (speaker-id labels). Wide enough
    /// for "S10".
    private static let rowLabelWidth: CGFloat = 28

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(String(localized: "cluster.heatmap.header"))
                    .font(.caption.bold())
                    .foregroundStyle(.secondary)
                Spacer()
                if !cluster.speakers.isEmpty {
                    Text("\(cluster.speakers.count)")
                        .font(.caption2.monospacedDigit())
                        .foregroundStyle(.tertiary)
                }
            }
            if cluster.speakers.isEmpty {
                Text(String(localized: "cluster.empty"))
                    .font(.caption)
                    .foregroundStyle(.tertiary)
                    .frame(maxWidth: .infinity, alignment: .leading)
            } else {
                grid
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .glassEffect(in: RoundedRectangle(cornerRadius: 12, style: .continuous))
    }

    @ViewBuilder
    private var grid: some View {
        let speakers = cluster.speakers
        VStack(alignment: .leading, spacing: Self.cellSpacing) {
            // Top header row: blank corner + speaker ids across.
            HStack(spacing: Self.cellSpacing) {
                Color.clear
                    .frame(width: Self.rowLabelWidth, height: Self.cellSize)
                ForEach(speakers, id: \.id) { spk in
                    Text(spk.id)
                        .font(.caption2.monospaced())
                        .foregroundStyle(speakerTint(for: spk.id))
                        .frame(width: Self.cellSize, height: Self.cellSize)
                        .lineLimit(1)
                        .minimumScaleFactor(0.6)
                }
            }
            ForEach(speakers, id: \.id) { rowSpk in
                HStack(spacing: Self.cellSpacing) {
                    Text(rowSpk.id)
                        .font(.caption2.monospaced())
                        .foregroundStyle(speakerTint(for: rowSpk.id))
                        .frame(width: Self.rowLabelWidth, height: Self.cellSize, alignment: .leading)
                        .lineLimit(1)
                    ForEach(speakers, id: \.id) { colSpk in
                        let d = Self.cosineDistance(rowSpk.centroid, colSpk.centroid)
                        ZStack {
                            RoundedRectangle(cornerRadius: 3, style: .continuous)
                                .fill(Self.heatColor(distance: d))
                            // Numeric label inside the cell. Pulled
                            // out into a Text only when the cell is
                            // large enough to read it — at 22 pt
                            // there's just barely room for two
                            // digits.
                            Text(String(format: "%.2f", d))
                                .font(.system(size: 8, weight: .semibold))
                                .foregroundStyle(.white.opacity(0.95))
                        }
                        .frame(width: Self.cellSize, height: Self.cellSize)
                    }
                }
            }
        }
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
        return max(0, min(1, 1 - dot))
    }

    /// Three-stop gradient: short distance (potential collision) →
    /// warm red, mid → orange/yellow, long distance (clean
    /// separation) → green. Matches the mental model "red is bad"
    /// — a red cell off the diagonal means the diarizer might be
    /// merging two real speakers.
    static func heatColor(distance: Float) -> Color {
        let clamped = Double(max(0, min(distance, 1)))
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
