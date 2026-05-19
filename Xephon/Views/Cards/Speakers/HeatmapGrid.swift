import SwiftUI
import XephonUtilities

/// Stateless layout + math helpers backing `SpeakerHeatmapCard`'s
/// pairwise-distance grid. Pulled out of the view so the gradient
/// thresholds, cell-side autoscaling, and cosine-distance
/// arithmetic are testable in isolation.
enum HeatmapGrid {
    /// Preferred side length of each grid cell when there's room.
    /// The grid auto-shrinks each cell below this when the
    /// available width can't fit N at full size, with a hard floor
    /// so cells remain visually distinguishable.
    static let maxCellSize: CGFloat = 22
    /// Floor for the per-cell side length. Below ~10 pt the
    /// numeric label drops out and the cells become pure color
    /// samples; at 6 pt they're still readable as a heatmap on
    /// iPad.
    static let minCellSize: CGFloat = 6
    /// Threshold below which the numeric "0.85" label is dropped
    /// because the glyph won't fit. The grid stays informative as
    /// a pure color matrix.
    static let labelDropCellSize: CGFloat = 14
    /// Gap between cells so each one reads as a separate sample;
    /// without this the grid looks like a continuous heat surface
    /// and the per-pair quantization is lost.
    static let cellSpacing: CGFloat = 2
    /// Width of the leftmost column (speaker-id labels). Wide
    /// enough for "S10".
    static let rowLabelWidth: CGFloat = 28

    /// Per-cell side length that fits N cells + the row-label
    /// gutter inside `available` width. Clamped to `[minCellSize,
    /// maxCellSize]` so small sessions don't waste space and huge
    /// sessions don't disappear into single-pixel cells.
    static func cellSide(available: CGFloat, n: CGFloat) -> CGFloat {
        guard n > 0, available > 0 else { return maxCellSize }
        let usable = available - rowLabelWidth - cellSpacing * (n + 1)
        let raw = usable / n
        return raw.clamped(to: minCellSize...maxCellSize)
    }

    /// Total grid height for `count` speakers at the size the grid
    /// will pick after measuring. Used to give the GeometryReader
    /// a concrete height so it doesn't expand to fill the parent.
    /// Computed assuming the worst case (`maxCellSize`) since
    /// that's always an upper bound — the actual rendered grid
    /// will be at most this tall.
    static func gridHeight(forSpeakerCount count: Int) -> CGFloat {
        // count + 1 row (header + N data rows), spaced.
        let rows = CGFloat(count + 1)
        return rows * maxCellSize + (rows - 1) * cellSpacing
    }

    /// Cosine distance for two L2-normalized vectors (the
    /// FluidAudio extractor's invariant). For unit-norm `a` and
    /// `b`, `cos_sim = a·b` and `cos_dist = 1 − cos_sim`. We clamp
    /// the result into `[0, 1]` even though the theoretical range
    /// is `[0, 2]` — FluidAudio's embeddings cluster tightly
    /// enough that negative similarities don't show up in practice
    /// and the extended range would only waste color budget on
    /// cells we never see.
    static func cosineDistance(_ a: [Float], _ b: [Float]) -> Float {
        let n = min(a.count, b.count)
        guard n > 0 else { return 1 }
        var dot: Float = 0
        for i in 0..<n { dot += a[i] * b[i] }
        return (1 - dot).clamped(to: 0...1)
    }

    /// Three-stop gradient: short distance (potential collision)
    /// → warm red, mid → orange/yellow, long distance (clean
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

    /// One-liner reading of where the distance falls on the heat
    /// scale, mirroring the same thresholds used by `heatColor`.
    /// Helps users back-project the number to "is this concerning?"
    /// without having to remember the gradient stops.
    static func distanceInterpretation(_ d: Float) -> String {
        if d < 0.4 {
            return String(localized: "cluster.heatmap.tip.close")
        } else if d < 0.7 {
            return String(localized: "cluster.heatmap.tip.mid")
        } else {
            return String(localized: "cluster.heatmap.tip.far")
        }
    }
}
