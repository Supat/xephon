import SwiftUI

/// Single-axis time-series projector for the three Canvas-backed
/// drawings in `AccommodationCohesionCard` (accommodation,
/// cohesion, drift). Each one previously hand-rolled the same
/// `(t - sessionStart) / span × width` → x and the value → y
/// arithmetic. Bundles the two so the draw helpers reduce to
/// iteration + path moves.
///
/// Y mapping has two flavors:
///
///  - `yFromBottom(_:maxValue:)` for non-negative magnitudes
///    plotted as a line/area rising from the baseline (used by
///    accommodation distance + cohesion variance).
///  - `yCenteredFromZero(_:maxAbs:)` for signed deviations plotted
///    around a midline at zero (used by the drift lines).
struct TimeSeriesProjector {
    let sessionStart: TimeInterval
    let sessionEnd: TimeInterval
    let canvasSize: CGSize
    /// Vertical padding above + below the data area so the line's
    /// stroke doesn't kiss the canvas edge. Matches the historical
    /// `size.height - 3` / `size.height - 6` constants.
    var verticalInset: CGFloat = 3

    private var span: Double { max(sessionEnd - sessionStart, 1.0) }
    private var availH: CGFloat { canvasSize.height - 2 * verticalInset }

    func x(at t: TimeInterval) -> CGFloat {
        CGFloat((t - sessionStart) / span) * canvasSize.width
    }

    /// Map a non-negative `value` to a y coordinate rising from
    /// the bottom of the canvas. `maxValue` defines the top of the
    /// usable range; callers fall back to a small floor (e.g.
    /// 0.05) so a flat session doesn't divide by zero.
    func yFromBottom(_ value: Double, maxValue: Double) -> CGFloat {
        guard maxValue > 0 else { return canvasSize.height - verticalInset }
        let scale = Double(availH) / maxValue
        return canvasSize.height - verticalInset - CGFloat(value * scale)
    }

    /// Map a signed `value` around the midline at zero. Positive
    /// values rise above; negative values sink below.
    func yCenteredFromZero(_ value: Double, maxAbs: Double) -> CGFloat {
        let midY = canvasSize.height * 0.5
        guard maxAbs > 0 else { return midY }
        return midY - CGFloat(value / maxAbs) * (midY - verticalInset)
    }
}
