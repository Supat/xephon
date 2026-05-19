import SwiftUI

/// Canvas-drawing primitives for `SpeakerClusterCard` that aren't
/// part of the scatter model itself — currently just the directional
/// focus arrow that pins the active observation among the cluster
/// dots. Pulled out so the card body stays focused on layout +
/// SwiftUI plumbing while these Path-arithmetic helpers stay
/// callable from anywhere they're useful.
enum ClusterCanvasDrawing {
    /// Short fixed-length arrow pointing at `target` from the
    /// upper-right (or whichever diagonal stays inside the canvas
    /// at the chosen length). Sized so the tip kisses the dot's
    /// outer edge without crowding it. Always white so the
    /// pointer reads against every speaker tint behind it — using
    /// the speaker's color made the arrow blend into the dot's
    /// own halo, defeating the purpose.
    ///
    /// `dotRadius` is the radius of the dot the arrow points at,
    /// used to stop the tip short by one pixel so it kisses the
    /// edge rather than plunging through.
    static func drawFocusArrow(
        ctx: GraphicsContext,
        in canvas: CGRect,
        to target: CGPoint,
        dotRadius: CGFloat
    ) {
        // 22pt total length, sitting at a 45° angle. Pick the
        // diagonal that fits in the canvas — usually upper-right;
        // mirror to other quadrants when the target hugs an edge
        // so the arrow doesn't draw off-screen.
        let armLength: CGFloat = 16
        let signX: CGFloat = (target.x + armLength + 4 > canvas.maxX) ? -1 : 1
        let signY: CGFloat = (target.y - armLength - 4 < canvas.minY) ? 1 : -1
        let diag = armLength / sqrt(2)
        let origin = CGPoint(
            x: target.x + signX * diag,
            y: target.y + signY * diag
        )
        let dx = target.x - origin.x
        let dy = target.y - origin.y
        let length = sqrt(dx * dx + dy * dy)
        guard length > 4 else { return }
        // Stop short of the dot so the tip kisses the edge rather
        // than plunging through. `dotRadius + 1pt` gap.
        let dotEdgeGap: CGFloat = dotRadius + 1
        let stopDistance = max(0, length - dotEdgeGap)
        let scale = stopDistance / length
        let trimmedEnd = CGPoint(
            x: origin.x + dx * scale,
            y: origin.y + dy * scale
        )
        var shaft = Path()
        shaft.move(to: origin)
        shaft.addLine(to: trimmedEnd)
        ctx.stroke(
            shaft,
            with: .color(.white.opacity(0.95)),
            lineWidth: 1.4
        )
        let headLength: CGFloat = 6
        let headHalfWidth: CGFloat = 3.5
        let ux = dx / length
        let uy = dy / length
        let base = CGPoint(
            x: trimmedEnd.x - ux * headLength,
            y: trimmedEnd.y - uy * headLength
        )
        let left = CGPoint(
            x: base.x + uy * headHalfWidth,
            y: base.y - ux * headHalfWidth
        )
        let right = CGPoint(
            x: base.x - uy * headHalfWidth,
            y: base.y + ux * headHalfWidth
        )
        var head = Path()
        head.move(to: trimmedEnd)
        head.addLine(to: left)
        head.addLine(to: right)
        head.closeSubpath()
        ctx.fill(head, with: .color(.white))
    }
}
