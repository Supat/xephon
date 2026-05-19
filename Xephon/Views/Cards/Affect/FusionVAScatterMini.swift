import SwiftUI
import XephonUtilities

/// Per-utterance mini-scatter showing how late fusion pulled the
/// V/A point into its final position. Three dots — acoustic (blue),
/// text (orange), fused (green) — with arrows from each source
/// toward the fused point so the geometry of the weighted average
/// reads at a glance.
///
/// Lives inside the row inspector next to the top-fused-labels
/// block, so a user can see in one place: (a) the final label and
/// how confident the pick was, and (b) which modality pulled the
/// V/A point. Together those are usually enough to explain a
/// surprising fused result without diving into the per-modality
/// probability bars below.
///
/// Coordinate space: V on the x-axis, A on the y-axis, both in
/// `[0, 1]`. Neutral (0.5, 0.5) sits at the canvas centre. Higher
/// arousal renders upward; higher valence renders rightward.
struct FusionVAScatterMini: View {
    let acoustic: (v: Float, a: Float)?
    let text: (v: Float, a: Float)?
    let fused: (v: Float, a: Float)?

    /// Endpoint tints — pulled from the same constants the
    /// FusionContributionStrip uses so this view can't drift out
    /// of sync with the strip's color story.
    private static let acousticTint = Color(
        red: FusionContributionStrip.acousticRGB.r,
        green: FusionContributionStrip.acousticRGB.g,
        blue: FusionContributionStrip.acousticRGB.b
    )
    private static let textTint = Color(
        red: FusionContributionStrip.textRGB.r,
        green: FusionContributionStrip.textRGB.g,
        blue: FusionContributionStrip.textRGB.b
    )
    private static let fusedTint: Color = .green
    /// Margin reserved around the data box for axis labels.
    /// Sized to fit "V" / "A" letter glyphs at caption2 plus the
    /// `+` / `−` polarity markers at each corner.
    private static let inset: CGFloat = 14
    private static let dotRadius: CGFloat = 4

    var body: some View {
        Canvas { ctx, size in
            let usable = CGRect(
                x: Self.inset,
                y: Self.inset,
                width: size.width - 2 * Self.inset,
                height: size.height - 2 * Self.inset
            )
            drawAxes(ctx: ctx, in: usable)
            drawAxisLabels(ctx: ctx, canvasSize: size, dataRect: usable)
            // Arrows from each source to fused, drawn first so dots
            // overlay them. Skipped when either endpoint is nil —
            // no meaningful "pull" line to render.
            if let fused = fused {
                let fusedPt = project(fused, in: usable)
                if let a = acoustic {
                    drawArrow(
                        ctx: ctx,
                        from: project(a, in: usable),
                        to: fusedPt,
                        tint: Self.acousticTint
                    )
                }
                if let t = text {
                    drawArrow(
                        ctx: ctx,
                        from: project(t, in: usable),
                        to: fusedPt,
                        tint: Self.textTint
                    )
                }
            }
            if let a = acoustic {
                drawDot(
                    ctx: ctx,
                    at: project(a, in: usable),
                    tint: Self.acousticTint
                )
            }
            if let t = text {
                drawDot(
                    ctx: ctx,
                    at: project(t, in: usable),
                    tint: Self.textTint
                )
            }
            if let fused = fused {
                drawDot(
                    ctx: ctx,
                    at: project(fused, in: usable),
                    tint: Self.fusedTint,
                    larger: true
                )
            }
        }
    }

    private func project(_ point: (v: Float, a: Float), in rect: CGRect) -> CGPoint {
        let x = rect.minX + CGFloat(point.v.clamped(to: 0...1)) * rect.width
        // Higher arousal = lower y in screen space.
        let y = rect.minY + (1 - CGFloat(point.a.clamped(to: 0...1))) * rect.height
        return CGPoint(x: x, y: y)
    }

    /// Axis labels around the data box.
    /// - "V" centered below the box (with `−` / `+` polarity at the
    ///   bottom corners — left = low valence, right = high).
    /// - "A" centered to the left of the box (with `−` / `+`
    ///   polarity at the left corners — top = high arousal, bottom
    ///   = low).
    /// Polarity markers carry more diagnostic weight than tick
    /// numbers here because the user reads the geometry, not the
    /// exact V/A coordinates — those are already in the meta line.
    private func drawAxisLabels(
        ctx: GraphicsContext,
        canvasSize: CGSize,
        dataRect: CGRect
    ) {
        let axisFont = Font.system(size: 8, weight: .semibold)
        let polarityFont = Font.system(size: 7, weight: .medium)
        // `.tertiary` only exists as a hierarchical ShapeStyle, not
        // a Color — fall back to a system tertiary-label Color so
        // the markers stay quiet without leaning on opacity tricks.
        let axisColor: Color = .secondary
        let polarityColor: Color = Color(uiColor: .tertiaryLabel)
        // Valence axis label below the box, centered.
        ctx.draw(
            Text("V").font(axisFont).foregroundStyle(axisColor),
            at: CGPoint(x: dataRect.midX, y: canvasSize.height - 2),
            anchor: .bottom
        )
        // Arousal axis label to the left, centered.
        ctx.draw(
            Text("A").font(axisFont).foregroundStyle(axisColor),
            at: CGPoint(x: 2, y: dataRect.midY),
            anchor: .leading
        )
        // Polarity markers in the four corners of the data box —
        // pulled slightly outside the box so they don't compete
        // with dots that land near a corner.
        let pad: CGFloat = 1
        ctx.draw(
            Text("−").font(polarityFont).foregroundStyle(polarityColor),
            at: CGPoint(x: dataRect.minX, y: dataRect.maxY + pad),
            anchor: .top
        )
        ctx.draw(
            Text("+").font(polarityFont).foregroundStyle(polarityColor),
            at: CGPoint(x: dataRect.maxX, y: dataRect.maxY + pad),
            anchor: .top
        )
        ctx.draw(
            Text("+").font(polarityFont).foregroundStyle(polarityColor),
            at: CGPoint(x: dataRect.minX - pad, y: dataRect.minY),
            anchor: .trailing
        )
        ctx.draw(
            Text("−").font(polarityFont).foregroundStyle(polarityColor),
            at: CGPoint(x: dataRect.minX - pad, y: dataRect.maxY),
            anchor: .trailing
        )
    }

    private func drawAxes(ctx: GraphicsContext, in rect: CGRect) {
        // Box frame so the scatter reads as a bounded canvas, not a
        // floating constellation. Tertiary opacity to stay quiet
        // against the row inspector's background.
        let border = Path(rect)
        ctx.stroke(border, with: .color(.secondary.opacity(0.18)), lineWidth: 0.5)
        // Cross-hair through neutral (V=0.5, A=0.5) — the reference
        // origin researchers expect when reading V/A planes.
        var v = Path()
        v.move(to: CGPoint(x: rect.midX, y: rect.minY))
        v.addLine(to: CGPoint(x: rect.midX, y: rect.maxY))
        var h = Path()
        h.move(to: CGPoint(x: rect.minX, y: rect.midY))
        h.addLine(to: CGPoint(x: rect.maxX, y: rect.midY))
        ctx.stroke(v, with: .color(.secondary.opacity(0.12)), lineWidth: 0.5)
        ctx.stroke(h, with: .color(.secondary.opacity(0.12)), lineWidth: 0.5)
    }

    private func drawArrow(
        ctx: GraphicsContext,
        from start: CGPoint,
        to end: CGPoint,
        tint: Color
    ) {
        // Skip degenerate arrows so the arrowhead math doesn't NaN
        // on coincident points (acoustic === fused when text is
        // missing, etc.).
        let dx = end.x - start.x
        let dy = end.y - start.y
        let length = sqrt(dx * dx + dy * dy)
        guard length > 1 else { return }
        // Stop the line just shy of the fused dot so the arrowhead
        // tip sits at the dot's edge instead of inside it.
        let stopDistance = max(0, length - Self.dotRadius - 1)
        let scale = stopDistance / length
        let trimmedEnd = CGPoint(
            x: start.x + dx * scale,
            y: start.y + dy * scale
        )
        var shaft = Path()
        shaft.move(to: start)
        shaft.addLine(to: trimmedEnd)
        ctx.stroke(
            shaft,
            with: .color(tint.opacity(0.6)),
            lineWidth: 1.2
        )
        // Arrowhead: a small filled triangle perpendicular to the
        // shaft. Sized small enough to read on a 100pt canvas.
        let headLength: CGFloat = 4
        let headHalfWidth: CGFloat = 2.5
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
        ctx.fill(head, with: .color(tint.opacity(0.8)))
    }

    private func drawDot(
        ctx: GraphicsContext,
        at point: CGPoint,
        tint: Color,
        larger: Bool = false
    ) {
        let r = larger ? Self.dotRadius + 1 : Self.dotRadius
        let rect = CGRect(
            x: point.x - r, y: point.y - r,
            width: 2 * r, height: 2 * r
        )
        ctx.fill(Path(ellipseIn: rect), with: .color(tint))
        ctx.stroke(
            Path(ellipseIn: rect),
            with: .color(.white.opacity(0.9)),
            lineWidth: 0.8
        )
    }
}
