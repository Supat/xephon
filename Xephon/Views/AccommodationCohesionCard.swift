import SwiftUI
import Fusion

/// Surfaces backlog items #8–#10 over the session's V/A trajectory.
/// Three Canvas line plots stacked on top of each other, all sharing
/// the same time axis so the eye can correlate phases across them:
///
///   1. **Accommodation** — mean cross-speaker V/A distance per
///      time bin. A falling line means consecutive cross-speaker
///      turns landed closer in V/A space over time (warming up).
///      The session's linear-fit slope shows as a numeric badge.
///   2. **Cohesion** — across-speaker V variance per bin. Low =
///      cohesive; high = fragmenting. Plotted with a soft fill so
///      transient cohesion peaks read as bands rather than spikes.
///   3. **Drift** — per-speaker V deviation from session mean over
///      time. Lines flattening toward 0 = accommodating; lines
///      that grow = anchoring.
///
/// All three series come from a single
/// `AccommodationCohesion.compute(utterances:)` pass; the card just
/// renders them. Sits below `InfluenceContagionCard` on the
/// speaker-analytics page.
struct AccommodationCohesionCard: View {
    let utterances: [UtteranceEstimate]

    private static let plotHeight: CGFloat = 56
    private static let speakerLegendDotSize: CGFloat = 8

    var body: some View {
        let result = AccommodationCohesion.compute(utterances: utterances)
        VStack(alignment: .leading, spacing: 14) {
            header(speakerCount: result.speakerIDs.count)
            if utterances.isEmpty || result.speakerIDs.isEmpty {
                Text(String(localized: "cohesion.empty"))
                    .font(.caption)
                    .foregroundStyle(.tertiary)
                    .frame(maxWidth: .infinity, alignment: .leading)
            } else {
                accommodationSection(
                    points: result.accommodation,
                    slope: result.accommodationSlope,
                    sessionStart: result.sessionStart,
                    sessionEnd: result.sessionEnd
                )
                Divider().opacity(0.4)
                cohesionSection(
                    points: result.cohesion,
                    sessionStart: result.sessionStart,
                    sessionEnd: result.sessionEnd
                )
                Divider().opacity(0.4)
                driftSection(
                    points: result.drift,
                    speakerIDs: result.speakerIDs,
                    sessionStart: result.sessionStart,
                    sessionEnd: result.sessionEnd
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
            Text(String(localized: "cohesion.header"))
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

    // MARK: - #8 Accommodation

    @ViewBuilder
    private func accommodationSection(
        points: [AccommodationCohesion.AccommodationPoint],
        slope: Double?,
        sessionStart: TimeInterval,
        sessionEnd: TimeInterval
    ) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                sectionTitle(String(localized: "cohesion.section.accommodation"))
                Spacer(minLength: 4)
                if let slope {
                    slopeBadge(slope)
                }
            }
            Canvas { ctx, size in
                drawAccommodationLine(
                    in: ctx,
                    size: size,
                    points: points,
                    sessionStart: sessionStart,
                    sessionEnd: sessionEnd
                )
            }
            .frame(height: Self.plotHeight)
            .frame(maxWidth: .infinity)
            Text(String(localized: "cohesion.footnote.accommodation"))
                .font(.caption2)
                .foregroundStyle(.tertiary)
        }
    }

    @ViewBuilder
    private func slopeBadge(_ slope: Double) -> some View {
        let warming = slope < 0
        let label = warming
            ? String(localized: "cohesion.slope.warming")
            : String(localized: "cohesion.slope.splitting")
        let tint: Color = warming ? .green : .orange
        HStack(spacing: 3) {
            Image(systemName: warming
                ? "arrow.down.right"
                : "arrow.up.right")
                .font(.caption2)
            Text(label)
                .font(.caption2.bold())
        }
        .foregroundStyle(tint)
        .padding(.horizontal, 6)
        .padding(.vertical, 1)
        .overlay(Capsule().strokeBorder(tint.opacity(0.45), lineWidth: 0.5))
    }

    private func drawAccommodationLine(
        in ctx: GraphicsContext,
        size: CGSize,
        points: [AccommodationCohesion.AccommodationPoint],
        sessionStart: TimeInterval,
        sessionEnd: TimeInterval
    ) {
        let valid = points.compactMap { p -> (Double, Double)? in
            guard let d = p.meanDistance else { return nil }
            return (p.midTime, d)
        }
        guard valid.count >= 2 else { return }
        let maxDist = valid.map(\.1).max() ?? 1
        let yScale = maxDist > 0 ? Double(size.height - 6) / maxDist : 0
        let span = max(sessionEnd - sessionStart, 1.0)
        var path = Path()
        for (idx, p) in valid.enumerated() {
            let x = CGFloat((p.0 - sessionStart) / span) * size.width
            let y = size.height - 3 - CGFloat(p.1 * yScale)
            let pt = CGPoint(x: x, y: y)
            if idx == 0 { path.move(to: pt) } else { path.addLine(to: pt) }
        }
        ctx.stroke(path, with: .color(.accentColor), lineWidth: 1.5)
    }

    // MARK: - #9 Cohesion

    @ViewBuilder
    private func cohesionSection(
        points: [AccommodationCohesion.CohesionPoint],
        sessionStart: TimeInterval,
        sessionEnd: TimeInterval
    ) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            sectionTitle(String(localized: "cohesion.section.cohesion"))
            Canvas { ctx, size in
                drawCohesionBand(
                    in: ctx,
                    size: size,
                    points: points,
                    sessionStart: sessionStart,
                    sessionEnd: sessionEnd
                )
            }
            .frame(height: Self.plotHeight)
            .frame(maxWidth: .infinity)
            Text(String(localized: "cohesion.footnote.cohesion"))
                .font(.caption2)
                .foregroundStyle(.tertiary)
        }
    }

    private func drawCohesionBand(
        in ctx: GraphicsContext,
        size: CGSize,
        points: [AccommodationCohesion.CohesionPoint],
        sessionStart: TimeInterval,
        sessionEnd: TimeInterval
    ) {
        let valid = points.compactMap { p -> (Double, Double)? in
            guard let v = p.valenceVariance else { return nil }
            return (p.midTime, v)
        }
        guard valid.count >= 2 else { return }
        let maxVar = valid.map(\.1).max() ?? 0
        guard maxVar > 0 else { return }
        let yScale = Double(size.height - 6) / maxVar
        let span = max(sessionEnd - sessionStart, 1.0)
        var line = Path()
        var fill = Path()
        for (idx, p) in valid.enumerated() {
            let x = CGFloat((p.0 - sessionStart) / span) * size.width
            let y = size.height - 3 - CGFloat(p.1 * yScale)
            let pt = CGPoint(x: x, y: y)
            if idx == 0 {
                line.move(to: pt)
                fill.move(to: CGPoint(x: x, y: size.height))
                fill.addLine(to: pt)
            } else {
                line.addLine(to: pt)
                fill.addLine(to: pt)
            }
            if idx == valid.count - 1 {
                fill.addLine(to: CGPoint(x: x, y: size.height))
                fill.closeSubpath()
            }
        }
        ctx.fill(fill, with: .color(.orange.opacity(0.18)))
        ctx.stroke(line, with: .color(.orange), lineWidth: 1.2)
    }

    // MARK: - #10 Drift

    @ViewBuilder
    private func driftSection(
        points: [AccommodationCohesion.DriftPoint],
        speakerIDs: [String],
        sessionStart: TimeInterval,
        sessionEnd: TimeInterval
    ) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            sectionTitle(String(localized: "cohesion.section.drift"))
            Canvas { ctx, size in
                drawDriftLines(
                    in: ctx,
                    size: size,
                    points: points,
                    speakerIDs: speakerIDs,
                    sessionStart: sessionStart,
                    sessionEnd: sessionEnd
                )
            }
            .frame(height: Self.plotHeight)
            .frame(maxWidth: .infinity)
            driftLegend(speakerIDs: speakerIDs)
            Text(String(localized: "cohesion.footnote.drift"))
                .font(.caption2)
                .foregroundStyle(.tertiary)
        }
    }

    @ViewBuilder
    private func driftLegend(speakerIDs: [String]) -> some View {
        HStack(spacing: 8) {
            ForEach(speakerIDs, id: \.self) { spk in
                HStack(spacing: 3) {
                    Circle()
                        .fill(speakerTint(for: spk))
                        .frame(
                            width: Self.speakerLegendDotSize,
                            height: Self.speakerLegendDotSize
                        )
                    Text(spk)
                        .font(.caption2.bold())
                        .foregroundStyle(speakerTint(for: spk))
                }
            }
            Spacer(minLength: 0)
        }
    }

    private func drawDriftLines(
        in ctx: GraphicsContext,
        size: CGSize,
        points: [AccommodationCohesion.DriftPoint],
        speakerIDs: [String],
        sessionStart: TimeInterval,
        sessionEnd: TimeInterval
    ) {
        // Range = max |deviation| across all bins + speakers, with
        // a small floor so a flat session doesn't divide by 0.
        var maxAbs = 0.05
        for p in points {
            for v in p.perSpeakerValenceDeviation.values {
                if abs(v) > maxAbs { maxAbs = abs(v) }
            }
        }
        let span = max(sessionEnd - sessionStart, 1.0)
        let midY = size.height * 0.5
        // Center reference line at zero deviation.
        var zero = Path()
        zero.move(to: CGPoint(x: 0, y: midY))
        zero.addLine(to: CGPoint(x: size.width, y: midY))
        ctx.stroke(
            zero,
            with: .color(.secondary.opacity(0.25)),
            style: StrokeStyle(lineWidth: 0.5, dash: [3, 3])
        )

        for spk in speakerIDs {
            var path = Path()
            var started = false
            for p in points {
                guard let v = p.perSpeakerValenceDeviation[spk] else { continue }
                let x = CGFloat((p.midTime - sessionStart) / span) * size.width
                let y = midY - CGFloat(v / maxAbs) * (midY - 3)
                let pt = CGPoint(x: x, y: y)
                if !started {
                    path.move(to: pt)
                    started = true
                } else {
                    path.addLine(to: pt)
                }
            }
            guard started else { continue }
            ctx.stroke(
                path,
                with: .color(speakerTint(for: spk)),
                lineWidth: 1.2
            )
        }
    }

    // MARK: - Shared

    @ViewBuilder
    private func sectionTitle(_ text: String) -> some View {
        Text(text)
            .font(.caption2.weight(.semibold))
            .foregroundStyle(.secondary)
            .textCase(.uppercase)
    }
}
