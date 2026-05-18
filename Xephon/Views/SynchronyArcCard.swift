import SwiftUI
import Fusion
import XephonUtilities

/// Time-binned session-arc visualization: per-speaker mean V (or A)
/// drawn as one line per speaker over the session timeline, with
/// the session aggregate drawn heavier on top. When the lines
/// cluster tightly around the aggregate, speakers are emotionally
/// converging — synchrony. When they spread, the speakers are
/// emotionally diverging.
///
/// Complement to `AffectiveSynchronyCard`: that card answers
/// "which pairs sync, structurally?"; this card answers "where in
/// the session does convergence happen, narratively?"
struct SynchronyArcCard: View {
    let utterances: [UtteranceEstimate]

    /// V/A toggle mirroring the sister card so the user reads
    /// "synchrony on V" or "synchrony on A" consistently across
    /// both views.
    @State private var axis: Axis = .valence

    enum Axis: Hashable {
        case valence, arousal
    }

    private static let canvasHeight: CGFloat = 140
    private static let canvasInset: CGFloat = 12
    private static let binCount: Int = 16

    var body: some View {
        let arc = AffectiveSynchrony.sessionArc(
            utterances: utterances,
            binCount: Self.binCount
        )
        VStack(alignment: .leading, spacing: 8) {
            header(speakerCount: arc.speakerIDs.count)
            if arc.bins.isEmpty || arc.speakerIDs.isEmpty {
                Text(String(localized: "synchronyArc.empty"))
                    .font(.caption)
                    .foregroundStyle(.tertiary)
                    .frame(maxWidth: .infinity, alignment: .leading)
            } else {
                Canvas { ctx, size in
                    drawArc(ctx: ctx, size: size, arc: arc)
                }
                .frame(height: Self.canvasHeight)
                .frame(maxWidth: .infinity)
                .background(
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .fill(Color(uiColor: .secondarySystemBackground))
                )
                speakerLegend(arc.speakerIDs)
                Text(String(localized: "synchronyArc.footnote"))
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .glassEffect(in: RoundedRectangle(cornerRadius: 12, style: .continuous))
    }

    @ViewBuilder
    private func header(speakerCount: Int) -> some View {
        HStack(spacing: 6) {
            Text(String(localized: "synchronyArc.header"))
                .font(.caption.bold())
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .minimumScaleFactor(0.8)
            Spacer(minLength: 4)
            if speakerCount > 0 {
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
            }
        }
    }

    private var axisLabel: String {
        switch axis {
        case .valence: return String(localized: "synchrony.metric.valence")
        case .arousal: return String(localized: "synchrony.metric.arousal")
        }
    }

    @ViewBuilder
    private func speakerLegend(_ ids: [String]) -> some View {
        // Pin the session-mean swatch on the left (always
        // visible, anchors the reading) and let the per-speaker
        // chips scroll horizontally when there are more than the
        // card's interior fits. Without this the row blows the
        // iPad portrait pane budget on 3+ speakers.
        HStack(spacing: 8) {
            HStack(spacing: 3) {
                Capsule()
                    .fill(Color(uiColor: .label))
                    .frame(width: 12, height: 3)
                Text(String(localized: "synchronyArc.legend.sessionMean"))
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    ForEach(ids, id: \.self) { id in
                        HStack(spacing: 3) {
                            Capsule()
                                .fill(speakerTint(for: id))
                                .frame(width: 12, height: 2)
                            Text(id)
                                .font(.caption2.monospaced())
                                .foregroundStyle(speakerTint(for: id))
                        }
                    }
                }
                .padding(.horizontal, 2)
            }
        }
    }

    /// Render the arc into the Canvas: a fine line per speaker
    /// (speakerTint at 0.65 alpha) plus the heavy session-mean
    /// line on top, all on a fixed [0, 1] y-axis with a midline at
    /// 0.5. Time runs left to right across the bin range. Gaps
    /// where a speaker had no utterance in a bin show as breaks
    /// in their line.
    private func drawArc(
        ctx: GraphicsContext,
        size: CGSize,
        arc: AffectiveSynchrony.ArcResult
    ) {
        let inset = Self.canvasInset
        let w = size.width - 2 * inset
        let h = size.height - 2 * inset

        // Midline at V=0.5 (or A=0.5) — Russell-VA neutral.
        let midY = inset + h * 0.5
        var midline = Path()
        midline.move(to: CGPoint(x: inset, y: midY))
        midline.addLine(to: CGPoint(x: inset + w, y: midY))
        ctx.stroke(
            midline,
            with: .color(Color.secondary.opacity(0.25)),
            lineWidth: 0.5
        )

        let binCount = arc.bins.count
        guard binCount > 1 else { return }

        func xFor(_ binIndex: Int) -> CGFloat {
            inset + w * CGFloat(binIndex) / CGFloat(binCount - 1)
        }
        func yFor(_ value: Double) -> CGFloat {
            let clamped = value.clamped(to: 0.0...1.0)
            return inset + (1 - CGFloat(clamped)) * h
        }
        func selectPer(_ bin: AffectiveSynchrony.ArcBin, spk: String) -> Double? {
            switch axis {
            case .valence: return bin.perSpeakerValence[spk]
            case .arousal: return bin.perSpeakerArousal[spk]
            }
        }
        func selectSession(_ bin: AffectiveSynchrony.ArcBin) -> Double? {
            switch axis {
            case .valence: return bin.sessionMeanValence
            case .arousal: return bin.sessionMeanArousal
            }
        }

        // Per-speaker lines first so the session aggregate draws
        // on top.
        for spk in arc.speakerIDs {
            var path = Path()
            var hasLastPoint = false
            for (i, bin) in arc.bins.enumerated() {
                guard let v = selectPer(bin, spk: spk) else {
                    hasLastPoint = false
                    continue
                }
                let pt = CGPoint(x: xFor(i), y: yFor(v))
                if hasLastPoint {
                    path.addLine(to: pt)
                } else {
                    path.move(to: pt)
                }
                hasLastPoint = true
            }
            ctx.stroke(
                path,
                with: .color(speakerTint(for: spk).opacity(0.65)),
                lineWidth: 1.0
            )
        }

        // Session aggregate
        var sessionPath = Path()
        var hasLast = false
        for (i, bin) in arc.bins.enumerated() {
            guard let v = selectSession(bin) else {
                hasLast = false
                continue
            }
            let pt = CGPoint(x: xFor(i), y: yFor(v))
            if hasLast {
                sessionPath.addLine(to: pt)
            } else {
                sessionPath.move(to: pt)
            }
            hasLast = true
        }
        ctx.stroke(
            sessionPath,
            with: .color(Color(uiColor: .label)),
            lineWidth: 1.8
        )
    }
}
