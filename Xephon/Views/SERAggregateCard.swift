import SwiftUI
import Fusion
import SERAcoustic
import SERText

/// Session-wide aggregate of the per-modality SER outputs:
///
/// 1. Plutchik wheel — 8 radial wedges, each wedge's outer radius
///    proportional to the mean intensity that text SER assigned to
///    that label across every utterance with a Plutchik vector.
/// 2. Acoustic 9-class histogram — horizontal bars sized by the
///    mean probability emotion2vec assigned to each class.
/// 3. Valence–Arousal scatter — one dot per utterance plotted on
///    the V (x) / A (y) plane, color-coded by speaker so emotion
///    arcs by person read at a glance.
///
/// Sits below `StatisticsCard` on the left pane's affect page.
/// Aggregates over `recorder.utterances` (the whole session, not
/// the filtered view) so the user can read the model's overall
/// bias regardless of which subset is currently on screen.
struct SERAggregateCard: View {
    let recorder: RecordingController
    /// Utterance currently in focus, or nil. Used to enlarge +
    /// halo the matching dot on the V×A scatter and draw an
    /// arrow at it so the user can find that one row's V/A
    /// position among hundreds of dots.
    var focusedUtteranceID: UUID?
    /// Called with the utterance id of the tapped V×A scatter dot.
    /// Each dot already carries its source utterance's id, so the
    /// callback receives an exact match — no embedding-distance
    /// fallback like the speaker cluster card needs. Nil disables
    /// tap-to-scroll entirely.
    var onTapUtterance: ((UUID) -> Void)?

    private static let wheelHeight: CGFloat = 180
    private static let scatterHeight: CGFloat = 160

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text(String(localized: "ser.aggregate.header"))
                    .font(.caption.bold())
                    .foregroundStyle(.secondary)
                Spacer()
                Text("\(recorder.utterances.count)")
                    .font(.caption2.monospacedDigit())
                    .foregroundStyle(.tertiary)
            }
            if recorder.utterances.isEmpty {
                Text(String(localized: "ser.aggregate.empty"))
                    .font(.caption)
                    .foregroundStyle(.tertiary)
                    .frame(maxWidth: .infinity, alignment: .leading)
            } else {
                plutchikSection
                acousticSection
                vaScatterSection
                shapeRationaleFootnote
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .glassEffect(in: RoundedRectangle(cornerRadius: 12, style: .continuous))
    }

    // MARK: - Plutchik wheel

    @ViewBuilder
    private var plutchikSection: some View {
        VStack(alignment: .leading, spacing: 4) {
            sectionHeader(String(localized: "ser.aggregate.plutchik"))
            let means = plutchikMeans
            if means.values.allSatisfy({ $0 == 0 }) {
                Text(String(localized: "ser.aggregate.plutchik.empty"))
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            } else {
                plutchikWheel(means: means)
                    .frame(height: Self.wheelHeight)
                    .frame(maxWidth: .infinity)
            }
        }
    }

    /// Mean intensity per Plutchik label across utterances that
    /// produced a text-SER vector. Labels missing from a row count
    /// as zero contribution (matches the per-utterance schema —
    /// missing-label-means-zero, not "missing"-as-NaN).
    private var plutchikMeans: [PlutchikScore.Label: Float] {
        var sums: [PlutchikScore.Label: Float] = [:]
        var count: Int = 0
        for utt in recorder.utterances {
            guard let probs = utt.plutchik?.probabilities else { continue }
            count += 1
            for label in PlutchikScore.Label.allCases {
                sums[label, default: 0] += probs[label] ?? 0
            }
        }
        guard count > 0 else { return [:] }
        var out: [PlutchikScore.Label: Float] = [:]
        for label in PlutchikScore.Label.allCases {
            out[label] = (sums[label] ?? 0) / Float(count)
        }
        return out
    }

    @ViewBuilder
    private func plutchikWheel(means: [PlutchikScore.Label: Float]) -> some View {
        Canvas { ctx, size in
            let cx = size.width / 2
            let cy = size.height / 2
            let maxR = min(cx, cy) - 28
            let labels = PlutchikScore.Label.allCases
            let step = 2 * .pi / Double(labels.count)
            // Auto-scale so the dominant class fills the outer ring.
            // Linear scaling against the absolute [0, 1] softmax range
            // makes every wedge tiny once the session grows: an
            // 8-class softmax averaged over many utterances is
            // bounded near 1/8 = 0.125, and stronger predictions
            // still rarely push any class mean past ~0.25. The
            // 0.05 floor prevents division blow-up on near-empty
            // sessions (peak ≈ 0 → infinite scale).
            let peak = max(means.values.max() ?? 0, 0.05)
            let scale = 1.0 / peak
            // Reference rings at 50% and 100% of the peak so the
            // user can still read relative intensities between
            // wedges; the absolute peak fraction is reported in
            // the corner so the absolute scale isn't lost.
            for fraction in [0.5, 1.0] {
                let r = maxR * CGFloat(fraction)
                let rect = CGRect(
                    x: cx - r, y: cy - r,
                    width: 2 * r, height: 2 * r
                )
                ctx.stroke(
                    Path(ellipseIn: rect),
                    with: .color(.secondary.opacity(0.18)),
                    lineWidth: 0.5
                )
            }
            // Wedges + labels.
            for (i, label) in labels.enumerated() {
                let mean = means[label] ?? 0
                let normalized = max(0, min(1, mean * scale))
                let r = maxR * CGFloat(normalized)
                // Anchor wedge 0 at the top, going clockwise.
                let startAngle = Double(i) * step - .pi / 2
                let endAngle = startAngle + step
                var path = Path()
                path.move(to: CGPoint(x: cx, y: cy))
                path.addArc(
                    center: CGPoint(x: cx, y: cy),
                    radius: r,
                    startAngle: Angle(radians: startAngle),
                    endAngle: Angle(radians: endAngle),
                    clockwise: false
                )
                path.closeSubpath()
                ctx.fill(
                    path,
                    with: .color(emotionTint(for: label.rawValue).opacity(0.7))
                )
                // Label at outer ring midpoint.
                let midAngle = (startAngle + endAngle) / 2
                let lx = cx + (maxR + 14) * CGFloat(cos(midAngle))
                let ly = cy + (maxR + 14) * CGFloat(sin(midAngle))
                let text = Text(label.rawValue.prefix(4).capitalized)
                    .font(.caption2.weight(.medium))
                    .foregroundStyle(emotionTint(for: label.rawValue))
                ctx.draw(text, at: CGPoint(x: lx, y: ly), anchor: .center)
            }
            // Anchor the absolute peak in the bottom-right so the
            // auto-scale can be back-projected to true probabilities
            // by eye.
            let peakPercent = Int((peak * 100).rounded())
            let peakText = Text("peak \(peakPercent)%")
                .font(.caption2)
                .foregroundStyle(.secondary)
            ctx.draw(
                peakText,
                at: CGPoint(x: size.width - 4, y: size.height - 4),
                anchor: .bottomTrailing
            )
        }
    }

    // MARK: - Acoustic histogram

    @ViewBuilder
    private var acousticSection: some View {
        VStack(alignment: .leading, spacing: 4) {
            sectionHeader(String(localized: "ser.aggregate.acoustic"))
            let means = acousticMeans
            if means.values.allSatisfy({ $0 == 0 }) {
                Text(String(localized: "ser.aggregate.acoustic.empty"))
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            } else {
                VStack(alignment: .leading, spacing: 2) {
                    ForEach(
                        CategoricalEmotion.Label.allCases,
                        id: \.self
                    ) { label in
                        acousticBar(label: label, value: means[label] ?? 0)
                    }
                }
            }
        }
    }

    /// Mean per-class probability across utterances that produced
    /// an acoustic-categorical softmax. Same missing-as-zero rule
    /// as `plutchikMeans`.
    private var acousticMeans: [CategoricalEmotion.Label: Float] {
        var sums: [CategoricalEmotion.Label: Float] = [:]
        var count: Int = 0
        for utt in recorder.utterances {
            guard let probs = utt.acousticCategorical?.probabilities else { continue }
            count += 1
            for label in CategoricalEmotion.Label.allCases {
                sums[label, default: 0] += probs[label] ?? 0
            }
        }
        guard count > 0 else { return [:] }
        var out: [CategoricalEmotion.Label: Float] = [:]
        for label in CategoricalEmotion.Label.allCases {
            out[label] = (sums[label] ?? 0) / Float(count)
        }
        return out
    }

    @ViewBuilder
    private func acousticBar(
        label: CategoricalEmotion.Label,
        value: Float
    ) -> some View {
        let tint = emotionTint(for: label.rawValue)
        HStack(spacing: 8) {
            Text(label.rawValue.capitalized)
                .font(.caption2.monospaced())
                .foregroundStyle(tint)
                .lineLimit(1)
                .truncationMode(.tail)
                .frame(width: 70, alignment: .leading)
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    Capsule().fill(tint.opacity(0.12))
                    Capsule()
                        .fill(tint.opacity(0.55))
                        .frame(width: geo.size.width * CGFloat(max(0, min(1, value))))
                }
            }
            .frame(height: 5)
            Text(String(format: "%.2f", value))
                .font(.caption2.monospacedDigit())
                .foregroundStyle(.secondary)
                .frame(width: 36, alignment: .trailing)
        }
    }

    // MARK: - V/A scatter

    @ViewBuilder
    private var vaScatterSection: some View {
        VStack(alignment: .leading, spacing: 4) {
            sectionHeader(String(localized: "ser.aggregate.va"))
            let points = vaPoints
            if points.isEmpty {
                Text(String(localized: "ser.aggregate.va.empty"))
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            } else {
                // GeometryReader supplies the same canvas size the
                // Canvas draws into, so `handleScatterTap` can
                // re-run the projection math identically and
                // resolve a tap to its nearest dot. The Canvas
                // already fills the bounded frame; wrapping in
                // GeometryReader is layout-neutral.
                GeometryReader { proxy in
                    vaScatter(points: points)
                        .contentShape(Rectangle())
                        .onTapGesture(coordinateSpace: .local) { location in
                            handleScatterTap(
                                at: location,
                                in: proxy.size,
                                points: points
                            )
                        }
                }
                .frame(height: Self.scatterHeight)
                .frame(maxWidth: .infinity)
                .background(
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .fill(Color(uiColor: .secondarySystemBackground))
                )
            }
        }
    }

    /// Find the V×A scatter dot closest to `location` within a
    /// fat-finger hit radius and forward its utterance id. Uses
    /// the same `inset`-based projection the draw path uses so
    /// the visible dots and the hit zones agree pixel-for-pixel.
    /// Misses in empty space silently do nothing.
    private func handleScatterTap(
        at location: CGPoint,
        in size: CGSize,
        points: [VAPoint]
    ) {
        guard let onTap = onTapUtterance, !points.isEmpty else { return }
        let inset: CGFloat = 14
        let w = size.width - 2 * inset
        let h = size.height - 2 * inset
        let hitRadius: CGFloat = 18
        let limit = hitRadius * hitRadius
        var bestID: UUID?
        var bestDist = limit
        for p in points {
            let cx = inset + CGFloat(max(0, min(1, p.valence))) * w
            let cy = inset + (1 - CGFloat(max(0, min(1, p.arousal)))) * h
            let dx = cx - location.x
            let dy = cy - location.y
            let d2 = dx * dx + dy * dy
            if d2 < bestDist {
                bestDist = d2
                bestID = p.id
            }
        }
        if let id = bestID { onTap(id) }
    }

    private struct VAPoint {
        let id: UUID
        let valence: Float
        let arousal: Float
        let speakerID: String
    }

    /// One dot per utterance with both fused V and fused A present.
    /// Rows missing either coordinate are dropped — there's no
    /// meaningful position to plot otherwise. The V/A space is
    /// canonically 0..1 in this codebase; we don't clamp here so
    /// out-of-range values would render at the canvas edges and
    /// flag the anomaly visually.
    private var vaPoints: [VAPoint] {
        recorder.utterances.compactMap { utt in
            guard let v = utt.fusedValence,
                  let a = utt.fusedArousal else { return nil }
            return VAPoint(
                id: utt.id,
                valence: v,
                arousal: a,
                speakerID: utt.speakerID
            )
        }
    }

    @ViewBuilder
    private func vaScatter(points: [VAPoint]) -> some View {
        Canvas { ctx, size in
            let inset: CGFloat = 14
            let w = size.width - 2 * inset
            let h = size.height - 2 * inset
            // Axis crosshairs at v=0.5, a=0.5 so the user can read
            // each dot's quadrant against the conventional Russell
            // valence/arousal axes (right = positive valence, up =
            // high arousal).
            let midX = inset + w * 0.5
            let midY = inset + h * 0.5
            var axes = Path()
            axes.move(to: CGPoint(x: inset, y: midY))
            axes.addLine(to: CGPoint(x: inset + w, y: midY))
            axes.move(to: CGPoint(x: midX, y: inset))
            axes.addLine(to: CGPoint(x: midX, y: inset + h))
            ctx.stroke(
                axes,
                with: .color(.secondary.opacity(0.25)),
                lineWidth: 0.5
            )
            // Dots. The focused row's dot renders bigger + ringed
            // so the user can spot it among hundreds in a long
            // session without an explicit arrow — the halo is
            // direction enough at this density.
            for p in points {
                let cx = inset + CGFloat(max(0, min(1, p.valence))) * w
                let cy = inset + (1 - CGFloat(max(0, min(1, p.arousal)))) * h
                let tint = speakerTint(for: p.speakerID)
                let isFocused = (p.id == focusedUtteranceID)
                let r: CGFloat = isFocused ? 5.5 : 3.5
                let rect = CGRect(
                    x: cx - r, y: cy - r,
                    width: 2 * r, height: 2 * r
                )
                if isFocused {
                    let haloR: CGFloat = 11
                    let haloRect = CGRect(
                        x: cx - haloR, y: cy - haloR,
                        width: 2 * haloR, height: 2 * haloR
                    )
                    ctx.fill(
                        Path(ellipseIn: haloRect),
                        with: .color(tint.opacity(0.25))
                    )
                    ctx.stroke(
                        Path(ellipseIn: haloRect),
                        with: .color(tint),
                        lineWidth: 1.6
                    )
                }
                ctx.fill(
                    Path(ellipseIn: rect),
                    with: .color(tint.opacity(isFocused ? 1.0 : 0.85))
                )
                if isFocused {
                    ctx.stroke(
                        Path(ellipseIn: rect),
                        with: .color(.white.opacity(0.95)),
                        lineWidth: 1
                    )
                }
            }
            // Axis labels at the corners — V↗ (right), A↑ (top).
            ctx.draw(
                Text("V→").font(.caption2).foregroundStyle(.tertiary),
                at: CGPoint(x: inset + w, y: midY + 8),
                anchor: .trailing
            )
            ctx.draw(
                Text("A↑").font(.caption2).foregroundStyle(.tertiary),
                at: CGPoint(x: midX + 8, y: inset),
                anchor: .leading
            )
        }
    }

    private func sectionHeader(_ text: String) -> some View {
        Text(text)
            .font(.caption2.weight(.semibold))
            .foregroundStyle(.tertiary)
            .textCase(.uppercase)
    }

    /// One-paragraph note explaining why the text and acoustic SER
    /// panels are drawn in different shapes (wheel vs bars). Lives
    /// at the bottom of the card so first-time readers don't wonder
    /// if the inconsistency is a bug. A divider above sets it off
    /// from the data visualizations.
    @ViewBuilder
    private var shapeRationaleFootnote: some View {
        VStack(alignment: .leading, spacing: 4) {
            Divider()
            Text(String(localized: "ser.aggregate.shape.footnote"))
                .font(.caption2)
                .foregroundStyle(.tertiary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}
