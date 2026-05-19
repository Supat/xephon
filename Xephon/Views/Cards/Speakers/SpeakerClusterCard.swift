import SwiftUI
import Diarization

/// 2D PCA scatter of the diarizer's speaker cluster — every
/// speaker's averaged centroid plus its retained raw observations,
/// projected onto the top two principal components of the pooled
/// embedding set. Sits below `SpeakerHeatmapCard` as the second
/// diagnostic view of the diarizer's internal state.
///
/// Centroids render as larger filled dots in each speaker's color;
/// observations render smaller and lower-opacity so a cloud around
/// its centroid reads as "this speaker's spread". When two clouds
/// overlap, the diarizer is having a hard time keeping the
/// speakers apart — paired with the heatmap, the user can see both
/// the pairwise number and the visual reason for it.
///
/// PCA fitting + halo / arrow / hit-test all live on
/// `SpeakerScatterModel`; this view is the rendering layer plus
/// the "Linked only" header toggle.
struct SpeakerClusterCard: View {
    let cluster: SpeakerClusterSnapshot
    /// Speaker id of the currently-focused utterance, or nil if no
    /// row is focused. Drives the highlight ring drawn around the
    /// matching centroid dot so the user can trace a transcript row
    /// to its diarizer-side cluster at a glance. Lives outside the
    /// PCA recompute trigger (`SpeakerScatterModel.snapshotKey`)
    /// so a focus change repaints without re-fitting the basis.
    var highlightedSpeakerID: String?
    /// Raw speaker embedding of the focused utterance. When set
    /// (and the matching observation is still in the snapshot's
    /// tail window), the halo + arrow shift from the speaker's
    /// centroid to the specific observation that came from this
    /// utterance — so the user sees "this row's node" rather than
    /// "this row's speaker's average". Nil falls back to the
    /// centroid behavior.
    var focusedEmbedding: [Float]?
    /// Called with the tapped node's identity when the user taps
    /// inside the scatter. The triple carries:
    ///
    ///  - `speakerID` so the callback can constrain its
    ///    embedding-to-utterance fallback search to that speaker
    ///    (overlapping clouds otherwise pull a tap on A's node
    ///    into one of B's utterances).
    ///  - `observationSegmentID`: the FluidAudio
    ///    `RawEmbedding.segmentId` of the tapped observation, or
    ///    nil if the user tapped a centroid (centroids aren't
    ///    individual observations). When present, the callback
    ///    can resolve observation → utterance by exact id lookup
    ///    against `RecordingController.utteranceObservationSegmentIDs`
    ///    instead of running an L2 argmin at tap time — robust
    ///    against later trimming and reorderings.
    ///  - `embedding`: the raw 256-D vector at the tapped point,
    ///    used as the fallback signal when the id lookup misses
    ///    (older session, observation tail-trimmed past the
    ///    captured row, or centroid taps).
    ///
    /// Nil disables tap-to-scroll entirely.
    var onTapNode: ((
        _ speakerID: String,
        _ observationSegmentID: UUID?,
        _ embedding: [Float]
    ) -> Void)?
    /// Set of diarizer observation `segmentId`s that have a
    /// utterance pinned to them — the values of the controller's
    /// `utteranceObservationSegmentIDs` map. Drives the
    /// observation-dot side of the "Linked only" toggle: when
    /// enabled, observation dots whose id isn't in this set are
    /// hidden. Nil leaves observations unfiltered.
    var linkedObservationIDs: Set<UUID>?
    /// Set of speaker ids referenced by at least one utterance —
    /// same source the roster + heatmap cards use. Drives the
    /// centroid side of the "Linked only" toggle: when enabled,
    /// a speaker's centroid dot is hidden if no utterance maps to
    /// that speaker (diarizer-DB orphans, promoted-but-unclaimed
    /// entries). Nil leaves centroids unfiltered.
    var linkedSpeakerIDs: Set<String>?

    @State private var model = SpeakerScatterModel()
    /// Header toggle: hide observation dots whose `segmentId` isn't
    /// in `linkedObservationIDs`. Kept here so flipping the toggle
    /// doesn't trigger a PCA refit — the basis stays anchored to
    /// the full point set, only the visibility filter changes,
    /// so the visible dots don't drift around when the toggle is
    /// flipped.
    @State private var hideUnlinkedObservations: Bool = false

    nonisolated private static let canvasHeight: CGFloat = 180
    nonisolated private static let centroidRadius: CGFloat = 5
    nonisolated private static let observationRadius: CGFloat = 2.5
    nonisolated private static let observationOpacity: Double = 0.45
    /// Outer radius of the focus-highlight halo drawn around the
    /// centroid that matches the currently-selected utterance's
    /// speaker. Sized about 2.4× the centroid radius so the ring
    /// reads as a clear emphasis without crowding adjacent nodes.
    nonisolated private static let highlightHaloRadius: CGFloat = 12
    /// Touch slop around each projected dot. 20pt is roughly the
    /// minimum tappable target Apple recommends and is generous
    /// against the 2.5pt observation radius without overlapping
    /// adjacent clouds in dense scatters.
    nonisolated private static let tapHitRadius: CGFloat = 20
    /// Padding inside the canvas so the dots don't kiss the edges.
    nonisolated private static let canvasInset: CGFloat = 10

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            header
            if cluster.speakers.isEmpty {
                Text(String(localized: "cluster.empty"))
                    .font(.caption)
                    .foregroundStyle(.tertiary)
                    .frame(maxWidth: .infinity, alignment: .leading)
            } else {
                canvasView
                    .frame(height: Self.canvasHeight)
                    .frame(maxWidth: .infinity)
                    .background(
                        RoundedRectangle(cornerRadius: 8, style: .continuous)
                            .fill(Color(uiColor: .secondarySystemBackground))
                    )
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .glassEffect(in: RoundedRectangle(cornerRadius: 12, style: .continuous))
        .task(id: SpeakerScatterModel.snapshotKey(for: cluster)) {
            await model.recompute(snapshot: cluster)
        }
    }

    @ViewBuilder
    private var header: some View {
        HStack(spacing: 6) {
            Text(String(localized: "cluster.scatter.header"))
                .font(.caption.bold())
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .minimumScaleFactor(0.8)
            Spacer(minLength: 4)
            // Only show the toggle when there's something to
            // filter — either at least one pinned observation
            // (observation-dot filtering) or at least one centroid
            // whose speaker isn't referenced by any utterance
            // (centroid filtering). Legacy sessions with neither
            // signal get a hidden toggle. Rendered as a plain
            // caption2 button instead of a native Toggle: the
            // card's header is otherwise all-text at caption2/
            // tertiary, and a system switch (even at .mini) blew
            // out the visual weight relative to the "PCA 2D" tag
            // beside it. On-state is signalled by the filled link
            // icon and accent tint; off-state stays secondary.
            if canFilterAnything {
                Button {
                    hideUnlinkedObservations.toggle()
                } label: {
                    Label(
                        String(localized: "cluster.scatter.linkedOnly"),
                        systemImage: hideUnlinkedObservations
                            ? "link.circle.fill"
                            : "link.circle"
                    )
                    .font(.caption2)
                    .labelStyle(.titleAndIcon)
                    .lineLimit(1)
                }
                .buttonStyle(.plain)
                .foregroundStyle(
                    hideUnlinkedObservations
                        ? AnyShapeStyle(Color.accentColor)
                        : AnyShapeStyle(HierarchicalShapeStyle.secondary)
                )
            }
            if !cluster.speakers.isEmpty {
                Text("PCA 2D")
                    .font(.caption2.monospaced())
                    .foregroundStyle(.tertiary)
            }
        }
    }

    @ViewBuilder
    private var canvasView: some View {
        if model.points.isEmpty {
            Text(String(localized: "cluster.warmingUp"))
                .font(.caption)
                .foregroundStyle(.tertiary)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        } else {
            // GeometryReader supplies the canvas size to the tap
            // handler. Without it the hit-test math has nothing to
            // project against — Canvas only exposes `size` inside
            // its draw closure, not to outside gestures.
            GeometryReader { proxy in
                clusterCanvas
                    .contentShape(Rectangle())
                    .onTapGesture(coordinateSpace: .local) { location in
                        handleTap(at: location, in: proxy.size)
                    }
            }
        }
    }

    @ViewBuilder
    private var clusterCanvas: some View {
        Canvas { ctx, size in
            guard let project = SpeakerScatterProjector(
                points: model.points, in: size, inset: Self.canvasInset
            ) else { return }
            drawOriginLines(ctx: ctx, project: project, size: size)
            drawAxisLabels(ctx: ctx, size: size)
            // Observations first so centroids stack on top.
            // `isPointVisible` honors the "Linked only" toggle —
            // when on, an observation only renders if its
            // `segmentId` is in the controller-supplied set, and
            // a centroid only renders if its speaker is in the
            // linked-speakers set. Filtering applies to draw,
            // halo, arrow, and tap pickling so all four agree.
            for p in model.points
                where !p.isCentroid && isPointVisible(p) {
                let r = Self.observationRadius
                let center = project(p)
                let rect = CGRect(
                    x: center.x - r, y: center.y - r,
                    width: 2 * r, height: 2 * r
                )
                ctx.fill(
                    Path(ellipseIn: rect),
                    with: .color(p.color.opacity(Self.observationOpacity))
                )
            }
            // Two distinct focus targets: the halo stays on the
            // speaker's centroid (so the speaker context reads at
            // a glance), while the arrow tips at the specific
            // observation that came from the focused utterance
            // (so the user sees "this row's node"). Halo skips
            // when the speaker has no centroid in this snapshot
            // (orphan / DB-only); arrow skips when we couldn't
            // pin a specific observation — older utterance whose
            // embedding wasn't captured OR snapshot tail-trimmed
            // past this utterance.
            let halo = model.haloTarget(
                project: project,
                focusedSpeakerID: highlightedSpeakerID,
                visibility: isPointVisible
            )
            let arrow = model.arrowTarget(
                in: cluster,
                project: project,
                focusedSpeakerID: highlightedSpeakerID,
                focusedEmbedding: focusedEmbedding,
                visibility: isPointVisible
            )
            for p in model.points where p.isCentroid && isPointVisible(p) {
                let r = Self.centroidRadius
                let center = project(p)
                let rect = CGRect(
                    x: center.x - r, y: center.y - r,
                    width: 2 * r, height: 2 * r
                )
                ctx.fill(Path(ellipseIn: rect), with: .color(p.color))
                ctx.stroke(
                    Path(ellipseIn: rect),
                    with: .color(.white.opacity(0.9)),
                    lineWidth: 1
                )
            }
            if let target = halo {
                let haloR = Self.highlightHaloRadius
                let haloRect = CGRect(
                    x: target.position.x - haloR,
                    y: target.position.y - haloR,
                    width: 2 * haloR, height: 2 * haloR
                )
                ctx.fill(
                    Path(ellipseIn: haloRect),
                    with: .color(target.color.opacity(0.30))
                )
                ctx.stroke(
                    Path(ellipseIn: haloRect),
                    with: .color(target.color),
                    lineWidth: 2
                )
            }
            // Directional pointer at the focused observation (not
            // the centroid — that's the halo's job). Drawn last
            // so it sits above every other dot.
            if let target = arrow {
                ClusterCanvasDrawing.drawFocusArrow(
                    ctx: ctx,
                    in: CGRect(origin: .zero, size: size),
                    to: target.position,
                    dotRadius: Self.centroidRadius
                )
            }
        }
    }

    /// PCA mean-centers the data so projected (0, 0) is the data
    /// centroid. Drawing axis lines through it gives the user a
    /// reference for "which side of the mean does this cluster
    /// sit on?" along each principal component. Skipped per-axis
    /// when the origin falls outside the visible data extent
    /// (e.g. all observations clustered on positive PC1) —
    /// drawing the line would clip off-canvas.
    private func drawOriginLines(
        ctx: GraphicsContext,
        project: SpeakerScatterProjector,
        size: CGSize
    ) {
        let originXFrac = (0 - project.minX) / project.dx
        let originYFrac = (0 - project.minY) / project.dy
        let originStyle: GraphicsContext.Shading = .color(.secondary.opacity(0.35))
        let dataRect = CGRect(
            x: Self.canvasInset,
            y: Self.canvasInset,
            width: project.availW,
            height: project.availH
        )
        if originXFrac >= 0 && originXFrac <= 1 {
            let x = Self.canvasInset + CGFloat(originXFrac) * project.availW
            var line = Path()
            line.move(to: CGPoint(x: x, y: dataRect.minY))
            line.addLine(to: CGPoint(x: x, y: dataRect.maxY))
            ctx.stroke(line, with: originStyle, lineWidth: 0.5)
        }
        if originYFrac >= 0 && originYFrac <= 1 {
            let y = Self.canvasInset + (1 - CGFloat(originYFrac)) * project.availH
            var line = Path()
            line.move(to: CGPoint(x: dataRect.minX, y: y))
            line.addLine(to: CGPoint(x: dataRect.maxX, y: y))
            ctx.stroke(line, with: originStyle, lineWidth: 0.5)
        }
    }

    /// Axis labels live just inside the data rect at diagonally-
    /// opposite corners — the canvas only reserves a 10pt inset
    /// around the data area, which isn't wide enough to fit the
    /// labels in the margin without clipping (PC2↑ pushed off
    /// the left edge previously). Caption2 / tertiary keeps them
    /// quiet behind the dots.
    private func drawAxisLabels(ctx: GraphicsContext, size: CGSize) {
        let dataRect = CGRect(
            x: Self.canvasInset,
            y: Self.canvasInset,
            width: size.width - 2 * Self.canvasInset,
            height: size.height - 2 * Self.canvasInset
        )
        ctx.draw(
            Text("PC1→").font(.caption2).foregroundStyle(.tertiary),
            at: CGPoint(x: dataRect.maxX - 2, y: dataRect.maxY - 2),
            anchor: .bottomTrailing
        )
        ctx.draw(
            Text("PC2↑").font(.caption2).foregroundStyle(.tertiary),
            at: CGPoint(x: dataRect.minX + 2, y: dataRect.minY + 2),
            anchor: .topLeading
        )
    }

    // MARK: - "Linked only" toggle

    /// Would the "Linked only" filter actually hide anything under
    /// the current snapshot? True when either an observation in
    /// the cluster has a `segmentId` not in `linkedObservationIDs`,
    /// or a speaker in the cluster isn't in `linkedSpeakerIDs`.
    /// Gates the header toggle so users don't see a control that's
    /// guaranteed to do nothing.
    private var canFilterAnything: Bool {
        if let speakers = linkedSpeakerIDs,
           cluster.speakers.contains(where: { !speakers.contains($0.id) }) {
            return true
        }
        if let observations = linkedObservationIDs,
           cluster.speakers.contains(where: { spk in
               spk.observationSegmentIDs.contains { !observations.contains($0) }
           }) {
            return true
        }
        return false
    }

    /// Whether `point` should be drawn / hit-tested under the
    /// current "Linked only" toggle state. Splits the test by
    /// point type:
    ///
    ///  - Centroid: visible unless the toggle is on AND the
    ///    controller supplied a `linkedSpeakerIDs` set AND this
    ///    centroid's speaker isn't in it (i.e. an orphan in the
    ///    diarizer DB with no utterance referencing it).
    ///  - Observation: visible unless the toggle is on AND a
    ///    `linkedObservationIDs` set is provided AND this point's
    ///    `segmentId` isn't in it. An observation with no
    ///    `segmentId` (older diarizer that doesn't expose ids)
    ///    is treated as unlinked — there's no way to prove it
    ///    maps to an utterance.
    private func isPointVisible(_ point: SpeakerScatterPoint) -> Bool {
        guard hideUnlinkedObservations else { return true }
        if point.isCentroid {
            guard let speakers = linkedSpeakerIDs else { return true }
            return speakers.contains(point.speakerID)
        }
        guard let ids = linkedObservationIDs else { return true }
        guard let sid = point.observationSegmentID else { return false }
        return ids.contains(sid)
    }

    // MARK: - Tap handling

    /// Tap-to-scroll: hit-test through the model, then resolve the
    /// chosen point's raw embedding from the cluster (centroid
    /// embedding for centroid hits, observation vector for
    /// observation hits) before forwarding to `onTapNode`. Misses
    /// (taps in the empty area between dots) silently do nothing.
    private func handleTap(at location: CGPoint, in size: CGSize) {
        guard let onTap = onTapNode,
              let project = SpeakerScatterProjector(
                points: model.points, in: size, inset: Self.canvasInset
              ),
              let hit = model.hitTest(
                at: location, project: project,
                hitRadius: Self.tapHitRadius,
                visibility: isPointVisible
              ),
              let speaker = cluster.speakers.first(where: { $0.id == hit.speakerID })
        else { return }
        let embedding: [Float]
        if hit.isCentroid {
            embedding = speaker.centroid
        } else if let idx = hit.observationIndex,
                  idx >= 0, idx < speaker.observations.count {
            embedding = speaker.observations[idx]
        } else {
            return
        }
        onTap(hit.speakerID, hit.observationSegmentID, embedding)
    }

}
