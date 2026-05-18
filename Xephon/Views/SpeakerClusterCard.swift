import SwiftUI
import Diarization

/// 2D PCA scatter of the diarizer's speaker cluster — every
/// speaker's averaged centroid plus its retained raw observations,
/// projected onto the top two principal components of the pooled
/// embedding set. Sits below `SpeakerHeatmapCard` as the second
/// diagnostic view of the diarizer's internal state.
///
/// Centroids render as larger filled dots in each speaker's color;
/// observations render smaller and lower-opacity so a cloud
/// around its centroid reads as "this speaker's spread". When two
/// clouds overlap, the diarizer is having a hard time keeping the
/// speakers apart — paired with the heatmap, the user can see both
/// the pairwise number and the visual reason for it.
///
/// PCA fits per snapshot refresh (~0.5 Hz while recording). To
/// prevent the axes from flipping between fits we sign-stabilize
/// each new basis against the previous one: if `v1·v1' < 0` we
/// flip `v1'`, same for `v2'`. Sufficient for the dominant frame-
/// to-frame failure; full Procrustes alignment isn't worth the
/// extra code for a debug visualization.
struct SpeakerClusterCard: View {
    let cluster: SpeakerClusterSnapshot
    /// Speaker id of the currently-focused utterance, or nil if no
    /// row is focused. Drives the highlight ring drawn around the
    /// matching centroid dot so the user can trace a transcript row
    /// to its diarizer-side cluster at a glance. Lives outside the
    /// PCA recompute trigger (`snapshotKey`) so a focus change
    /// repaints without re-fitting the basis.
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

    /// Cached projection from the most recent PCA fit. Empty until
    /// the first snapshot with ≥ 2 distinct points arrives. Kept on
    /// the view so frame re-renders don't recompute — the recompute
    /// happens explicitly inside `.task(id:)` when the snapshot
    /// changes.
    @State private var points: [Point] = []
    /// Header toggle: hide observation dots whose `segmentId` isn't
    /// in `linkedObservationIDs`. Kept here so flipping the toggle
    /// doesn't trigger a PCA refit — the basis stays anchored to
    /// the full point set, only the visibility filter changes,
    /// so the visible dots don't drift around when the toggle is
    /// flipped.
    @State private var hideUnlinkedObservations: Bool = false
    /// Previous basis kept so the next fit can sign-stabilize
    /// against it (prevents axis flips between consecutive PCA
    /// runs on near-identical data). Empty arrays = no prior fit.
    @State private var prevV1: [Float] = []
    @State private var prevV2: [Float] = []

    nonisolated private static let canvasHeight: CGFloat = 180
    nonisolated private static let centroidRadius: CGFloat = 5
    nonisolated private static let observationRadius: CGFloat = 2.5
    nonisolated private static let observationOpacity: Double = 0.45
    /// Outer radius of the focus-highlight halo drawn around the
    /// centroid that matches the currently-selected utterance's
    /// speaker. Sized about 2.4× the centroid radius so the ring
    /// reads as a clear emphasis without crowding adjacent nodes.
    nonisolated private static let highlightHaloRadius: CGFloat = 12

    /// Halo target — always the highlighted speaker's centroid.
    /// Encodes "this row's speaker" context regardless of whether
    /// we have a specific observation to pin down.
    private func findHaloTarget(
        in points: [Point],
        project: Projector
    ) -> (position: CGPoint, color: Color)? {
        guard let speakerID = highlightedSpeakerID else { return nil }
        guard let p = points.first(where: {
            $0.speakerID == speakerID && $0.isCentroid
        }),
              isPointVisible(p) else { return nil }
        return (position: project(p), color: p.color)
    }

    /// Arrow target — the specific observation that best matches
    /// the focused utterance's stored embedding. Nil when we don't
    /// have an embedding (older utterance / no diarizer / mic-mode
    /// import) or the snapshot's tail no longer contains the
    /// matching observation. The arrow stays absent rather than
    /// fall back to the centroid because the halo already marks
    /// the centroid — a centroid-pointed arrow would be redundant
    /// emphasis on the same point.
    private func findArrowTarget(
        in points: [Point],
        project: Projector
    ) -> (position: CGPoint, color: Color)? {
        guard let speakerID = highlightedSpeakerID,
              let embedding = focusedEmbedding,
              let bestIndex = nearestObservationIndex(
                speakerID: speakerID, to: embedding
              ),
              let p = points.first(where: {
                $0.speakerID == speakerID
                    && $0.isCentroid == false
                    && $0.observationIndex == bestIndex
              }),
              isPointVisible(p) else { return nil }
        return (position: project(p), color: p.color)
    }

    /// Argmin Euclidean distance from `query` over the highlighted
    /// speaker's observations in the snapshot. Nil when the
    /// speaker has no observations in the current tail window
    /// (snapshot capped at `clusterObservationsPerSpeaker` and the
    /// utterance's window has aged out).
    private func nearestObservationIndex(
        speakerID: String,
        to query: [Float]
    ) -> Int? {
        guard let speaker = cluster.speakers.first(where: { $0.id == speakerID }),
              !speaker.observations.isEmpty else { return nil }
        var bestIndex: Int?
        var bestDist: Float = .infinity
        for (i, obs) in speaker.observations.enumerated() {
            let n = min(obs.count, query.count)
            guard n > 0 else { continue }
            var sum: Float = 0
            for j in 0..<n {
                let d = obs[j] - query[j]
                sum += d * d
            }
            if sum < bestDist {
                bestDist = sum
                bestIndex = i
            }
        }
        return bestIndex
    }

    /// Would the "Linked only" filter actually hide anything under
    /// the current snapshot? True when either an observation in the
    /// cluster has a `segmentId` not in `linkedObservationIDs`, or a
    /// speaker in the cluster isn't in `linkedSpeakerIDs`. Gates the
    /// header toggle so users don't see a control that's guaranteed
    /// to do nothing.
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
    private func isPointVisible(_ point: Point) -> Bool {
        guard hideUnlinkedObservations else { return true }
        if point.isCentroid {
            guard let speakers = linkedSpeakerIDs else { return true }
            return speakers.contains(point.speakerID)
        }
        guard let ids = linkedObservationIDs else { return true }
        guard let sid = point.observationSegmentID else { return false }
        return ids.contains(sid)
    }

    /// Tap-to-scroll hit testing. Re-runs the same projection the
    /// Canvas draws with so screen coordinates line up with what
    /// the user sees, picks the closest point within
    /// `tapHitRadius`, then forwards its raw embedding to
    /// `onTapNode`. Misses (taps in the empty area between dots)
    /// silently do nothing — no spurious scroll if the user just
    /// pans past the scatter.
    private func handleTap(at location: CGPoint, in size: CGSize) {
        guard let onTap = onTapNode,
              let project = Projector(points: points, in: size, inset: Self.canvasInset)
        else { return }
        let limit = Self.tapHitRadius * Self.tapHitRadius
        var bestDist = limit
        var bestPoint: Point?
        for p in points where isPointVisible(p) {
            let c = project(p)
            let ddx = c.x - location.x
            let ddy = c.y - location.y
            let d2 = ddx * ddx + ddy * ddy
            if d2 < bestDist {
                bestDist = d2
                bestPoint = p
            }
        }
        guard let hit = bestPoint,
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

    /// Touch slop around each projected dot. 20pt is roughly the
    /// minimum tappable target Apple recommends and is generous
    /// against the 2.5pt observation radius without overlapping
    /// adjacent clouds in dense scatters.
    nonisolated private static let tapHitRadius: CGFloat = 20

    /// Maps unscaled `Point.x`/`y` (PCA-projected) into canvas
    /// coordinates. Bundles the data-extent + inset math so the
    /// Canvas body and `handleTap` share one source of truth —
    /// previously each computed the bounds and projection inline,
    /// which let the two drift if either was edited in isolation.
    fileprivate struct Projector {
        let minX: Float
        let minY: Float
        let dx: Float
        let dy: Float
        let availW: CGFloat
        let availH: CGFloat
        let inset: CGFloat

        init?(points: [Point], in size: CGSize, inset: CGFloat) {
            guard !points.isEmpty else { return nil }
            var minX: Float = .infinity, maxX: Float = -.infinity
            var minY: Float = .infinity, maxY: Float = -.infinity
            for p in points {
                minX = min(minX, p.x); maxX = max(maxX, p.x)
                minY = min(minY, p.y); maxY = max(maxY, p.y)
            }
            self.minX = minX
            self.minY = minY
            self.dx = max(maxX - minX, 1e-6)
            self.dy = max(maxY - minY, 1e-6)
            self.availW = size.width - 2 * inset
            self.availH = size.height - 2 * inset
            self.inset = inset
        }

        func callAsFunction(_ p: Point) -> CGPoint {
            let nx = (p.x - minX) / dx
            let ny = (p.y - minY) / dy
            return CGPoint(
                x: inset + CGFloat(nx) * availW,
                y: inset + (1 - CGFloat(ny)) * availH
            )
        }
    }

    /// Short fixed-length arrow pointing at `target` from the
    /// upper-right (or whichever diagonal stays inside the canvas
    /// at the chosen length). Sized so the tip kisses the dot's
    /// outer edge without crowding it. Always white so the
    /// pointer reads against every speaker tint behind it — using
    /// the speaker's color made the arrow blend into the dot's
    /// own halo, defeating the purpose.
    nonisolated private static func drawFocusArrow(
        ctx: GraphicsContext,
        in canvas: CGRect,
        to target: CGPoint
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
        // than plunging through. Dot radius (5pt) + 1pt gap.
        let dotEdgeGap: CGFloat = centroidRadius + 1
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
    /// Padding inside the canvas so the dots don't kiss the edges.
    nonisolated private static let canvasInset: CGFloat = 10

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 6) {
                Text(String(localized: "cluster.scatter.header"))
                    .font(.caption.bold())
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .minimumScaleFactor(0.8)
                Spacer(minLength: 4)
                // Only show the toggle when there's something to
                // filter — either at least one pinned observation
                // (observation-dot filtering) or at least one
                // centroid whose speaker isn't referenced by any
                // utterance (centroid filtering). Legacy sessions
                // with neither signal get a hidden toggle.
                // Rendered as a plain caption2 button instead of a
                // native Toggle: the card's header is otherwise
                // all-text at caption2/tertiary, and a system
                // switch (even at .mini) blew out the visual
                // weight relative to the "PCA 2D" tag beside it.
                // On-state is signalled by the filled link icon
                // and accent tint; off-state stays secondary.
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
        .task(id: snapshotKey) {
            await recompute()
        }
    }

    @ViewBuilder
    private var canvasView: some View {
        if points.isEmpty {
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
                guard let project = Projector(
                    points: points, in: size, inset: Self.canvasInset
                ) else { return }
                let minX = project.minX
                let minY = project.minY
                let dx = project.dx
                let dy = project.dy
                let availW = project.availW
                let availH = project.availH
                // PCA mean-centers the data so projected (0, 0) is
                // the data centroid. Drawing axis lines through it
                // gives the user a reference for "which side of
                // the mean does this cluster sit on?" along each
                // principal component. Skipped per-axis when the
                // origin falls outside the visible data extent
                // (e.g. all observations clustered on positive
                // PC1) — drawing the line would clip off-canvas.
                let dataRect = CGRect(
                    x: Self.canvasInset,
                    y: Self.canvasInset,
                    width: availW,
                    height: availH
                )
                let originXFrac = (0 - minX) / dx
                let originYFrac = (0 - minY) / dy
                let originStyle: GraphicsContext.Shading =
                    .color(.secondary.opacity(0.35))
                if originXFrac >= 0 && originXFrac <= 1 {
                    let x = Self.canvasInset + CGFloat(originXFrac) * availW
                    var line = Path()
                    line.move(to: CGPoint(x: x, y: dataRect.minY))
                    line.addLine(to: CGPoint(x: x, y: dataRect.maxY))
                    ctx.stroke(line, with: originStyle, lineWidth: 0.5)
                }
                if originYFrac >= 0 && originYFrac <= 1 {
                    let y = Self.canvasInset + (1 - CGFloat(originYFrac)) * availH
                    var line = Path()
                    line.move(to: CGPoint(x: dataRect.minX, y: y))
                    line.addLine(to: CGPoint(x: dataRect.maxX, y: y))
                    ctx.stroke(line, with: originStyle, lineWidth: 0.5)
                }
                // Axis labels live just inside the data rect at
                // diagonally-opposite corners — the canvas only
                // reserves a 10pt inset around the data area,
                // which isn't wide enough to fit the labels in
                // the margin without clipping (PC2↑ pushed off
                // the left edge previously). Caption2 / tertiary
                // keeps them quiet behind the dots.
                ctx.draw(
                    Text("PC1→").font(.caption2).foregroundStyle(.tertiary),
                    at: CGPoint(
                        x: dataRect.maxX - 2,
                        y: dataRect.maxY - 2
                    ),
                    anchor: .bottomTrailing
                )
                ctx.draw(
                    Text("PC2↑").font(.caption2).foregroundStyle(.tertiary),
                    at: CGPoint(
                        x: dataRect.minX + 2,
                        y: dataRect.minY + 2
                    ),
                    anchor: .topLeading
                )
                // Observations first so centroids stack on top.
                // `isPointVisible` honors the "Linked only" toggle
                // — when on, an observation only renders if its
                // `segmentId` is in the controller-supplied set,
                // and a centroid only renders if its speaker is in
                // the linked-speakers set. Filtering applies to
                // draw, halo, arrow, and tap pickling so all four
                // agree.
                for p in points
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
                // speaker's centroid (so the speaker context reads
                // at a glance), while the arrow tips at the
                // specific observation that came from the focused
                // utterance (so the user sees "this row's node").
                // Halo skips when the speaker has no centroid in
                // this snapshot (orphan / DB-only); arrow skips
                // when we couldn't pin a specific observation —
                // older utterance whose embedding wasn't captured
                // OR snapshot tail-trimmed past this utterance.
                let haloTarget = findHaloTarget(in: points, project: project)
                let arrowTarget = findArrowTarget(in: points, project: project)
                for p in points where p.isCentroid && isPointVisible(p) {
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
                if let target = haloTarget {
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
                // Directional pointer at the focused observation
                // (not the centroid — that's the halo's job).
                // Drawn last so it sits above every other dot.
                if let target = arrowTarget {
                    Self.drawFocusArrow(
                        ctx: ctx,
                        in: CGRect(origin: .zero, size: size),
                        to: target.position
                    )
                }
        }
    }

    /// Key the `.task(id:)` modifier watches. Changes when the
    /// snapshot's speaker roster or any observation-count changes —
    /// no need to recompute when literally nothing in the cluster
    /// has shifted between two timer ticks.
    private var snapshotKey: String {
        cluster.speakers
            .map { "\($0.id):\($0.observations.count)" }
            .joined(separator: ",")
    }

    /// Recompute the PCA projection on a detached task so the body
    /// re-render isn't blocked by the eigenvector iterations.
    /// Reads the current snapshot + previous basis, computes off
    /// main, then hops back to write the result.
    private func recompute() async {
        let snapshot = cluster
        let prior = (v1: prevV1, v2: prevV2)
        let result = await Task.detached(priority: .userInitiated) {
            Self.computeProjection(snapshot: snapshot, previousBasis: prior)
        }.value
        await MainActor.run {
            self.points = result.points
            self.prevV1 = result.v1
            self.prevV2 = result.v2
        }
    }

    // MARK: - Math

    fileprivate struct Point: Sendable {
        let x: Float
        let y: Float
        let speakerID: String
        let color: Color
        let isCentroid: Bool
        /// Index into the source speaker's `observations` array
        /// when `isCentroid == false`; nil for the centroid Point.
        /// Used to find the projected coordinate of a specific
        /// observation when the focused-utterance embedding picks
        /// a particular one out.
        let observationIndex: Int?
        /// Stable diarizer `RawEmbedding.segmentId` for this
        /// observation; nil for centroids and for snapshots
        /// produced by diarizers that don't expose stable per-
        /// observation ids. Carried into the tap callback so the
        /// controller can look up the emitting utterance by id
        /// rather than running another embedding-distance search.
        let observationSegmentID: UUID?
    }

    private struct ProjectionResult: Sendable {
        let points: [Point]
        let v1: [Float]
        let v2: [Float]
    }

    /// Project every centroid + observation onto the top-2 PCs of
    /// the pooled embedding set. The detached-task entry point —
    /// stays nonisolated so SwiftUI's `Task.detached` can call it
    /// without an explicit hop.
    nonisolated private static func computeProjection(
        snapshot: SpeakerClusterSnapshot,
        previousBasis: (v1: [Float], v2: [Float])
    ) -> ProjectionResult {
        struct Row {
            let vec: [Float]
            let speakerID: String
            let isCentroid: Bool
            let observationIndex: Int?
            let observationSegmentID: UUID?
        }
        var rows: [Row] = []
        for spk in snapshot.speakers {
            rows.append(Row(
                vec: spk.centroid,
                speakerID: spk.id,
                isCentroid: true,
                observationIndex: nil,
                observationSegmentID: nil
            ))
            // `observationSegmentIDs` is parallel to `observations`
            // when the underlying diarizer fills it; falls back to
            // index-aligned nils for snapshots from a diarizer
            // that doesn't expose ids (mock / older path).
            let ids = spk.observationSegmentIDs
            for (i, obs) in spk.observations.enumerated() {
                rows.append(Row(
                    vec: obs,
                    speakerID: spk.id,
                    isCentroid: false,
                    observationIndex: i,
                    observationSegmentID: i < ids.count ? ids[i] : nil
                ))
            }
        }
        guard rows.count >= 2, let d = rows.first?.vec.count, d > 0 else {
            return ProjectionResult(points: [], v1: [], v2: [])
        }

        // Mean-center.
        var mean = [Float](repeating: 0, count: d)
        for row in rows {
            for j in 0..<d { mean[j] += row.vec[j] }
        }
        let invN = 1.0 / Float(rows.count)
        for j in 0..<d { mean[j] *= invN }
        let centered: [[Float]] = rows.map { row in
            var c = row.vec
            for j in 0..<min(d, c.count) { c[j] -= mean[j] }
            return c
        }

        // Power iteration for top-2 eigenvectors of C = X^T X.
        // We never materialize C — `matvec(v) = X^T (X v)` is two
        // O(N·d) sweeps and avoids the d² covariance build.
        func matvec(_ v: [Float]) -> [Float] {
            var u = [Float](repeating: 0, count: centered.count)
            for i in 0..<centered.count {
                let row = centered[i]
                var s: Float = 0
                let m = min(d, row.count)
                for j in 0..<m { s += row[j] * v[j] }
                u[i] = s
            }
            var cv = [Float](repeating: 0, count: d)
            for i in 0..<centered.count {
                let row = centered[i]
                let coef = u[i]
                let m = min(d, row.count)
                for j in 0..<m { cv[j] += row[j] * coef }
            }
            return cv
        }

        func powerIterate(deflateAgainst: [Float]?, maxIter: Int) -> [Float] {
            var v = [Float](repeating: 0, count: d)
            // Deterministic seed → frame-to-frame stability when the
            // input is nearly identical.
            v[0] = 1
            for _ in 0..<maxIter {
                var cv = matvec(v)
                if let u = deflateAgainst, u.count == d {
                    var coef: Float = 0
                    for j in 0..<d { coef += cv[j] * u[j] }
                    for j in 0..<d { cv[j] -= coef * u[j] }
                }
                var norm: Float = 0
                for j in 0..<d { norm += cv[j] * cv[j] }
                norm = sqrt(norm)
                if norm < 1e-8 { return v }
                let inv = 1.0 / norm
                for j in 0..<d { cv[j] *= inv }
                v = cv
            }
            return v
        }

        var v1 = powerIterate(deflateAgainst: nil, maxIter: 40)
        var v2 = powerIterate(deflateAgainst: v1, maxIter: 40)

        // Sign-stabilize against the previous basis so consecutive
        // refits don't visually flip the cloud across an axis. Only
        // applies when we have a non-empty prior.
        if previousBasis.v1.count == d {
            var dotV1: Float = 0
            for j in 0..<d { dotV1 += v1[j] * previousBasis.v1[j] }
            if dotV1 < 0 { for j in 0..<d { v1[j] = -v1[j] } }
        }
        if previousBasis.v2.count == d {
            var dotV2: Float = 0
            for j in 0..<d { dotV2 += v2[j] * previousBasis.v2[j] }
            if dotV2 < 0 { for j in 0..<d { v2[j] = -v2[j] } }
        }

        // Project every centered row onto (v1, v2).
        var points: [Point] = []
        points.reserveCapacity(rows.count)
        for i in 0..<rows.count {
            let row = centered[i]
            var x: Float = 0
            var y: Float = 0
            let m = min(d, row.count)
            for j in 0..<m {
                x += row[j] * v1[j]
                y += row[j] * v2[j]
            }
            points.append(Point(
                x: x, y: y,
                speakerID: rows[i].speakerID,
                color: speakerTint(for: rows[i].speakerID),
                isCentroid: rows[i].isCentroid,
                observationIndex: rows[i].observationIndex,
                observationSegmentID: rows[i].observationSegmentID
            ))
        }
        return ProjectionResult(points: points, v1: v1, v2: v2)
    }
}
