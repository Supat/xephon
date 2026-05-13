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

    /// Cached projection from the most recent PCA fit. Empty until
    /// the first snapshot with ≥ 2 distinct points arrives. Kept on
    /// the view so frame re-renders don't recompute — the recompute
    /// happens explicitly inside `.task(id:)` when the snapshot
    /// changes.
    @State private var points: [Point] = []
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
    /// Padding inside the canvas so the dots don't kiss the edges.
    nonisolated private static let canvasInset: CGFloat = 10

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(String(localized: "cluster.scatter.header"))
                    .font(.caption.bold())
                    .foregroundStyle(.secondary)
                Spacer()
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
            Canvas { ctx, size in
                // Compute the data extent once so we can scale the
                // whole cloud into the canvas. `Point.x` and `.y` are
                // unscaled projections; we map to canvas coords here.
                var minX: Float = .infinity
                var maxX: Float = -.infinity
                var minY: Float = .infinity
                var maxY: Float = -.infinity
                for p in points {
                    minX = min(minX, p.x); maxX = max(maxX, p.x)
                    minY = min(minY, p.y); maxY = max(maxY, p.y)
                }
                let dx = max(maxX - minX, 1e-6)
                let dy = max(maxY - minY, 1e-6)
                let availW = size.width - 2 * Self.canvasInset
                let availH = size.height - 2 * Self.canvasInset
                func project(_ p: Point) -> CGPoint {
                    let nx = (p.x - minX) / dx
                    let ny = (p.y - minY) / dy
                    return CGPoint(
                        x: Self.canvasInset + CGFloat(nx) * availW,
                        y: Self.canvasInset + (1 - CGFloat(ny)) * availH
                    )
                }
                // Observations first so centroids stack on top.
                for p in points where !p.isCentroid {
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
                for p in points where p.isCentroid {
                    let r = Self.centroidRadius
                    let center = project(p)
                    let rect = CGRect(
                        x: center.x - r, y: center.y - r,
                        width: 2 * r, height: 2 * r
                    )
                    // Highlight ring around the focused-utterance's
                    // centroid: a wider halo in the speaker's own
                    // tint behind the filled dot so the matching
                    // node pops without obscuring its position.
                    // Drawn before the fill + standard white stroke
                    // so those overlay it cleanly. Skipped when no
                    // utterance is focused or when the focused row's
                    // speaker has no centroid in this snapshot
                    // (orphan after reassignment / DB-only entry).
                    if let highlight = highlightedSpeakerID,
                       p.speakerID == highlight {
                        let haloR = Self.highlightHaloRadius
                        let haloRect = CGRect(
                            x: center.x - haloR, y: center.y - haloR,
                            width: 2 * haloR, height: 2 * haloR
                        )
                        ctx.fill(
                            Path(ellipseIn: haloRect),
                            with: .color(p.color.opacity(0.30))
                        )
                        ctx.stroke(
                            Path(ellipseIn: haloRect),
                            with: .color(p.color),
                            lineWidth: 2
                        )
                    }
                    ctx.fill(Path(ellipseIn: rect), with: .color(p.color))
                    ctx.stroke(
                        Path(ellipseIn: rect),
                        with: .color(.white.opacity(0.9)),
                        lineWidth: 1
                    )
                }
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
        struct Row { let vec: [Float]; let speakerID: String; let isCentroid: Bool }
        var rows: [Row] = []
        for spk in snapshot.speakers {
            rows.append(Row(vec: spk.centroid, speakerID: spk.id, isCentroid: true))
            for obs in spk.observations {
                rows.append(Row(vec: obs, speakerID: spk.id, isCentroid: false))
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
                isCentroid: rows[i].isCentroid
            ))
        }
        return ProjectionResult(points: points, v1: v1, v2: v2)
    }
}
