import SwiftUI
import Diarization

/// One node in the PCA scatter — either a speaker centroid or one
/// retained observation. Carries the projected coordinates plus
/// enough identity to round-trip a tap back to the originating
/// utterance.
struct SpeakerScatterPoint: Sendable {
    let x: Float
    let y: Float
    let speakerID: String
    let color: Color
    let isCentroid: Bool
    /// Index into the source speaker's `observations` array when
    /// `isCentroid == false`; nil for centroid points. Used to find
    /// the projected coordinate of a specific observation when the
    /// focused-utterance embedding picks one out.
    let observationIndex: Int?
    /// Stable diarizer `RawEmbedding.segmentId` for this
    /// observation; nil for centroids and for snapshots produced by
    /// diarizers that don't expose stable per-observation ids.
    /// Carried into the tap callback so the controller can look up
    /// the emitting utterance by id rather than running an
    /// embedding-distance search.
    let observationSegmentID: UUID?
}

/// Maps unscaled `SpeakerScatterPoint.x`/`y` (PCA-projected) into
/// canvas coordinates. Bundles the data-extent + inset math so the
/// Canvas body and the model's hit/halo/arrow logic share one
/// source of truth — previously each computed the bounds inline,
/// which let the two drift if either was edited in isolation.
struct SpeakerScatterProjector {
    let minX: Float
    let minY: Float
    let dx: Float
    let dy: Float
    let availW: CGFloat
    let availH: CGFloat
    let inset: CGFloat

    init?(points: [SpeakerScatterPoint], in size: CGSize, inset: CGFloat) {
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

    func callAsFunction(_ p: SpeakerScatterPoint) -> CGPoint {
        let nx = (p.x - minX) / dx
        let ny = (p.y - minY) / dy
        return CGPoint(
            x: inset + CGFloat(nx) * availW,
            y: inset + (1 - CGFloat(ny)) * availH
        )
    }
}

/// 2D PCA projection of a diarizer speaker-cluster snapshot.
///
/// The owning `SpeakerClusterCard` view triggers `recompute(snapshot:)`
/// inside `.task(id: snapshotKey(for:))` whenever the snapshot's
/// roster or observation counts change. PCA runs on a detached
/// task so the view body re-render isn't blocked by the
/// eigenvector iterations; the resulting `points` array drives the
/// Canvas redraw on the main actor.
///
/// Sign-stabilization against the previous basis (`prevV1`/`prevV2`,
/// kept `@ObservationIgnored` because the view never reads them
/// directly) prevents axis flips between consecutive refits on
/// near-identical data. Sufficient for the dominant frame-to-frame
/// failure; full Procrustes alignment isn't worth the extra code
/// for a debug visualization.
///
/// Halo / arrow / hit-test helpers take a pre-built
/// `SpeakerScatterProjector` plus a visibility predicate so the
/// view can apply its "Linked only" toggle without the model
/// needing to know about the toggle's state. The same predicate
/// drives the Canvas draw, halo, arrow, and tap pickling — all
/// four agree about which dots are live.
@MainActor
@Observable
final class SpeakerScatterModel {
    /// Cached projection from the most recent PCA fit. Empty until
    /// the first snapshot with ≥ 2 distinct points arrives.
    private(set) var points: [SpeakerScatterPoint] = []

    @ObservationIgnored private var prevV1: [Float] = []
    @ObservationIgnored private var prevV2: [Float] = []

    /// Key the view's `.task(id:)` modifier watches. Changes when
    /// the snapshot's speaker roster or any observation-count
    /// changes — no need to recompute when literally nothing in
    /// the cluster has shifted between two timer ticks.
    static func snapshotKey(for cluster: SpeakerClusterSnapshot) -> String {
        cluster.speakers
            .map { "\($0.id):\($0.observations.count)" }
            .joined(separator: ",")
    }

    /// Recompute the PCA projection on a detached task so the body
    /// re-render isn't blocked by the eigenvector iterations.
    /// Reads the current snapshot + previous basis, computes off
    /// main, then writes the result back here.
    func recompute(snapshot: SpeakerClusterSnapshot) async {
        let prior = (v1: prevV1, v2: prevV2)
        let result = await Task.detached(priority: .userInitiated) {
            Self.computeProjection(snapshot: snapshot, previousBasis: prior)
        }.value
        self.points = result.points
        self.prevV1 = result.v1
        self.prevV2 = result.v2
    }

    // MARK: - Halo / arrow / hit test

    /// Halo target — always the highlighted speaker's centroid.
    /// Encodes "this row's speaker" context regardless of whether
    /// we have a specific observation to pin down.
    func haloTarget(
        project: SpeakerScatterProjector,
        focusedSpeakerID: String?,
        visibility: (SpeakerScatterPoint) -> Bool
    ) -> (position: CGPoint, color: Color)? {
        guard let speakerID = focusedSpeakerID else { return nil }
        guard let p = points.first(where: {
            $0.speakerID == speakerID && $0.isCentroid
        }),
              visibility(p) else { return nil }
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
    func arrowTarget(
        in cluster: SpeakerClusterSnapshot,
        project: SpeakerScatterProjector,
        focusedSpeakerID: String?,
        focusedEmbedding: [Float]?,
        visibility: (SpeakerScatterPoint) -> Bool
    ) -> (position: CGPoint, color: Color)? {
        guard let speakerID = focusedSpeakerID,
              let embedding = focusedEmbedding,
              let bestIndex = Self.nearestObservationIndex(
                in: cluster, speakerID: speakerID, to: embedding
              ),
              let p = points.first(where: {
                $0.speakerID == speakerID
                    && $0.isCentroid == false
                    && $0.observationIndex == bestIndex
              }),
              visibility(p) else { return nil }
        return (position: project(p), color: p.color)
    }

    /// Tap hit test. Picks the closest visible point within
    /// `hitRadius`, or nil when the tap landed in empty space.
    /// View resolves the chosen point back to a raw embedding
    /// (it has the cluster) before forwarding to the tap callback.
    func hitTest(
        at location: CGPoint,
        project: SpeakerScatterProjector,
        hitRadius: CGFloat,
        visibility: (SpeakerScatterPoint) -> Bool
    ) -> SpeakerScatterPoint? {
        let limit = hitRadius * hitRadius
        var bestDist = limit
        var bestPoint: SpeakerScatterPoint?
        for p in points where visibility(p) {
            let c = project(p)
            let ddx = c.x - location.x
            let ddy = c.y - location.y
            let d2 = ddx * ddx + ddy * ddy
            if d2 < bestDist {
                bestDist = d2
                bestPoint = p
            }
        }
        return bestPoint
    }

    // MARK: - Math

    private struct ProjectionResult: Sendable {
        let points: [SpeakerScatterPoint]
        let v1: [Float]
        let v2: [Float]
    }

    /// Argmin Euclidean distance from `query` over `speakerID`'s
    /// observations in the snapshot. Nil when the speaker has no
    /// observations in the current tail window (snapshot capped at
    /// `clusterObservationsPerSpeaker` and the utterance's window
    /// has aged out).
    nonisolated private static func nearestObservationIndex(
        in cluster: SpeakerClusterSnapshot,
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

    /// Project every centroid + observation onto the top-2 PCs of
    /// the pooled embedding set. The detached-task entry point —
    /// stays `nonisolated` so `Task.detached` can call it without
    /// an explicit hop.
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
        var points: [SpeakerScatterPoint] = []
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
            points.append(SpeakerScatterPoint(
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
