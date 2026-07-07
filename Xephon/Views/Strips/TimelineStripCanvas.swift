import SwiftUI
import XephonUtilities

/// Shared scaffold for the session timeline strips (diarization,
/// emotion, fusion contribution, keyword): the Liquid Glass capsule
/// base track, the primary-stroke selection overlay, the capsule
/// clip, and the whole-strip tap-to-audio-time gesture. Each strip
/// supplies only its run rectangles via `runs(width)`.
///
/// The base track provides the glass frame (rounded capsule profile
/// + edge refraction + tinted backdrop); runs are expected to be
/// flat fills — per-run glass blur dominated the strips and
/// softened their colors. `contentShape(Rectangle())` keeps gaps
/// between runs tappable, and the local coordinate space maps
/// `location.x / width` directly onto the strip's audio-time axis.
struct TimelineStripCanvas<Runs: View>: View {
    let totalDuration: TimeInterval
    let selectedRange: (start: TimeInterval, end: TimeInterval)?
    /// Fires with the tapped audio-time, clamped to
    /// `[0, totalDuration]`. Optional so a strip is useful even
    /// without a tap-routing parent.
    let onTapAtTime: ((TimeInterval) -> Void)?
    var height: CGFloat = 6
    var trackTintOpacity: Double = 0.06
    @ViewBuilder let runs: (_ width: CGFloat) -> Runs

    var body: some View {
        GeometryReader { geo in
            ZStack(alignment: .leading) {
                Capsule()
                    .glassEffect(
                        .regular.tint(.secondary.opacity(trackTintOpacity)),
                        in: Capsule()
                    )
                    .frame(height: height)
                runs(geo.size.width)
            }
            // Selection overlay — plain primary-color stroke kept
            // outside any glass material so the edge stays crisp
            // against the colored runs underneath. Identical across
            // strips so all highlighters line up at the focused
            // row's audio range.
            .overlay(alignment: .leading) {
                if let sel = selectedRange, totalDuration > 0 {
                    let clampedStart = sel.start.clamped(to: 0...totalDuration)
                    let clampedEnd = sel.end.clamped(to: clampedStart...totalDuration)
                    let x = geo.size.width * CGFloat(clampedStart / totalDuration)
                    let w = geo.size.width * CGFloat((clampedEnd - clampedStart) / totalDuration)
                    RoundedRectangle(cornerRadius: 3, style: .continuous)
                        .strokeBorder(Color.primary.opacity(0.9), lineWidth: 1.5)
                        .frame(width: max(2, w), height: height)
                        .offset(x: x)
                }
            }
            .clipShape(Capsule())
            .contentShape(Rectangle())
            .onTapGesture(coordinateSpace: .local) { location in
                guard let onTapAtTime, totalDuration > 0, geo.size.width > 0 else { return }
                let t = totalDuration * Double(location.x / geo.size.width)
                onTapAtTime(t.clamped(to: 0...totalDuration))
            }
        }
        .frame(height: height)
    }
}
