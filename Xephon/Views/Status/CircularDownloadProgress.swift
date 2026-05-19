import SwiftUI
import XephonUtilities

/// Compact circular fill indicator for the Settings card's
/// summarizer-download row. Renders a thin grey track with an
/// accent-tinted arc filling clockwise as `downloaded / total`
/// approaches 1. When the total isn't known yet (race between
/// "task started" and "first didWriteData callback"), falls back
/// to the system's indeterminate spinner so the row never reads
/// as "0 of 0".
struct CircularDownloadProgress: View {
    let downloaded: Int64
    let total: Int64

    /// Stroke width tuned to read cleanly at the Settings card's
    /// caption-line height (~14 pt). Anything thinner blurs on
    /// non-Retina; anything thicker eats the visible track.
    private static let lineWidth: CGFloat = 2.5
    private static let diameter: CGFloat = 18

    var body: some View {
        if total > 0 {
            let fraction = (Double(downloaded) / Double(total)).clamped(to: 0...1)
            ZStack {
                Circle()
                    .stroke(
                        Color.secondary.opacity(0.25),
                        lineWidth: Self.lineWidth
                    )
                Circle()
                    .trim(from: 0, to: fraction)
                    .stroke(
                        Color.accentColor,
                        style: StrokeStyle(
                            lineWidth: Self.lineWidth,
                            lineCap: .round
                        )
                    )
                    // Trim starts at 3 o'clock by default; rotate
                    // -90° so the arc begins at 12 o'clock, which
                    // is the conventional progress-fill origin.
                    .rotationEffect(.degrees(-90))
                    .animation(.linear(duration: 0.15), value: fraction)
            }
            .frame(width: Self.diameter, height: Self.diameter)
            .accessibilityLabel(
                Text("\(Int(fraction * 100)) percent downloaded")
            )
        } else {
            // No total yet — show the spinner so the user sees
            // *something* is happening rather than a dim ring.
            ProgressView()
                .controlSize(.mini)
                .frame(width: Self.diameter, height: Self.diameter)
        }
    }
}
