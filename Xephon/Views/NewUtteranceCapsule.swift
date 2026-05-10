import SwiftUI

/// Floating "new utterance available" affordance shown over the bottom
/// of the transcript when a new entry has arrived but the user has
/// scrolled the list away from the latest row. Tapping scrolls back
/// to the most recent utterance. Hidden when the latest is in view.
struct NewUtteranceCapsule: View {
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 6) {
                Image(systemName: "arrow.down")
                    .font(.caption.bold())
                Text(String(localized: "transcript.newUtterance"))
                    .font(.subheadline.weight(.semibold))
            }
            .foregroundStyle(.primary)
            .padding(.horizontal, 14)
            .padding(.vertical, 8)
            .background(.ultraThinMaterial, in: Capsule())
            .overlay(Capsule().strokeBorder(.tertiary, lineWidth: 0.5))
            .shadow(color: .black.opacity(0.15), radius: 6, y: 2)
        }
        .buttonStyle(.plain)
    }
}
