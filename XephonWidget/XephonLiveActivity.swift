import ActivityKit
import WidgetKit
import SwiftUI

/// Lock Screen + Dynamic Island Live Activity for an active Xephon
/// session. The host app updates `XephonActivityAttributes.ContentState`
/// each time an utterance finalizes; this widget renders the latest
/// state. All updates flow locally — no APNs.
struct XephonLiveActivity: Widget {
    var body: some WidgetConfiguration {
        ActivityConfiguration(for: XephonActivityAttributes.self) { context in
            // Lock Screen / banner presentation.
            LockScreenView(state: context.state, attrs: context.attributes)
                .activityBackgroundTint(Color.black.opacity(0.65))
                .activitySystemActionForegroundColor(Color.white)

        } dynamicIsland: { context in
            DynamicIsland {
                DynamicIslandExpandedRegion(.leading) {
                    Image(systemName: context.state.isAnalyzing
                          ? "waveform.badge.mic"
                          : "waveform.and.mic")
                        .foregroundStyle(.tint)
                        .font(.title3)
                }
                DynamicIslandExpandedRegion(.trailing) {
                    Text("\(context.state.utteranceCount)")
                        .font(.body.monospacedDigit().bold())
                        .foregroundStyle(.tint)
                }
                DynamicIslandExpandedRegion(.center) {
                    if let label = context.state.topLabel {
                        Text(label.capitalized(with: Locale(identifier: "en_US")))
                            .font(.caption.bold())
                            .foregroundStyle(emotionTint(for: label))
                    } else {
                        Text("Calibrating…")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
                DynamicIslandExpandedRegion(.bottom) {
                    HStack(spacing: 12) {
                        Text(formatClock(context.state.elapsedSeconds))
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(.secondary)
                        if let v = context.state.valence {
                            vaTag("V", value: v)
                        }
                        if let a = context.state.arousal {
                            vaTag("A", value: a)
                        }
                        Spacer()
                    }
                }
            } compactLeading: {
                Image(systemName: "waveform.and.mic")
                    .foregroundStyle(.tint)
            } compactTrailing: {
                Text("\(context.state.utteranceCount)")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.tint)
            } minimal: {
                Image(systemName: "waveform")
                    .foregroundStyle(.tint)
            }
            .keylineTint(.accentColor)
        }
    }
}

private struct LockScreenView: View {
    let state: XephonActivityAttributes.ContentState
    let attrs: XephonActivityAttributes

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Image(systemName: state.isAnalyzing
                      ? "waveform.badge.mic"
                      : "waveform.and.mic")
                    .foregroundStyle(.tint)
                Text(state.isAnalyzing ? "Xephon · finalizing" : "Xephon · recording")
                    .font(.caption.bold())
                Spacer()
                Text(attrs.sourceLabel)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }

            HStack(alignment: .firstTextBaseline) {
                if let label = state.topLabel {
                    Text(label.capitalized(with: Locale(identifier: "en_US")))
                        .font(.title3.bold())
                        .foregroundStyle(emotionTint(for: label))
                } else {
                    Text("Calibrating…")
                        .font(.title3.bold())
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Text("\(state.utteranceCount) utts · \(formatClock(state.elapsedSeconds))")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }

            if state.valence != nil || state.arousal != nil {
                HStack(spacing: 12) {
                    if let v = state.valence { vaTag("V", value: v) }
                    if let a = state.arousal { vaTag("A", value: a) }
                    Spacer()
                }
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
    }
}

@ViewBuilder
private func vaTag(_ axis: String, value: Float) -> some View {
    let eps: Float = 0.05
    let tint: Color = value >  eps ? .green
                    : value < -eps ? .red
                                   : .gray
    Text(String(format: "%@ %+.2f", axis, value))
        .font(.caption.monospacedDigit())
        .foregroundStyle(tint)
}

/// Local copy of the main app's `formatClock` — widget extensions
/// don't link the main module, and the helper is small enough that
/// duplicating beats setting up a third shared target.
private func formatClock(_ seconds: TimeInterval) -> String {
    let total = Int(max(0, seconds))
    let h = total / 3600
    let m = (total % 3600) / 60
    let s = total % 60
    if h > 0 { return String(format: "%d:%02d:%02d", h, m, s) }
    return String(format: "%d:%02d", m, s)
}

/// Local copy of the emotion palette — same rationale as `formatClock`.
private func emotionTint(for raw: String) -> Color {
    switch raw.lowercased() {
    case "happy", "joy", "joyful":               return .yellow
    case "sad", "sadness":                       return .blue
    case "angry", "anger":                       return .red
    case "fear", "fearful", "afraid":            return .purple
    case "disgust", "disgusted":                 return Color(red: 0.45, green: 0.55, blue: 0.20)
    case "surprise", "surprised":                return .orange
    case "trust":                                return .green
    case "anticipation":                         return Color(red: 0.95, green: 0.55, blue: 0.10)
    case "neutral", "calm":                      return .gray
    default:                                     return Color.secondary
    }
}
