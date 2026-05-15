import SwiftUI
import Fusion
import SERText

extension SwitchingTextSER.Backend {
    /// Short tag rendered on each utterance row's text-backend badge.
    var badgeLabel: String {
        switch self {
        case .deberta:          return "DeBERTa"
        case .foundationModels: return "Apple FM"
        }
    }
}

/// Lightweight view-model for the text-SER badge rendered on each
/// `UtteranceRow`. Carries the chip label plus a flag for the Apple
/// FoundationModels guardrail-decline state, which is rendered in
/// place of the regular "Apple FM" chip with a warning tint.
struct TextBackendBadge: Equatable {
    let label: String
    let isGuardrail: Bool
}

/// Live `mm:ss` readout of the wall-clock time elapsed since `start`.
/// Uses `TimelineView(.periodic)` so the value ticks once per second
/// without driving the rest of the view's body to re-evaluate —
/// scoped re-renders matter inside sheets that also host LLM
/// generation, where unnecessary `body` evaluations on the parent
/// can stall the main thread under memory pressure.
struct ElapsedTimeLabel: View {
    let start: Date

    var body: some View {
        TimelineView(.periodic(from: start, by: 1.0)) { context in
            Text(format(elapsed: context.date.timeIntervalSince(start)))
                .monospacedDigit()
        }
    }

    private func format(elapsed: TimeInterval) -> String {
        let total = max(0, Int(elapsed))
        let m = total / 60
        let s = total % 60
        return String(format: "%d:%02d", m, s)
    }
}

extension URL: @retroactive Identifiable {
    public var id: String { absoluteString }
}

/// `ParseableFormatStyle` that renders a `TimeInterval` audio offset
/// via `formatClock(_:)` (adaptive M:SS.s / MM:SS.s / H:MM:SS) and
/// parses the same shapes back. Used by `EditUtteranceSheet`'s start /
/// end fields so the user sees `1:23:45.6` instead of `5025.60 s`
/// while still being able to nudge by `±0.1 s` via the buttons or
/// type a value directly.
///
/// Parser accepts:
///   - Plain decimal seconds (`123.45`) — backward-compat with the
///     prior raw-seconds field and the most natural input when
///     someone has the exact number.
///   - `MM:SS[.s]` (`12:34.5`)
///   - `H:MM:SS[.s]` (`1:23:45.6`)
///
/// Negative values throw — clamping happens at the binding layer.
struct ClockTimeFormatStyle: ParseableFormatStyle {
    var parseStrategy: ClockTimeParseStrategy { ClockTimeParseStrategy() }
    func format(_ value: TimeInterval) -> String { formatClock(value) }
}

struct ClockTimeParseStrategy: ParseStrategy {
    enum Failure: Error { case invalid }

    func parse(_ value: String) throws -> TimeInterval {
        let trimmed = value.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty else { throw Failure.invalid }
        let parts = trimmed.split(separator: ":", omittingEmptySubsequences: false)
        switch parts.count {
        case 1:
            guard let s = Double(parts[0]) else { throw Failure.invalid }
            return max(0, s)
        case 2:
            guard let m = Double(parts[0]),
                  let s = Double(parts[1]),
                  m >= 0, s >= 0 else { throw Failure.invalid }
            return m * 60 + s
        case 3:
            guard let h = Double(parts[0]),
                  let m = Double(parts[1]),
                  let s = Double(parts[2]),
                  h >= 0, m >= 0, s >= 0 else { throw Failure.invalid }
            return h * 3600 + m * 60 + s
        default:
            throw Failure.invalid
        }
    }
}

/// Format an audio-time offset (seconds) as a clock string. Adapts to length:
///   < 1 min  → `M:SS.s`     (e.g. `0:05.2`)
///   < 1 hour → `MM:SS.s`    (e.g. `12:34.5`)
///   ≥ 1 hour → `H:MM:SS`    (e.g. `1:23:45`)
/// Hour-or-greater drops the fractional part to keep the row compact;
/// sub-second precision rarely matters at that scale.
func formatClock(_ seconds: TimeInterval) -> String {
    let clamped = max(0, seconds)
    let total = Int(clamped)
    let h = total / 3600
    let m = (total % 3600) / 60
    let s = total % 60
    if h > 0 {
        return String(format: "%d:%02d:%02d", h, m, s)
    }
    let frac = clamped - Double(total)
    return String(format: "%d:%02d.%d", m, s, Int(frac * 10))
}

/// SI suffixes for raw sample counts. 16 kHz audio crosses millions in
/// minutes, so plain integers fill the status line — `123K` / `1.23M` is
/// easier to scan. Decimal precision tapers as magnitude grows.
func formatCount(_ n: Int) -> String {
    let v = Double(n)
    if abs(v) < 1_000          { return "\(n)" }
    if abs(v) < 1_000_000      { return formatSI(v / 1_000,         "K") }
    if abs(v) < 1_000_000_000  { return formatSI(v / 1_000_000,     "M") }
    return                       formatSI(v / 1_000_000_000, "G")
}

private func formatSI(_ v: Double, _ suffix: String) -> String {
    if v >= 100 { return String(format: "%.0f%@", v, suffix) }
    if v >= 10  { return String(format: "%.1f%@", v, suffix) }
    return        String(format: "%.2f%@", v, suffix)
}

/// Render a stored speaker ID for display. The diarizer-tracker
/// writes `S01`, `S02`, … internally; the display label is the
/// stored id verbatim, unless the user has supplied a custom name
/// (taken as-is).
func formatSpeakerLabel(
    _ stored: String,
    customName: String? = nil
) -> String {
    if let name = customName?.trimmingCharacters(in: .whitespacesAndNewlines),
       !name.isEmpty {
        return name
    }
    return stored
}

/// Per-speaker tint pulled from a small palette indexed by the speaker
/// number embedded in the stored ID (`S01` → 1). The palette is meant
/// to read as a row of colored chips at a glance; ordering matches the
/// rough warmth gradient (cool → warm → neutral) so consecutive
/// speakers stay distinguishable.
func speakerTint(for stored: String) -> Color {
    let palette: [Color] = [
        .blue, .orange, .green, .pink, .purple,
        .teal, .indigo, .brown, .mint, .red,
    ]
    let digits = stored.drop(while: { !$0.isNumber })
    let n = Int(digits) ?? 1
    let idx = max(0, n - 1) % palette.count
    return palette[idx]
}

/// Color tint for a fused emotion label. Tracks the conventional Plutchik
/// wheel where it overlaps; falls back to grey for unknown/neutral labels.
/// Shared by `UtteranceRow`, `SummaryCard`, and `StatisticsRow` so the
/// color mapping is consistent across the UI. Named `emotionTint` (not
/// `color`) to avoid clashing with implicit `color`-named members
/// SwiftUI exposes inside View closures.
func emotionTint(for raw: String) -> Color {
    switch raw.lowercased() {
    case "happy", "joy", "joyful":               return .yellow
    case "sad", "sadness":                       return .blue
    case "angry", "anger":                       return .red
    case "fear", "fearful", "afraid":            return .purple
    case "disgust", "disgusted":                 return Color(red: 0.45, green: 0.55, blue: 0.20)
    // Plutchik's wheel pairs surprise (cyan) opposite anticipation
    // (orange). The previous tint was system .orange, which sat on
    // top of anticipation's orange and made the two wedges of the
    // SER aggregate wheel indistinguishable. Cyan/light-blue is
    // Plutchik canonical and well-separated from both sadness
    // (.blue, darker / more saturated) and trust (.green).
    case "surprise", "surprised":                return Color(red: 0.20, green: 0.75, blue: 0.95)
    case "trust":                                return .green
    case "anticipation":                         return Color(red: 0.95, green: 0.55, blue: 0.10)
    case "neutral", "calm":                      return .gray
    default:                                     return Color.secondary
    }
}
