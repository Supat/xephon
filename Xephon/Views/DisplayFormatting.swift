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

extension URL: @retroactive Identifiable {
    public var id: String { absoluteString }
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

/// Render a stored speaker ID for display. The diarizer-tracker writes
/// `S01`, `S02`, … internally; the row label uses an `M` prefix
/// ("multi") when there are two or more distinct speakers in the
/// session, leaving `S` ("single") for the simple case. Number is
/// preserved either way.
func formatSpeakerLabel(_ stored: String, multiSpeaker: Bool) -> String {
    let prefix = multiSpeaker ? "M" : "S"
    if let trail = stored.dropFirst().first, trail.isNumber {
        return "\(prefix)\(stored.dropFirst())"
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
    case "surprise", "surprised":                return .orange
    case "trust":                                return .green
    case "anticipation":                         return Color(red: 0.95, green: 0.55, blue: 0.10)
    case "neutral", "calm":                      return .gray
    default:                                     return Color.secondary
    }
}
