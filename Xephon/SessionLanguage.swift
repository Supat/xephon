import Foundation

/// Languages the session can target. Drives the ASR locale, the
/// Foundation Models prompt opener, and the per-language gating of
/// the DeBERTa-WRIME text SER (Japanese-only). The user picks one
/// from the Settings card; the choice survives across launches via
/// `UserDefaults`.
///
/// Kept as a small standalone enum (rather than nested inside
/// `RecordingController`) so views, persistence, and the pipeline
/// can all reference it without pulling the controller into module
/// boundaries that don't otherwise need it.
public enum SessionLanguage: String, CaseIterable, Sendable, Codable {
    case japanese = "ja"
    case english = "en"

    /// `Locale` handed to the Apple SpeechTranscriber + offline
    /// SpeechAnalyzerTranscriber. We use BCP-47 region-tagged forms
    /// so SpeechTranscriber resolves to a concrete shipped variant
    /// rather than the unspecified language root.
    public var locale: Locale {
        switch self {
        case .japanese: return Locale(identifier: "ja_JP")
        case .english:  return Locale(identifier: "en_US")
        }
    }

    /// Human-readable label baked into the FoundationModelsSER
    /// prompt opener ("You are a Japanese-language affect annotator.").
    /// Keep these in English regardless of the user's UI locale —
    /// the prompt itself is in English and Apple FM responds in
    /// kind.
    public var label: String {
        switch self {
        case .japanese: return "Japanese"
        case .english:  return "English"
        }
    }

    /// User-facing display name for the language picker, localized
    /// via Localizable.strings so the chip reads in the user's UI
    /// language.
    public var displayName: String {
        switch self {
        case .japanese: return String(localized: "language.japanese")
        case .english:  return String(localized: "language.english")
        }
    }

    /// UserDefaults key under which the user's last-chosen session
    /// language is persisted.
    private static let defaultsKey = "xephon.sessionLanguage"

    /// Load the persisted language, or fall back to `.japanese`
    /// (the original default) when nothing is stored or the stored
    /// value is unrecognized (e.g. a future build wrote a new case).
    public static func loadFromDefaults() -> SessionLanguage {
        let raw = UserDefaults.standard.string(forKey: defaultsKey) ?? ""
        return SessionLanguage(rawValue: raw) ?? .japanese
    }

    /// Persist this choice so the next launch starts at the same
    /// language. Cheap; called from the controller's setter.
    public func saveToDefaults() {
        UserDefaults.standard.set(rawValue, forKey: Self.defaultsKey)
    }
}
