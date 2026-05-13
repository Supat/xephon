import Foundation

/// Helper for telling on-device LLMs which language to respond in.
/// The summarizer + reviewer paths both hit this when building their
/// prompts so the LLM's freeform output (topic / mood / per-speaker
/// paragraph / issue reason) matches the user's app-language pick
/// in iPadOS Settings instead of drifting toward Chinese (Qwen3) or
/// the prompt's English (Apple FM) regardless of audio language.
///
/// Reads `Bundle.main.preferredLocalizations.first` — the
/// localization actually resolved for this app launch, which
/// follows the user's iPadOS app-language override AND falls back
/// to system language when no override exists. This is the same
/// signal `String(localized:)` uses, so prompt language stays in
/// lock-step with the UI strings the user sees.
public enum SummarizerLocale {
    /// Human-readable language name, always *in English*, for the
    /// app's effective UI language. Returning the English-side
    /// name ("Japanese", "English") gives the LLM a stable token
    /// cue regardless of whether the app itself is in Japanese —
    /// embedding `日本語` in an English instruction line has been
    /// observed to cause Qwen to mirror back the script rather
    /// than treat it as a directive.
    public static var responseLanguageNameInEnglish: String {
        let code = Bundle.main.preferredLocalizations.first ?? "en"
        let englishLocale = Locale(identifier: "en")
        return englishLocale.localizedString(forLanguageCode: code)
            ?? code.capitalized
    }

    /// One-line directive ready to drop into a prompt. Phrased
    /// hard ("Respond in X. Use no other language under any
    /// circumstance.") because softer wording leaks Chinese on
    /// Qwen3 when the input data is Japanese.
    public static var responseLanguageInstruction: String {
        "Respond in \(responseLanguageNameInEnglish). Use no other language under any circumstance, regardless of the language of the input transcript."
    }
}
