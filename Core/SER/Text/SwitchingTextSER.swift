import Foundation
import XephonLogging

/// Runtime-switchable text SER. Holds both the bundled DeBERTa-WRIME
/// (`RoBERTa-base` today, see `DeBERTaWRIME.swift`) and the Apple Foundation
/// Models fallback, and forwards `classify(_:)` to whichever backend is
/// currently selected. Backend changes are honored on the next call.
public actor SwitchingTextSER: TextSER {
    public enum Backend: String, Sendable, Hashable, CaseIterable, Codable {
        case deberta
        case foundationModels
    }

    private var preferredBackend: Backend
    private let deberta: (any TextSER)?
    private let foundationModels: any TextSER
    /// ISO-639 language code the current session is targeting (e.g.
    /// `"ja"`, `"en"`). DeBERTa-WRIME is Japanese-only, so when the
    /// language is anything else, it drops out of `availableBackends`
    /// and `currentBackend` falls back to `.foundationModels`
    /// regardless of `preferredBackend`. Nil treats the session as
    /// Japanese (the original default) for backward compatibility.
    private var sessionLanguageCode: String?

    /// Whether the language-specific text SER (DeBERTa-WRIME) is
    /// usable for the active session. DeBERTa is fine-tuned on
    /// Japanese tweet emotion data and doesn't transfer to other
    /// languages; gating it here keeps the rest of the pipeline
    /// agnostic to that fact.
    private var debertaIsLanguageMatched: Bool {
        guard let code = sessionLanguageCode else { return true }
        return code.lowercased() == "ja"
    }

    /// Backends actually available right now. Drops `.deberta` when
    /// the model wasn't bundled OR when the session language isn't
    /// Japanese.
    public var availableBackends: [Backend] {
        if deberta != nil, debertaIsLanguageMatched {
            return [.deberta, .foundationModels]
        }
        return [.foundationModels]
    }

    /// Effective backend after fallbacks are applied.
    public var currentBackend: Backend {
        if preferredBackend == .deberta && (deberta == nil || !debertaIsLanguageMatched) {
            return .foundationModels
        }
        return preferredBackend
    }

    public init(
        deberta: (any TextSER)? = nil,
        foundationModels: any TextSER,
        initial: Backend? = nil
    ) {
        self.deberta = deberta
        self.foundationModels = foundationModels
        self.preferredBackend = initial ?? (deberta == nil ? .foundationModels : .deberta)
    }

    public func setBackend(_ backend: Backend) {
        preferredBackend = backend
        AppLog.serText.info("Text SER backend → \(backend.rawValue, privacy: .public)")
    }

    /// Tell the switcher which language the current session is in.
    /// Forwards to `FoundationModelsSER.setLanguage` so the prompt
    /// opener follows along (Japanese, English, …), and re-evaluates
    /// the DeBERTa gating on the next `availableBackends` /
    /// `currentBackend` query. Pass `nil` to revert to the legacy
    /// "treat as Japanese" behavior.
    public func setLanguage(code: String?, label: String?) async {
        sessionLanguageCode = code
        await (foundationModels as? FoundationModelsSER)?.setLanguage(label)
        AppLog.serText.info(
            "Text SER language → \(code ?? "nil", privacy: .public)"
        )
    }

    public func classify(_ text: String) async throws -> PlutchikScore {
        switch currentBackend {
        case .deberta:
            // currentBackend already gates this — deberta is non-nil here.
            return try await deberta!.classify(text)
        case .foundationModels:
            return try await foundationModels.classify(text)
        }
    }
}
