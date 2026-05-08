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

    /// Backends actually available right now. If DeBERTa wasn't bundled, only
    /// `.foundationModels` is listed.
    public var availableBackends: [Backend] {
        if deberta != nil { return [.deberta, .foundationModels] }
        return [.foundationModels]
    }

    /// Effective backend after fallbacks are applied.
    public var currentBackend: Backend {
        if preferredBackend == .deberta && deberta == nil {
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
