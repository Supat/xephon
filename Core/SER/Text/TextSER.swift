import Foundation

// Plutchik 8 binary emotion vector for WRIME.
public struct PlutchikScore: Sendable, Hashable, Codable {
    public enum Label: String, Sendable, CaseIterable, Hashable, Codable {
        case joy, sadness, anticipation, surprise, anger, fear, disgust, trust
    }
    public let probabilities: [Label: Float]

    public init(probabilities: [Label: Float]) {
        self.probabilities = probabilities
    }
}

public protocol TextSER: Actor {
    func classify(_ text: String) async throws -> PlutchikScore
}
