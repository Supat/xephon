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

    // Custom Codable so each label becomes its own JSON key — Swift's default
    // Dictionary encoding only treats `String`/`Int` keys as object keys, and
    // bails to a flat ["joy", 0.71, "sadness", 0.18, …] array for enum keys.
    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: DynamicKey.self)
        for label in Label.allCases {
            if let value = probabilities[label],
               let key = DynamicKey(stringValue: label.rawValue) {
                try container.encode(value, forKey: key)
            }
        }
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: DynamicKey.self)
        var probs: [Label: Float] = [:]
        for key in container.allKeys {
            if let label = Label(rawValue: key.stringValue),
               let value = try? container.decode(Float.self, forKey: key) {
                probs[label] = value
            }
        }
        self.probabilities = probs
    }
}

public protocol TextSER: Actor {
    func classify(_ text: String) async throws -> PlutchikScore
}

// Shared dynamic-key helper for emotion-dictionary Codable conformances.
struct DynamicKey: CodingKey {
    let stringValue: String
    var intValue: Int? { nil }

    init?(stringValue: String) {
        self.stringValue = stringValue
    }

    init?(intValue: Int) {
        return nil
    }
}
