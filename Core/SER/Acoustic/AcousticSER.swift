import Foundation
import Audio

// Dimensional V/A/D in [0, 1]. Matches the audeering W2V2 output range.
public struct VADScore: Sendable, Hashable, Codable {
    public let valence: Float
    public let arousal: Float
    public let dominance: Float

    public init(valence: Float, arousal: Float, dominance: Float) {
        self.valence = valence
        self.arousal = arousal
        self.dominance = dominance
    }
}

// 9-class softmax from emotion2vec+:
// angry, disgusted, fearful, happy, neutral, other, sad, surprised, unknown.
public struct CategoricalEmotion: Sendable, Hashable, Codable {
    public enum Label: String, Sendable, CaseIterable, Hashable, Codable {
        case angry, disgusted, fearful, happy, neutral, other, sad, surprised, unknown
    }
    public let probabilities: [Label: Float]

    public init(probabilities: [Label: Float]) {
        self.probabilities = probabilities
    }

    // See note on PlutchikScore: Swift's Dictionary Codable only treats
    // String/Int keys as object keys, so we flatten to named fields.
    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: AcousticDynamicKey.self)
        for label in Label.allCases {
            if let value = probabilities[label],
               let key = AcousticDynamicKey(stringValue: label.rawValue) {
                try container.encode(value, forKey: key)
            }
        }
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: AcousticDynamicKey.self)
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

public protocol DimensionalAcousticSER: Actor {
    func score(_ buffer: AudioChunk) async throws -> VADScore
}

public protocol CategoricalAcousticSER: Actor {
    func score(_ buffer: AudioChunk) async throws -> CategoricalEmotion
}

struct AcousticDynamicKey: CodingKey {
    let stringValue: String
    var intValue: Int? { nil }
    init?(stringValue: String) { self.stringValue = stringValue }
    init?(intValue: Int) { return nil }
}
