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

/// Per-utterance demographics estimate from the audeering W2V2
/// age-gender model. Carries the raw model outputs verbatim — view
/// + export layers can render either the regressed age in years
/// (`age * 100`) or the argmax gender label as they see fit.
public struct AgeGenderEstimate: Sendable, Hashable, Codable {
    public enum Gender: String, Sendable, Hashable, Codable, CaseIterable {
        case child, female, male
    }
    /// Regressed age normalized to `[0, 1]` (model trained against
    /// 0–100 years; multiply by 100 for years).
    public let age: Float
    /// Softmax distribution over the three gender classes,
    /// summing to 1.0.
    public let genderProbabilities: [Gender: Float]

    public init(age: Float, genderProbabilities: [Gender: Float]) {
        self.age = age
        self.genderProbabilities = genderProbabilities
    }

    /// Argmax label across the three classes. Nil only when the
    /// distribution is empty (defensive — a well-formed estimate
    /// has all three keys).
    public var topGender: Gender? {
        genderProbabilities.max(by: { $0.value < $1.value })?.key
    }

    /// Convenience: model output `age ∈ [0, 1]` mapped to years.
    public var ageYears: Float { age * 100 }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: AcousticDynamicKey.self)
        if let ageKey = AcousticDynamicKey(stringValue: "age") {
            try container.encode(age, forKey: ageKey)
        }
        for label in Gender.allCases {
            if let value = genderProbabilities[label],
               let key = AcousticDynamicKey(stringValue: label.rawValue) {
                try container.encode(value, forKey: key)
            }
        }
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: AcousticDynamicKey.self)
        var age: Float = 0
        var probs: [Gender: Float] = [:]
        for key in container.allKeys {
            if key.stringValue == "age",
               let v = try? container.decode(Float.self, forKey: key) {
                age = v
            } else if let label = Gender(rawValue: key.stringValue),
                      let v = try? container.decode(Float.self, forKey: key) {
                probs[label] = v
            }
        }
        self.age = age
        self.genderProbabilities = probs
    }
}

public protocol AgeGenderSER: Actor {
    func estimate(_ buffer: AudioChunk) async throws -> AgeGenderEstimate
}

struct AcousticDynamicKey: CodingKey {
    let stringValue: String
    var intValue: Int? { nil }
    init?(stringValue: String) { self.stringValue = stringValue }
    init?(intValue: Int) { return nil }
}
