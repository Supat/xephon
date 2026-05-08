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
}

public protocol DimensionalAcousticSER: Actor {
    func score(_ buffer: AudioChunk) async throws -> VADScore
}

public protocol CategoricalAcousticSER: Actor {
    func score(_ buffer: AudioChunk) async throws -> CategoricalEmotion
}
