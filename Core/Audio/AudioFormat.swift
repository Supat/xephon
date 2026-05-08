import Foundation

// 16 kHz mono Float32 throughout the pipeline. Resample at capture, never mid-pipeline.
public enum PipelineAudio {
    public static let sampleRate: Double = 16_000
    public static let channelCount: Int = 1
}

public struct AudioChunk: Sendable {
    public let samples: [Float]
    public let sampleRate: Double
    public let timestamp: TimeInterval

    public init(samples: [Float], sampleRate: Double = PipelineAudio.sampleRate, timestamp: TimeInterval) {
        self.samples = samples
        self.sampleRate = sampleRate
        self.timestamp = timestamp
    }
}
