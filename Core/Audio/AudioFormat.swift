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
    /// Per-source-channel perceptual level (0…1, dB-scaled RMS),
    /// measured BEFORE the downmix to mono. Empty when not provided
    /// (file capture, synthesized chunks). Used to drive a multi-bar
    /// level meter so a stereo USB mic shows L/R separately rather
    /// than a single mixed-down meter that hides imbalance.
    public let channelLevels: [Float]

    public init(
        samples: [Float],
        sampleRate: Double = PipelineAudio.sampleRate,
        timestamp: TimeInterval,
        channelLevels: [Float] = []
    ) {
        self.samples = samples
        self.sampleRate = sampleRate
        self.timestamp = timestamp
        self.channelLevels = channelLevels
    }
}
