import Foundation

/// Sample-domain counterpart of `SpeechLeveler` (the live engine's
/// DynamicsProcessor node) for ASR-bound audio that never runs
/// through the live AVAudioEngine graph: `AudioFileCapture`'s
/// processed stream (file-opened analysis) and the re-evaluation /
/// hand-edit ASR re-feeds.
///
/// Feed-forward compressor with makeup gain on 16 kHz mono Float32
/// samples. The curve mirrors `SpeechLeveler`'s intent — threshold
/// −35 dB, ~6:1 above threshold (approximating the AU's 6 dB head
/// room), attack 5 ms, release 150 ms, makeup +12 dB, no downward
/// expansion. Not bit-identical to the AudioUnit, but close enough
/// that live-pass and re-fed ASR hear similarly levelled speech.
///
/// Scope discipline (same as the live graph): ASR inputs ONLY.
/// SER / fusion re-runs, speaker-embedding extraction, and the
/// diarizer always receive raw audio — energy is an arousal cue and
/// embeddings must not see processed samples (see
/// docs/speech_enhancement_research.md).
///
/// Streaming-capable: envelope state persists across `process`
/// calls, so hold one instance per stream and feed consecutive
/// chunks. NOT Sendable — confine each instance to one task/actor.
public final class SoftwareSpeechLeveler {
    private let thresholdDB: Float = -35
    private let ratio: Float = 6
    private let makeupDB: Float = 12
    private let attackCoeff: Float
    private let releaseCoeff: Float
    /// Running envelope in dB. Seeded at silence so a fresh instance
    /// treats the first samples as quiet — full makeup gain, settling
    /// within roughly one attack constant (~5 ms).
    private var envelopeDB: Float = -80

    public init(sampleRate: Double) {
        // One-pole smoothing: env += k·(level − env), with
        // k = 1 − e^(−1 / (τ·fs)).
        attackCoeff = 1 - exp(Float(-1 / (0.005 * sampleRate)))
        releaseCoeff = 1 - exp(Float(-1 / (0.15 * sampleRate)))
    }

    /// Level one chunk of mono samples. O(n) scalar loop — at 16 kHz
    /// this is negligible next to the ASR inference it feeds, even at
    /// file-mode's faster-than-realtime pump rates.
    public func process(_ samples: [Float]) -> [Float] {
        var out = samples
        for i in 0..<out.count {
            let levelDB = 20 * log10(max(abs(out[i]), 1e-6))
            let k = levelDB > envelopeDB ? attackCoeff : releaseCoeff
            envelopeDB += k * (levelDB - envelopeDB)
            let over = envelopeDB - thresholdDB
            let reductionDB = over > 0 ? over * (1 - 1 / ratio) : 0
            out[i] *= pow(10, (makeupDB - reductionDB) / 20)
        }
        return out
    }

    /// One-shot convenience for slice-based ASR re-feeds
    /// (re-evaluation, hand-edit re-transcription): a copy of `chunk`
    /// with levelled samples and identical timing/metadata. Fresh
    /// envelope per call — the ~5 ms settle at the slice head is
    /// inaudible to the recognizer.
    public static func levelledCopy(of chunk: AudioChunk) -> AudioChunk {
        let leveler = SoftwareSpeechLeveler(sampleRate: chunk.sampleRate)
        return AudioChunk(
            samples: leveler.process(chunk.samples),
            sampleRate: chunk.sampleRate,
            timestamp: chunk.timestamp,
            channelLevels: chunk.channelLevels
        )
    }
}
