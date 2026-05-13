import Accelerate
import Foundation

/// Per-utterance zero-mean, unit-variance normalization — what
/// `Wav2Vec2FeatureExtractor` does when `do_normalize: true` is set in
/// `preprocessor_config.json`. Both audeering W2V2 checkpoints we use
/// (V/A/D and age-gender) ship with that flag enabled, so feeding raw
/// audio in directly silently shifts the input distribution off what
/// the model was trained on. For the age-gender head the symptom was
/// catastrophic (the softmax collapsed onto one class regardless of
/// speaker); for the V/A/D regression it's more subtle but still
/// out-of-distribution.
///
/// Matches the HF reference implementation byte-for-byte:
///     normed = (x - mean(x)) / sqrt(var(x) + 1e-7)
/// using *population* variance (divisor N), which is what NumPy's
/// `.var()` produces and what HF calls into.
enum Wav2Vec2Preprocess {
    static func normalize(_ samples: [Float]) -> [Float] {
        let n = vDSP_Length(samples.count)
        guard n > 0 else { return samples }
        var mean: Float = 0
        var meanSquare: Float = 0
        vDSP_meanv(samples, 1, &mean, n)
        vDSP_measqv(samples, 1, &meanSquare, n)
        let variance = max(meanSquare - mean * mean, 0)
        let invStd = 1.0 / sqrt(variance + 1e-7)
        var negMean = -mean
        var out = [Float](repeating: 0, count: samples.count)
        vDSP_vsadd(samples, 1, &negMean, &out, 1, n)
        var scale = invStd
        vDSP_vsmul(out, 1, &scale, &out, 1, n)
        return out
    }
}
