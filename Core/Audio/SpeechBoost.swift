import Foundation
@preconcurrency import AVFoundation

/// Speech-band EQ for the ASR-bound branch of the capture graph. The raw
/// branch (used by SER) is intentionally unprocessed — paralinguistic cues
/// like jitter, shimmer, and spectral tilt that the dimensional SER relies on
/// would be flattened by aggressive enhancement.
///
/// Curve (4 parametric bands, applied in order):
///   80 Hz  high-pass   — removes rumble, HVAC hum, table thuds
///   250 Hz low-shelf   −3 dB, reduces room boom
///   2.5 kHz peak       +6 dB, Q≈1.0 — sharpens consonants
///   8 kHz  low-pass    — trims hiss above the speech band
public enum SpeechBoost {
    public static func makeEQ() -> AVAudioUnitEQ {
        let eq = AVAudioUnitEQ(numberOfBands: 4)

        let highPass = eq.bands[0]
        highPass.filterType = .highPass
        highPass.frequency = 80
        highPass.bypass = false

        let lowShelf = eq.bands[1]
        lowShelf.filterType = .lowShelf
        lowShelf.frequency = 250
        lowShelf.gain = -3.0
        lowShelf.bypass = false

        let presence = eq.bands[2]
        presence.filterType = .parametric
        presence.frequency = 2500
        presence.gain = 6.0
        presence.bandwidth = 1.0
        presence.bypass = false

        let lowPass = eq.bands[3]
        lowPass.filterType = .lowPass
        lowPass.frequency = 8000
        lowPass.bypass = false

        eq.globalGain = 0
        return eq
    }
}
