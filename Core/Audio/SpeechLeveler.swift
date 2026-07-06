import Foundation
@preconcurrency import AVFoundation
import AudioToolbox

/// AGC-style speech leveler for the ASR-bound branch of the capture
/// graph — Apple's DynamicsProcessor AudioUnit configured as a
/// compressor with makeup gain. Lifts quiet / distant speakers toward
/// a stable level so the transcriber stops missing speech the
/// diarizer (raw branch) can clearly see, while squashing loud
/// speakers instead of clipping them.
///
/// Level-domain processing only: unlike neural denoising, broadband
/// gain adds no spectral artifacts, so it sidesteps the
/// artifact-degrades-robust-ASR failure mode documented in
/// docs/speech_enhancement_research.md. The raw branch (SER,
/// diarizer, speaker embeddings) stays untouched — energy is an
/// arousal cue for the SER models, and the embeddings path must not
/// see processed audio.
///
/// Starting curve (tune against the held-out calibration set per
/// CLAUDE.md's eval rule; log deltas to docs/eval_log.md):
///   threshold  −35 dB — compression kicks in above this, so normal
///                       speech (≈ −25 dB RMS) is gently leveled and
///                       loud speech is firmly held down
///   head room    6 dB — soft-ish knee; peaks land ≈ threshold+headroom
///   attack     5 ms   — fast enough to catch plosives
///   release  150 ms   — slow enough not to pump between syllables
///   makeup    +12 dB  — the actual "boost": quiet speech below the
///                       threshold passes at unity and gains the full
///                       +12; leveled peaks stay well under 0 dBFS
///   expansion  1:1    — defeats the downward expander: a gate would
///                       re-attenuate exactly the quiet speech this
///                       node exists to lift
public enum SpeechLeveler {
    public static func makeLeveler() -> AVAudioUnitEffect {
        let desc = AudioComponentDescription(
            componentType: kAudioUnitType_Effect,
            componentSubType: kAudioUnitSubType_DynamicsProcessor,
            componentManufacturer: kAudioUnitManufacturer_Apple,
            componentFlags: 0,
            componentFlagsMask: 0
        )
        let unit = AVAudioUnitEffect(audioComponentDescription: desc)
        let au = unit.audioUnit
        func set(_ param: AudioUnitParameterID, _ value: AudioUnitParameterValue) {
            AudioUnitSetParameter(au, param, kAudioUnitScope_Global, 0, value, 0)
        }
        set(kDynamicsProcessorParam_Threshold, -35)
        set(kDynamicsProcessorParam_HeadRoom, 6)
        set(kDynamicsProcessorParam_AttackTime, 0.005)
        set(kDynamicsProcessorParam_ReleaseTime, 0.15)
        set(kDynamicsProcessorParam_OverallGain, 12)
        set(kDynamicsProcessorParam_ExpansionRatio, 1)
        return unit
    }
}
