import Foundation
@preconcurrency import AVFoundation
import XephonLogging

/// Short UI feedback cues, played via a freshly-constructed
/// `AVAudioPlayer` per call from `/System/Library/Audio/UISounds/`
/// files. Falls back to a self-generated WAV (sine burst written
/// to temp dir) if no candidate system file exists on this iOS
/// version.
///
/// Earlier iterations with system files were silently muted because
/// the player was **cached** at first use and re-played later. The
/// cached `prepareToPlay`'d state appears to get invalidated when
/// the audio session deactivates / reactivates between recording
/// and re-eval contexts — same hazard the existing
/// `RecordingController.togglePlayback` explicitly avoids by
/// constructing a fresh `AVAudioPlayer` per tap. This file now
/// matches that lifecycle exactly:
///   1. Resolve a stable source URL once (system file or temp WAV).
///   2. Construct a fresh `AVAudioPlayer(contentsOf: url)` on each
///      play call.
///   3. Retain the player on `activePlayer` BEFORE calling
///      `play()` — ARC could otherwise dealloc the local
///      reference on the next line.
///   4. Session in `.playback`, `setActive(true)`.
@MainActor
enum UISounds {
    /// Revert long-press confirmation. `Tink.caf` is the canonical
    /// soft tap; it's the right "subtle action committed" feel for
    /// a held-button revert.
    static func playRevert() {
        play(url: revertURL, label: "revert")
    }

    /// Recording start. `begin_record.caf` is iOS's canonical
    /// recording-start chime.
    static func playRecordingStart() {
        play(url: recordingStartURL, label: "recordingStart")
    }

    /// Recording stop. `end_record.caf` is the canonical
    /// recording-stop chime.
    static func playRecordingStop() {
        play(url: recordingStopURL, label: "recordingStop")
    }

    /// Retain the active player so ARC doesn't dealloc it before
    /// `play()` finishes producing audio.
    private static var activePlayer: AVAudioPlayer?

    private static let revertURL: URL? = resolveSource(
        name: "revert",
        candidates: [
            "/System/Library/Audio/UISounds/Tink.caf",
            "/System/Library/Audio/UISounds/begin_record.caf",
        ],
        fallbackFrequency: 1200,
        fallbackDurationSec: 0.10
    )

    private static let recordingStartURL: URL? = resolveSource(
        name: "recordingStart",
        candidates: [
            "/System/Library/Audio/UISounds/begin_record.caf",
            "/System/Library/Audio/UISounds/begin_video_record.caf",
        ],
        fallbackFrequency: 880,
        fallbackDurationSec: 0.18
    )

    private static let recordingStopURL: URL? = resolveSource(
        name: "recordingStop",
        candidates: [
            "/System/Library/Audio/UISounds/end_record.caf",
            "/System/Library/Audio/UISounds/end_video_record.caf",
        ],
        fallbackFrequency: 440,
        fallbackDurationSec: 0.20
    )

    /// Return the first existing system file path, or — if none
    /// resolve on this iOS version — synthesize a sine-burst WAV
    /// in temp dir and return that URL as the fallback source.
    private static func resolveSource(
        name: String,
        candidates: [String],
        fallbackFrequency: Double,
        fallbackDurationSec: Double
    ) -> URL? {
        for path in candidates {
            if FileManager.default.fileExists(atPath: path) {
                let url = URL(fileURLWithPath: path)
                AppLog.app.info(
                    "UISounds[\(name, privacy: .public)]: system source \(path, privacy: .public)"
                )
                return url
            }
        }
        AppLog.app.info(
            "UISounds[\(name, privacy: .public)]: no system file present; falling back to synthesized WAV"
        )
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("xephon-cue-\(name).wav")
        do {
            let data = sineBurstWAV(
                frequency: fallbackFrequency,
                durationSec: fallbackDurationSec
            )
            try data.write(to: url, options: [.atomic])
            return url
        } catch {
            AppLog.app.warning(
                "UISounds[\(name, privacy: .public)]: fallback WAV write failed: \(String(describing: error), privacy: .public)"
            )
            return nil
        }
    }

    /// Hot path: configure session, construct fresh player, retain,
    /// play. Mirrors `RecordingController.togglePlayback` step for
    /// step.
    private static func play(url: URL?, label: String) {
        let session = AVAudioSession.sharedInstance()
        let priorCategory = session.category
        let priorOutputs = session.currentRoute.outputs
            .map(\.portType.rawValue)
            .joined(separator: ",")
        AppLog.app.info(
            "UISounds[\(label, privacy: .public)]: play priorCategory=\(priorCategory.rawValue, privacy: .public) outputs=\(priorOutputs, privacy: .public)"
        )
        do {
            try session.setCategory(.playback, mode: .default, options: [])
            try session.setActive(true, options: [])
        } catch {
            AppLog.app.warning(
                "UISounds[\(label, privacy: .public)]: session setup failed: \(String(describing: error), privacy: .public)"
            )
        }
        guard let url else {
            AppLog.app.warning(
                "UISounds[\(label, privacy: .public)]: no source URL"
            )
            return
        }
        do {
            let player = try AVAudioPlayer(contentsOf: url)
            player.volume = 1.0
            // Retain BEFORE play(). ARC could otherwise dealloc
            // `player` on the next line — the immediate symptom is
            // `play()` returning true with no audible output.
            // Existing `togglePlayback` solves this the same way.
            activePlayer = player
            player.prepareToPlay()
            let didPlay = player.play()
            AppLog.app.info(
                "UISounds[\(label, privacy: .public)]: play() returned \(didPlay, privacy: .public) duration=\(player.duration, privacy: .public)s"
            )
        } catch {
            AppLog.app.warning(
                "UISounds[\(label, privacy: .public)]: AVAudioPlayer init failed: \(String(describing: error), privacy: .public)"
            )
        }
    }

    /// Build a mono 16-bit PCM WAV with a sine burst and
    /// exponential-decay envelope. Used only when no system audio
    /// file resolves on this iOS version.
    private static func sineBurstWAV(
        frequency: Double,
        durationSec: Double,
        sampleRate: UInt32 = 44100
    ) -> Data {
        let frameCount = Int(Double(sampleRate) * durationSec)
        var samples = [Int16](repeating: 0, count: frameCount)
        let twoPiF = 2.0 * .pi * frequency
        for i in 0..<frameCount {
            let t = Double(i) / Double(sampleRate)
            let envelope = exp(-t * 6.0)
            let value = sin(twoPiF * t) * envelope * 0.7
            samples[i] = Int16(max(-1.0, min(1.0, value)) * 32767)
        }

        let bitsPerSample: UInt16 = 16
        let channels: UInt16 = 1
        let byteRate = sampleRate * UInt32(channels) * UInt32(bitsPerSample / 8)
        let blockAlign = channels * (bitsPerSample / 8)
        let dataSize = UInt32(frameCount) * UInt32(blockAlign)
        let chunkSize = 36 + dataSize

        var out = Data()
        out.append(contentsOf: "RIFF".utf8)
        out.append(le32(chunkSize))
        out.append(contentsOf: "WAVE".utf8)
        out.append(contentsOf: "fmt ".utf8)
        out.append(le32(16))
        out.append(le16(1))
        out.append(le16(channels))
        out.append(le32(sampleRate))
        out.append(le32(byteRate))
        out.append(le16(blockAlign))
        out.append(le16(bitsPerSample))
        out.append(contentsOf: "data".utf8)
        out.append(le32(dataSize))
        samples.withUnsafeBufferPointer { bufPtr in
            let raw = UnsafeRawBufferPointer(bufPtr)
            out.append(contentsOf: raw)
        }
        return out
    }

    private static func le16(_ v: UInt16) -> Data {
        var le = v.littleEndian
        return Data(bytes: &le, count: 2)
    }

    private static func le32(_ v: UInt32) -> Data {
        var le = v.littleEndian
        return Data(bytes: &le, count: 4)
    }
}
