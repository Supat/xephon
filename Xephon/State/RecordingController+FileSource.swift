import Foundation
@preconcurrency import AVFoundation
import Audio
import XephonLogging

// File-backed source mode split out of RecordingController.swift.
// Two paired entry points: `startFromFile` swaps in an
// `AudioFileCapture` and kicks the analysis loop; `resetToMicrophone`
// undoes that swap. Same cross-file extension pattern as the
// other splits.
extension RecordingController {



    /// Switch to a file-backed audio source and immediately begin
    /// streaming it through the same pipeline used for the microphone.
    /// File analysis runs **non-realtime** under the buffer-pipeline
    /// pump: chunks emit as fast as downstream consumers can swallow,
    /// with retry-on-drop backpressure. Chunk timestamps are still
    /// anchored to the source file's audio-time axis, so the pipeline
    /// sees a continuous monotonic file-time clock — see
    /// `RollingAudioBuffer`'s anchor-based mapping for the detail.
    func startFromFile(_ url: URL) async {
        guard phase == .idle else { return }
        // Acquire the playback scope synchronously, before any await
        // hop. The picker's implicit grant for the URL is freshest
        // right after the dialog dismisses; by the time the analysis
        // task and AudioFileCapture's own scope ref have run, opening
        // a new ref later (e.g. when the user taps Playback) can fail
        // with permErr (-54). Holding our own ref here keeps the URL
        // readable for the lifetime of `playbackSourceURL`.
        setPlaybackSourceURL(url)
        // Probe the file's length so the status line can render a
        // completion percentage. AVAudioFile open is cheap and is
        // immediately discarded — the capture pump opens its own.
        // Falls back to nil when the file can't be parsed; the UI
        // hides the percentage in that case.
        fileTotalAudioDuration = {
            guard let f = try? AVAudioFile(forReading: url) else { return nil }
            let rate = f.processingFormat.sampleRate
            guard rate > 0, f.length > 0 else { return nil }
            return TimeInterval(Double(f.length) / rate)
        }()
        sourceMode = .file(url)
        // File analysis runs as fast as the consumers can drain — see
        // AudioFileCapture's doc-comment. `asrLatencyMeaningful` stays
        // false here because chunk timestamps (file-time) decouple
        // from wall-clock when the pump runs faster than 1×.
        asrLatencyMeaningful = false
        capture = AudioFileCapture(fileURL: url)
        availableInputs = []
        currentInputUID = nil
        await start()
    }

    /// Restore the microphone as the active source. Called automatically when
    /// a file-backed session ends.
    func resetToMicrophone() async {
        guard phase == .idle else { return }
        sourceMode = .microphone
        capture = micCapture
        await refreshInputs()
    }

}
