import Foundation
@preconcurrency import AVFoundation
import Audio
import ASR
import Diarization
import Fusion
import XephonLogging
import XephonUtilities

// Per-utterance re-evaluation flow split out of
// RecordingController.swift to keep the controller's source file
// readable. Same cross-file extension pattern as
// `RecordingController+HandEdit.swift` — no own state, just
// operations over `utterances`, `utteranceEmbeddings`, and the
// pipeline. The trailing `revertReevaluation`,
// `isChronologicallyOrdered`, and `readAudioChunkForReevaluation`
// helpers move with the section because they're tightly coupled
// to re-evaluation (revert is the inverse; the static helpers
// support the audio + sort paths). The static reader is shared
// with `RecordingController+HandEdit.swift` — both call sites
// reach it via the type, so locality doesn't matter.
extension RecordingController {



    /// Front-only padding applied before the original utterance's
    /// start when re-feeding audio to offline ASR. The streaming
    /// pass's finalizer cuts segments at the volatile-stabilization
    /// boundary, which often clips the first phoneme of an utterance;
    /// 500 ms of lead-in gives offline ASR a chance to recover it.
    /// No back padding is applied in the long-utterance path — the
    /// segment's tail is already preserved by streaming, and the
    /// sentence-aware trim in `AnalysisPipeline.reevaluate` drops
    /// anything past the last terminator anyway.
    static let reevaluationPaddingSec: TimeInterval = 0.5

    /// USB-C audio plug/unplug polling cadence while idle — there's
    /// no public notification for it, so we diff `availableInputs`
    /// at this rate. 2 s is comfortably below human reaction time
    /// for a "plug it in, tap record" workflow.
    static let inputPollIntervalSec: TimeInterval = 2.0
    /// Continuous-diarize outer-loop tick. Diarize fires on every
    /// stride boundary; this is just the polling cadence between
    /// checks, so it can be much finer than the stride itself.
    static let continuousDiarizeTickSec: TimeInterval = 0.1
    /// Wait when the ASR pump is ahead of the diarize cursor by
    /// more than `maxDiarizeLagSeconds`. Sleeping a beat here lets
    /// diarize catch up instead of growing the buffer unbounded.
    static let diarizeBackpressurePollSec: TimeInterval = 0.1
    /// `volatileText` UI poll cadence — 5 Hz reads fluid without
    /// taxing the analyzer actor.
    static let volatilePumpIntervalSec: TimeInterval = 0.2
    /// Drain wait while `stop()` waits for the continuous-diarize
    /// task to cover the final captured audio.
    static let diarizeDrainPollSec: TimeInterval = 0.2
    /// Slot-availability poll for the bounded concurrent-SER pool.
    /// Tiny because SER tasks complete in tens of ms typically.
    static let serSlotWaitSec: TimeInterval = 0.005

    /// Below this duration, the original streaming utterance is
    /// probably an incomplete fragment (the volatile-stabilization
    /// boundary cut mid-sentence). Re-evaluate then enters a retry
    /// loop, growing the back pad in steps until offline ASR
    /// produces a transcript containing a sentence terminator.
    private static let shortUtteranceThresholdSec: TimeInterval = 1.0
    /// Initial and per-iteration step for back padding when retrying
    /// the short-utterance case.
    private static let reevaluationBackPadStepSec: TimeInterval = 1.0
    /// Hard cap on back padding so a recording with no clean sentence
    /// boundary anywhere ahead doesn't keep growing the read forever.
    /// 10 s is comfortably longer than any realistic Japanese
    /// conversational sentence.
    private static let reevaluationMaxBackPadSec: TimeInterval = 10.0

    /// Surrounding-context pad (each side) for the audio chunk fed
    /// to the diarizer when re-resolving a speaker after a hand-
    /// edit commit or re-evaluation. Sortformer's input
    /// `chunkDuration` is 10 s; a typical edited slice is 2–5 s,
    /// which gets zero-padded — `createSegmentIfValid` then drops
    /// segments whose duration falls under `minSpeechDuration`,
    /// causing `runDiarization` to return empty. Giving the model
    /// real surrounding audio (the same audio it ran on during
    /// streaming) reliably yields segments. We still vote for
    /// speakers in the user-supplied range only; the wider window
    /// just gives Sortformer something to chew on. ~4 s on each
    /// side brings most slices comfortably above the 10 s window.
    static let rediarizePadSec: TimeInterval = 4.0

    /// Per-call invariants shared by the long/short paths and the
    /// finalize step. Bundled so each helper takes one context
    /// argument instead of seven matching positional params.
    private struct ReevaluationContext {
        let utteranceID: UUID
        let pipeline: AnalysisPipeline
        let url: URL
        let originalStart: TimeInterval
        let originalEnd: TimeInterval
        let speakerID: String
        let volatileHandler: @Sendable @MainActor (String) -> Void
    }

    /// Re-feed the utterance's audio (padded by `reevaluationPaddingSec`
    /// on each side) to offline ASR, then run SER + fusion on the new
    /// result and replace the utterance in `utterances` in place. The
    /// utterance's `id`, `start`, `end`, `speakerID`, and `speechBoost`
    /// are preserved so list position, selection, and Save/Load
    /// identity all hold across the re-evaluation.
    ///
    /// No-op when there's no source audio (mic mode), when the
    /// session is mid-recording / mid-analysis, or when another
    /// re-evaluation is already in flight. The button gates these
    /// in the UI; the guards here are defense-in-depth.
    func reevaluate(_ utterance: UtteranceEstimate) async {
        guard reevaluatingUtteranceID == nil else { return }
        guard phase == .idle else { return }
        guard let url = playbackSourceURL else { return }

        reevaluatingUtteranceID = utterance.id
        volatileText = ""
        defer {
            reevaluatingUtteranceID = nil
            volatileText = ""
        }

        // Anchor padding to the *truly-original* utterance bounds —
        // the pre-first-reeval snapshot when one exists, otherwise
        // the current row's start/end (which on the first pass *is*
        // the original). Each re-evaluation produces a corrected
        // start/end; sourcing padding anchors from the snapshot
        // keeps re-evaluation idempotent (without this, a retry
        // anchored to the corrected values would compound the shift
        // and grow the utterance's duration without bound).
        let snapshot = preReevaluationSnapshots[utterance.id]
        let ctx = ReevaluationContext(
            utteranceID: utterance.id,
            pipeline: await ensurePipeline(),
            url: url,
            originalStart: snapshot?.start ?? utterance.start,
            originalEnd: snapshot?.end ?? utterance.end,
            speakerID: utterance.speakerID,
            volatileHandler: { [weak self] text in
                // Stream the offline ASR's rolling hypothesis into
                // the same `volatileText` slot the live ASR uses so
                // the pipeline panel's preview animates the same way
                // during a re-evaluation as during recording.
                self?.volatileText = text
            }
        )
        let originalDuration = ctx.originalEnd - ctx.originalStart

        do {
            let result: (fresh: UtteranceEstimate, chunk: AudioChunk)?
            if originalDuration >= Self.shortUtteranceThresholdSec {
                result = try await reevaluateLong(ctx: ctx)
            } else {
                AppLog.app.info(
                    "reevaluate: short utterance (\(originalDuration, privacy: .public)s) — entering back-pad retry loop"
                )
                result = try await reevaluateShortWithRetry(ctx: ctx)
            }
            guard let (fresh, chunk) = result else { return }
            await finalizeReevaluation(fresh: fresh, chunk: chunk, ctx: ctx)
        } catch {
            AppLog.app.error("reevaluate failed: \(String(describing: error), privacy: .public)")
        }
    }

    /// Long-path body: single read of `[extendedStart, originalEnd]`,
    /// single `pipeline.reevaluate` call. Returns nil on empty audio
    /// or no-transcript.
    private func reevaluateLong(
        ctx: ReevaluationContext
    ) async throws -> (fresh: UtteranceEstimate, chunk: AudioChunk)? {
        let extendedStart = max(0, ctx.originalStart - Self.reevaluationPaddingSec)
        let url = ctx.url
        let chunk = try await Task.detached(priority: .userInitiated) {
            try Self.readAudioChunkForReevaluation(
                fileURL: url, start: extendedStart, end: ctx.originalEnd
            )
        }.value
        guard !chunk.samples.isEmpty else {
            AppLog.app.warning("reevaluate: extended audio range was empty")
            return nil
        }
        guard let (fresh, _) = try await ctx.pipeline.reevaluate(
            audio: chunk,
            originalStart: ctx.originalStart,
            originalEnd: ctx.originalEnd,
            speakerID: ctx.speakerID,
            onVolatileText: ctx.volatileHandler
        ) else {
            AppLog.app.warning("reevaluate: offline ASR returned no transcript")
            return nil
        }
        return (fresh, chunk)
    }

    /// Short-path body: grow back-padding in steps and rerun offline
    /// ASR until the transcript contains a sentence terminator (or
    /// the pad cap is reached, or the file is exhausted). No front
    /// pad on this path — when the original is shorter than 1 s the
    /// streaming pass usually finalized late so `start` already sits
    /// well inside the sentence; adding lead-in drags in the
    /// previous sentence's tail and its terminator would fool the
    /// `segmentsContainFullSentence` check.
    private func reevaluateShortWithRetry(
        ctx: ReevaluationContext
    ) async throws -> (fresh: UtteranceEstimate, chunk: AudioChunk)? {
        var backPad: TimeInterval = Self.reevaluationBackPadStepSec
        var previousSampleCount = -1
        var matchedSegments: [ASRSegment]?
        var matchedAudio: AudioChunk?

        while backPad <= Self.reevaluationMaxBackPadSec {
            let currentEnd = ctx.originalEnd + backPad
            let url = ctx.url
            let originalStart = ctx.originalStart
            let chunk = try await Task.detached(priority: .userInitiated) {
                try Self.readAudioChunkForReevaluation(
                    fileURL: url, start: originalStart, end: currentEnd
                )
            }.value
            if chunk.samples.isEmpty {
                AppLog.app.warning("reevaluate: empty read at backPad=\(backPad, privacy: .public)s")
                break
            }
            if chunk.samples.count == previousSampleCount {
                // File-end clamping returned the same audio as the
                // previous iteration; growing further just repeats.
                AppLog.app.info(
                    "reevaluate: file exhausted at backPad=\(backPad, privacy: .public)s; stopping retry"
                )
                break
            }
            previousSampleCount = chunk.samples.count

            let segments = try await ctx.pipeline.transcribeForReevaluation(
                audio: chunk, onVolatileText: ctx.volatileHandler
            )
            if AnalysisPipeline.segmentsContainFullSentence(segments) {
                AppLog.app.info(
                    "reevaluate: found full sentence at backPad=\(backPad, privacy: .public)s"
                )
                matchedSegments = segments
                matchedAudio = chunk
                break
            }
            AppLog.app.info(
                "reevaluate: no terminator at backPad=\(backPad, privacy: .public)s; growing"
            )
            backPad += Self.reevaluationBackPadStepSec
        }

        guard let segments = matchedSegments, let chunk = matchedAudio else {
            AppLog.app.warning(
                "reevaluate: no full sentence found within \(Self.reevaluationMaxBackPadSec, privacy: .public)s of back pad; preserving original"
            )
            return nil
        }
        guard let (fresh, _) = try await ctx.pipeline.reevaluateFromSegments(
            segments: segments,
            audio: chunk,
            originalStart: ctx.originalStart,
            originalEnd: ctx.originalEnd,
            speakerID: ctx.speakerID
        ) else {
            AppLog.app.warning("reevaluate: pipeline rejected segments after retry")
            return nil
        }
        return (fresh, chunk)
    }

    /// Shared post-processing: re-resolve the speaker over the
    /// corrected window, apply the fresh estimate to `utterances`,
    /// then refresh the per-utterance embedding so the cluster
    /// scatter's focus arrow lands on the post-reeval observation.
    private func finalizeReevaluation(
        fresh: UtteranceEstimate,
        chunk: AudioChunk,
        ctx: ReevaluationContext
    ) async {
        let speaker = await rediarizedSpeaker(
            url: ctx.url,
            pipeline: ctx.pipeline,
            correctedStart: fresh.start,
            correctedEnd: fresh.end,
            fallback: ctx.speakerID
        )
        applyReevaluation(
            utteranceID: ctx.utteranceID,
            fresh: fresh.withSpeakerID(speaker)
        )
        if let embedding = await ctx.pipeline.extractSpeakerEmbedding(
            audio: chunk.samples
        ) {
            utteranceEmbeddings[ctx.utteranceID] = embedding
            await pinObservationSegmentID(
                utteranceID: ctx.utteranceID,
                embedding: embedding,
                speakerID: speaker
            )
        }
    }

    /// Re-resolve the speaker for an utterance's corrected
    /// `[start, end]` window after re-evaluation. Reads a
    /// `rediarizePadSec`-wider chunk so Sortformer has enough
    /// surrounding speech to actually emit segments, then votes
    /// for the speaker covering the corrected range against the
    /// fresh segments only (so the cumulative timeline's ~5
    /// observations from the streaming pass don't outvote a
    /// genuinely-different verdict). Returns `fallback` on any
    /// read failure, on empty audio, or when the diarizer
    /// produces no segments for the slice.
    private func rediarizedSpeaker(
        url: URL,
        pipeline: AnalysisPipeline,
        correctedStart: TimeInterval,
        correctedEnd: TimeInterval,
        fallback: String
    ) async -> String {
        let totalDur: TimeInterval = fileTotalAudioDuration
            ?? max(correctedEnd + Self.rediarizePadSec, correctedEnd)
        let diarStart = max(0, correctedStart - Self.rediarizePadSec)
        let diarEnd = min(totalDur, correctedEnd + Self.rediarizePadSec)
        do {
            let chunk = try await Task.detached(priority: .userInitiated) {
                try Self.readAudioChunkForReevaluation(
                    fileURL: url,
                    start: diarStart,
                    end: diarEnd
                )
            }.value
            guard !chunk.samples.isEmpty else { return fallback }
            let speakers = await pipeline.resolveSpeakersForRanges(
                audio: chunk,
                ranges: [(start: correctedStart, end: correctedEnd)],
                fallback: fallback
            )
            return speakers.first ?? fallback
        } catch {
            AppLog.app.warning(
                "reevaluate: speaker re-diarize read failed: \(String(describing: error), privacy: .public)"
            )
            return fallback
        }
    }

    private func applyReevaluation(utteranceID: UUID, fresh: UtteranceEstimate) {
        guard let index = utterances.firstIndex(where: { $0.id == utteranceID }) else {
            return
        }
        let original = utterances[index]
        // Capture the truly-original (pre-first-reeval) snapshot so a
        // later long-press can restore it. Only on the FIRST re-eval
        // per row — subsequent re-evals leave the existing snapshot
        // alone, so revert always returns to the streaming result,
        // not the previous re-eval.
        if preReevaluationSnapshots[utteranceID] == nil {
            preReevaluationSnapshots[utteranceID] = original
        }
        // Carry `fresh.speakerID` — the caller in `reevaluate`
        // ran the diarizer on a context-padded window covering
        // the corrected `[fresh.start, fresh.end]` range and
        // stamped the resolved verdict on `fresh` via
        // `withSpeakerID`. The original streaming-pass speaker
        // assignment isn't authoritative for the corrected window,
        // so trusting `fresh` lets a re-evaluation fix speaker-
        // boundary errors at the same time it fixes the
        // transcript.
        utterances[index] = Self.mergedEstimate(
            id: original.id,
            speakerID: fresh.speakerID,
            speechBoost: original.speechBoost,
            fresh: fresh,
            wasReevaluated: true,
            wasHandEdited: nil
        )
        // If the corrected start moved the row out of chronological
        // order with its neighbours, re-sort. Cheap (small N, stable
        // sort) and keeps the list consistent for filtering /
        // selection /scroll. No-op when the shift was small enough
        // that the row's still in the right place.
        if !Self.isChronologicallyOrdered(utterances, around: index) {
            utterances.sort { $0.start < $1.start }
        }
        commitUtteranceChanges()
    }

    /// Restore the utterance to its pre-first-reeval state. Invoked
    /// by a 5-second long-press on the re-evaluate button. No-op
    /// when there's no snapshot for this id (the row was never
    /// re-evaluated) or when a re-evaluation is in flight (would
    /// race the upcoming write). Re-sorts if the original start
    /// time moved the row out of order, and bumps
    /// `utterancesVersion` so the filter memo invalidates.
    func revertReevaluation(_ utterance: UtteranceEstimate) {
        guard reevaluatingUtteranceID == nil else { return }
        guard phase == .idle else { return }
        guard let snapshot = preReevaluationSnapshots[utterance.id] else {
            return
        }
        // If this row spawned siblings via a multi-sentence hand-edit
        // commit, drop them so the revert leaves a single row again
        // rather than orphan sentences alongside the restored
        // original.
        if let siblingIDs = handEditChildren.removeValue(forKey: utterance.id), !siblingIDs.isEmpty {
            let drop = Set(siblingIDs)
            utterances.removeAll { drop.contains($0.id) }
        }
        guard let index = utterances.firstIndex(where: { $0.id == utterance.id }) else {
            preReevaluationSnapshots.removeValue(forKey: utterance.id)
            return
        }
        AppLog.app.info(
            "reevaluate: reverting utterance \(utterance.id, privacy: .public) to original snapshot"
        )
        utterances[index] = snapshot
        preReevaluationSnapshots.removeValue(forKey: utterance.id)
        if !Self.isChronologicallyOrdered(utterances, around: index) {
            utterances.sort { $0.start < $1.start }
        }
        commitUtteranceChanges()
    }

    /// True when the element at `index` is in chronological order
    /// relative to its immediate neighbours. Cheap O(1) check used
    /// by `applyReevaluation` to skip the sort when the corrected
    /// timestamp didn't actually move the row past anyone.
    static func isChronologicallyOrdered(
        _ utterances: [UtteranceEstimate],
        around index: Int
    ) -> Bool {
        if index > 0, utterances[index - 1].start > utterances[index].start {
            return false
        }
        if index + 1 < utterances.count, utterances[index].start > utterances[index + 1].start {
            return false
        }
        return true
    }

    /// Read a sub-range of `fileURL` and resample to the pipeline's
    /// 16 kHz mono Float32 format. Used by `reevaluate` — runs on a
    /// detached task so the file I/O and AVAudioConverter pass don't
    /// hitch the MainActor. `start` and `end` are absolute seconds in
    /// the file's native timeline; both are clamped to `[0, duration]`
    /// here so a padded-past-EOF range comes back as a partial chunk
    /// rather than failing.
    nonisolated static func readAudioChunkForReevaluation(
        fileURL: URL,
        start: TimeInterval,
        end: TimeInterval
    ) throws -> AudioChunk {
        let file: AVAudioFile
        do {
            file = try AVAudioFile(forReading: fileURL)
        } catch {
            throw AudioError.underlying(error)
        }
        let inputFormat = file.processingFormat
        let inputRate = inputFormat.sampleRate
        guard inputRate > 0 else {
            throw AudioError.unsupportedFormat(
                expected: ">0 Hz",
                got: "\(inputRate) Hz"
            )
        }
        let totalDuration = TimeInterval(Double(file.length) / inputRate)
        let clampedStart = start.clamped(to: 0...totalDuration)
        let clampedEnd = end.clamped(to: clampedStart...totalDuration)
        let startFrame = AVAudioFramePosition(clampedStart * inputRate)
        let endFrame = AVAudioFramePosition(clampedEnd * inputRate)
        let inputFrames = AVAudioFrameCount(max(0, endFrame - startFrame))
        guard inputFrames > 0 else {
            return AudioChunk(
                samples: [],
                sampleRate: PipelineAudio.sampleRate,
                timestamp: clampedStart
            )
        }

        file.framePosition = startFrame
        guard let inputBuffer = AVAudioPCMBuffer(
            pcmFormat: inputFormat,
            frameCapacity: inputFrames
        ) else {
            throw AudioError.unsupportedFormat(
                expected: "input PCM buffer",
                got: "alloc failed"
            )
        }
        do {
            try file.read(into: inputBuffer, frameCount: inputFrames)
        } catch {
            throw AudioError.underlying(error)
        }

        guard let outputFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: PipelineAudio.sampleRate,
            channels: AVAudioChannelCount(PipelineAudio.channelCount),
            interleaved: false
        ) else {
            throw AudioError.unsupportedFormat(
                expected: "16 kHz mono Float32",
                got: "alloc failed"
            )
        }

        let sameFormat = inputFormat.sampleRate == outputFormat.sampleRate
            && inputFormat.channelCount == outputFormat.channelCount
            && inputFormat.commonFormat == outputFormat.commonFormat
            && inputFormat.isInterleaved == outputFormat.isInterleaved
        if sameFormat {
            guard let channelData = inputBuffer.floatChannelData else {
                throw AudioError.unsupportedFormat(
                    expected: "Float32 channel data",
                    got: "nil"
                )
            }
            let count = Int(inputBuffer.frameLength)
            let samples = Array(UnsafeBufferPointer(start: channelData[0], count: count))
            return AudioChunk(
                samples: samples,
                sampleRate: PipelineAudio.sampleRate,
                timestamp: clampedStart
            )
        }

        guard let converter = AVAudioConverter(from: inputFormat, to: outputFormat) else {
            throw AudioError.unsupportedFormat(
                expected: String(describing: outputFormat),
                got: String(describing: inputFormat)
            )
        }
        converter.primeMethod = .none

        let ratio = outputFormat.sampleRate / inputFormat.sampleRate
        let outCapacity = AVAudioFrameCount(
            (Double(inputBuffer.frameLength) * ratio).rounded(.up)
        ) + 1024
        guard let outBuffer = AVAudioPCMBuffer(
            pcmFormat: outputFormat,
            frameCapacity: outCapacity
        ) else {
            throw AudioError.unsupportedFormat(
                expected: "output PCM buffer",
                got: "alloc failed"
            )
        }

        var convError: NSError?
        // `@unchecked Sendable` is safe: one-shot latch consumed only
        // by the AVAudioConverter input block, which the converter
        // calls serially on a single thread per `convert(...)` call.
        final class Once: @unchecked Sendable { var fired = false }
        let once = Once()
        let block: AVAudioConverterInputBlock = { _, status in
            if once.fired {
                status.pointee = .endOfStream
                return nil
            }
            once.fired = true
            status.pointee = .haveData
            return inputBuffer
        }
        let result = converter.convert(to: outBuffer, error: &convError, withInputFrom: block)
        if result == .error {
            throw AudioError.underlying(
                convError ?? NSError(domain: "Reevaluate", code: -1)
            )
        }

        guard let channelData = outBuffer.floatChannelData else {
            throw AudioError.unsupportedFormat(
                expected: "Float32 channel data",
                got: "nil"
            )
        }
        let count = Int(outBuffer.frameLength)
        let samples = Array(UnsafeBufferPointer(start: channelData[0], count: count))
        return AudioChunk(
            samples: samples,
            sampleRate: PipelineAudio.sampleRate,
            timestamp: clampedStart
        )
    }

}
