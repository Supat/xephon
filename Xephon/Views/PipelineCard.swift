import SwiftUI

struct PipelineCard: View {
    let recorder: RecordingController

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(String(localized: "pipeline.header"))
                .font(.caption.bold())
                .foregroundStyle(.secondary)

            StageRow(
                icon: "mic.fill",
                name: String(localized: "pipeline.capture"),
                state: captureState,
                metric: captureMetric
            )
            VStack(alignment: .leading, spacing: 2) {
                StageRow(
                    icon: "waveform.and.mic",
                    name: String(localized: "pipeline.asr"),
                    state: asrState,
                    metric: asrMetric
                )
                // Always render the volatile-preview region, even when empty,
                // so the pipeline panel keeps a fixed height and doesn't
                // shift the rows below as text streams in. `reservesSpace`
                // pads the Text to its line limit regardless of content;
                // `.head` truncation keeps the most recent words visible
                // when the preview overflows the 3-line budget.
                Text(recorder.volatileText.isEmpty ? " " : "“\(recorder.volatileText)…”")
                    .font(.caption2.italic())
                    .foregroundStyle(.tertiary)
                    .padding(.leading, 24)
                    .lineLimit(3, reservesSpace: true)
                    .truncationMode(.head)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            StageRow(
                icon: "person.2.fill",
                name: String(localized: "pipeline.diarizer"),
                state: diarizerState,
                metric: diarizerMetric
            )
            StageRow(
                icon: "waveform",
                name: String(localized: "pipeline.acousticSER"),
                state: perSegmentState(latency: recorder.lastAcousticDuration),
                metric: latencyMetric(recorder.lastAcousticDuration)
            )
            StageRow(
                icon: "text.bubble.fill",
                name: String(localized: "pipeline.textSER"),
                state: perSegmentState(latency: recorder.lastTextDuration),
                metric: latencyMetric(recorder.lastTextDuration)
            )
            StageRow(
                icon: "circle.hexagongrid.fill",
                name: String(localized: "pipeline.fusion"),
                state: fusionState,
                metric: Text(recorder.utterances.isEmpty ? "—" : "\(recorder.utterances.count) utts")
            )
            StageRow(
                icon: "square.and.arrow.up",
                name: String(localized: "pipeline.export"),
                state: recorder.lastExportAt == nil ? .idle : .ready,
                metric: exportMetric
            )
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
    }

    // MARK: derived state

    private var captureState: StageRow.State {
        recorder.isRecording ? .active(recorder.inputLevel) : .idle
    }
    /// Rolling buffer size, not session elapsed time. Once the trim
    /// kicks in (after ~60 s of audio = the diarization context window)
    /// this metric stops growing and oscillates near the cap, which is
    /// the more useful signal — it shows the buffer's actual memory
    /// footprint instead of restating the wall clock that's already
    /// visible in the status row above.
    private var captureMetric: Text {
        Text(recorder.isRecording
            ? "\(formatCount(recorder.bufferedSamples)) buf"
            : "—")
    }

    private var asrState: StageRow.State {
        if recorder.isRecording { return .pending }
        if !recorder.utterances.isEmpty { return .ready }
        return .idle
    }
    /// Real-time / mic mode shows the most recent finalize latency
    /// (wall clock from end-of-utterance to ASR-final-emitted). 200–800 ms
    /// is the typical range on M-class. Fast-pace mode falls through to
    /// the utterance count since audio time isn't wall-clock-aligned
    /// there and the latency number would be meaningless.
    /// The leading `backward.frame` glyph reads as "look back to the
    /// most recent finalize" — replaces the verbose "last:" text.
    private var asrMetric: Text {
        if let latency = recorder.lastASRFinalizeLatency {
            let ms = Int((latency * 1000).rounded())
            return Text("\(Image(systemName: "backward.frame")) \(ms) ms")
        }
        return Text(recorder.utterances.isEmpty ? "—" : "\(recorder.utterances.count)")
    }

    /// Per-segment stages (Acoustic SER, Text SER): active while a
    /// segment is in flight, ready once a result has landed, idle
    /// before the first one. Same pattern as fusion/diarizer/ASR —
    /// .ready means "we have output and aren't busy right now" and
    /// will flip back to .active when the next segment arrives.
    private func perSegmentState(latency: TimeInterval?) -> StageRow.State {
        if recorder.inflightSegments > 0 { return .active(0) }
        return latency == nil ? .idle : .ready
    }

    private var fusionState: StageRow.State {
        if recorder.inflightSegments > 0 { return .active(0) }
        return recorder.utterances.isEmpty ? .idle : .ready
    }

    /// Diarizer runs first on each segment; its glyph follows the
    /// same active/ready/idle pattern as fusion. "Ready" means the
    /// last chunk produced at least one speaker — the row reflects
    /// per-chunk activity, not a session-wide accumulator.
    private var diarizerState: StageRow.State {
        if recorder.inflightSegments > 0 { return .active(0) }
        return recorder.lastChunkSpeakerCount == 0 ? .idle : .ready
    }

    /// Distinct speakers in the most recently-diarized chunk. `spk`
    /// matches the `utts` shorthand on the fusion row for visual
    /// rhythm. Distinct from a cumulative session count: this row's
    /// job is to show what the diarizer just did, not the total.
    private var diarizerMetric: Text {
        let count = recorder.lastChunkSpeakerCount
        return Text(count == 0 ? "—" : "\(count) spk")
    }

    private func latencyMetric(_ value: TimeInterval?) -> Text {
        guard let value else { return Text("—") }
        if value >= 1 { return Text(String(format: "%.2f s", value)) }
        return Text(String(format: "%.0f ms", value * 1000))
    }

    private var exportMetric: Text {
        guard let date = recorder.lastExportAt else { return Text("—") }
        let interval = -date.timeIntervalSinceNow
        if interval < 60 { return Text(String(format: "%.0fs ago", interval)) }
        return Text(String(format: "%.0fm ago", interval / 60))
    }
}

struct StageRow: View {
    enum State {
        case idle
        case pending
        case active(Float)   // 0...1 intensity
        case ready
    }

    let icon: String
    let name: String
    let state: State
    /// Right-aligned per-stage value. `Text` (not `String`) so callers
    /// can embed SF Symbols inline via `Text("\(Image(systemName:)) …")`.
    let metric: Text

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: icon)
                .font(.caption)
                .frame(width: 16)
                .foregroundStyle(iconColor)
            Text(name)
                .font(.caption)
                .lineLimit(1)
            Spacer(minLength: 4)
            metric
                .font(.caption.monospaced())
                .foregroundStyle(.secondary)
                .lineLimit(1)
            stateGlyph
                .font(.caption2)
                .frame(width: 14, alignment: .center)
        }
    }

    private var iconColor: Color {
        switch state {
        case .idle: return .secondary
        case .pending: return .blue
        case .active: return .green
        case .ready: return .accentColor
        }
    }

    @ViewBuilder
    private var stateGlyph: some View {
        switch state {
        case .idle:
            Image(systemName: "circle")
                .foregroundStyle(.tertiary)
        case .pending:
            Text("⋯")
                .foregroundStyle(.blue)
        case .active(let intensity):
            Image(systemName: "circle.fill")
                .foregroundStyle(.green.opacity(Double(0.4 + intensity * 0.6)))
        case .ready:
            Image(systemName: "checkmark.circle.fill")
                .foregroundStyle(.green)
        }
    }
}
