import SwiftUI
import Diarization

/// Modal sheet for mid-session diarizer tuning. Surfaced only
/// while the recording is paused — the audio engine is already
/// stopped, the diarizer task isn't being ticked concurrently, and
/// the user is looking for an interactive feedback loop on speaker
/// thresholds.
///
/// Two sliders plus an Apply button. Tapping Apply runs through
/// `RecordingController.applyDiarizerTuning(_:)`, which swaps the
/// diarizer's model, re-diarizes the captured audio, and reassigns
/// every existing utterance's speakerID. The user can keep
/// adjusting and re-applying without leaving the sheet.
struct DiarizerTuningSheet: View {
    let recorder: RecordingController
    let onDismiss: () -> Void

    /// Local copy of the tuning values, decoupled from the
    /// controller's snapshot so the sliders feel responsive even
    /// while an Apply is in flight (we re-sync from the controller
    /// after each apply completes). Defaults match
    /// `FluidAudioDiarizer.conversationalConfig`.
    @State private var clusteringThreshold: Float = 0.6
    @State private var minSpeechDuration: Float = 0.5

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                ScrollView {
                    VStack(alignment: .leading, spacing: 20) {
                        intro
                        clusteringSection
                        minSpeechSection
                        if let error = recorder.errorMessage {
                            Text(error)
                                .font(.footnote)
                                .foregroundStyle(.red)
                                .padding(.top, 4)
                        }
                    }
                    .padding(20)
                }
                Divider()
                applyBar
            }
            .background(Color(uiColor: .systemBackground))
            .navigationTitle(String(localized: "tuning.title"))
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button(String(localized: "summary.done"), action: onDismiss)
                }
            }
            .task {
                // Prime sliders from the live diarizer state on
                // first appearance so the UI doesn't lie about
                // the active values.
                await recorder.refreshDiarizerTuning()
                if let live = recorder.diarizerTuning {
                    clusteringThreshold = live.clusteringThreshold
                    minSpeechDuration = live.minSpeechDuration
                }
            }
        }
    }

    @ViewBuilder
    private var intro: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(String(localized: "tuning.intro.title"))
                .font(.headline)
            Text(String(localized: "tuning.intro.body"))
                .font(.footnote)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    @ViewBuilder
    private var clusteringSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(String(localized: "tuning.clustering.title"))
                    .font(.callout.weight(.semibold))
                Spacer()
                Text(String(format: "%.2f", clusteringThreshold))
                    .font(.callout.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            // Range chosen to match FluidAudio's documented sane
            // window (0.5–0.9). Step 0.01 gives visible movement
            // while keeping the slider from feeling twitchy.
            Slider(value: $clusteringThreshold, in: 0.5...0.9, step: 0.01)
                .disabled(recorder.diarizerTuningApplyInProgress)
            Text(String(localized: "tuning.clustering.hint"))
                .font(.caption2)
                .foregroundStyle(.tertiary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .glassEffect(in: RoundedRectangle(cornerRadius: 10, style: .continuous))
    }

    @ViewBuilder
    private var minSpeechSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(String(localized: "tuning.minSpeech.title"))
                    .font(.callout.weight(.semibold))
                Spacer()
                Text(String(format: "%.2f s", minSpeechDuration))
                    .font(.callout.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            Slider(value: $minSpeechDuration, in: 0.2...2.0, step: 0.05)
                .disabled(recorder.diarizerTuningApplyInProgress)
            Text(String(localized: "tuning.minSpeech.hint"))
                .font(.caption2)
                .foregroundStyle(.tertiary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .glassEffect(in: RoundedRectangle(cornerRadius: 10, style: .continuous))
    }

    @ViewBuilder
    private var applyBar: some View {
        // Apply rebuilds the diarizer model AND re-runs diarization
        // over the captured audio, which can take a moment on long
        // sessions — show a spinner so the user knows we haven't
        // hung. Local slider state stays editable during the apply
        // so the user can queue a follow-up tweak.
        Button {
            let tuning = DiarizationTuning(
                clusteringThreshold: clusteringThreshold,
                minSpeechDuration: minSpeechDuration
            )
            Task {
                await recorder.applyDiarizerTuning(tuning)
            }
        } label: {
            HStack(spacing: 8) {
                if recorder.diarizerTuningApplyInProgress {
                    ProgressView()
                        .controlSize(.small)
                } else {
                    Image(systemName: "wand.and.stars")
                }
                Text(String(localized: "tuning.apply"))
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 12)
        }
        .buttonStyle(.borderedProminent)
        .disabled(
            recorder.diarizerTuningApplyInProgress
                || !recorder.isPaused
                || valuesMatchLive
        )
        .padding(.horizontal, 20)
        .padding(.vertical, 10)
    }

    /// True when the local slider values equal the diarizer's
    /// live tuning — i.e. there's nothing to apply. Compared on
    /// 2-decimal precision so floating-point drift in the slider
    /// doesn't keep the button hot after an apply.
    private var valuesMatchLive: Bool {
        guard let live = recorder.diarizerTuning else { return false }
        return abs(live.clusteringThreshold - clusteringThreshold) < 0.005
            && abs(live.minSpeechDuration - minSpeechDuration) < 0.025
    }
}
