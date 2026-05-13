import SwiftUI
import Fusion

/// Modal sheet for hand-editing an utterance's transcript and audio
/// range. Raised from `UtteranceRow` via a long-press on the
/// transcript Text. Confirming Commit hands the edited values to
/// `RecordingController.commitHandEdit`, which re-runs SER + fusion
/// on the new audio slice and stamps the row's `wasHandEdited`
/// flag.
///
/// Designed to live as a `.sheet` (not `.alert`) because the
/// content includes multi-line text input + two Stepper-backed
/// time controls that don't fit in an alert.
struct EditUtteranceSheet: View {
    /// Minimum gap (s) the Stepper enforces between start and end
    /// so `commitEnabled` (which requires `editedEnd > editedStart`)
    /// stays satisfiable when the user nudges either bound right
    /// up against the other. Also matches the Stepper's `step`
    /// (0.1) one notch above zero, so a "drag start up against
    /// end" hits a Stepper boundary, not a Stepper dead-stop.
    private static let stepperMinGapSec: TimeInterval = 0.05

    let utterance: UtteranceEstimate
    /// Total length of the source audio in seconds — used to clamp
    /// the Stepper ranges so the user can't dial past EOF. `nil`
    /// when the source duration isn't known (rare; happens if
    /// `AVAudioFile` couldn't probe the file). In that case the
    /// upper bound falls back to `start + 60s` for the spinner.
    let maxDuration: TimeInterval?
    /// True when the session has source audio backing the row
    /// (file mode or an imported file-mode `.xph`). False for
    /// live / imported mic-mode sessions, where there's no audio
    /// to slice or preview — the dialog then hides the play
    /// button and time spinners, and the commit re-runs only
    /// text SER + fusion, inheriting the parent's time range and
    /// acoustic scores.
    let audioEditingEnabled: Bool
    let onPlayRange: (TimeInterval, TimeInterval) -> Void
    let onStopRange: () -> Void
    /// Live "is a preview playing right now" flag from the
    /// controller. Drives the play/stop icon swap on the preview
    /// button so the user knows they can tap to interrupt.
    let isPreviewPlaying: Bool
    let onCommit: (String, TimeInterval, TimeInterval) -> Void
    let onCancel: () -> Void

    @State private var editedText: String
    @State private var editedStart: TimeInterval
    @State private var editedEnd: TimeInterval

    init(
        utterance: UtteranceEstimate,
        maxDuration: TimeInterval?,
        audioEditingEnabled: Bool,
        onPlayRange: @escaping (TimeInterval, TimeInterval) -> Void,
        onStopRange: @escaping () -> Void,
        isPreviewPlaying: Bool,
        onCommit: @escaping (String, TimeInterval, TimeInterval) -> Void,
        onCancel: @escaping () -> Void
    ) {
        self.utterance = utterance
        self.maxDuration = maxDuration
        self.audioEditingEnabled = audioEditingEnabled
        self.onPlayRange = onPlayRange
        self.onStopRange = onStopRange
        self.isPreviewPlaying = isPreviewPlaying
        self.onCommit = onCommit
        self.onCancel = onCancel
        _editedText = State(initialValue: utterance.transcript)
        _editedStart = State(initialValue: utterance.start)
        _editedEnd = State(initialValue: utterance.end)
    }

    private var upperBound: TimeInterval {
        // Clamp range to file duration when known; otherwise allow
        // a generous headroom so the stepper isn't useless on rows
        // whose duration probe failed.
        maxDuration ?? max(utterance.end + 60, 60)
    }

    /// Commit is valid when text isn't whitespace-only. With
    /// audio editing enabled (file-mode sessions), the time range
    /// must also have positive duration; in audio-disabled
    /// (mic-mode) mode the range comes from the parent utterance
    /// and is always valid.
    private var commitEnabled: Bool {
        let textOK = !editedText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        if audioEditingEnabled {
            return textOK && editedEnd > editedStart
        }
        return textOK
    }

    var body: some View {
        NavigationStack {
            // Unified visual style: one outer system background, no
            // `GroupBox` cards (which paint their own chrome that
            // clashes with the sheet's). Section headers are plain
            // secondary Text; each section's content sits in a soft
            // `.secondarySystemBackground` rounded fill that matches
            // the timestamp control's own background, so the dialog
            // reads as one coherent surface instead of cards-on-cards.
            VStack(alignment: .leading, spacing: 18) {
                sectionHeader(String(localized: "edit.transcript.header"))
                TextEditor(text: $editedText)
                    .font(.body)
                    .scrollContentBackground(.hidden)
                    .frame(minHeight: 120)
                    .padding(10)
                    .glassEffect(in: RoundedRectangle(cornerRadius: 10, style: .continuous))

                if audioEditingEnabled {
                    sectionHeader(String(localized: "edit.range.header"))
                    HStack(alignment: .center, spacing: 16) {
                        Button {
                            if isPreviewPlaying {
                                onStopRange()
                            } else {
                                onPlayRange(editedStart, editedEnd)
                            }
                        } label: {
                            Image(systemName: isPreviewPlaying
                                ? "stop.circle.fill"
                                : "play.circle.fill"
                            )
                                .font(.largeTitle)
                                .symbolRenderingMode(.hierarchical)
                                .foregroundStyle(Color.accentColor)
                        }
                        .buttonStyle(.borderless)
                        .disabled(editedEnd <= editedStart)

                        VStack(spacing: 8) {
                            timeControl(
                                label: String(localized: "edit.range.start"),
                                value: $editedStart,
                                in: 0...max(0, editedEnd - Self.stepperMinGapSec)
                            )
                            timeControl(
                                label: String(localized: "edit.range.end"),
                                value: $editedEnd,
                                in: min(editedStart + Self.stepperMinGapSec, upperBound)...upperBound
                            )
                        }
                    }
                    .padding(12)
                    .glassEffect(in: RoundedRectangle(cornerRadius: 10, style: .continuous))
                }
            }
            .padding(20)
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
            // Opaque sheet backdrop. Without this, the inner
            // `.glassEffect` cards sample whatever's underneath the
            // sheet (the transcript list) and refract its colors
            // through — a list full of completed-state green
            // re-evaluate markers tints the whole sheet green.
            // Glass is meant to layer over a neutral surface, not
            // over arbitrary app content.
            .background(Color(uiColor: .systemBackground))
            .navigationTitle(String(localized: "edit.title"))
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button(String(localized: "edit.cancel")) { onCancel() }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button(String(localized: "edit.commit")) {
                        onCommit(editedText, editedStart, editedEnd)
                    }
                    .disabled(!commitEnabled)
                }
            }
        }
    }

    /// Section header rendered as plain secondary text — no
    /// GroupBox chrome, no Divider. Matches the inline header
    /// style iOS Settings uses inside its grouped tables.
    @ViewBuilder
    private func sectionHeader(_ text: String) -> some View {
        Text(text)
            .font(.subheadline.weight(.semibold))
            .foregroundStyle(.secondary)
            .textCase(.uppercase)
            .padding(.horizontal, 4)
    }

    /// One time control = label + numeric TextField + unit suffix
    /// + a pair of -/+ buttons, all wrapped in a single soft-fill
    /// rounded rectangle so the four subviews read as one control.
    ///
    /// We swapped out `Stepper` for explicit `minus`/`plus` buttons
    /// because the system Stepper's internal tap-target padding
    /// doesn't compress when the iPad keyboard takes over half the
    /// screen — the layout would push the `+` to the row's edge
    /// with a wide gap from the value, and the Stepper's intrinsic
    /// height didn't match the .title3 TextField so the vertical
    /// centers drifted. Custom buttons let us pin spacing AND
    /// vertical baseline.
    @ViewBuilder
    private func timeControl(
        label: String,
        value: Binding<TimeInterval>,
        in range: ClosedRange<TimeInterval>
    ) -> some View {
        HStack(spacing: 8) {
            Text(label)
                .font(.caption2.weight(.semibold))
                .foregroundStyle(.secondary)
            Spacer(minLength: 4)
            TextField(
                "0.00",
                value: value,
                format: .number.precision(.fractionLength(2))
            )
            .textFieldStyle(.plain)
            .keyboardType(.decimalPad)
            .multilineTextAlignment(.trailing)
            .font(.title3.monospacedDigit())
            // Wide enough for `9999.99` (≈ 2.7 hours of audio) in
            // .title3 monospaced. Previous 64 pt clipped anything
            // past ~`99.99`, which the screenshot showed for a
            // 100 s+ timestamp.
            .frame(minWidth: 100, alignment: .trailing)
            Text("s")
                .font(.caption)
                .foregroundStyle(.tertiary)
            stepButton(systemImage: "minus") {
                let next = max(range.lowerBound, value.wrappedValue - 0.1)
                value.wrappedValue = (next * 100).rounded() / 100
            }
            .disabled(value.wrappedValue <= range.lowerBound)
            stepButton(systemImage: "plus") {
                let next = min(range.upperBound, value.wrappedValue + 0.1)
                value.wrappedValue = (next * 100).rounded() / 100
            }
            .disabled(value.wrappedValue >= range.upperBound)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .glassEffect(in: RoundedRectangle(cornerRadius: 10, style: .continuous))
    }

    /// Fixed-size -/+ button used in place of the system `Stepper`
    /// so the time control row stays tidy at any width — see
    /// `timeControl(label:value:in:)` for the rationale.
    @ViewBuilder
    private func stepButton(
        systemImage: String,
        action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            Image(systemName: systemImage)
                .font(.callout.weight(.semibold))
                .frame(width: 28, height: 28)
        }
        .buttonStyle(.bordered)
        .buttonBorderShape(.circle)
    }
}
