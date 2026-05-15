import SwiftUI
import Fusion
import SERAcoustic
import SERText
import XephonLogging

/// Custom vertical alignment that anchors the playback button to the
/// center of the row's main content only — independent of whether the
/// detail panel is expanded below. Without this, stretching the button
/// to the HStack's full height made it slide down when the row grew.
extension VerticalAlignment {
    private struct UtteranceRowMainContent: AlignmentID {
        static func defaultValue(in d: ViewDimensions) -> CGFloat {
            d[VerticalAlignment.center]
        }
    }
    static let utteranceRowMainContent = VerticalAlignment(UtteranceRowMainContent.self)
}

struct UtteranceRow: View {
    /// Playback availability for this row's audio. Driven by the
    /// recorder's source mode + analysis state — the row itself
    /// doesn't decide, it just renders the supplied state.
    enum PlaybackAvailability: Equatable {
        /// No source file (mic-recorded session); hide the button.
        case unavailable
        /// File source present but analysis is still running; show
        /// a disabled button so the user knows playback is coming.
        case disabled
        /// File source present and analysis idle; tapping plays.
        case idle
        /// File source present and this utterance is currently
        /// playing back; tapping stops.
        case playing
    }

    /// Re-evaluate availability mirrors `PlaybackAvailability` but
    /// without a toggle state — re-evaluate is one-shot. `.running`
    /// renders a spinner for the row whose re-evaluation is in flight;
    /// `.completed` keeps the button tappable but tints it green so
    /// the user can see which entries have been refreshed.
    enum ReevaluateAvailability: Equatable {
        case unavailable
        case disabled
        case idle
        case running
        case completed
    }

    let number: Int
    let utterance: UtteranceEstimate
    let isExpanded: Bool
    let onToggleExpanded: () -> Void
    let playback: PlaybackAvailability
    let onPlaybackToggle: () -> Void
    let reevaluate: ReevaluateAvailability
    let onReevaluate: () -> Void
    let onRevert: () -> Void
    /// Custom name for this row's speaker if the user renamed it,
    /// nil otherwise. Drives the chip display.
    let speakerCustomName: String?
    /// True when the cumulative diarizer timeline's verdict for
    /// this row's `[start, end]` disagrees with the row's stored
    /// `speakerID`. Renders a small warning glyph after the chip
    /// — purely informational, no tap action — so the user can
    /// spot rows where re-running the diarizer (or a manual
    /// reassignment) would change the answer. Computed once per
    /// render by `TranscriptList` against the controller's
    /// snapshot.
    let hasSpeakerMismatch: Bool
    /// Speaker ids currently appearing in the session, sorted.
    /// Drives the chip's reassign menu (we filter out this row's
    /// current speaker at render time so it reads as a "switch to"
    /// list rather than a picker with one disabled entry).
    let knownSpeakerIDs: [String]
    /// Resolves a stored speaker id (e.g. `S01`) to its custom
    /// display name when the user has renamed it. Returns nil for
    /// default-named speakers. Used by the chip menu so a
    /// reassignment target reads `S02 Alice` instead of just `S02`.
    let speakerDisplayName: (String) -> String?
    /// Live acoustic-modality weight from the controller's fusion
    /// settings. Used by the inspector's V/A and label
    /// contribution-share summaries so they reflect what fusion
    /// WOULD do under the current slider values, not the
    /// compiled-in defaults.
    let fusionAcousticWeight: Float
    /// Live text-modality weight floor — same plumbing as
    /// `fusionAcousticWeight`.
    let fusionTextWeightFloor: Float
    /// Fires when the user picks a different speaker from the chip
    /// menu with **Teach diarizer** *off*. ContentView calls
    /// `recorder.reassignSpeaker` — pure row-level annotation
    /// override, no diarizer-state changes.
    let onReassignSpeaker: (String) -> Void
    /// Fires when the user picks a different speaker from the chip
    /// menu with **Teach diarizer** *on*. ContentView calls
    /// `recorder.correctUtteranceSpeaker` — folds the row's audio
    /// embedding into the target speaker's centroid in the
    /// diarizer DB and rewrites the cumulative timeline range, so
    /// future re-eval / hand-edit on similar audio matches the
    /// target.
    let onCorrectSpeaker: (String) -> Void
    /// Fires when the user picks "Promote New Speaker" from the
    /// chip menu. ContentView calls
    /// `recorder.promoteUtteranceToNewSpeaker` — the controller
    /// extracts an embedding from this row's audio, registers it
    /// in the diarizer's SpeakerManager DB under a fresh id,
    /// reassigns the row, and rewrites the cumulative timeline.
    let onPromoteNewSpeaker: () -> Void
    /// Fires when the user taps "Affirm Speaker" — confirms the
    /// row's current speaker assignment is correct and reinforces
    /// it in the diarizer DB. ContentView calls
    /// `recorder.affirmUtteranceSpeaker` which extracts the row's
    /// audio embedding and folds it into the current speaker's
    /// centroid via EMA. No row reassignment, no timeline rewrite.
    let onAffirmSpeaker: () -> Void
    /// Fires when the user picks "Rename Speaker…" from the chip
    /// menu. ContentView raises the existing rename alert
    /// pre-filled with the current override.
    let onRenameSpeaker: () -> Void
    /// Fires when the user long-presses the transcript text.
    /// ContentView raises the Edit Utterance sheet. Confirming the
    /// sheet's Commit hands the edited values to
    /// `recorder.commitHandEdit`.
    let onEditTranscript: () -> Void
    /// Fires when the user taps anywhere on the row that isn't an
    /// inner interactive element (buttons, badges with their own
    /// long-press, etc.). The parent uses this to toggle the row's
    /// focus: tap an unfocused row to focus it, tap the focused row
    /// to unfocus. List's built-in tap-to-select is bypassed because
    /// it can't model the "tap-to-unfocus" half of the toggle.
    let onTap: () -> Void
    /// Fires when the user long-presses the mismatch warning glyph.
    /// The parent recomputes the dominant speaker for this row's
    /// range from the cumulative timeline and reassigns the row,
    /// closing the disagreement that drew the glyph in the first
    /// place. Only attached when `hasSpeakerMismatch == true`.
    let onCorrectMismatch: () -> Void
    /// Shared "Teach diarizer" toggle bound to the recorder so
    /// flipping it in one row's popover propagates to every other
    /// row's popover. Backed by `RecordingController.teaching-
    /// Diarizer`, which is session-only (resets to off on launch).
    @Binding var teachingDiarizer: Bool

    /// Set by a 2-second long-press on the re-evaluate button to
    /// suppress the upcoming tap action (so a held press doesn't
    /// also re-trigger a fresh re-evaluation on release). Reset to
    /// false the next time the button is tapped without a hold.
    @State private var revertJustFired: Bool = false

    /// Drives the per-row speaker chip action sheet. Bound to a
    /// `.confirmationDialog` attached to the chip Text so iPad
    /// anchors the popover to the chip's frame instead of falling
    /// back to a screen-default position.
    @State private var showingSpeakerMenu: Bool = false

    // V/A from fusion are in [0, 1] with 0.5 = neutral. Re-center to [-1, +1]
    // so positive vs negative read naturally and 0 maps to "neutral grey".
    private static let neutralEpsilon: Float = 0.05

    /// Threshold for "open an editor or pop a menu" long-presses
    /// (speaker chip, transcript text). 0.5 s is short enough to
    /// feel responsive but long enough to not fight List
    /// tap-to-select.
    private static let editLongPressSec: Double = 0.5
    /// Threshold for "revert this row" long-presses (re-evaluate
    /// button, Edited badge). 2 s is deliberately past the
    /// edit-press threshold so a momentary hold doesn't blow away
    /// state — the user has to commit to the gesture.
    private static let revertLongPressSec: Double = 2.0
    /// Tint strength (0…1) applied to the Liquid Glass capsule
    /// when the speaker color is laid over it. 0.4 gives a clearly
    /// readable speaker identity while letting the underlying
    /// glass blur the popover backdrop through.
    private static let menuCapsuleTintOpacity: Double = 0.4
    private static let menuCapsuleHPadding: CGFloat = 10
    private static let menuCapsuleVPadding: CGFloat = 5

    var body: some View {
        HStack(alignment: .utteranceRowMainContent, spacing: 8) {
            leadingButtonColumn
            // Hold-to-expand surface excludes the leading button
            // column so a tap or hold near the play / re-evaluate
            // buttons doesn't also toggle the expansion. Inner 0.5 s
            // long-press gestures on the speaker chip and transcript
            // text fire first when held on those elements (their
            // dialogs open at 0.5 s, well before this 2 s threshold).
            VStack(alignment: .leading, spacing: 8) {
                HStack(alignment: .top) {
                    mainContentLeft
                    Spacer(minLength: 8)
                    mainContentRight
                }
                .alignmentGuide(.utteranceRowMainContent) { d in
                    // Anchor the button to the vertical center of just
                    // this main-content row; the detail section below
                    // doesn't participate, so expansion doesn't shift
                    // the button.
                    d[VerticalAlignment.center]
                }
                if isExpanded {
                    detailSection
                        .transition(.opacity.combined(with: .move(edge: .top)))
                }
            }
            .contentShape(Rectangle())
            // Expansion is no longer a row-wide long-press —
            // tapping the emotion label badge in `mainContentRight`
            // toggles the detail panel. Long-pressing empty space
            // was too easy to discover by accident and too obscure
            // to find on purpose; the badge gives the gesture an
            // explicit, visible target.
        }
        .padding(.vertical, 4)
        .contentShape(Rectangle())
        // Tap anywhere on the row toggles its focus (set as the
        // selected utterance, or clear when re-tapping the already-
        // focused row). Overrides List's built-in tap-to-select —
        // which can't model the second half of the toggle — and
        // writes directly to the same `selection:` binding so
        // keyboard arrow-key navigation and the row highlight keep
        // working off `selectedUtteranceID`. Inner buttons use
        // `buttonStyle(.borderless)` so their taps don't bubble up
        // here; inner long-press gestures don't consume quick taps,
        // so a tap on the chip / transcript text still focuses the
        // row before its own long-press window opens.
        .onTapGesture(perform: onTap)
        .animation(.easeInOut(duration: 0.18), value: isExpanded)
    }

    @ViewBuilder
    private var mainContentLeft: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 6) {
                Text("#\(number)")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
                // Speaker chip — half-second long-press on the chip
                // text raises the Rename alert. `.contextMenu` was
                // the earlier approach but SwiftUI escalates a
                // context menu attached to a leaf inside a List row
                // to the row's full bounds, so long-pressing the
                // re-evaluate button (3 s for revert) tripped the
                // chip's menu at the 0.5 s threshold and never
                // reached 3 s. An explicit `.onLongPressGesture` on
                // just the Text stays inside the chip's hit area,
                // so the re-evaluate button's long-press is
                // untouched. Reset-name is folded into the alert
                // (clear the TextField + Save) rather than a
                // separate affordance.
                Text(formatSpeakerLabel(
                    utterance.speakerID,
                    customName: speakerCustomName
                ))
                    .font(.caption.bold())
                    .foregroundStyle(speakerTint(for: utterance.speakerID))
                    // Plain tap (was a 0.5 s long-press). Long-press
                    // was a hidden gesture with no visual hint that
                    // anything would happen; making the chip a
                    // standard tappable element matches the way the
                    // filter-bar chips behave and what users
                    // actually try first.
                    .contentShape(Rectangle())
                    .onTapGesture {
                        showingSpeakerMenu = true
                    }
                    // Custom popover (not `confirmationDialog`) so we
                    // can render reassignment targets as the same
                    // colored capsules the filter bar uses, instead
                    // of the system action-sheet's plain text rows.
                    // Anchored to the chip's frame; on compact
                    // widths SwiftUI would normally adapt to a
                    // sheet, but `.presentationCompactAdaptation(.popover)`
                    // forces the popover form so anchoring stays
                    // consistent across iPad and iPhone.
                    //
                    // No explicit `arrowEdge:` so SwiftUI picks the
                    // edge with the most room — a chip near the
                    // bottom of the list flips to render above
                    // instead of getting clipped off-screen, and a
                    // chip near the right edge of the row leans
                    // its arrow rightward instead of running the
                    // popover past the trailing margin.
                    .popover(isPresented: $showingSpeakerMenu) {
                        speakerMenuPopover
                            .presentationCompactAdaptation(.popover)
                    }
                if hasSpeakerMismatch {
                    // The cumulative diarizer timeline disagrees
                    // with this row's stored speaker. Surfaced as
                    // a caution glyph right after the chip; a
                    // 0.5 s long-press accepts the timeline's
                    // verdict and reassigns the row, closing the
                    // disagreement in one gesture. Re-evaluating
                    // the row or picking a different speaker from
                    // the chip menu still works as before.
                    Image(systemName: "exclamationmark.triangle.fill")
                        .font(.caption2)
                        .foregroundStyle(.orange)
                        .symbolRenderingMode(.hierarchical)
                        .accessibilityLabel(
                            "Diarizer disagrees with assigned speaker"
                        )
                        .accessibilityHint(
                            "Long-press to accept the diarizer's verdict and reassign this row"
                        )
                        .onLongPressGesture(minimumDuration: Self.editLongPressSec) {
                            AppLog.app.info("mismatch glyph long-press → correct speaker")
                            UISounds.playRevert()
                            onCorrectMismatch()
                        }
                }
                if utterance.speechBoost == true {
                    Label("Boost", systemImage: "waveform.badge.plus")
                        .labelStyle(.iconOnly)
                        .font(.caption2)
                        .padding(.horizontal, 5)
                        .padding(.vertical, 1)
                        .foregroundStyle(.orange)
                        .glassEffect(.regular.tint(.orange.opacity(0.35)), in: Capsule())
                }
                // Hand-edit marker: appears immediately after Boost
                // when the row's transcript/range was committed via
                // the Edit Utterance dialog. Re-evaluating the row
                // clears the flag (the controller's
                // `applyReevaluation` rebuilds the row without
                // `wasHandEdited`). A 2-second long-press reverts
                // the row to its pre-hand-edit snapshot via the
                // same `onRevert` path the re-evaluate button uses
                // — both store into `preReevaluationSnapshots`, so
                // a single revert handler covers both.
                if utterance.wasHandEdited == true {
                    Label(String(localized: "edit.badge"), systemImage: "pencil.tip")
                        .labelStyle(.titleAndIcon)
                        .font(.caption2)
                        .padding(.horizontal, 5)
                        .padding(.vertical, 1)
                        .foregroundStyle(.blue)
                        .glassEffect(.regular.tint(.blue.opacity(0.35)), in: Capsule())
                        .onLongPressGesture(minimumDuration: Self.revertLongPressSec) {
                            AppLog.app.info("edit badge long-press → revert")
                            UISounds.playRevert()
                            onRevert()
                        }
                }
            }
            // Transcript text — long-press (0.5 s) raises the Edit
            // Utterance dialog. Hit-test stays inside the Text
            // bounds (same pattern as the speaker chip) so the
            // re-evaluate button's 3 s revert long-press elsewhere
            // in the row isn't pre-empted.
            Text(utterance.transcript.isEmpty ? "—" : utterance.transcript)
                .font(.body)
                .onLongPressGesture(minimumDuration: Self.editLongPressSec) {
                    onEditTranscript()
                }
        }
    }

    /// Content of the chip's reassignment popover. Renders every
    /// other known speaker as a tappable capsule using the same
    /// tint / shape / typography the filter bar uses, plus a final
    /// "Rename Speaker…" row that hands off to the existing rename
    /// alert via `onRenameSpeaker`. We don't show the current
    /// speaker as a tappable target (would be a no-op); it's
    /// surfaced in the header for orientation instead.
    @ViewBuilder
    private var speakerMenuPopover: some View {
        let currentLabel = formatSpeakerLabel(
            utterance.speakerID,
            customName: speakerCustomName
        )
        let others = knownSpeakerIDs.filter { $0 != utterance.speakerID }
        VStack(alignment: .leading, spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                Text(String(localized: "speaker.action.title"))
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Text(currentLabel)
                    .font(.caption.bold())
                    .foregroundStyle(speakerTint(for: utterance.speakerID))
            }
            // Affirm: teach the diarizer that the current assignment
            // is correct. Requires source audio (mic-mode rows have
            // nothing to slice + embed), so we hide the capsule on
            // those — mirroring how the reassign / promote actions
            // upstream silently no-op in mic mode.
            if playback != .unavailable {
                Button {
                    onAffirmSpeaker()
                    showingSpeakerMenu = false
                } label: {
                    affirmSpeakerCapsule
                }
                .buttonStyle(.plain)
            }
            Text(String(localized: "speaker.action.reassign.header"))
                .font(.caption2.weight(.semibold))
                .foregroundStyle(.secondary)
                .textCase(.uppercase)
            // "Teach diarizer" toggle. Off (default) keeps the
            // capsule taps as pure annotation reassignment — the
            // historical low-cost behavior. Flipping on promotes
            // them to corrective: the row's audio embedding is
            // folded into the target speaker's centroid and the
            // cumulative timeline gets rewritten, so future
            // re-eval / hand-edit on similar audio matches the
            // chosen speaker. Resets to off on dismiss so the
            // heavier behavior isn't applied accidentally on
            // the next open.
            Toggle(isOn: $teachingDiarizer) {
                Label {
                    Text(String(localized: "speaker.action.teach"))
                        .font(.caption)
                } icon: {
                    Image(systemName: "graduationcap.fill")
                        .font(.caption)
                }
            }
            .toggleStyle(.switch)
            .controlSize(.mini)
            // Lay the capsules out horizontally, wrapping after
            // every 5 entries so a session with many speakers
            // doesn't blow the popover into a single very wide
            // strip. Chunked HStacks read row-by-row left-to-
            // right exactly like the filter bar.
            VStack(alignment: .leading, spacing: 6) {
                let rows = Self.chunked(others, size: Self.speakerMenuRowSize)
                ForEach(Array(rows.enumerated()), id: \.offset) { _, row in
                    HStack(spacing: 6) {
                        ForEach(row, id: \.self) { spk in
                            Button {
                                performReassign(to: spk)
                            } label: {
                                speakerMenuCapsule(spk)
                            }
                            .buttonStyle(.plain)
                        }
                    }
                }
                Button {
                    onPromoteNewSpeaker()
                    showingSpeakerMenu = false
                } label: {
                    promoteNewSpeakerCapsule
                }
                .buttonStyle(.plain)
            }
            Divider()
            Button {
                onRenameSpeaker()
                showingSpeakerMenu = false
            } label: {
                Label(
                    String(localized: "speaker.action.rename"),
                    systemImage: "pencil"
                )
                .font(.callout)
            }
            .buttonStyle(.plain)
        }
        .padding(16)
        .frame(minWidth: 220, alignment: .leading)
    }

    /// Dispatch the user's capsule tap to either the pure-
    /// annotation reassignment or the corrective version,
    /// depending on the **Teach diarizer** toggle. Always closes
    /// the popover so the user sees the row update.
    private func performReassign(to spk: String) {
        if teachingDiarizer {
            onCorrectSpeaker(spk)
        } else {
            onReassignSpeaker(spk)
        }
        showingSpeakerMenu = false
    }

    /// Max reassignment capsules per row in the speaker popover.
    /// Five keeps the popover narrow enough to anchor cleanly to a
    /// chip in iPad split view, while still showing a typical
    /// 3–6-speaker session without much wrapping.
    private static let speakerMenuRowSize: Int = 5

    /// Split `array` into consecutive chunks of `size`. The final
    /// chunk may be shorter. Returns an empty array when the input
    /// is empty so `ForEach` over the result is a no-op.
    private static func chunked<T>(_ array: [T], size: Int) -> [[T]] {
        guard size > 0, !array.isEmpty else { return [] }
        var result: [[T]] = []
        var index = 0
        while index < array.count {
            let end = min(index + size, array.count)
            result.append(Array(array[index..<end]))
            index = end
        }
        return result
    }

    /// Styling applied to every capsule inside the speaker
    /// popover: tinted Liquid Glass capsule with the speaker's
    /// color as the glass tint. Interactive variant so the system
    /// renders the press-and-bounce affordance.
    @ViewBuilder
    private func popoverCapsule<Content: View>(
        tint: Color,
        @ViewBuilder content: () -> Content
    ) -> some View {
        content()
            .font(.caption.bold())
            .padding(.horizontal, Self.menuCapsuleHPadding)
            .padding(.vertical, Self.menuCapsuleVPadding)
            .foregroundStyle(tint)
            .glassEffect(
                .regular.tint(tint.opacity(Self.menuCapsuleTintOpacity)).interactive(),
                in: Capsule()
            )
    }

    /// One reassignment-target capsule, tinted by the destination
    /// speaker's color so the menu reads as the same vocabulary as
    /// the filter bar (though slightly bolder — popover entries are
    /// actions, filter chips are indicators).
    @ViewBuilder
    private func speakerMenuCapsule(_ spk: String) -> some View {
        popoverCapsule(tint: speakerTint(for: spk)) {
            Text(formatSpeakerLabel(
                spk,
                customName: speakerDisplayName(spk)
            ))
        }
    }

    /// "Promote New Speaker" capsule — visually distinct from
    /// "New Speaker" because the action is deeper: extracts the
    /// row's audio embedding, registers it under a fresh id in
    /// the diarizer's SpeakerManager database, and rewrites the
    /// cumulative timeline. The accent tint flags this as the
    /// "teach the diarizer about this voice" option.
    @ViewBuilder
    private var promoteNewSpeakerCapsule: some View {
        popoverCapsule(tint: .accentColor) {
            Label {
                Text(String(localized: "speaker.action.promoteNewSpeaker"))
            } icon: {
                Image(systemName: "person.crop.circle.badge.plus")
            }
        }
    }

    /// "Affirm Speaker" capsule. Green-tinted to read as a positive /
    /// confirming action, distinct from the orange-ish reassign
    /// capsules and the accent-blue Promote-New capsule. The
    /// `checkmark.seal.fill` glyph mirrors Apple's house language
    /// for "validated / confirmed" in Mail and the App Store.
    @ViewBuilder
    private var affirmSpeakerCapsule: some View {
        popoverCapsule(tint: .green) {
            Label {
                Text(String(localized: "speaker.action.affirm"))
            } icon: {
                Image(systemName: "checkmark.seal.fill")
            }
        }
    }

    @ViewBuilder
    private var mainContentRight: some View {
        VStack(alignment: .trailing, spacing: 4) {
            HStack(spacing: 6) {
                if let backendBadge {
                    let tint: Color = backendBadge.isGuardrail ? .orange : .secondary
                    Text(backendBadge.label)
                        .font(.caption2)
                        .padding(.horizontal, 5)
                        .padding(.vertical, 1)
                        .overlay(
                            Capsule().strokeBorder(tint.opacity(0.45), lineWidth: 0.5)
                        )
                        .foregroundStyle(tint)
                }
                Text("\(formatClock(utterance.start))–\(formatClock(utterance.end))")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            HStack(spacing: 8) {
                if let label = utterance.fusedTopLabel {
                    let tint = emotionTint(for: label)
                    Text(label.capitalized(with: Locale(identifier: "en_US")))
                        .font(.caption)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(tint.opacity(0.18), in: Capsule())
                        .foregroundStyle(tint)
                        // Tap-to-expand. `contentShape(Capsule())`
                        // pins the hit area to the visible badge so
                        // the gesture doesn't bleed onto the V/A
                        // pills next to it. Replaces the previous
                        // row-wide long-press.
                        .contentShape(Capsule())
                        .onTapGesture { onToggleExpanded() }
                }
                if let v = utterance.fusedValence {
                    vaLabel("V", value: v)
                }
                if let a = utterance.fusedArousal {
                    vaLabel("A", value: a)
                }
            }
        }
    }

    @ViewBuilder
    private var detailSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            Divider()
            // Two rows: meta numbers up top, fusion attribution +
            // disagreement badge below. Splitting prevents the
            // single-row layout from wrapping inelegantly once we
            // added Fused V/A/D alongside the acoustic V/A/D.
            HStack(spacing: 12) {
                if let conf = utterance.asrConfidence {
                    metaLine("ASR conf", String(format: "%.2f", conf))
                }
                if let vad = utterance.dimensional {
                    metaLine(
                        "Acoustic V/A/D",
                        String(
                            format: "%.2f · %.2f · %.2f",
                            vad.valence, vad.arousal, vad.dominance
                        )
                    )
                }
                if let fused = fusedVADSummary {
                    metaLine("Fused V/A/D", fused)
                }
            }
            HStack(spacing: 12) {
                if let weightText = fusionWeightSummary {
                    metaLine("Fusion V/A", weightText)
                }
                if let labelText = labelFusionSummary {
                    metaLine("Fusion label", labelText)
                }
                if disagreesAcrossModalities {
                    disagreementBadge
                }
            }
            if let ageGender = utterance.ageGender {
                HStack(spacing: 12) {
                    metaLine(
                        "Age",
                        String(format: "%.0f", ageGender.ageYears)
                    )
                    if let top = ageGender.topGender,
                       let p = ageGender.genderProbabilities[top] {
                        metaLine(
                            "Gender",
                            String(format: "%@ (%.0f%%)", top.rawValue, p * 100)
                        )
                    }
                }
            }

            // Outcome row: top fused labels on the left, the V/A
            // three-pull scatter on the right. Pairs the "what
            // label?" and "where in V/A space?" diagnostics on the
            // same horizontal strip so they read together. The
            // scatter only shows when both modalities contributed
            // — without two pulls the geometry is degenerate.
            HStack(alignment: .top, spacing: 16) {
                if let candidates = topFusedLabels, !candidates.isEmpty {
                    VStack(alignment: .leading, spacing: 3) {
                        sectionHeader("Top fused labels")
                        ForEach(candidates, id: \.label) { entry in
                            ProbabilityBar(
                                label: entry.label.capitalized(with: Locale(identifier: "en_US")),
                                value: entry.score
                            )
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
                if let scatter = fusionScatterInputs {
                    VStack(alignment: .leading, spacing: 3) {
                        sectionHeader("V/A pulls")
                        FusionVAScatterMini(
                            acoustic: scatter.acoustic,
                            text: scatter.text,
                            fused: scatter.fused
                        )
                        .frame(width: 110, height: 110)
                    }
                }
            }

            HStack(alignment: .top, spacing: 16) {
                if let cat = utterance.acousticCategorical, !cat.probabilities.isEmpty {
                    VStack(alignment: .leading, spacing: 3) {
                        sectionHeader("Acoustic SER (emotion2vec)")
                        ForEach(acousticEntries(cat.probabilities), id: \.label) { entry in
                            ProbabilityBar(label: entry.label, value: entry.value)
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }

                if let pl = utterance.plutchik, !pl.probabilities.isEmpty {
                    VStack(alignment: .leading, spacing: 3) {
                        sectionHeader("Text SER (\(textBackendName))")
                        ForEach(plutchikEntries(pl.probabilities), id: \.label) { entry in
                            ProbabilityBar(label: entry.label, value: entry.value)
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
            }
        }
    }

    /// Inputs to the V/A three-pull mini-scatter, or nil when there
    /// isn't a meaningful geometry to render. We gate on BOTH
    /// modalities contributing — single-modality rows have a fused
    /// point coincident with the source and the scatter would draw
    /// one dot on top of another with no arrows, which is uglier
    /// than just omitting the view.
    private var fusionScatterInputs: (
        acoustic: (v: Float, a: Float)?,
        text: (v: Float, a: Float)?,
        fused: (v: Float, a: Float)?
    )? {
        guard let dim = utterance.dimensional,
              let plutchik = utterance.plutchik else { return nil }
        let fusedV = utterance.fusedValence
        let fusedA = utterance.fusedArousal
        guard let v = fusedV, let a = fusedA else { return nil }
        let textV = LateFusion.plutchikToValence(plutchik)
        let textA = LateFusion.plutchikToArousal(plutchik)
        return (
            acoustic: (v: dim.valence, a: dim.arousal),
            text: (v: textV, a: textA),
            fused: (v: v, a: a)
        )
    }

    /// Top 3 normalized fused-label candidates so the user can see
    /// not just *which* label won but *how confidently* — a 0.42 vs
    /// 0.39 runner-up reads very differently from 0.85 vs 0.05.
    /// Nil when neither modality contributed score (no top label
    /// to attribute anyway).
    private var topFusedLabels: [(label: String, score: Float)]? {
        guard let scored = LateFusion.labelFusionScores(
            acoustic: utterance.acousticCategorical,
            plutchik: utterance.plutchik,
            asrConfidence: utterance.asrConfidence ?? 0.5,
            acousticWeight: fusionAcousticWeight,
            textWeightFloor: fusionTextWeightFloor
        ), !scored.isEmpty else { return nil }
        return Array(scored.prefix(3))
    }

    /// Compact "0.62 · 0.48 · 0.55" rendering of the fused V/A/D
    /// numbers. Returned nil only when none of the three components
    /// fused — in which case the row's fusion summary is also nil
    /// and there's nothing to attribute.
    private var fusedVADSummary: String? {
        let v = utterance.fusedValence
        let a = utterance.fusedArousal
        let d = utterance.fusedDominance
        guard v != nil || a != nil || d != nil else { return nil }
        func fmt(_ x: Float?) -> String {
            x.map { String(format: "%.2f", $0) } ?? "—"
        }
        return "\(fmt(v)) · \(fmt(a)) · \(fmt(d))"
    }

    /// True when the top acoustic class (mapped through
    /// `plutchikToAcousticLabelMapping`) and the top Plutchik class
    /// (also mapped through the same table for comparability) name
    /// different acoustic-label buckets. Used to flag rows where
    /// the two modalities openly disagree — the fused label hides
    /// this signal; the inspector should not.
    ///
    /// Excludes acoustic "other"/"unknown" buckets (those are sink
    /// classes — disagreement there is uninformative) and only
    /// counts text classes that have a mapping (trust /
    /// anticipation route to "other" too, see LateFusion's
    /// mapping doc).
    private var disagreesAcrossModalities: Bool {
        guard let acoustic = utterance.acousticCategorical?.probabilities,
              let plutchik = utterance.plutchik?.probabilities else {
            return false
        }
        let validAcoustic = acoustic
            .filter { $0.key != .unknown && $0.key != .other }
        guard let topAcoustic = validAcoustic.max(by: { $0.value < $1.value })?.key else {
            return false
        }
        let mappedText: [(label: String, value: Float)] = plutchik.compactMap { entry in
            guard let mapped = LateFusion.plutchikToAcousticLabelMapping[entry.key],
                  mapped != "other" else { return nil }
            return (label: mapped, value: entry.value)
        }
        guard let topText = mappedText.max(by: { $0.value < $1.value })?.label else {
            return false
        }
        return topText != topAcoustic.rawValue
    }

    @ViewBuilder
    private var disagreementBadge: some View {
        Label("Modality disagreement", systemImage: "exclamationmark.triangle.fill")
            .font(.caption2.weight(.semibold))
            .labelStyle(.titleAndIcon)
            .foregroundStyle(.orange)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(
                Capsule().fill(Color.orange.opacity(0.18))
            )
    }

    /// Compact summary of how V/A fusion weighted the two sides for
    /// this utterance. Nil when neither modality contributed (the
    /// fused V/A would be nil too, so there's nothing to attribute).
    /// When only one modality was present, reports that side as 100%
    /// so the user can see which side carried the result.
    private var fusionWeightSummary: String? {
        let hasAcoustic = utterance.dimensional != nil
        let hasText = utterance.plutchik != nil
        switch (hasAcoustic, hasText) {
        case (false, false):
            return nil
        case (true, false):
            return "Acoustic 100% (no text)"
        case (false, true):
            return "Text 100% (no acoustic)"
        case (true, true):
            let share = LateFusion.vaFusionShare(
                asrConfidence: utterance.asrConfidence ?? 0.5,
                acousticWeight: fusionAcousticWeight,
                textWeightFloor: fusionTextWeightFloor
            )
            return String(
                format: "Acoustic %.0f%% · Text %.0f%%",
                share.acoustic * 100,
                share.text * 100
            )
        }
    }

    /// Compact summary of each modality's overall influence on the
    /// fused-label argmax. Nil when there's no top label or when
    /// neither modality contributed any score (defensive — a
    /// well-formed estimate should always have at least one side
    /// of input).
    private var labelFusionSummary: String? {
        guard utterance.fusedTopLabel != nil else { return nil }
        guard let share = LateFusion.labelFusionShare(
            acoustic: utterance.acousticCategorical,
            plutchik: utterance.plutchik,
            asrConfidence: utterance.asrConfidence ?? 0.5,
            acousticWeight: fusionAcousticWeight,
            textWeightFloor: fusionTextWeightFloor
        ) else { return nil }
        return String(
            format: "Acoustic %.0f%% · Text %.0f%%",
            share.acoustic * 100,
            share.text * 100
        )
    }

    private var textBackendName: String {
        guard let raw = utterance.textBackend else { return "Plutchik" }
        if raw == SwitchingTextSER.foundationModelsGuardrailBackend {
            return String(localized: "textSER.appleFMViolation")
        }
        guard let backend = SwitchingTextSER.Backend(rawValue: raw) else {
            return "Plutchik"
        }
        return backend.badgeLabel
    }

    private func metaLine(_ key: String, _ value: String) -> some View {
        HStack(spacing: 4) {
            Text(key)
                .font(.caption2)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.caption.monospacedDigit())
                .foregroundStyle(.primary)
        }
    }

    private func sectionHeader(_ text: String) -> some View {
        Text(text)
            .font(.caption2.bold())
            .foregroundStyle(.secondary)
            .padding(.top, 2)
    }

    private func acousticEntries(
        _ probs: [CategoricalEmotion.Label: Float]
    ) -> [(label: String, value: Float)] {
        probs
            .map { (label: $0.key.rawValue, value: $0.value) }
            .sorted { $0.value > $1.value }
    }

    private func plutchikEntries(
        _ probs: [PlutchikScore.Label: Float]
    ) -> [(label: String, value: Float)] {
        probs
            .map { (label: $0.key.rawValue, value: $0.value) }
            .sorted { $0.value > $1.value }
    }

    private var backendBadge: TextBackendBadge? {
        guard let raw = utterance.textBackend else { return nil }
        if raw == SwitchingTextSER.foundationModelsGuardrailBackend {
            return TextBackendBadge(
                label: String(localized: "textSER.appleFMViolation"),
                isGuardrail: true
            )
        }
        guard let backend = SwitchingTextSER.Backend(rawValue: raw) else { return nil }
        return TextBackendBadge(label: backend.badgeLabel, isGuardrail: false)
    }

    @ViewBuilder
    private func vaLabel(_ axis: String, value: Float) -> some View {
        let centered = value * 2 - 1
        Text(String(format: "%@ %+.2f", axis, centered))
            .font(.caption.monospacedDigit())
            .foregroundStyle(color(for: centered))
    }

    private func color(for centered: Float) -> Color {
        if centered > Self.neutralEpsilon { return .green }
        if centered < -Self.neutralEpsilon { return .red }
        return .gray
    }

    /// Playback + re-evaluate stacked vertically at the row's leading
    /// edge. Hidden entirely when both buttons are unavailable (mic
    /// mode), so the row's text content reflows to the left rather
    /// than carrying the HStack's spacer when there's no audio.
    @ViewBuilder
    private var leadingButtonColumn: some View {
        let hidden = playback == .unavailable && reevaluate == .unavailable
        if !hidden {
            VStack(spacing: 4) {
                playbackButton
                reevaluateButton
            }
        }
    }

    @ViewBuilder
    private var playbackButton: some View {
        switch playback {
        case .unavailable:
            EmptyView()
        case .disabled, .idle, .playing:
            Button(action: {
                AppLog.app.info("playback button tapped (state=\(String(describing: self.playback), privacy: .public))")
                onPlaybackToggle()
            }) {
                Image(systemName: playback == .playing ? "stop.circle.fill" : "play.circle.fill")
                    .font(.title3)
                    .foregroundStyle(playback == .disabled ? Color.secondary : Color.accentColor)
                    .symbolRenderingMode(.hierarchical)
            }
            // `.borderless` keeps the tap from bubbling up and toggling
            // List selection — `.plain` doesn't on every platform.
            .buttonStyle(.borderless)
            .disabled(playback == .disabled)
        }
    }

    @ViewBuilder
    private var reevaluateButton: some View {
        switch reevaluate {
        case .unavailable:
            EmptyView()
        case .running:
            ProgressView()
                .controlSize(.small)
                // Match the playback icon's footprint so the column
                // doesn't shift width while a re-evaluation is in
                // flight.
                .frame(width: 22, height: 22)
        case .disabled, .idle, .completed:
            // A 2-second long-press on a `.completed` row reverts to
            // the pre-first-reeval snapshot. Implemented as a
            // simultaneous LongPressGesture alongside the Button's
            // own tap so the gesture system doesn't have to
            // disambiguate up front — the long press fires at the
            // 2 s mark, sets `revertJustFired`, and the Button's
            // release-fired tap action sees the flag and skips
            // `onReevaluate`. Short taps still trigger re-eval
            // normally.
            Button(action: {
                if revertJustFired {
                    revertJustFired = false
                    return
                }
                AppLog.app.info("reevaluate button tapped (state=\(String(describing: self.reevaluate), privacy: .public))")
                onReevaluate()
            }) {
                Image(systemName: "arrow.clockwise.circle.fill")
                    .font(.title3)
                    .foregroundStyle(reevaluateTint)
                    .symbolRenderingMode(.hierarchical)
            }
            .buttonStyle(.borderless)
            .disabled(reevaluate == .disabled)
            .simultaneousGesture(
                LongPressGesture(minimumDuration: Self.revertLongPressSec)
                    .onEnded { _ in
                        guard reevaluate == .completed else { return }
                        AppLog.app.info("reevaluate long-press → revert")
                        revertJustFired = true
                        // Audible confirmation — iPad has no Taptic
                        // Engine so the sensoryFeedback below is a
                        // no-op there. The helper guards the
                        // AVAudioSession state so the sound isn't
                        // swallowed when a recent recording left
                        // the session in `.record`.
                        UISounds.playRevert()
                        onRevert()
                    }
            )
            // Haptic confirmation when supported (e.g. Mac
            // Designed-for-iPad with Force Touch). No-op on iPad
            // proper; left in because it's cheap and the
            // closure-form trigger ensures only the false → true
            // edge produces feedback.
            .sensoryFeedback(trigger: revertJustFired) { _, new in
                new ? .success : nil
            }
        }
    }

    /// Foreground tint for the re-evaluate button. `.completed` rows
    /// stay green even when re-disabled (e.g. another row is mid-
    /// re-evaluation) so the completion marker doesn't flicker away
    /// the moment a new pass starts elsewhere in the list.
    private var reevaluateTint: Color {
        switch reevaluate {
        case .completed: return .green
        case .disabled: return .secondary
        default: return .accentColor
        }
    }
}

private struct ProbabilityBar: View {
    let label: String
    let value: Float

    var body: some View {
        let tint = emotionTint(for: label)
        let fraction = Double(max(0, min(1, value)))
        HStack(spacing: 8) {
            // Fixed-width column with tail truncation so a long
            // outlier like "Anticipation" (12 chars) gets clipped
            // to align with neighbours like "Joy" / "Fear" /
            // "Trust" instead of pushing the bar right and
            // breaking column alignment with the other category's
            // bars rendered next to it.
            Text(label.capitalized(with: Locale(identifier: "en_US")))
                .font(.caption2.monospaced())
                .foregroundStyle(tint)
                .lineLimit(1)
                .truncationMode(.tail)
                .frame(width: 80, alignment: .leading)
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    Capsule().fill(tint.opacity(0.12))
                    Capsule()
                        .fill(tint.opacity(0.55))
                        .frame(width: geo.size.width * fraction)
                }
            }
            .frame(height: 5)
            Text(String(format: "%.2f", value))
                .font(.caption2.monospacedDigit())
                .foregroundStyle(.secondary)
                .frame(minWidth: 36, alignment: .trailing)
        }
    }
}
