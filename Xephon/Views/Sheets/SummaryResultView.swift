import SwiftUI
import Summarizer

/// Shared body content for both the overall-session and per-
/// section summary sheets. Both surfaces render the same
/// `SessionSummary` schema (setting / topic / overall mood /
/// per-speaker / footer) plus the same "generating" /
/// "empty" / "regenerate" affordances — only the navigation
/// title, the placeholder copy, and the Markdown export
/// filename differ between them. The wrapping sheets keep
/// those bits; this view owns the body.
///
/// Vertical layout, top-to-bottom:
///   1. Content area: result paragraphs, the "generating"
///      spinner, or the empty-state placeholder, depending
///      on state.
///   2. Regenerate bar (only when a summary already exists,
///      one is currently generating, or the summarizer is
///      ready). Before that the user is configuring the
///      summarizer below, not deciding whether to re-run.
struct SummaryResultView: View {
    let recorder: RecordingController
    let summary: SessionSummary?
    let isGenerating: Bool
    /// Localized copy shown under the spinner while a pass
    /// is in flight. Differs between sheets so the wording
    /// can reference "session" vs "section" naturally.
    let generatingMessage: String
    /// Localized copy shown when the sheet opens with no
    /// cached summary AND the summarizer isn't yet ready
    /// (so the auto-fire policy hasn't kicked in). Differs
    /// between sheets for the same reason as the generating
    /// message.
    let emptyMessage: String
    let onRegenerate: () -> Void

    var body: some View {
        VStack(spacing: 0) {
            // Backend badge sits at the top of the body across
            // all three states (result / generating / empty) so
            // it's always visible — the result-view footer also
            // lists the model name, but a top-of-sheet chip
            // makes the on-device vs remote distinction obvious
            // before the user reads any content. Particularly
            // important when the user has LM Studio enabled,
            // since the remote path has privacy implications
            // (post-ASR text leaves the device).
            HStack {
                LLMBackendBadge(recorder: recorder)
                Spacer()
            }
            .padding(.horizontal, 20)
            .padding(.top, 12)
            .padding(.bottom, 6)

            Group {
                if let summary {
                    resultView(for: summary)
                } else if isGenerating {
                    generatingView
                } else {
                    emptyView
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)

            // Regenerate bar only makes sense once we have a
            // summary or one is in flight — before that the user
            // is configuring the summarizer below, not deciding
            // whether to re-run. The first generation is kicked
            // off either by the toolbar Summarize button's auto-
            // fire (when the summarizer is already ready) or by
            // the user toggling the bottom controls into a ready
            // state, which lights up this bar.
            if summary != nil || isGenerating || recorder.summarizerReady {
                Divider()
                regenerateBar
            }
        }
    }

    @ViewBuilder
    private func resultView(for summary: SessionSummary) -> some View {
        ScrollView(.vertical, showsIndicators: true) {
            VStack(alignment: .leading, spacing: 18) {
                if let setting = summary.inferredSetting?
                    .trimmingCharacters(in: .whitespacesAndNewlines),
                   !setting.isEmpty {
                    section(
                        header: String(localized: "summary.section.setting"),
                        body: setting
                    )
                }
                // Meeting mode drops affect: no overall-mood section,
                // and a structured Topics breakdown sits between the
                // overview and the per-speaker talking points.
                let isMeeting = summary.mode == .meeting
                    || (summary.topics?.isEmpty == false)
                section(
                    header: String(localized: "summary.section.topic"),
                    body: summary.topic
                )
                if let topics = summary.topics, !topics.isEmpty {
                    VStack(alignment: .leading, spacing: 8) {
                        sectionHeader(String(localized: "summary.section.topics"))
                        ForEach(Array(topics.enumerated()), id: \.offset) { _, t in
                            topicCard(t)
                        }
                    }
                }
                if !isMeeting {
                    section(
                        header: String(localized: "summary.section.overallMood"),
                        body: summary.overallMood
                    )
                }
                if !summary.perSpeaker.isEmpty {
                    VStack(alignment: .leading, spacing: 8) {
                        sectionHeader(String(localized: "summary.section.perSpeaker"))
                        ForEach(summary.perSpeaker, id: \.speakerID) { entry in
                            speakerCard(entry)
                        }
                    }
                }
                footer(
                    model: summary.model,
                    mode: summary.mode,
                    generatedAt: summary.generatedAt
                )
            }
            .padding(20)
        }
    }

    @ViewBuilder
    private var regenerateBar: some View {
        // Label flips between first-run ("Summarize") and rerun
        // ("Regenerate") so the affordance makes sense whether the
        // user is generating for the first time from inside the
        // sheet or re-running over an existing result.
        let label = summary == nil
            ? String(localized: "summary.summarize")
            : String(localized: "summary.regenerate")
        Button {
            onRegenerate()
        } label: {
            HStack(spacing: 8) {
                if isGenerating {
                    ProgressView()
                        .controlSize(.small)
                    if let start = recorder.summarizerInferenceStart {
                        ElapsedTimeLabel(start: start)
                            .foregroundStyle(.secondary)
                    }
                    // Regenerating over an EXISTING summary renders
                    // this bar, not `generatingView` — the live
                    // progress line must appear in both states.
                    LLMLiveProgressLabel(progress: recorder.summarizerLiveProgress)
                } else {
                    Image(systemName: "arrow.clockwise")
                }
                Text(label)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 12)
        }
        .disabled(isGenerating || !recorder.summarizerReady)
        .padding(.horizontal, 20)
        .padding(.vertical, 10)
    }

    @ViewBuilder
    private var generatingView: some View {
        VStack(spacing: 16) {
            ProgressView()
                .controlSize(.large)
            Text(generatingMessage)
                .font(.callout)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
            if let start = recorder.summarizerInferenceStart {
                ElapsedTimeLabel(start: start)
                    .font(.title3.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            LLMLiveProgressLabel(progress: recorder.summarizerLiveProgress)
        }
        .padding(40)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    @ViewBuilder
    private var emptyView: some View {
        VStack(spacing: 12) {
            Image(systemName: "text.book.closed")
                .font(.largeTitle)
                .foregroundStyle(.secondary)
            Text(emptyMessage)
                .font(.callout)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
        .padding(40)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    @ViewBuilder
    private func section(header: String, body: String) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            sectionHeader(header)
            Text(body.isEmpty ? "—" : body)
                .font(.body)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    @ViewBuilder
    private func sectionHeader(_ text: String) -> some View {
        Text(text)
            .font(.subheadline.weight(.semibold))
            .foregroundStyle(.secondary)
            .textCase(.uppercase)
    }

    /// One meeting topic: title, who raised it, and each speaker's
    /// position. Meeting mode only (`summary.topics`).
    @ViewBuilder
    private func topicCard(_ topic: SessionSummary.TopicSummary) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(topic.title.isEmpty ? "—" : topic.title)
                .font(.callout.bold())
                .fixedSize(horizontal: false, vertical: true)
            if let by = topic.raisedBy?.trimmingCharacters(in: .whitespacesAndNewlines),
               !by.isEmpty {
                Text(String(format: String(localized: "summary.topic.raisedBy"), by))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            ForEach(Array(topic.positions.enumerated()), id: \.offset) { _, p in
                if !p.stance.isEmpty {
                    Text(positionAttributed(speaker: p.speaker, stance: p.stance))
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .glassEffect(in: RoundedRectangle(cornerRadius: 10, style: .continuous))
    }

    /// "Speaker: stance" with the speaker bolded, as one inline-
    /// flowing run. AttributedString instead of the `Text + Text`
    /// concatenation operator (deprecated in iOS 26).
    private func positionAttributed(speaker: String, stance: String) -> AttributedString {
        var name = AttributedString("\(speaker): ")
        name.font = .subheadline.weight(.semibold)
        var rest = AttributedString(stance)
        rest.font = .subheadline
        name.append(rest)
        return name
    }

    @ViewBuilder
    private func speakerCard(_ entry: SessionSummary.SpeakerSummary) -> some View {
        // Meeting mode: bullet talking points, no mood chip.
        let points = entry.talkingPoints?.filter { !$0.isEmpty } ?? []
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Text(displayName(for: entry))
                    .font(.callout.bold())
                    .foregroundStyle(speakerTint(for: entry.speakerID))
                Spacer(minLength: 6)
                if points.isEmpty, !entry.dominantMood.isEmpty {
                    Text(entry.dominantMood)
                        .font(.caption)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(
                            Capsule().fill(speakerTint(for: entry.speakerID).opacity(0.18))
                        )
                        .foregroundStyle(speakerTint(for: entry.speakerID))
                }
            }
            if !points.isEmpty {
                ForEach(Array(points.enumerated()), id: \.offset) { _, point in
                    Text("• \(point)")
                        .font(.body)
                        .fixedSize(horizontal: false, vertical: true)
                }
            } else {
                Text(entry.summary.isEmpty ? "—" : entry.summary)
                    .font(.body)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .glassEffect(in: RoundedRectangle(cornerRadius: 10, style: .continuous))
    }

    private func displayName(for entry: SessionSummary.SpeakerSummary) -> String {
        if let name = entry.speakerName?.trimmingCharacters(in: .whitespacesAndNewlines),
           !name.isEmpty {
            return name
        }
        return entry.speakerID
    }

    @ViewBuilder
    private func footer(
        model: String,
        mode: SummarizeMode?,
        generatedAt: Date
    ) -> some View {
        // "Model: qwen3-8b-4bit · Deep summary" — the mode suffix
        // is dropped for legacy summaries persisted before the
        // mode field was added (decoded as nil), so the footer
        // doesn't render "(unknown)" or similar awkwardness on
        // older `.xph` bundles.
        let modelLine: String = {
            let base = String(format: String(localized: "summary.footer.model"), model)
            guard let mode else { return base }
            let modeLabel: String
            switch mode {
            case .trailing:  modeLabel = String(localized: "summary.footer.mode.trailing")
            case .heuristic: modeLabel = String(localized: "summary.footer.mode.heuristic")
            case .deep:      modeLabel = String(localized: "summary.footer.mode.deep")
            case .meeting:   modeLabel = String(localized: "summary.footer.mode.meeting")
            case .meetingExperimental:
                modeLabel = String(localized: "summary.footer.mode.meetingExperimental")
            case .all:       modeLabel = String(localized: "summary.footer.mode.all")
            }
            return "\(base) · \(modeLabel)"
        }()
        HStack(alignment: .top) {
            VStack(alignment: .leading, spacing: 2) {
                Text(modelLine)
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                Text(generatedAt.formatted(date: .abbreviated, time: .shortened))
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
            Spacer(minLength: 12)
            VStack(alignment: .trailing, spacing: 2) {
                // `sparkles` is Apple's house glyph for AI-generated
                // content (Apple Intelligence affordances all use
                // it), so it reads as "this came from a model"
                // without needing explanatory text.
                HStack(spacing: 4) {
                    Image(systemName: "sparkles")
                    Text(String(localized: "summary.footer.aiGenerated"))
                }
                .font(.caption2)
                .foregroundStyle(.tertiary)
                Text(String(localized: "summary.footer.aiCaveat"))
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
        }
        .padding(.top, 8)
    }
}
