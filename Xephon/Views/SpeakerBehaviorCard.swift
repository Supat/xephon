import SwiftUI
import Fusion

/// Per-speaker behavioral fingerprint card. Each speaker row shows
/// five rank-normalized mini bars under a single header row of
/// abbreviated metric labels (Talk · V · Lab · Lead · Resp). Bar
/// fill = the speaker's tint, height = rank-normalized value in
/// [0, 1] within the session.
///
/// Reading the fingerprint plainly substitutes for explicit
/// archetype labels — "high Talk + high Lead + low Lab" reads as
/// anchor without needing a chip to say so, and k-means archetype
/// assignment over ≤ 4 speakers is noisy enough that the auto-label
/// would mislead more than it helps. A speaker-clustering scatter
/// becomes worth building when sessions routinely exceed ~5
/// distinct speakers; not yet.
struct SpeakerBehaviorCard: View {
    let utterances: [UtteranceEstimate]

    /// Inspector popover state — keyed by speaker id so tapping a
    /// row toggles a single raw-value popover at a time.
    @State private var inspectedSpeakerID: String?

    private static let barHeight: CGFloat = 18
    // Wide enough for the 2–3-char metric labels ("Lab" / "Ld") to
    // sit inside their column rather than overflowing into the
    // neighbor. `.lineLimit(1).minimumScaleFactor(0.7)` on the
    // label text guards the JA abbreviations too (発話 / 変動 etc.).
    private static let barWidth: CGFloat = 22
    private static let barSpacing: CGFloat = 4
    private static let labelColumnWidth: CGFloat = 44

    var body: some View {
        let profiles = SpeakerBehavior.computeProfiles(utterances: utterances)
        VStack(alignment: .leading, spacing: 8) {
            header(profileCount: profiles.count)
            if profiles.isEmpty {
                Text(String(localized: "behavior.empty"))
                    .font(.caption)
                    .foregroundStyle(.tertiary)
                    .frame(maxWidth: .infinity, alignment: .leading)
            } else {
                legendRow
                ForEach(profiles, id: \.speakerID) { profile in
                    profileRow(profile)
                }
                footnoteBlock
                    .padding(.top, 4)
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .glassEffect(in: RoundedRectangle(cornerRadius: 12, style: .continuous))
    }

    @ViewBuilder
    private func header(profileCount: Int) -> some View {
        HStack(spacing: 6) {
            Text(String(localized: "behavior.header"))
                .font(.caption.bold())
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .minimumScaleFactor(0.8)
            Spacer(minLength: 4)
            if profileCount > 0 {
                Text("\(profileCount)")
                    .font(.caption2.monospacedDigit())
                    .foregroundStyle(.tertiary)
            }
        }
    }

    /// Header row of metric letter labels above the bars. Lined up
    /// pixel-for-pixel with the bar columns below by sharing the
    /// same `barWidth` / `barSpacing` geometry. The leading column
    /// is empty (the speaker chip sits there in the data rows).
    @ViewBuilder
    private var legendRow: some View {
        HStack(spacing: Self.barSpacing) {
            Color.clear
                .frame(width: Self.labelColumnWidth, height: 12)
            ForEach(Self.metricLabels(), id: \.0) { label in
                Text(label.1)
                    .font(.caption2.weight(.semibold))
                    .foregroundStyle(.tertiary)
                    .textCase(.uppercase)
                    .lineLimit(1)
                    .minimumScaleFactor(0.7)
                    .frame(width: Self.barWidth)
                    .accessibilityLabel(Text(label.2))
            }
            Spacer(minLength: 0)
        }
    }

    /// One row per speaker: chip + five mini bars + small
    /// utterance-count tail. Tap → toggles the raw-values popover.
    @ViewBuilder
    private func profileRow(_ profile: SpeakerBehavior.Profile) -> some View {
        let tint = speakerTint(for: profile.speakerID)
        Button {
            inspectedSpeakerID = (inspectedSpeakerID == profile.speakerID)
                ? nil : profile.speakerID
        } label: {
            HStack(spacing: Self.barSpacing) {
                Text(profile.speakerID)
                    .font(.caption.monospaced())
                    .foregroundStyle(tint)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 1)
                    .background(tint.opacity(0.15), in: Capsule())
                    .frame(width: Self.labelColumnWidth, alignment: .leading)
                miniBar(profile.normalized.talkTimeShare, tint: tint)
                miniBar(profile.normalized.meanValence, tint: tint)
                miniBar(profile.normalized.lability, tint: tint)
                miniBar(profile.normalized.leadership, tint: tint)
                miniBar(profile.normalized.respondQuickness, tint: tint)
                Spacer(minLength: 4)
                Text("\(profile.utteranceCount)")
                    .font(.caption2.monospacedDigit())
                    .foregroundStyle(.tertiary)
            }
            .padding(.horizontal, 4)
            .padding(.vertical, 2)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .popover(
            isPresented: Binding(
                get: { inspectedSpeakerID == profile.speakerID },
                set: { isPresented in
                    if !isPresented { inspectedSpeakerID = nil }
                }
            ),
            arrowEdge: .top
        ) {
            inspectorPopover(profile)
                .presentationCompactAdaptation(.popover)
        }
    }

    /// Vertical mini bar — fills from the bottom up to the
    /// normalized value. Nil values render the empty track only,
    /// so a row missing a metric reads as "no signal" rather than
    /// vanishing entirely (which would misalign the columns).
    @ViewBuilder
    private func miniBar(_ value: Double?, tint: Color) -> some View {
        ZStack(alignment: .bottom) {
            RoundedRectangle(cornerRadius: 1, style: .continuous)
                .fill(Color(uiColor: .tertiarySystemFill))
                .frame(width: Self.barWidth, height: Self.barHeight)
            if let v = value {
                let clamped = max(0, min(1, v))
                RoundedRectangle(cornerRadius: 1, style: .continuous)
                    .fill(tint)
                    .frame(
                        width: Self.barWidth,
                        height: Self.barHeight * CGFloat(clamped)
                    )
            }
        }
        .frame(width: Self.barWidth, height: Self.barHeight)
    }

    @ViewBuilder
    private func inspectorPopover(
        _ profile: SpeakerBehavior.Profile
    ) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 6) {
                Text(profile.speakerID)
                    .font(.caption.monospaced())
                    .foregroundStyle(speakerTint(for: profile.speakerID))
                Text(String.localizedStringWithFormat(
                    String(localized: "behavior.utterances.count"),
                    profile.utteranceCount
                ))
                .font(.caption2)
                .foregroundStyle(.secondary)
            }
            Divider()
            metricRow(
                label: String(localized: "behavior.metric.talkShare"),
                rawDisplay: percentString(profile.raw.talkTimeShare)
            )
            metricRow(
                label: String(localized: "behavior.metric.valence"),
                rawDisplay: formatScalar(profile.raw.meanValence)
            )
            metricRow(
                label: String(localized: "behavior.metric.lability"),
                rawDisplay: formatScalar(profile.raw.lability)
            )
            metricRow(
                label: String(localized: "behavior.metric.leadership"),
                rawDisplay: formatSigned(profile.raw.leadership)
            )
            metricRow(
                label: String(localized: "behavior.metric.responseLatency"),
                rawDisplay: formatSeconds(profile.raw.medianResponseLatency)
            )
        }
        .padding(12)
        .frame(minWidth: 240)
    }

    @ViewBuilder
    private func metricRow(label: String, rawDisplay: String) -> some View {
        HStack {
            Text(label)
                .font(.caption2)
                .foregroundStyle(.secondary)
            Spacer(minLength: 8)
            Text(rawDisplay)
                .font(.caption2.monospacedDigit())
                .foregroundStyle(.primary)
        }
    }

    /// Two-line footer: a `T = Talk share · V = Mean valence · …`
    /// legend decoding the abbreviated column headers, plus the
    /// existing rank-normalization note. The legend is built from
    /// `metricLabels()` so the column headers and the legend can't
    /// drift apart — same source of truth.
    @ViewBuilder
    private var footnoteBlock: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(legendText)
                .font(.caption2)
                .foregroundStyle(.tertiary)
                .fixedSize(horizontal: false, vertical: true)
            Text(String(localized: "behavior.footnote"))
                .font(.caption2)
                .foregroundStyle(.tertiary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private var legendText: String {
        Self.metricLabels()
            .map { "\($0.1) = \($0.2)" }
            .joined(separator: " · ")
    }

    // MARK: - Metric labels

    /// `(id, abbreviation, full localized label)` for each column.
    /// IDs are stable so SwiftUI's `ForEach` has a unique key without
    /// pulling the abbreviation into the hash.
    private static func metricLabels() -> [(String, String, String)] {
        [
            ("talk", String(localized: "behavior.metric.talk.abbr"),
                     String(localized: "behavior.metric.talkShare")),
            ("v", String(localized: "behavior.metric.v.abbr"),
                  String(localized: "behavior.metric.valence")),
            ("lab", String(localized: "behavior.metric.lab.abbr"),
                    String(localized: "behavior.metric.lability")),
            ("lead", String(localized: "behavior.metric.lead.abbr"),
                     String(localized: "behavior.metric.leadership")),
            ("resp", String(localized: "behavior.metric.resp.abbr"),
                     String(localized: "behavior.metric.responseLatency")),
        ]
    }

    // MARK: - Formatters

    private func percentString(_ value: Double) -> String {
        String(format: "%.0f%%", value * 100)
    }

    private func formatScalar(_ value: Double?) -> String {
        guard let v = value else { return "—" }
        return String(format: "%.2f", v)
    }

    private func formatSigned(_ value: Double?) -> String {
        guard let v = value else { return "—" }
        return String(format: "%+.2f", v)
    }

    private func formatSeconds(_ value: Double?) -> String {
        guard let v = value else { return "—" }
        return String(format: "%.2fs", v)
    }
}
