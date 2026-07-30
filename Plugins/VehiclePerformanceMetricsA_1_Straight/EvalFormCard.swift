import SwiftUI
import XephonPluginKit

/// The plugin's control-pane page: coverage readout, run control,
/// and the per-item review list with evidence-row reveal (chip tap
/// jumps + highlights the transcript row). Scores
/// follow the visual policy: stated values plain, inferred values
/// marked and tinted, conflicts called out.
struct EvalFormCard: View {
    @Bindable var model: EvalFormModel

    /// Evidence lists expanded past the one-line collapse — keyed
    /// by item id (or the supplementary key). View-local state:
    /// collapsing again on session/draft change is fine.
    @State private var expandedEvidenceKeys: Set<String> = []

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            header
            switch model.phase {
            case .running(let step):
                HStack(spacing: 8) {
                    ProgressView()
                    Text(step)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            case .failed(let reason):
                Text(reason)
                    .font(.caption2)
                    .foregroundStyle(.red)
            case .idle:
                EmptyView()
            }
            if let draft = model.draft, model.draftMatchesTemplate {
                if model.draftIsStale {
                    Text(String(localized: "evalform.stale", bundle: .module))
                        .font(.caption2)
                        .foregroundStyle(.orange)
                }
                metadataSection(draft)
                itemsSection(draft)
                supplementarySection(draft)
                undetectedSection(draft)
                if let lastExport = model.lastExport {
                    Text(verbatim: "Export: \(lastExport)")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }
            } else {
                if !model.draftMatchesTemplate {
                    Text(String(localized: "evalform.templateMismatch", bundle: .module))
                        .font(.caption2)
                        .foregroundStyle(.orange)
                        .fixedSize(horizontal: false, vertical: true)
                }
                coverageSection
            }
            rubricFootnote
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .glassEffect(in: RoundedRectangle(cornerRadius: 12, style: .continuous))
    }

    /// The sheet's ※ magnitude rubric, as printed on the form —
    /// always visible so scores (and the model's inferred
    /// suggestions) are read against the same calibration.
    @ViewBuilder
    private var rubricFootnote: some View {
        if let anchors = model.template.strengthScale.anchors, !anchors.isEmpty {
            VStack(alignment: .leading, spacing: 1) {
                ForEach(anchors, id: \.magnitude) { anchor in
                    Text(verbatim: "※ \(String(format: "%g", anchor.magnitude)): \(anchor.meaning)")
                        .font(.system(size: 8))
                        .foregroundStyle(.tertiary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            .padding(.top, 4)
        }
    }

    // MARK: - Sections

    private var header: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(model.template.name)
                .font(.caption.bold())
                .foregroundStyle(.secondary)
            HStack(spacing: 12) {
                Button {
                    Task { await model.run() }
                } label: {
                    Label(
                        String(localized: "evalform.run", bundle: .module),
                        systemImage: "wand.and.sparkles"
                    )
                }
                .disabled(!runEnabled)
                Menu {
                    Button(String(localized: "evalform.export.markdown", bundle: .module)) {
                        model.exportMarkdown()
                    }
                    Button(String(localized: "evalform.export.csv", bundle: .module)) {
                        model.exportCSV()
                    }
                } label: {
                    Label(
                        String(localized: "evalform.export", bundle: .module),
                        systemImage: "square.and.arrow.up"
                    )
                }
                .disabled(!model.canExport)
            }
            .font(.caption)
            .buttonStyle(.bordered)
            // Template + section tooling. Menu keeps the row
            // compact; every action is reversible or additive.
            HStack(spacing: 12) {
                Menu {
                    Button {
                        model.importTemplatePack()
                    } label: {
                        Label(
                            String(localized: "evalform.importTemplate", bundle: .module),
                            systemImage: "square.and.arrow.down.on.square"
                        )
                    }
                    if model.usesImportedTemplate {
                        Button {
                            model.resetTemplateToDefault()
                        } label: {
                            Label(
                                String(localized: "evalform.resetTemplate", bundle: .module),
                                systemImage: "arrow.uturn.backward"
                            )
                        }
                    }
                } label: {
                    Label(
                        String(localized: "evalform.template", bundle: .module),
                        systemImage: "doc.badge.gearshape"
                    )
                }
                Button {
                    model.detectRoadSections()
                } label: {
                    Label(
                        String(localized: "evalform.detectSections", bundle: .module),
                        systemImage: "road.lanes"
                    )
                }
                if let added = model.lastSectionDetection {
                    Text(verbatim: "+\(added)")
                        .font(.caption2.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
            }
            .font(.caption)
            if case .unavailable(let reason) = model.inferenceAvailability {
                Text(reason)
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private var runEnabled: Bool {
        if case .running = model.phase { return false }
        if case .unavailable = model.inferenceAvailability { return false }
        return true
    }

    /// Pre-run readout: how many rows mention each item's
    /// vocabulary — what a run would work with.
    private var coverageSection: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(String(localized: "evalform.coverage", bundle: .module))
                .font(.caption2.weight(.semibold))
                .foregroundStyle(.tertiary)
                .textCase(.uppercase)
            ForEach(model.candidateCounts, id: \.item.id) { entry in
                HStack {
                    Text(verbatim: "\(entry.item.number). \(entry.item.titleJa)")
                        .font(.caption2)
                        .lineLimit(1)
                    Spacer(minLength: 8)
                    Text(verbatim: "\(entry.count)")
                        .font(.caption2.monospacedDigit())
                        .foregroundStyle(entry.count == 0 ? .tertiary : .secondary)
                }
            }
        }
    }

    @ViewBuilder
    private func metadataSection(_ draft: EvalFormDraft) -> some View {
        if !draft.metadata.isEmpty {
            VStack(alignment: .leading, spacing: 2) {
                Text(String(localized: "evalform.metadata", bundle: .module))
                    .font(.caption2.weight(.semibold))
                    .foregroundStyle(.tertiary)
                    .textCase(.uppercase)
                ForEach(
                    model.template.metadataFields.filter { draft.metadata[$0] != nil },
                    id: \.self
                ) { field in
                    Text(verbatim: "\(field): \(draft.metadata[field] ?? "")")
                        .font(.caption2)
                }
            }
        }
    }

    @ViewBuilder
    private func supplementarySection(_ draft: EvalFormDraft) -> some View {
        if let supplementary = draft.supplementaryComment {
            VStack(alignment: .leading, spacing: 3) {
                Text(String(localized: "evalform.supplementary", bundle: .module))
                    .font(.caption2.weight(.semibold))
                    .foregroundStyle(.tertiary)
                    .textCase(.uppercase)
                Text(supplementary)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                if let evidence = draft.supplementaryEvidenceRows, !evidence.isEmpty {
                    evidenceChips(evidence, key: "supplementary")
                }
            }
        }
    }

    /// The reviewer's to-fill list — same `EvalFormCoverage` data
    /// the exports render, so the card and the report agree on
    /// what remains manual.
    @ViewBuilder
    private func undetectedSection(_ draft: EvalFormDraft) -> some View {
        let undetected = EvalFormCoverage.undetected(
            draft: draft,
            template: model.template
        )
        if !undetected.isEmpty {
            VStack(alignment: .leading, spacing: 3) {
                Text(String(localized: "evalform.undetected", bundle: .module))
                    .font(.caption2.weight(.semibold))
                    .foregroundStyle(.tertiary)
                    .textCase(.uppercase)
                if !undetected.missingHeaderFields.isEmpty {
                    Text(verbatim: "ヘッダ: \(undetected.missingHeaderFields.joined(separator: ", "))")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                ForEach(undetected.items, id: \.itemID) { gaps in
                    if let item = model.template.items.first(where: { $0.id == gaps.itemID }) {
                        Text(verbatim: "\(item.number). \(item.titleJa): \(EvalFormCoverage.gapPhrase(gaps))")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
            }
        }
    }

    private func itemsSection(_ draft: EvalFormDraft) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            ForEach(model.template.items) { item in
                itemRow(
                    item: item,
                    result: draft.items.first { $0.itemID == item.id }
                )
                if item.id != model.template.items.last?.id {
                    Divider()
                }
            }
        }
    }

    @ViewBuilder
    private func itemRow(
        item: EvalFormTemplate.Item,
        result: EvalFormDraft.ItemResult?
    ) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack(alignment: .firstTextBaseline) {
                // Reviewer-confirmed toggle (payload v2 state).
                Button {
                    model.toggleReviewed(item.id)
                } label: {
                    Image(
                        systemName: model.isReviewed(item.id)
                            ? "checkmark.circle.fill"
                            : "circle"
                    )
                    .font(.caption)
                    .foregroundStyle(
                        model.isReviewed(item.id)
                            ? AnyShapeStyle(.green)
                            : AnyShapeStyle(.tertiary)
                    )
                }
                .buttonStyle(.plain)
                .accessibilityLabel(
                    Text(String(localized: "evalform.reviewed", bundle: .module))
                )
                Text(verbatim: "\(item.number). \(item.titleJa)")
                    .font(.caption.weight(.semibold))
                    .lineLimit(1)
                Spacer(minLength: 8)
                scoreBadge(result)
                if let preference = result?.likeDislike {
                    Text(verbatim: "♥\(preference)")
                        .font(.caption2.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
            }
            // Where this item's cited evidence was actually driven
            // — full road names from the fill-time segmentation.
            if let roads = model.draft?.roadsForItem(item.id), !roads.isEmpty {
                Text(verbatim: "走行路: \(roads.joined(separator: ", "))")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
            // The sheet's two scales, as printed on the form:
            // strength (強い −1 … +1 弱い) and preference (嫌い 1 …
            // 9 好き). Empty tracks read as unfilled rows.
            StrengthScaleView(
                scale: model.template.strengthScale,
                stated: result?.strengthScore,
                inferred: result?.strengthScoreInferred
            )
            .padding(.top, 2)
            PreferenceScaleView(
                scale: model.template.preferenceScale,
                value: result?.likeDislike
            )
            if let comment = result?.comment {
                Text(comment)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if let evidence = result?.evidenceRows, !evidence.isEmpty {
                evidenceChips(evidence, key: item.id)
            }
            ForEach(result?.conflicts ?? [], id: \.self) { conflict in
                Text(verbatim: "⚠ \(conflict)")
                    .font(.caption2)
                    .foregroundStyle(.orange)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    /// Roughly one grid line of chips — past this the list
    /// collapses behind a "+N" toggle. 6, not 7: the portrait
    /// pane is narrower and 7 chips + the toggle overflow it.
    private static let evidenceCollapseLimit = 6

    /// Evidence chips, grouped by road provenance when the draft
    /// carries it (fill-time callout segmentation): a small road
    /// label heads each group; rows outside any road group under
    /// "—". Flat grid when the session had no callouts.
    @ViewBuilder
    private func evidenceChips(_ rows: [Int], key: String) -> some View {
        let roadByRow = model.draft?.roadByRow ?? [:]
        if roadByRow.isEmpty {
            chipGrid(rows, key: key)
        } else {
            VStack(alignment: .leading, spacing: 3) {
                ForEach(roadGroups(rows, roadByRow: roadByRow), id: \.road) { group in
                    VStack(alignment: .leading, spacing: 1) {
                        Text(verbatim: group.road)
                            .font(.system(size: 8).weight(.semibold))
                            .foregroundStyle(.tertiary)
                        chipGrid(group.rows, key: "\(key)|\(group.road)")
                    }
                }
            }
        }
    }

    private struct RoadGroup {
        let road: String
        let rows: [Int]
    }

    /// Groups in first-appearance order over the (sorted) evidence
    /// rows; rows with no assignment land under "—".
    private func roadGroups(
        _ rows: [Int],
        roadByRow: [Int: String]
    ) -> [RoadGroup] {
        var order: [String] = []
        var byRoad: [String: [Int]] = [:]
        for row in rows {
            let road = roadByRow[row] ?? "—"
            if byRoad[road] == nil { order.append(road) }
            byRoad[road, default: []].append(row)
        }
        return order.map { RoadGroup(road: $0, rows: byRoad[$0] ?? []) }
    }

    /// One flat chip grid with the collapse behaviour. Tap = jump
    /// the transcript list to that row and highlight it (host
    /// reveal). An adaptive grid, not an HStack — the context
    /// window can put dozens of rows behind one item, and an
    /// overflowing HStack compresses each chip into a vertical
    /// character stack in the narrow pane. Lists longer than ~one
    /// line collapse to the first chips plus a "+N" expander;
    /// expanded lists get a collapse chip.
    private func chipGrid(_ rows: [Int], key: String) -> some View {
        let isExpanded = expandedEvidenceKeys.contains(key)
        // No toggle when it would hide a single chip — showing the
        // chip costs the same space as the "+1".
        let collapsible = rows.count > Self.evidenceCollapseLimit + 1
        let visible = (collapsible && !isExpanded)
            ? Array(rows.prefix(Self.evidenceCollapseLimit))
            : rows
        return LazyVGrid(
            columns: [GridItem(.adaptive(minimum: 42), spacing: 4)],
            alignment: .leading,
            spacing: 2
        ) {
            ForEach(visible, id: \.self) { row in
                Button {
                    model.revealRow(row)
                } label: {
                    Text(verbatim: "[\(row)]")
                        .font(.caption2.monospacedDigit())
                        .lineLimit(1)
                        .fixedSize()
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .buttonStyle(.plain)
                .foregroundStyle(.tint)
            }
            if collapsible {
                Button {
                    withAnimation(.easeInOut(duration: 0.15)) {
                        if isExpanded {
                            expandedEvidenceKeys.remove(key)
                        } else {
                            expandedEvidenceKeys.insert(key)
                        }
                    }
                } label: {
                    if isExpanded {
                        Image(systemName: "chevron.up")
                            .font(.caption2)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    } else {
                        Text(verbatim: "+\(rows.count - Self.evidenceCollapseLimit)")
                            .font(.caption2.monospacedDigit().weight(.semibold))
                            .lineLimit(1)
                            .fixedSize()
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                }
                .buttonStyle(.plain)
                .foregroundStyle(.secondary)
                .accessibilityLabel(Text(String(
                    localized: isExpanded
                        ? "evalform.evidence.collapse"
                        : "evalform.evidence.expand",
                    bundle: .module
                )))
            }
        }
    }

    @ViewBuilder
    private func scoreBadge(_ result: EvalFormDraft.ItemResult?) -> some View {
        if let stated = result?.strengthScore {
            Text(verbatim: formatted(stated))
                .font(.caption.monospacedDigit().weight(.semibold))
        } else if let inferred = result?.strengthScoreInferred {
            // Inferred = suggestion, visually distinct + suffixed.
            Text(verbatim: "\(formatted(inferred))?")
                .font(.caption.monospacedDigit())
                .foregroundStyle(.orange)
        } else {
            Text(verbatim: "—")
                .font(.caption.monospacedDigit())
                .foregroundStyle(.tertiary)
        }
    }

    private func formatted(_ value: Double) -> String {
        let sign = value > 0 ? "+" : ""
        return "\(sign)\(String(format: "%g", value))"
    }
}
