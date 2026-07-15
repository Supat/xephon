import SwiftUI
import XephonPluginKit

/// The plugin's control-pane page: coverage readout, run control,
/// and the per-item review list with evidence-row playback. Scores
/// follow the visual policy: stated values plain, inferred values
/// marked and tinted, conflicts called out.
struct EvalFormCard: View {
    @Bindable var model: EvalFormModel

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
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .glassEffect(in: RoundedRectangle(cornerRadius: 12, style: .continuous))
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
                    evidenceChips(evidence)
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
                evidenceChips(evidence)
            }
            ForEach(result?.conflicts ?? [], id: \.self) { conflict in
                Text(verbatim: "⚠ \(conflict)")
                    .font(.caption2)
                    .foregroundStyle(.orange)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    /// Evidence chips: tap = toggle row playback (same semantics
    /// as a transcript row's play button). An adaptive grid, not
    /// an HStack — the context window can put dozens of rows
    /// behind one item, and an overflowing HStack compresses each
    /// chip into a vertical character stack in the narrow pane.
    private func evidenceChips(_ rows: [Int]) -> some View {
        LazyVGrid(
            columns: [GridItem(.adaptive(minimum: 42), spacing: 4)],
            alignment: .leading,
            spacing: 2
        ) {
            ForEach(rows, id: \.self) { row in
                Button {
                    model.playRow(row)
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
