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
            if let draft = model.draft {
                if model.draftIsStale {
                    Text(String(localized: "evalform.stale", bundle: .module))
                        .font(.caption2)
                        .foregroundStyle(.orange)
                }
                metadataSection(draft)
                itemsSection(draft)
                if let lastExport = model.lastExport {
                    Text(verbatim: "Export: \(lastExport)")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }
            } else {
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
                Button {
                    model.exportMarkdown()
                } label: {
                    Label(
                        String(localized: "evalform.export", bundle: .module),
                        systemImage: "square.and.arrow.up"
                    )
                }
                .disabled(model.draft == nil)
            }
            .font(.caption)
            .buttonStyle(.bordered)
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
                    Text("\(entry.item.number). \(entry.item.titleJa)")
                        .font(.caption2)
                        .lineLimit(1)
                    Spacer(minLength: 8)
                    Text("\(entry.count)")
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
                Text("\(item.number). \(item.titleJa)")
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
            if let comment = result?.comment {
                Text(comment)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if let evidence = result?.evidenceRows, !evidence.isEmpty {
                // Evidence chips: tap = toggle row playback (same
                // semantics as a transcript row's play button).
                HStack(spacing: 4) {
                    ForEach(evidence, id: \.self) { row in
                        Button {
                            model.playRow(row)
                        } label: {
                            Text(verbatim: "[\(row)]")
                                .font(.caption2.monospacedDigit())
                        }
                        .buttonStyle(.plain)
                        .foregroundStyle(.tint)
                    }
                }
            }
            ForEach(result?.conflicts ?? [], id: \.self) { conflict in
                Text(verbatim: "⚠ \(conflict)")
                    .font(.caption2)
                    .foregroundStyle(.orange)
                    .fixedSize(horizontal: false, vertical: true)
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
