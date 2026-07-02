import SwiftUI
import Speech
import Summarizer

/// Card listing every model the analysis pipeline can call into,
/// alongside its current install / availability state. Sits below
/// `SummarizerCard` on the 4th left-pane page so users can see at
/// a glance which stages are live and which fell back / failed.
///
/// Status is read directly off `recorder` — no caching needed; each
/// row's value is a cheap bool lookup on the pipeline snapshot built
/// at pre-warm + the existing summarizer install flags.
struct ModelsCard: View {
    let recorder: RecordingController

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text(String(localized: "models.header"))
                    .font(.caption.bold())
                    .foregroundStyle(.secondary)
                Spacer()
            }
            VStack(spacing: 6) {
                ForEach(rows) { row in
                    ModelStatusRowView(row: row, onDownload: downloadAction(for: row))
                }
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .glassEffect(in: RoundedRectangle(cornerRadius: 12, style: .continuous))
    }

    private var rows: [ModelStatusRow] {
        // License values mirror `docs/models.md`. `restricted = true`
        // flags weights whose terms preclude commercial use (the
        // audEERING dim V/A/D and age/gender heads are CC-BY-NC);
        // those render in orange so a user surveying the pipeline
        // sees the redistribution constraint at a glance instead of
        // having to consult the docs. Apple SDK / Apache / MIT
        // licenses render in secondary tint — informational only.
        var out: [ModelStatusRow] = []
        out.append(ModelStatusRow(
            id: "speech",
            title: String(localized: "models.speech.title"),
            detail: String(localized: "models.speech.detail"),
            license: "Apple SDK",
            licenseIsRestricted: false,
            status: SpeechTranscriber.isAvailable ? .ready : .unavailable
        ))
        out.append(ModelStatusRow(
            id: "diarizer",
            title: String(localized: "models.diarizer.title"),
            detail: String(localized: "models.diarizer.detail"),
            license: "Apache-2.0 (Sortformer) · MIT (Silero VAD)",
            licenseIsRestricted: false,
            status: recorder.pipelineHasDiarizer ? .ready : .failed
        ))
        out.append(ModelStatusRow(
            id: "acousticVAD",
            title: String(localized: "models.dimensional.title"),
            detail: String(localized: "models.dimensional.detail"),
            license: "CC-BY-NC 4.0 (research-only)",
            licenseIsRestricted: true,
            status: recorder.pipelineHasDimensionalSER ? .ready : .failed
        ))
        out.append(ModelStatusRow(
            id: "acousticCategorical",
            title: String(localized: "models.categorical.title"),
            detail: String(localized: "models.categorical.detail"),
            license: "Apache-2.0",
            licenseIsRestricted: false,
            status: recorder.pipelineHasCategoricalSER ? .ready : .failed
        ))
        out.append(ModelStatusRow(
            id: "textSER",
            title: String(localized: "models.textSER.title"),
            detail: String(localized: "models.textSER.detail"),
            license: "MIT (base) · WRIME fine-tune TBD",
            licenseIsRestricted: false,
            status: recorder.pipelineHasDeBERTaTextSER ? .ready : .failed
        ))
        out.append(ModelStatusRow(
            id: "demographics",
            title: String(localized: "models.demographics.title"),
            detail: String(localized: "models.demographics.detail"),
            license: "CC-BY-NC 4.0 (research-only)",
            licenseIsRestricted: true,
            status: recorder.pipelineHasAgeGenderSER ? .ready : .failed
        ))
        out.append(ModelStatusRow(
            id: "summarizerAppleFM",
            title: String(localized: "models.summarizerAppleFM.title"),
            detail: String(localized: "models.summarizerAppleFM.detail"),
            license: "Apple SDK",
            licenseIsRestricted: false,
            status: recorder.summarizerAppleFMAvailable ? .ready : .unavailable
        ))
        out.append(ModelStatusRow(
            id: "summarizerQwen",
            title: String(localized: "models.summarizerQwen.title"),
            detail: String(localized: "models.summarizerQwen.detail"),
            license: "Apache-2.0",
            licenseIsRestricted: false,
            status: mlxStatus(installed: recorder.summarizerQwenInstalled, backend: .qwen)
        ))
        out.append(ModelStatusRow(
            id: "summarizerLlamaSwallow",
            title: String(localized: "models.summarizerLlamaSwallow.title"),
            detail: String(localized: "models.summarizerLlamaSwallow.detail"),
            license: "Llama 3 Community License · tokyotech-llm terms",
            licenseIsRestricted: false,
            status: mlxStatus(installed: recorder.summarizerLlamaSwallowInstalled, backend: .llamaSwallow)
        ))
        return out
    }

    /// Per-MLX-backend status. Paints "downloading" for whichever
    /// backend's weights are actually in flight (explicit Download
    /// button or auto-trigger), not just the active one.
    private func mlxStatus(installed: Bool, backend: SummarizerBackend) -> ModelStatus {
        if recorder.summarizerDownloadingBackend == backend { return .downloading }
        return installed ? .ready : .notInstalled
    }

    /// Download action for a row, when it's a not-installed MLX
    /// summarizer (the only on-demand-downloadable models). Nil for
    /// every other row.
    private func downloadAction(for row: ModelStatusRow) -> (() -> Void)? {
        guard row.status == .notInstalled else { return nil }
        let backend: SummarizerBackend
        switch row.id {
        case "summarizerQwen":          backend = .qwen
        case "summarizerLlamaSwallow":  backend = .llamaSwallow
        default:                        return nil
        }
        return { Task { await recorder.downloadSummarizerModel(backend) } }
    }
}

/// One row in the models card. `license` is the canonical
/// short-form string (e.g. "Apache-2.0", "CC-BY-NC 4.0"); when
/// `licenseIsRestricted` is true the row paints the license line
/// in orange so the user sees at a glance that the weights aren't
/// available for commercial reuse.
struct ModelStatusRow: Identifiable, Equatable {
    let id: String
    let title: String
    let detail: String
    let license: String
    let licenseIsRestricted: Bool
    let status: ModelStatus
}

/// Per-row state. Drives the trailing badge tint + icon + label.
enum ModelStatus: Equatable {
    case ready
    case downloading
    case unavailable
    case notInstalled
    case failed

    var badgeText: String {
        switch self {
        case .ready:        return String(localized: "models.status.ready")
        case .downloading:  return String(localized: "models.status.downloading")
        case .unavailable:  return String(localized: "models.status.unavailable")
        case .notInstalled: return String(localized: "models.status.notInstalled")
        case .failed:       return String(localized: "models.status.failed")
        }
    }

    var glyph: String {
        switch self {
        case .ready:        return "checkmark.circle.fill"
        case .downloading:  return "arrow.down.circle.fill"
        case .unavailable:  return "minus.circle.fill"
        case .notInstalled: return "circle.dashed"
        case .failed:       return "exclamationmark.triangle.fill"
        }
    }

    var tint: Color {
        switch self {
        case .ready:        return .green
        case .downloading:  return .accentColor
        case .unavailable:  return .secondary
        case .notInstalled: return .secondary
        case .failed:       return .orange
        }
    }
}

/// One row. Title + detail on the left, status badge on the right.
private struct ModelStatusRowView: View {
    let row: ModelStatusRow
    /// Non-nil for a not-installed, on-demand-downloadable model;
    /// renders a Download button in place of the status badge.
    var onDownload: (() -> Void)?

    var body: some View {
        HStack(alignment: .center, spacing: 10) {
            VStack(alignment: .leading, spacing: 2) {
                Text(row.title)
                    .font(.callout)
                    .lineLimit(1)
                    .minimumScaleFactor(0.8)
                Text(row.detail)
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                    .lineLimit(2)
                HStack(spacing: 4) {
                    Image(systemName: row.licenseIsRestricted
                        ? "exclamationmark.shield.fill"
                        : "scale.3d"
                    )
                    .font(.caption2)
                    Text(row.license)
                        .font(.caption2)
                        .lineLimit(1)
                        .minimumScaleFactor(0.85)
                }
                .foregroundStyle(row.licenseIsRestricted ? Color.orange : Color.secondary)
                .accessibilityElement(children: .combine)
                .accessibilityLabel(
                    String(
                        format: String(localized: "models.license.a11y"),
                        row.license
                    )
                )
            }
            Spacer(minLength: 6)
            // Not-installed downloadable models get an explicit
            // Download button in place of the redundant "Not
            // installed" badge; everything else shows the status
            // badge (incl. the in-flight "Downloading" state, which
            // replaces the button while the fetch runs).
            if let onDownload {
                Button(action: onDownload) {
                    Label(
                        String(localized: "models.download"),
                        systemImage: "arrow.down.circle"
                    )
                    .font(.caption2)
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .accessibilityLabel(Text(String(
                    format: String(localized: "models.download.a11y"),
                    row.title
                )))
            } else {
                HStack(spacing: 4) {
                    Image(systemName: row.status.glyph)
                        .font(.caption2)
                        .foregroundStyle(row.status.tint)
                    Text(row.status.badgeText)
                        .font(.caption2)
                        .foregroundStyle(row.status.tint)
                        .lineLimit(1)
                }
            }
        }
    }
}
