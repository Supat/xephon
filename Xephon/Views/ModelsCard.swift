import SwiftUI
import Speech

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
                    ModelStatusRowView(row: row)
                }
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .glassEffect(in: RoundedRectangle(cornerRadius: 12, style: .continuous))
    }

    private var rows: [ModelStatusRow] {
        var out: [ModelStatusRow] = []
        out.append(ModelStatusRow(
            id: "speech",
            title: String(localized: "models.speech.title"),
            detail: String(localized: "models.speech.detail"),
            status: SpeechTranscriber.isAvailable ? .ready : .unavailable
        ))
        out.append(ModelStatusRow(
            id: "diarizer",
            title: String(localized: "models.diarizer.title"),
            detail: String(localized: "models.diarizer.detail"),
            status: recorder.pipelineHasDiarizer ? .ready : .failed
        ))
        out.append(ModelStatusRow(
            id: "acousticVAD",
            title: String(localized: "models.dimensional.title"),
            detail: String(localized: "models.dimensional.detail"),
            status: recorder.pipelineHasDimensionalSER ? .ready : .failed
        ))
        out.append(ModelStatusRow(
            id: "acousticCategorical",
            title: String(localized: "models.categorical.title"),
            detail: String(localized: "models.categorical.detail"),
            status: recorder.pipelineHasCategoricalSER ? .ready : .failed
        ))
        out.append(ModelStatusRow(
            id: "textSER",
            title: String(localized: "models.textSER.title"),
            detail: String(localized: "models.textSER.detail"),
            status: recorder.pipelineHasDeBERTaTextSER ? .ready : .failed
        ))
        out.append(ModelStatusRow(
            id: "demographics",
            title: String(localized: "models.demographics.title"),
            detail: String(localized: "models.demographics.detail"),
            status: recorder.pipelineHasAgeGenderSER ? .ready : .failed
        ))
        out.append(ModelStatusRow(
            id: "summarizerAppleFM",
            title: String(localized: "models.summarizerAppleFM.title"),
            detail: String(localized: "models.summarizerAppleFM.detail"),
            status: recorder.summarizerAppleFMAvailable ? .ready : .unavailable
        ))
        out.append(ModelStatusRow(
            id: "summarizerQwen",
            title: String(localized: "models.summarizerQwen.title"),
            detail: String(localized: "models.summarizerQwen.detail"),
            status: qwenStatus
        ))
        return out
    }

    private var qwenStatus: ModelStatus {
        if recorder.summarizerDownloading { return .downloading }
        return recorder.summarizerModelInstalled ? .ready : .notInstalled
    }
}

/// One row in the models card.
struct ModelStatusRow: Identifiable, Equatable {
    let id: String
    let title: String
    let detail: String
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
            }
            Spacer(minLength: 6)
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
