import SwiftUI

/// First-launch model hydration UI. Shown until the controller flips
/// `modelsReady = true`. Renders per-file status (installed / bundled /
/// downloading / completed / failed) and surfaces a Retry button when
/// the overall phase is `.failed`.
struct SetupView: View {
    @Bindable var controller: RecordingController

    var body: some View {
        let state = controller.modelDownload
        VStack(spacing: 24) {
            VStack(spacing: 8) {
                Image(systemName: "waveform.and.mic")
                    .font(.system(size: 48))
                    .foregroundStyle(.tint)
                Text("Setting up Xephon")
                    .font(.title2.bold())
                Text("Downloading the on-device emotion models. About \(formatMB(ModelManifest.approximateTotalBytes)) over Wi-Fi — only happens once.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                    .fixedSize(horizontal: false, vertical: true)
            }

            VStack(alignment: .leading, spacing: 12) {
                ForEach(ModelManifest.entries) { entry in
                    EntryRow(entry: entry, state: state)
                }
            }
            .frame(maxWidth: 480)

            ProgressView(value: state.fractionComplete) {
                if let label = state.currentEntry, case .running = state.phase {
                    Text(label).font(.caption.monospacedDigit())
                } else if case .failed(let msg) = state.phase {
                    Text(msg).font(.caption).foregroundStyle(.red)
                } else {
                    Text("Preparing…").font(.caption).foregroundStyle(.secondary)
                }
            }
            .progressViewStyle(.linear)
            .frame(maxWidth: 480)

            if case .failed = state.phase {
                Button {
                    Task { await controller.retryModelDownload() }
                } label: {
                    Label("Retry", systemImage: "arrow.clockwise")
                        .font(.body.bold())
                        .padding(.horizontal, 24)
                        .padding(.vertical, 10)
                }
                .buttonStyle(.borderedProminent)
            }
        }
        .padding(32)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

private struct EntryRow: View {
    let entry: ModelEntry
    let state: ModelDownloadState

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                Text(entry.displayName)
                    .font(.subheadline.bold())
                ForEach(entry.files, id: \.assetName) { file in
                    FileLine(file: file, state: state)
                }
            }
            Spacer()
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 10))
    }
}

private struct FileLine: View {
    let file: ModelFile
    let state: ModelDownloadState

    var body: some View {
        HStack(spacing: 6) {
            statusIcon
            Text(file.assetName)
                .font(.caption.monospaced())
                .lineLimit(1)
                .truncationMode(.middle)
            Spacer(minLength: 8)
            statusText
        }
        .foregroundStyle(.secondary)
    }

    @ViewBuilder
    private var statusIcon: some View {
        switch state.fileStatus[file.assetName] ?? .pending {
        case .pending:
            Image(systemName: "circle.dashed").foregroundStyle(.secondary)
        case .satisfied:
            Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
        case .downloading:
            ProgressView().controlSize(.small)
        case .completed:
            Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
        case .failed:
            Image(systemName: "xmark.octagon.fill").foregroundStyle(.red)
        }
    }

    @ViewBuilder
    private var statusText: some View {
        switch state.fileStatus[file.assetName] ?? .pending {
        case .pending:
            Text("waiting").font(.caption2)
        case .satisfied(let source):
            Text(source).font(.caption2)
        case .downloading:
            Text("\(formatMB(state.fileBytes[file.assetName] ?? 0)) / \(formatMB(state.fileExpected[file.assetName] ?? file.approximateBytes))")
                .font(.caption2.monospacedDigit())
        case .completed:
            Text(formatMB(file.approximateBytes)).font(.caption2.monospacedDigit())
        case .failed(let msg):
            Text(msg).font(.caption2).foregroundStyle(.red).lineLimit(1)
        }
    }
}

private func formatMB(_ bytes: Int64) -> String {
    let mb = Double(bytes) / 1024 / 1024
    if mb >= 100 { return String(format: "%.0f MB", mb) }
    if mb >= 10  { return String(format: "%.1f MB", mb) }
    if mb >= 1   { return String(format: "%.1f MB", mb) }
    let kb = Double(bytes) / 1024
    return String(format: "%.0f KB", kb)
}
