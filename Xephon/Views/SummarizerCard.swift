import SwiftUI
import Summarizer

/// Card hosting the on-device session-summarizer configuration:
/// enable toggle, backend picker (Apple FM vs MLX Qwen), install /
/// progress state, and the Remove-model affordance for reclaiming
/// the ~4.6 GB working set.
///
/// Lives on the left pane's 4th TabView page so the controls have
/// a stable home outside the summary sheet — the sheet itself is
/// now strictly a result viewer.
struct SummarizerCard: View {
    let recorder: RecordingController

    /// Drives the destructive-action confirmation dialog for the
    /// Remove-model glyph. Deleting the Qwen weights forces a
    /// ~4.6 GB re-download, so a tap-confirm gate avoids the
    /// "oops I meant to tap something else" failure mode that
    /// borderless icon buttons are especially prone to.
    @State private var showingRemoveModelConfirm: Bool = false

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text(String(localized: "summarizer.header"))
                    .font(.caption.bold())
                    .foregroundStyle(.secondary)
                Spacer()
            }
            HStack(spacing: 12) {
                Text(String(localized: "settings.summarizer.enable"))
                    .font(.callout)
                Spacer(minLength: 0)
                if recorder.summarizerEnabled {
                    summarizerBackendPicker
                }
                Toggle(
                    "",
                    isOn: Binding(
                        get: { recorder.summarizerEnabled },
                        set: { newValue in
                            Task { await recorder.setSummarizerEnabled(newValue) }
                        }
                    )
                )
                .labelsHidden()
                .toggleStyle(.switch)
            }
            if recorder.summarizerEnabled {
                summarizerStatusLine
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .glassEffect(in: RoundedRectangle(cornerRadius: 12, style: .continuous))
    }

    @ViewBuilder
    private var summarizerBackendPicker: some View {
        Picker(
            String(localized: "settings.summarizer.backend"),
            selection: Binding(
                get: { recorder.summarizerBackend },
                set: { newValue in
                    Task { await recorder.setSummarizerBackend(newValue) }
                }
            )
        ) {
            Text(String(localized: "settings.summarizer.backend.appleFM"))
                .tag(SummarizerBackend.appleFM)
            Text(String(localized: "settings.summarizer.backend.qwen"))
                .tag(SummarizerBackend.qwen)
        }
        .pickerStyle(.menu)
        .labelsHidden()
    }

    @ViewBuilder
    private var summarizerStatusLine: some View {
        switch recorder.summarizerBackend {
        case .appleFM:
            if recorder.summarizerAppleFMAvailable {
                HStack(spacing: 6) {
                    Image(systemName: "checkmark.circle.fill")
                        .font(.caption2)
                        .foregroundStyle(.green)
                    Text(String(localized: "settings.summarizer.ready"))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            } else {
                Text(String(localized: "settings.summarizer.appleFM.unavailable"))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        case .qwen:
            if recorder.summarizerDownloading {
                HStack(spacing: 8) {
                    CircularDownloadProgress(
                        downloaded: recorder.modelDownload.downloadedBytes,
                        total: recorder.modelDownload.totalBytes
                    )
                    Text(downloadProgressText)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            } else if recorder.summarizerModelInstalled {
                HStack(spacing: 6) {
                    Image(systemName: "checkmark.circle.fill")
                        .font(.caption2)
                        .foregroundStyle(.green)
                    Text(String(localized: "settings.summarizer.ready"))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer(minLength: 6)
                    Button {
                        showingRemoveModelConfirm = true
                    } label: {
                        Label(
                            String(localized: "settings.summarizer.remove"),
                            systemImage: "trash"
                        )
                        .labelStyle(.iconOnly)
                        .font(.caption)
                    }
                    .buttonStyle(.borderless)
                    .accessibilityLabel(Text(String(localized: "settings.summarizer.remove")))
                    .confirmationDialog(
                        String(localized: "settings.summarizer.removeConfirm.title"),
                        isPresented: $showingRemoveModelConfirm,
                        titleVisibility: .visible
                    ) {
                        Button(
                            String(localized: "settings.summarizer.removeConfirm.confirm"),
                            role: .destructive
                        ) {
                            Task { await recorder.removeSummarizerModel() }
                        }
                        Button(
                            String(localized: "settings.summarizer.removeConfirm.cancel"),
                            role: .cancel
                        ) {}
                    } message: {
                        Text(String(localized: "settings.summarizer.removeConfirm.message"))
                    }
                }
            } else {
                Text(String(localized: "settings.summarizer.notInstalled"))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }

    /// "Downloading Qwen3 · 412 MB of 4.6 GB" or similar — the
    /// fraction comes from the circular indicator next to it, so
    /// the text is byte counts, not a percent.
    private var downloadProgressText: String {
        let downloaded = recorder.modelDownload.downloadedBytes
        let total = recorder.modelDownload.totalBytes
        if total > 0, downloaded > 0 {
            return String(
                format: String(localized: "settings.summarizer.downloading.bytes"),
                Self.formatBytes(downloaded),
                Self.formatBytes(total)
            )
        }
        return String(localized: "settings.summarizer.downloading")
    }

    /// `1.7 GB`, `412 MB`, etc. — tracks Apple's convention for
    /// human-readable byte sizes (decimal SI).
    private static func formatBytes(_ bytes: Int64) -> String {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .decimal
        formatter.includesUnit = true
        formatter.allowedUnits = [.useMB, .useGB]
        return formatter.string(fromByteCount: bytes)
    }
}
