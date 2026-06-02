import SwiftUI
import Summarizer

/// Compact chip showing which LLM backend is currently active for
/// the surfacing sheet (Session Summary or Transcription Review).
/// On-device backends (Apple FM, Qwen3, Llama-3-Swallow) read as
/// "On-device · <name>"; the LM Studio remote backend reads as
/// "Remote · LM Studio · <host>:<port>" so the user can verify at
/// a glance both *that* a remote is in use AND *which* one.
///
/// Visual: an SF Symbol that connotes the destination ("cpu" for
/// on-device, "network" for remote), the prefix label, and the
/// backend / endpoint detail. Background tint matches the
/// destination (subtle gray for on-device, accent for remote) so
/// the difference is visible at a glance even if the user isn't
/// reading the text — the remote case is the one with privacy
/// implications, so it gets the more salient treatment.
struct LLMBackendBadge: View {
    let recorder: RecordingController

    var body: some View {
        HStack(spacing: 6) {
            Image(systemName: glyph)
                .font(.caption2)
            Text(prefix)
                .font(.caption2.weight(.semibold))
            Text(detail)
                .font(.caption2)
                .lineLimit(1)
                .truncationMode(.middle)
        }
        .foregroundStyle(tint)
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(
            Capsule().fill(tint.opacity(0.15))
        )
        .accessibilityElement(children: .combine)
        .accessibilityLabel(Text("\(prefix) \(detail)"))
    }

    private var isRemote: Bool {
        recorder.summarizerBackend == .lmStudio
    }

    private var glyph: String {
        isRemote ? "network" : "cpu"
    }

    private var tint: Color {
        isRemote ? .accentColor : .secondary
    }

    private var prefix: String {
        isRemote
            ? String(localized: "summary.backend.remote")
            : String(localized: "summary.backend.local")
    }

    /// Backend-specific detail line. On-device gets the human
    /// model name; remote gets "LM Studio · host:port" so the
    /// user can confirm which server is the target without
    /// leaving the sheet.
    private var detail: String {
        switch recorder.summarizerBackend {
        case .appleFM:
            return String(localized: "settings.summarizer.backend.appleFM")
        case .qwen:
            return String(localized: "settings.summarizer.backend.qwen")
        case .llamaSwallow:
            return String(localized: "settings.summarizer.backend.llamaSwallow")
        case .lmStudio:
            let endpoint = recorder.lmStudioSettings.baseURL?.absoluteString
                ?? String(localized: "settings.summarizer.lmStudio.notConfigured")
            return "LM Studio · \(endpoint)"
        }
    }
}
