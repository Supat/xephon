import SwiftUI

/// Surfaces per-modality construction failures captured during pipeline
/// pre-warm. Without this banner, a model that fails to load (e.g. an
/// FP16 ONNX graph that ORT rejects) silently disappears from the
/// available SER backends — the user sees a degraded picker with no
/// hint as to why.
struct PipelineDiagnosticsBanner: View {
    let messages: [String]

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(.yellow)
            VStack(alignment: .leading, spacing: 2) {
                Text("Some models didn't load")
                    .font(.caption.bold())
                ForEach(messages, id: \.self) { msg in
                    Text(msg)
                        .font(.caption2.monospaced())
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                        .truncationMode(.middle)
                }
            }
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .glassEffect(.regular.tint(.yellow.opacity(0.4)), in: RoundedRectangle(cornerRadius: 10, style: .continuous))
    }
}
