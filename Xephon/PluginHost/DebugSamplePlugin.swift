#if DEBUG
import SwiftUI
import XephonPluginKit
import XephonLogging

/// Debug-only demonstration plugin exercising every Phase 0/1
/// extension point end-to-end on a real device: session events →
/// observable state, snapshot reads, payload persistence through
/// Save → Open, a control-pane page, row playback, a text export
/// through the root picker, and a File-menu item. It is the living
/// example for plugin authors until the A1 Eval plugin (VehiclePerformanceMetricsA_1_Straight) exists — keep it
/// small and idiomatic.
///
/// Deliberately NOT localized beyond `String(localized:)` pass-
/// throughs of already-shipped keys: this UI never ships (Run
/// builds Release; see `xephonInstalledPlugins`).
struct DebugSamplePlugin: XephonPlugin {
    static let id = PluginID("xephon.debug.sample")
    static let displayName = "Sample (Debug)"
    static let payloadVersion = 1

    func activate(host: any PluginHost) -> PluginHandle {
        let model = DebugSampleModel(host: host)
        return PluginHandle(
            onSessionEvent: { model.handle($0) },
            pages: [
                PluginPageDescriptor(
                    id: "xephon.debug.sample.page",
                    title: Self.displayName,
                    systemImage: "puzzlepiece.extension"
                ) {
                    AnyView(DebugSampleCard(model: model))
                }
            ],
            menuCommands: [
                PluginMenuCommand(
                    id: "xephon.debug.sample.export",
                    title: "Export Plugin Sample…",
                    systemImage: "puzzlepiece.extension"
                ) {
                    model.exportSummary()
                }
            ]
        )
    }
}

/// The plugin's own state — event log, persisted note, export
/// status. Owns the host reference; the page and menu command both
/// drive this model.
@MainActor
@Observable
final class DebugSampleModel {
    private let host: any PluginHost

    private(set) var eventLog: [String] = []
    private(set) var lastExport: String = "—"
    /// Round-trips through the .xph payload so Save → Open
    /// verification is a text field away.
    var note: String {
        didSet { persistNote() }
    }

    init(host: any PluginHost) {
        self.host = host
        self.note = Self.decodeNote(
            from: host.storage(for: DebugSamplePlugin.self).sessionPayloadData
        )
    }

    func handle(_ event: SessionEvent) {
        switch event {
        case .sessionLoaded:
            eventLog.append("loaded")
            // Payload was restored with the session — re-read it.
            note = Self.decodeNote(
                from: host.storage(for: DebugSamplePlugin.self).sessionPayloadData
            )
        case .sessionCleared:
            eventLog.append("cleared")
            note = ""
        case .utterancesChanged(let version):
            eventLog.append("changed v\(version)")
        }
        if eventLog.count > 8 { eventLog.removeFirst(eventLog.count - 8) }
    }

    var snapshotSummary: String {
        let snap = host.session.snapshot()
        return "\(snap.utterances.count) utterances · v\(snap.utterancesVersion) · \(snap.speakerNames.count) named speakers"
    }

    var firstUtteranceID: UUID? {
        host.session.snapshot().utterances.first?.id
    }

    func playFirstUtterance() {
        guard let id = firstUtteranceID else { return }
        host.requestPlayback(utteranceID: id)
    }

    func exportSummary() {
        let snap = host.session.snapshot()
        let lines = snap.utterances.map { "\($0.speakerID)\t\($0.transcript)" }
        let body = "Xephon sample plugin export\n\(snapshotSummary)\n\n"
            + lines.joined(separator: "\n")
        host.export.presentExport(
            data: Data(body.utf8),
            contentType: .plainText,
            suggestedFilename: "xephon-plugin-sample.txt"
        ) { [weak self] outcome in
            self?.lastExport = String(describing: outcome)
        }
    }

    private func persistNote() {
        let storage = host.storage(for: DebugSamplePlugin.self)
        storage.sessionPayloadData = note.isEmpty ? nil : Data(note.utf8)
    }

    private static func decodeNote(from data: Data?) -> String {
        data.flatMap { String(data: $0, encoding: .utf8) } ?? ""
    }
}

/// The page card. Same glass-card look as the built-in cards.
struct DebugSampleCard: View {
    @Bindable var model: DebugSampleModel

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Sample plugin")
                .font(.caption.bold())
                .foregroundStyle(.secondary)
            Text(model.snapshotSummary)
                .font(.caption)
            TextField("Note (persists in .xph)", text: $model.note)
                .font(.caption)
                .textFieldStyle(.roundedBorder)
            HStack(spacing: 12) {
                Button("Play first row") { model.playFirstUtterance() }
                    .disabled(model.firstUtteranceID == nil)
                Button("Export…") { model.exportSummary() }
            }
            .font(.caption)
            .buttonStyle(.bordered)
            Text("Last export: \(model.lastExport)")
                .font(.caption2)
                .foregroundStyle(.tertiary)
            if !model.eventLog.isEmpty {
                Text("Events: \(model.eventLog.joined(separator: " · "))")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .glassEffect(in: RoundedRectangle(cornerRadius: 12, style: .continuous))
    }
}
#endif
