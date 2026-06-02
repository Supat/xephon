import SwiftUI
import Summarizer

/// "Remote LLM Server" subsection on the Settings card. Lets the
/// user toggle the LM Studio backend on, point it at a
/// host / port / model, and verify the configuration with a
/// one-shot connectivity probe.
///
/// Per CLAUDE.md's local-first / remote-open posture this is the
/// only place where the LM Studio path can be turned on. The
/// Summarizer card's backend picker only shows the LM Studio
/// option once `enabled` is true AND a `baseURL` resolves.
struct LMStudioServerSection: View {
    @Bindable var settings: LMStudioSettings

    /// Latest test-connection result. Drives the inline status
    /// line below the Test button. Cleared when the user mutates
    /// host / port so a stale "OK" can't linger after
    /// the user repointed the client. Mutating `modelID` does NOT
    /// reset the probe — the list of discovered models is the
    /// reason we ran the probe, so we want it to keep driving the
    /// Picker even as the user switches selections.
    @State private var probe: ProbeResult = .idle
    @State private var probeTask: Task<Void, Never>?
    /// Model IDs returned by the most recent successful probe.
    /// Drives the modelRow's choice of UI: empty → TextField
    /// (manual entry fallback), non-empty → Picker. Persisted in
    /// @State only — re-discovering on launch is one tap and
    /// keeps us honest about which models are actually loaded
    /// right now.
    @State private var availableModels: [String] = []

    enum ProbeResult: Equatable {
        case idle
        case running
        case ok(modelsFound: Int, includesConfigured: Bool)
        case failed(message: String)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Toggle(isOn: $settings.enabled) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(String(localized: "settings.lmStudio.enable"))
                        .font(.callout)
                    Text(String(localized: "settings.lmStudio.enable.caption"))
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            if settings.enabled {
                hostPortRow
                modelRow
                timeoutRow
                structuredOutputToggle
                testRow
                privacyNote
            }
        }
        .onChange(of: settings.host) { _, _ in resetProbe() }
        .onChange(of: settings.port) { _, _ in resetProbe() }
    }

    /// A change to host / port invalidates both the probe status
    /// AND the model list (we may now be pointing at an entirely
    /// different server). Pulled into a helper so the two
    /// onChange handlers stay symmetric.
    private func resetProbe() {
        probe = .idle
        availableModels = []
    }

    @ViewBuilder
    private var hostPortRow: some View {
        HStack(spacing: 8) {
            VStack(alignment: .leading, spacing: 2) {
                Text(String(localized: "settings.lmStudio.host"))
                    .font(.caption)
                    .foregroundStyle(.secondary)
                TextField(
                    "localhost",
                    text: $settings.host
                )
                .textFieldStyle(.roundedBorder)
                .autocorrectionDisabled()
                .textInputAutocapitalization(.never)
            }
            VStack(alignment: .leading, spacing: 2) {
                Text(String(localized: "settings.lmStudio.port"))
                    .font(.caption)
                    .foregroundStyle(.secondary)
                TextField(
                    "1234",
                    value: $settings.port,
                    formatter: Self.portFormatter
                )
                .textFieldStyle(.roundedBorder)
                .keyboardType(.numberPad)
                .frame(width: 80)
            }
        }
    }

    @ViewBuilder
    private var modelRow: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(String(localized: "settings.lmStudio.model"))
                .font(.caption)
                .foregroundStyle(.secondary)
            if availableModels.isEmpty {
                // Manual-entry fallback: surfaced before the
                // first Test press OR when the most recent probe
                // failed. Keep the field editable so the user
                // can stash a model id ahead of the first
                // round-trip.
                TextField(
                    String(localized: "settings.lmStudio.model.placeholder"),
                    text: $settings.modelID
                )
                .textFieldStyle(.roundedBorder)
                .autocorrectionDisabled()
                .textInputAutocapitalization(.never)
            } else {
                // Discovered-models picker. "" sentinel maps to
                // "Auto (loaded model)" — LM Studio falls back
                // to whatever is loaded when the request
                // omits / blanks the model field. If the
                // currently-set modelID isn't in the discovered
                // list (model unloaded since the last probe, or
                // the user typed something custom) the picker
                // still selects it via an explicit `.tag` on a
                // synthesized "Other" row so the binding stays
                // stable; otherwise SwiftUI would silently
                // overwrite the value.
                Picker(
                    String(localized: "settings.lmStudio.model"),
                    selection: $settings.modelID
                ) {
                    Text(String(localized: "settings.lmStudio.model.auto"))
                        .tag("")
                    ForEach(availableModels, id: \.self) { id in
                        Text(verbatim: id).tag(id)
                    }
                    if !settings.modelID.isEmpty,
                       !availableModels.contains(settings.modelID) {
                        Text(String(
                            format: String(localized: "settings.lmStudio.model.unlisted"),
                            settings.modelID
                        ))
                        .tag(settings.modelID)
                    }
                }
                .pickerStyle(.menu)
                .labelsHidden()
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
    }

    @ViewBuilder
    private var structuredOutputToggle: some View {
        Toggle(isOn: $settings.useStructuredOutput) {
            VStack(alignment: .leading, spacing: 2) {
                Text(String(localized: "settings.lmStudio.structured"))
                    .font(.callout)
                Text(String(localized: "settings.lmStudio.structured.caption"))
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    @ViewBuilder
    private var timeoutRow: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text(String(localized: "settings.lmStudio.timeout"))
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
                Text(String(format: "%.0f s", settings.requestTimeoutSeconds))
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.tertiary)
            }
            Slider(
                value: $settings.requestTimeoutSeconds,
                in: 15...600,
                step: 15
            )
        }
    }

    @ViewBuilder
    private var testRow: some View {
        HStack(spacing: 8) {
            Button {
                runProbe()
            } label: {
                Label(
                    String(localized: "settings.lmStudio.test"),
                    systemImage: "network"
                )
            }
            .buttonStyle(.bordered)
            .disabled(settings.baseURL == nil || probe == .running)
            probeStatus
            Spacer()
        }
    }

    @ViewBuilder
    private var probeStatus: some View {
        switch probe {
        case .idle:
            EmptyView()
        case .running:
            ProgressView().controlSize(.small)
        case .ok(let count, let includesConfigured):
            HStack(spacing: 4) {
                Image(systemName: "checkmark.circle.fill")
                    .foregroundStyle(.green)
                if settings.modelID.isEmpty {
                    Text(String(format: String(localized: "settings.lmStudio.test.ok.any"), count))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                } else if includesConfigured {
                    Text(String(localized: "settings.lmStudio.test.ok.match"))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                } else {
                    Text(String(format: String(localized: "settings.lmStudio.test.ok.noMatch"), count))
                        .font(.caption)
                        .foregroundStyle(.orange)
                }
            }
        case .failed(let message):
            HStack(spacing: 4) {
                Image(systemName: "xmark.octagon.fill")
                    .foregroundStyle(.red)
                Text(message)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
            }
        }
    }

    @ViewBuilder
    private var privacyNote: some View {
        Text(String(localized: "settings.lmStudio.privacy"))
            .font(.caption2)
            .foregroundStyle(.tertiary)
            .fixedSize(horizontal: false, vertical: true)
    }

    private func runProbe() {
        probeTask?.cancel()
        guard let baseURL = settings.baseURL else { return }
        let modelID = settings.modelID
        let timeout = min(15.0, settings.requestTimeoutSeconds)
        probe = .running
        probeTask = Task { @MainActor in
            // 15 s implicit timeout for the probe — long enough
            // to forgive a sluggish Mac, short enough that a
            // misconfigured host doesn't make the user wait the
            // full inference timeout to learn it's wrong.
            //
            // The Task is `@MainActor`-explicit so the state
            // mutations after `await client.listModels()`
            // resume on MainActor without a redundant
            // `MainActor.run` hop. The `await` inside the
            // listModels call DOES hop off MainActor (the
            // client is its own actor), so the UI stays
            // responsive during the network round-trip.
            let client = LMStudioClient(
                configuration: LMStudioClient.Configuration(
                    baseURL: baseURL,
                    modelID: modelID,
                    requestTimeoutSeconds: timeout
                )
            )
            do {
                let ids = try await client.listModels()
                if Task.isCancelled { return }
                let matches = modelID.isEmpty || ids.contains(modelID)
                // Store the list FIRST so the picker (driven
                // by `availableModels`) refreshes in the same
                // body re-eval as the status badge.
                availableModels = ids
                probe = .ok(modelsFound: ids.count, includesConfigured: matches)
            } catch is CancellationError {
                return
            } catch {
                if Task.isCancelled { return }
                availableModels = []
                // Cap displayed error length so a pathological
                // body (e.g. server returns a long HTML 502
                // page) can't push the status row's
                // 2-line-clamped Text into a layout edge case.
                let raw = String(describing: error)
                let trimmed = raw.count > 240
                    ? raw.prefix(237) + "…"
                    : Substring(raw)
                probe = .failed(message: String(trimmed))
            }
        }
    }

    private static let portFormatter: NumberFormatter = {
        let f = NumberFormatter()
        f.numberStyle = .none
        f.allowsFloats = false
        f.minimum = 1
        f.maximum = 65535
        return f
    }()
}
