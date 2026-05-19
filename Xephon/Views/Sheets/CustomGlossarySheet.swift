import SwiftUI
import UniformTypeIdentifiers
import SERText

/// Modal sheet for managing the user's custom glossary — the
/// per-term emotion bias the text-SER stage applies to each
/// utterance's Plutchik distribution. Raised from `SettingsCard`'s
/// "Custom Glossary" row.
///
/// Layout mirrors `TranscriptionReviewSheet`:
///   1. Content area (ScrollView of entry cards, or an empty
///      state when no entries are configured yet).
///   2. Divider.
///   3. Action bar with the global "Apply Bias" toggle.
///
/// Toolbar carries Done plus an overflow Menu hosting Add /
/// Import / Export — same three-dot disclosure pattern as the
/// other sheets.
///
/// All edits write back to the bound `GlossaryStore` immediately;
/// the store persists on every mutation and pings its `onChange`
/// hook so the controller can push the new lexicon snapshot to
/// the text-SER actor without a sheet-dismiss. On dismiss, the
/// controller also replays the active lexicon against the
/// stored utterances so weight tweaks become visible without
/// re-running text SER.
struct CustomGlossarySheet: View {
    @Bindable var store: GlossaryStore
    let onDismiss: () -> Void

    @State private var showingImporter = false
    @State private var showingExporter = false
    @State private var pendingExportDocument: GlossaryFileDocument?
    @State private var ioError: String?

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                applyBiasHeader
                Divider()
                content
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
                Divider()
                actionBar
            }
            // Opaque sheet backdrop — same reason as
            // `EditUtteranceSheet`: without this, the inner
            // `.glassEffect` cards sample whatever's underneath
            // the sheet and refract its colors through.
            .background(Color(uiColor: .systemBackground))
            .navigationTitle(String(localized: "glossary.title"))
            .toolbarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button(String(localized: "glossary.done"), action: onDismiss)
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Menu {
                        Button {
                            showingImporter = true
                        } label: {
                            Label(
                                String(localized: "glossary.import"),
                                systemImage: "square.and.arrow.down"
                            )
                        }
                        Button {
                            pendingExportDocument = GlossaryFileDocument(
                                document: store.exportDocument()
                            )
                            showingExporter = true
                        } label: {
                            Label(
                                String(localized: "glossary.export"),
                                systemImage: "square.and.arrow.up"
                            )
                        }
                        .disabled(store.entries.isEmpty)
                    } label: {
                        Image(systemName: "ellipsis.circle")
                    }
                }
            }
            .fileImporter(
                isPresented: $showingImporter,
                allowedContentTypes: [.json]
            ) { result in
                handleImport(result: result)
            }
            .fileExporter(
                isPresented: $showingExporter,
                document: pendingExportDocument,
                contentType: .json,
                defaultFilename: "glossary"
            ) { result in
                if case .failure(let error) = result {
                    ioError = String(describing: error)
                }
                pendingExportDocument = nil
            }
            .alert(
                String(localized: "glossary.error.title"),
                isPresented: Binding(
                    get: { ioError != nil },
                    set: { if !$0 { ioError = nil } }
                ),
                presenting: ioError
            ) { _ in
                Button(String(localized: "glossary.error.ok"), role: .cancel) {
                    ioError = nil
                }
            } message: { msg in
                Text(msg)
            }
        }
    }

    @ViewBuilder
    private var content: some View {
        if store.entries.isEmpty {
            emptyView
        } else {
            entryList
        }
    }

    @ViewBuilder
    private var entryList: some View {
        ScrollView(.vertical, showsIndicators: true) {
            VStack(alignment: .leading, spacing: 12) {
                ForEach($store.entries) { $entry in
                    entryCard(entry: $entry)
                }
            }
            .padding(20)
        }
    }

    @ViewBuilder
    private func entryCard(entry: Binding<LexiconBiasEntry>) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 8) {
                labelChip(entry.wrappedValue.label)
                Spacer(minLength: 4)
                Button(role: .destructive) {
                    // Defer the mutation past the current
                    // view-body cycle. Removing from
                    // `store.entries` synchronously while the
                    // ForEach is still holding a Binding into
                    // the deleted element crashes on the next
                    // binding read with an out-of-range index —
                    // a well-known SwiftUI gotcha. Capturing the
                    // id, hopping to the next MainActor tick,
                    // and removing there gives the ForEach a
                    // chance to drop the row's view first.
                    let id = entry.wrappedValue.id
                    Task { @MainActor in
                        store.entries.removeAll { $0.id == id }
                    }
                } label: {
                    Label(
                        String(localized: "glossary.delete"),
                        systemImage: "trash"
                    )
                    .labelStyle(.iconOnly)
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }
            TextField(
                String(localized: "glossary.entry.term.placeholder"),
                text: entry.term
            )
            .font(.body)
            .scrollContentBackground(.hidden)
            .padding(8)
            .background(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .fill(Color(uiColor: .secondarySystemBackground))
            )
            .autocorrectionDisabled()
            .textInputAutocapitalization(.never)

            HStack(spacing: 12) {
                Picker(
                    String(localized: "glossary.entry.label"),
                    selection: entry.label
                ) {
                    ForEach(PlutchikScore.Label.allCases, id: \.self) { label in
                        Text(Self.localizedLabel(label)).tag(label)
                    }
                }
                .pickerStyle(.menu)
                .labelsHidden()
                // Size the menu button to its longest possible
                // selection ("Anticipation") so the trailing
                // weight controls can't compress it into wrapping
                // when that label is currently picked.
                .fixedSize(horizontal: true, vertical: false)

                Spacer(minLength: 0)

                Text(String(localized: "glossary.entry.weight"))
                    .font(.caption2.weight(.semibold))
                    .foregroundStyle(.secondary)
                    .textCase(.uppercase)
                Text(String(format: "%.1f", entry.wrappedValue.weight))
                    .font(.body.monospacedDigit())
                    .foregroundStyle(.secondary)
                    .frame(width: 36, alignment: .trailing)
                Stepper(
                    "",
                    value: entry.weight,
                    in: 0.0...1.0,
                    step: 0.1
                )
                .labelsHidden()
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .glassEffect(in: RoundedRectangle(cornerRadius: 10, style: .continuous))
    }

    /// Per-entry emotion chip — same tint family used elsewhere
    /// for Plutchik labels. Tints chosen to map roughly onto
    /// Plutchik wheel quadrants so a glance at the chip reads
    /// "happy bucket" / "anger bucket" / etc. without naming.
    @ViewBuilder
    private func labelChip(_ label: PlutchikScore.Label) -> some View {
        Text(Self.localizedLabel(label))
            .font(.caption2.weight(.semibold))
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(
                Capsule().fill(Self.tint(for: label).opacity(0.18))
            )
            .foregroundStyle(Self.tint(for: label))
    }

    @ViewBuilder
    private var emptyView: some View {
        VStack(spacing: 12) {
            Image(systemName: "book.closed")
                .font(.largeTitle)
                .foregroundStyle(.secondary)
            Text(String(localized: "glossary.empty.title"))
                .font(.callout)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
            Text(String(localized: "glossary.empty.hint"))
                .font(.caption)
                .foregroundStyle(.tertiary)
                .multilineTextAlignment(.center)
        }
        .padding(40)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    /// Top header carrying the global "Apply Bias" toggle. Lives
    /// above the entry list so it reads as a setting that
    /// governs everything below it, with the hint footer making
    /// the off-vs-on semantics explicit.
    @ViewBuilder
    private var applyBiasHeader: some View {
        VStack(alignment: .leading, spacing: 6) {
            Toggle(isOn: $store.isEnabled) {
                Label(
                    String(localized: "glossary.enable"),
                    systemImage: "book.closed"
                )
            }
            .toggleStyle(.switch)
            Text(String(localized: "glossary.enable.hint"))
                .font(.caption2)
                .foregroundStyle(.tertiary)
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 12)
    }

    /// Bottom action bar carrying the Add Entry button —
    /// equivalent placement to the review sheet's Review /
    /// Re-review button: the primary forward action on the
    /// sheet, full-width, separated by a Divider.
    @ViewBuilder
    private var actionBar: some View {
        Button {
            store.add(
                LexiconBiasEntry(
                    term: "",
                    label: .joy,
                    weight: 0.5
                )
            )
        } label: {
            HStack(spacing: 8) {
                Image(systemName: "plus.circle.fill")
                Text(String(localized: "glossary.add"))
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 12)
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 10)
    }

    private func handleImport(result: Result<URL, any Error>) {
        switch result {
        case .success(let url):
            // The picker hands back a security-scoped URL on iPadOS;
            // we have to ask for access before reading or the load
            // silently returns an empty / permission-denied result.
            let needsScope = url.startAccessingSecurityScopedResource()
            defer {
                if needsScope { url.stopAccessingSecurityScopedResource() }
            }
            do {
                let data = try Data(contentsOf: url)
                try store.importJSONData(data)
            } catch {
                ioError = String(describing: error)
            }
        case .failure(let error):
            // CocoaError code 256 = "user cancelled" which the
            // picker raises as a failure. Skip surfacing it — the
            // user knows they cancelled.
            let nsError = error as NSError
            if nsError.code != NSUserCancelledError {
                ioError = String(describing: error)
            }
        }
    }

    /// Plutchik label names use the localizable strings defined
    /// alongside the rest of the affect labels. Falls back to the
    /// raw enum string when a string-table entry is missing so a
    /// stale localizer build doesn't strand the user with empty
    /// menu items.
    static func localizedLabel(_ label: PlutchikScore.Label) -> String {
        switch label {
        case .joy:          return String(localized: "plutchik.joy")
        case .sadness:      return String(localized: "plutchik.sadness")
        case .anticipation: return String(localized: "plutchik.anticipation")
        case .surprise:     return String(localized: "plutchik.surprise")
        case .anger:        return String(localized: "plutchik.anger")
        case .fear:         return String(localized: "plutchik.fear")
        case .disgust:      return String(localized: "plutchik.disgust")
        case .trust:        return String(localized: "plutchik.trust")
        }
    }

    /// Plutchik-wheel-inspired chip tints. Aligned with the
    /// existing modality-disagreement chip palette (orange / red
    /// for negatives; blue / green for positives) so glossary
    /// chips don't clash visually with other badges on the row.
    static func tint(for label: PlutchikScore.Label) -> Color {
        switch label {
        case .joy:          return .yellow
        case .sadness:      return .blue
        case .anticipation: return .orange
        case .surprise:     return .cyan
        case .anger:        return .red
        case .fear:         return .purple
        case .disgust:      return .green
        case .trust:        return .mint
        }
    }
}

/// `FileDocument` adapter for glossary JSON export. Read-only
/// (the importer goes through `.fileImporter` directly so it can
/// bypass the bidirectional FileDocument lifecycle and just read
/// the URL into a `Data` once). `JSON` is the only allowed
/// content type — keeps the picker's file list focused.
struct GlossaryFileDocument: FileDocument {
    static var readableContentTypes: [UTType] { [.json] }
    static var writableContentTypes: [UTType] { [.json] }

    var document: GlossaryDocument

    init(document: GlossaryDocument) {
        self.document = document
    }

    init(configuration: ReadConfiguration) throws {
        guard let data = configuration.file.regularFileContents else {
            throw CocoaError(.fileReadCorruptFile)
        }
        self.document = try JSONDecoder().decode(
            GlossaryDocument.self, from: data
        )
    }

    func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(document)
        return FileWrapper(regularFileWithContents: data)
    }
}
