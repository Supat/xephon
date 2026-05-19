import SwiftUI
import UniformTypeIdentifiers
import SERText

/// Modal sheet for managing the user's custom glossary — the
/// per-term emotion bias the text-SER stage applies to each
/// utterance's Plutchik distribution. Raised from `SettingsCard`'s
/// "Custom Glossary…" row.
///
/// Layout:
///   1. Global enable toggle. Off = the lexicon is bypassed
///      entirely; entries stay parked for later.
///   2. List of entries, one per row: term TextField, emotion
///      picker, weight stepper. Swipe-to-delete removes a row.
///   3. "Add entry" button below the list.
///   4. Toolbar Import / Export buttons surface JSON file I/O
///      via `.fileImporter` / `.fileExporter`.
///
/// All edits write back to the bound `GlossaryStore` immediately;
/// the store persists on every mutation and pings its `onChange`
/// hook so the controller can push the new lexicon snapshot to
/// the text-SER actor without a sheet-dismiss.
struct CustomGlossarySheet: View {
    @Bindable var store: GlossaryStore
    let onDismiss: () -> Void

    @State private var showingImporter = false
    @State private var showingExporter = false
    @State private var pendingExportDocument: GlossaryFileDocument?
    @State private var ioError: String?

    var body: some View {
        NavigationStack {
            List {
                Section {
                    Toggle(isOn: $store.isEnabled) {
                        Label(
                            String(localized: "glossary.enable"),
                            systemImage: "book.closed"
                        )
                    }
                } footer: {
                    Text(String(localized: "glossary.enable.hint"))
                }

                Section {
                    ForEach($store.entries) { $entry in
                        entryRow(entry: $entry)
                    }
                    .onDelete { offsets in
                        store.remove(at: offsets)
                    }
                    Button {
                        store.add(
                            LexiconBiasEntry(
                                term: "",
                                label: .joy,
                                weight: 0.5
                            )
                        )
                    } label: {
                        Label(
                            String(localized: "glossary.add"),
                            systemImage: "plus.circle.fill"
                        )
                    }
                } header: {
                    Text(String(localized: "glossary.entries"))
                } footer: {
                    if store.entries.isEmpty {
                        Text(String(localized: "glossary.empty.hint"))
                    } else {
                        Text(String(localized: "glossary.weight.hint"))
                    }
                }
            }
            .navigationTitle(String(localized: "glossary.title"))
            .toolbarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
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
    private func entryRow(entry: Binding<LexiconBiasEntry>) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            TextField(
                String(localized: "glossary.entry.term.placeholder"),
                text: entry.term
            )
            .textFieldStyle(.roundedBorder)
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

                Spacer(minLength: 0)

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
        .padding(.vertical, 4)
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
