import SwiftUI
import UniformTypeIdentifiers

/// User-managed keyword bank, rendered inline on the keywords
/// page (not a sheet). Header carries an overflow Menu with
/// Import / Export — same three-dot disclosure pattern the
/// glossary sheet uses, kept inside the card so the page chrome
/// stays thin.
///
/// All mutations write back to the bound `KeywordStore`
/// immediately; the store persists on the next MainActor tick
/// and pings its `onChange` hook for any downstream consumer
/// that opts in.
struct KeywordsCard: View {
    @Bindable var store: KeywordStore

    @State private var newKeyword: String = ""
    @State private var showingImporter = false
    @State private var showingExporter = false
    @State private var pendingExportDocument: KeywordsFileDocument?
    @State private var ioError: String?
    @FocusState private var newKeywordFocused: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            header
            inputRow
            Divider()
            content
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .glassEffect(in: RoundedRectangle(cornerRadius: 12, style: .continuous))
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
            defaultFilename: "keywords"
        ) { result in
            if case .failure(let error) = result {
                ioError = String(describing: error)
            }
            pendingExportDocument = nil
        }
        .alert(
            String(localized: "keywords.error.title"),
            isPresented: Binding(
                get: { ioError != nil },
                set: { if !$0 { ioError = nil } }
            ),
            presenting: ioError
        ) { _ in
            Button(String(localized: "keywords.error.ok"), role: .cancel) {
                ioError = nil
            }
        } message: { msg in
            Text(msg)
        }
    }

    @ViewBuilder
    private var header: some View {
        HStack {
            Text(String(localized: "keywords.header"))
                .font(.caption.bold())
                .foregroundStyle(.secondary)
            Spacer()
            Text("\(store.keywords.count)")
                .font(.caption.monospacedDigit())
                .foregroundStyle(.tertiary)
            Menu {
                Button {
                    showingImporter = true
                } label: {
                    Label(
                        String(localized: "keywords.import"),
                        systemImage: "square.and.arrow.down"
                    )
                }
                Button {
                    pendingExportDocument = KeywordsFileDocument(
                        document: store.exportDocument()
                    )
                    showingExporter = true
                } label: {
                    Label(
                        String(localized: "keywords.export"),
                        systemImage: "square.and.arrow.up"
                    )
                }
                .disabled(store.keywords.isEmpty)
                Divider()
                Button(role: .destructive) {
                    store.removeAll()
                } label: {
                    Label(
                        String(localized: "keywords.removeAll"),
                        systemImage: "trash"
                    )
                }
                .disabled(store.keywords.isEmpty)
            } label: {
                Image(systemName: "ellipsis.circle")
                    .foregroundStyle(.secondary)
            }
        }
    }

    @ViewBuilder
    private var inputRow: some View {
        HStack(spacing: 6) {
            TextField(
                String(localized: "keywords.add.placeholder"),
                text: $newKeyword
            )
            .textInputAutocapitalization(.never)
            .autocorrectionDisabled()
            .focused($newKeywordFocused)
            .onSubmit(submitNewKeyword)
            Button(action: submitNewKeyword) {
                Image(systemName: "plus.circle.fill")
                    .font(.title3)
                    .foregroundStyle(submitDisabled ? Color.secondary : Color.accentColor)
                    .symbolRenderingMode(.hierarchical)
            }
            .buttonStyle(.borderless)
            .disabled(submitDisabled)
        }
    }

    @ViewBuilder
    private var content: some View {
        if store.keywords.isEmpty {
            emptyState
        } else {
            keywordList
        }
    }

    @ViewBuilder
    private var emptyState: some View {
        VStack(spacing: 4) {
            Text(String(localized: "keywords.empty.title"))
                .font(.callout)
                .foregroundStyle(.secondary)
            Text(String(localized: "keywords.empty.hint"))
                .font(.caption2)
                .foregroundStyle(.tertiary)
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 12)
    }

    @ViewBuilder
    private var keywordList: some View {
        VStack(alignment: .leading, spacing: 4) {
            ForEach(store.keywords) { keyword in
                HStack(spacing: 8) {
                    Text(keyword.text)
                        .font(.body)
                        .frame(maxWidth: .infinity, alignment: .leading)
                    Button {
                        store.remove(id: keyword.id)
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundStyle(.secondary)
                            .symbolRenderingMode(.hierarchical)
                    }
                    .buttonStyle(.borderless)
                    .accessibilityLabel(
                        String(
                            format: String(localized: "keywords.remove.a11y"),
                            keyword.text
                        )
                    )
                }
                .padding(.horizontal, 8)
                .padding(.vertical, 6)
                .background(
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .fill(Color(uiColor: .secondarySystemBackground))
                )
            }
        }
    }

    private var submitDisabled: Bool {
        newKeyword.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private func submitNewKeyword() {
        let trimmed = newKeyword.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        store.add(trimmed)
        newKeyword = ""
        // Keep focus so the user can chain additions without
        // re-tapping the field.
        newKeywordFocused = true
    }

    private func handleImport(result: Result<URL, any Error>) {
        switch result {
        case .success(let url):
            let didStart = url.startAccessingSecurityScopedResource()
            defer {
                if didStart { url.stopAccessingSecurityScopedResource() }
            }
            do {
                let data = try Data(contentsOf: url)
                try store.importJSONData(data)
            } catch {
                ioError = String(describing: error)
            }
        case .failure(let error):
            ioError = String(describing: error)
        }
    }
}

/// `FileDocument` adapter for keyword JSON export. Same shape as
/// `GlossaryFileDocument` — read path delegates to `JSONDecoder`,
/// write path to the store's pretty-printed export.
struct KeywordsFileDocument: FileDocument {
    static var readableContentTypes: [UTType] { [.json] }
    static var writableContentTypes: [UTType] { [.json] }

    var document: KeywordDocument

    init(document: KeywordDocument) {
        self.document = document
    }

    init(configuration: ReadConfiguration) throws {
        guard let data = configuration.file.regularFileContents else {
            throw CocoaError(.fileReadCorruptFile)
        }
        self.document = try JSONDecoder().decode(KeywordDocument.self, from: data)
    }

    func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(document)
        return FileWrapper(regularFileWithContents: data)
    }
}
