import SwiftUI
import UniformTypeIdentifiers
import Export

/// UTType for our `.xph` session bundle. Mirrored in
/// `project.yml`'s `UTExportedTypeDeclarations` so iOS picks it up
/// system-wide on app install. `UTType(exportedAs:)` returns the
/// system-registered type when one matches the identifier, so the
/// document picker uses the same UTType to decide which files to
/// enable / grey out.
extension UTType {
    static let xephonSession = UTType(
        exportedAs: "com.supatsaetia.xephon.session",
        conformingTo: .data
    )
}

/// SwiftUI document adapter for `SessionDocument`. Used as the type
/// parameter on `.fileExporter` so the system file-save sheet can
/// write our binary plist to whatever location the user picks, and
/// as the importer target on `.fileImporter` for round-tripping.
struct SessionFileDocument: FileDocument {
    static var readableContentTypes: [UTType] { [.xephonSession] }
    static var writableContentTypes: [UTType] { [.xephonSession] }

    var session: SessionDocument

    init(session: SessionDocument) {
        self.session = session
    }

    init(configuration: ReadConfiguration) throws {
        guard let data = configuration.file.regularFileContents else {
            throw CocoaError(.fileReadCorruptFile)
        }
        self.session = try SessionBundle.decode(data)
    }

    func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper {
        let data = try SessionBundle.encode(session)
        return FileWrapper(regularFileWithContents: data)
    }
}

/// Bundles the `.fileExporter` + error-alert modifiers for session
/// save into a single attachment. Pulling them out of ContentView's
/// `var body` keeps the body small enough for the Swift type-checker.
/// Import is routed through ContentView's main `.fileImporter` via a
/// shared `filePickerMode` switch — two `.fileImporter` modifiers
/// stacked on the same view chain silently collide on iPadOS 26
/// (the menu fires, the binding flips, the picker initializes, but
/// nothing presents).
struct SessionIOModifier: ViewModifier {
    @Binding var showingSaveSession: Bool
    @Binding var pendingSaveDocument: SessionFileDocument?
    @Binding var sessionIOError: String?
    let defaultFilename: String

    func body(content: Content) -> some View {
        content
            .fileExporter(
                isPresented: $showingSaveSession,
                document: pendingSaveDocument,
                contentType: .xephonSession,
                defaultFilename: defaultFilename
            ) { result in
                if case .failure(let error) = result {
                    sessionIOError = String(describing: error)
                }
                pendingSaveDocument = nil
            }
            .alert(
                "Session I/O Error",
                isPresented: Binding(
                    get: { sessionIOError != nil },
                    set: { if !$0 { sessionIOError = nil } }
                ),
                presenting: sessionIOError
            ) { _ in
                Button("OK", role: .cancel) { sessionIOError = nil }
            } message: { msg in
                Text(msg)
            }
    }
}
