import Foundation
import SwiftUI
import UniformTypeIdentifiers

/// App-level singleton for routing every file-import / file-export
/// request through one `.fileImporter` and one `.fileExporter`
/// modifier attached at the ContentView root. Features that need
/// a picker call `presentImport` / `presentExport`; the
/// coordinator stores the request, the app-root modifier sees its
/// binding flip true, the picker presents, and the completion is
/// routed back through the request's closure.
///
/// Why centralize: SwiftUI's `.fileImporter` / `.fileExporter`
/// modifiers can interfere with each other when several coexist
/// in the same view hierarchy — sibling modifiers in a tab page,
/// a card body crowded with alerts, etc. The symptom is silent:
/// the trigger fires, state flips, nothing presents. Funneling
/// every request through one modifier pair eliminates the
/// conflict by construction. Adding a new caller is a one-line
/// `presentImport` / `presentExport` call instead of attaching
/// another pair of modifiers somewhere.
@MainActor
@Observable
public final class FilePickerCoordinator {
    /// A single export request held while the system picker is on
    /// screen. The `document` carries the bytes + UTType for
    /// `.fileExporter`'s `document:` parameter; the request itself
    /// also stores the originally-requested contentType + filename
    /// so the binding can read them when SwiftUI evaluates the
    /// modifier even after the request is being cleared.
    public struct ExportRequest {
        public let document: DataFileDocument
        public let contentType: UTType
        public let defaultFilename: String
        public let onResult: (Result<URL, any Error>) -> Void
    }

    /// A single import request. `onResult` fires once with the
    /// picker's verdict (the URL on success, the underlying error
    /// on user cancellation or framework failure).
    public struct ImportRequest {
        public let allowedTypes: [UTType]
        public let onResult: (Result<URL, any Error>) -> Void
    }

    /// Presentation flags drive the app-root `.fileImporter` /
    /// `.fileExporter` bindings via `$`-syntax. They're kept
    /// SEPARATE from the request payloads so the picker dismiss
    /// (which SwiftUI signals by setting `isPresented` false)
    /// doesn't accidentally clear `importRequest` before the
    /// completion handler runs — that race was what made every
    /// import silently no-op when the binding's set side cleared
    /// the request and the completion handler then read nil.
    /// Now: presentation flag is the binding; the request data
    /// outlives the flag flip and is only cleared inside `finish*`
    /// which the completion handler calls explicitly.
    public var isImporterPresented: Bool = false
    public var isExporterPresented: Bool = false
    public private(set) var importRequest: ImportRequest?
    public private(set) var exportRequest: ExportRequest?

    public init() {}

    /// Show the system file importer with `allowedTypes`. `onResult`
    /// runs on the MainActor with `.success(url)` for a picked
    /// file (security scope is the CALLER's responsibility — wrap
    /// the URL in `startAccessingSecurityScopedResource` /
    /// `stopAccessingSecurityScopedResource` before reading) or
    /// `.failure(error)` for user cancel / framework error.
    public func presentImport(
        allowedTypes: [UTType],
        onResult: @escaping (Result<URL, any Error>) -> Void
    ) {
        importRequest = ImportRequest(
            allowedTypes: allowedTypes,
            onResult: onResult
        )
        isImporterPresented = true
    }

    /// Show the system file exporter with `data` written as the
    /// file contents. `contentType` and `defaultFilename` map
    /// straight through to the picker. `onResult` mirrors the
    /// import contract.
    public func presentExport(
        data: Data,
        contentType: UTType,
        defaultFilename: String,
        onResult: @escaping (Result<URL, any Error>) -> Void
    ) {
        exportRequest = ExportRequest(
            document: DataFileDocument(data: data, contentType: contentType),
            contentType: contentType,
            defaultFilename: defaultFilename,
            onResult: onResult
        )
        isExporterPresented = true
    }

    /// Called from the app-root `.fileImporter` completion. Reads
    /// the request, clears both bookkeeping properties, then
    /// dispatches the result. Order matters: read first, then
    /// clear, so the closure handed to `presentImport` always
    /// sees its own original payload regardless of what SwiftUI
    /// has already done to the `isImporterPresented` binding.
    public func finishImport(_ result: Result<URL, any Error>) {
        let request = importRequest
        importRequest = nil
        isImporterPresented = false
        request?.onResult(result)
    }

    /// Mirror of `finishImport` for the export completion.
    public func finishExport(_ result: Result<URL, any Error>) {
        let request = exportRequest
        exportRequest = nil
        isExporterPresented = false
        request?.onResult(result)
    }
}

/// Generic `FileDocument` wrapper backing the centralized
/// exporter. Callers hand the coordinator raw `Data` + a UTType;
/// the wrapper holds them in a single concrete type the SwiftUI
/// `.fileExporter` modifier can accept without each call site
/// needing its own custom `FileDocument` struct. Read path is
/// supported so the same type could in principle be reused for
/// imports, but the coordinator's import flow returns a URL
/// instead (so callers can decide their own decoding pipeline).
public struct DataFileDocument: FileDocument, Sendable {
    // FileDocument's `readable/writableContentTypes` are static —
    // there's no instance-level override — so we whitelist every
    // content type any caller in the app might pass into
    // `presentExport(contentType:)`. Missing types log
    // "Attempting to export a document using a content type
    // (...) not included in its writableContentTypes" and abort
    // the export. `.data` stays as the generic fallback;
    // `.xephonSession` is the app's own `.xph` UTType (declared
    // in SessionFileDocument.swift).
    public static var readableContentTypes: [UTType] { [.data, .json, .plainText, .xephonSession] }
    public static var writableContentTypes: [UTType] { [.data, .json, .plainText, .xephonSession] }

    public var data: Data
    public var contentType: UTType

    public init(data: Data, contentType: UTType) {
        self.data = data
        self.contentType = contentType
    }

    public init(configuration: ReadConfiguration) throws {
        guard let data = configuration.file.regularFileContents else {
            throw CocoaError(.fileReadCorruptFile)
        }
        self.data = data
        self.contentType = .data
    }

    public func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper {
        FileWrapper(regularFileWithContents: data)
    }
}
