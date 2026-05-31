import Foundation
import UniformTypeIdentifiers

/// UTType for our `.xph` session bundle. Mirrored in
/// `project.yml`'s `UTExportedTypeDeclarations` so iOS picks it up
/// system-wide on app install. `UTType(exportedAs:)` returns the
/// system-registered type when one matches the identifier, so the
/// document picker uses the same UTType to decide which files to
/// enable / grey out.
///
/// Session save/load no longer route through a SwiftUI
/// `FileDocument` adapter — `SessionFileCoordinator.saveSession`
/// encodes the bundle bytes inline and hands them to
/// `FilePickerCoordinator.presentExport`, and the import side
/// receives a URL from the coordinator and decodes it directly.
/// The UTType remains here so the picker knows what content type
/// to advertise.
extension UTType {
    static let xephonSession = UTType(
        exportedAs: "com.supatsaetia.xephon.session",
        conformingTo: .data
    )
}
