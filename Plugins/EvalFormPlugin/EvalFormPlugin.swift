import Foundation
import SwiftUI
import XephonPluginKit

/// The evaluation-form auto-fill plugin — first shipped consumer of
/// the plugin architecture (docs/plugin_architecture.md §5, design
/// in docs/eval_form_autofill_research.md). Fills the A-1 ride-
/// quality sheet from the session transcript: deterministic
/// spoken-score capture first, schema-constrained per-item LLM
/// extraction second, human review always.
public struct EvalFormPlugin: XephonPlugin {
    public static let id = PluginID("xephon.evalform")
    public static var displayName: String {
        String(localized: "evalform.displayName", bundle: .module)
    }
    /// v2 added `reviewedItemIDs` to the draft; v1 payloads migrate
    /// on read (see `EvalFormDraft.restore(data:storedVersion:)`).
    public static let payloadVersion = 2

    public init() {}

    public func activate(host: any PluginHost) -> PluginHandle {
        let model = EvalFormModel(host: host)
        // Seed the ACTIVE sheet's vocabulary (imported pack or the
        // embedded A-1) into the keyword bank so the app's existing
        // matching, homophone review, and timeline strips light up
        // for the evaluation vocabulary. Idempotent by contract.
        model.seedKeywords()
        return PluginHandle(
            onSessionEvent: { model.handle($0) },
            pages: [
                PluginPageDescriptor(
                    id: "xephon.evalform.page",
                    title: Self.displayName,
                    systemImage: "checklist"
                ) {
                    AnyView(EvalFormCard(model: model))
                }
            ],
            menuCommands: [
                PluginMenuCommand(
                    id: "xephon.evalform.export",
                    title: String(localized: "evalform.menu.export", bundle: .module),
                    systemImage: "checklist"
                ) {
                    model.exportMarkdown()
                }
            ]
        )
    }
}
