import SwiftUI

/// Settings card sitting above `PipelineCard`. Hosts the session-
/// language picker, the speech-boost toggle (Capture's EQ), and the
/// text-SER backend picker — controls that configure how the pipeline
/// runs but aren't part of the live stage visualization itself. The
/// session-summarizer controls live on `SessionSummarySheet` so they
/// sit next to the artifact they affect.
///
/// Language and Text SER share a row when there's enough horizontal
/// space (landscape, regular iPad layout). In portrait — where the
/// left pane shrinks to ~1/3 of screen width — they stack vertically
/// so neither dropdown gets truncated. `ViewThatFits` picks the
/// first variant whose horizontal extent fits the available width,
/// falling back to the stacked layout otherwise. Speech-boost (a
/// toggle, distinct affordance) sits below on its own line.
struct SettingsCard<
    Language: View,
    SpeechBoost: View,
    TextSER: View
>: View {
    @ViewBuilder let languagePicker: () -> Language
    @ViewBuilder let speechBoostToggle: () -> SpeechBoost
    @ViewBuilder let textSERPicker: () -> TextSER

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            ViewThatFits(in: .horizontal) {
                HStack(alignment: .top, spacing: 16) {
                    languagePicker()
                    textSERPicker()
                }
                VStack(alignment: .leading, spacing: 12) {
                    languagePicker()
                    textSERPicker()
                }
            }
            speechBoostToggle()
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .glassEffect(in: RoundedRectangle(cornerRadius: 12, style: .continuous))
    }
}
