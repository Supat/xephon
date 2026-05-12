import SwiftUI

/// Settings card sitting above `PipelineCard`. Hosts the session-
/// language picker, the speech-boost toggle (Capture's EQ), the
/// text-SER backend picker, and the optional on-device summarizer
/// controls — controls that configure how the pipeline runs but
/// aren't part of the live stage visualization itself.
struct SettingsCard<
    Language: View,
    SpeechBoost: View,
    TextSER: View,
    Summarizer: View
>: View {
    @ViewBuilder let languagePicker: () -> Language
    @ViewBuilder let speechBoostToggle: () -> SpeechBoost
    @ViewBuilder let textSERPicker: () -> TextSER
    @ViewBuilder let summarizerControls: () -> Summarizer

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            languagePicker()
            speechBoostToggle()
            textSERPicker()
            summarizerControls()
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .glassEffect(in: RoundedRectangle(cornerRadius: 12, style: .continuous))
    }
}
