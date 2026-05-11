import SwiftUI

/// Settings card sitting above `PipelineCard`. Hosts the speech-boost
/// toggle (Capture's EQ) and the text-SER backend picker — controls that
/// configure how the pipeline runs but aren't part of the live stage
/// visualization itself.
struct SettingsCard<SpeechBoost: View, TextSER: View>: View {
    @ViewBuilder let speechBoostToggle: () -> SpeechBoost
    @ViewBuilder let textSERPicker: () -> TextSER

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            speechBoostToggle()
            textSERPicker()
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
    }
}
