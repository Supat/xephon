import SwiftUI
import Diarization

/// Read-only directory of every speaker the session knows about and
/// the user-supplied display name (if any) attached to each one.
/// Lives on the left pane's cluster-diagnostics page below the
/// heatmap so the three views read together: "this is the cloud,
/// this is how close each pair is, this is who each id maps to."
///
/// Roster is the union of (a) ids referenced by at least one
/// utterance and (b) ids in the diarizer's internal speaker DB —
/// the latter covers entries FluidAudio retained from the
/// streaming pass that haven't been claimed by a row yet, so the
/// user can see them before deciding what to do with them.
struct SpeakerRosterCard: View {
    let recorder: RecordingController
    let cluster: SpeakerClusterSnapshot

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(String(localized: "roster.header"))
                    .font(.caption.bold())
                    .foregroundStyle(.secondary)
                Spacer()
                if !speakerIDs.isEmpty {
                    Text("\(speakerIDs.count)")
                        .font(.caption2.monospacedDigit())
                        .foregroundStyle(.tertiary)
                }
            }
            if speakerIDs.isEmpty {
                Text(String(localized: "roster.empty"))
                    .font(.caption)
                    .foregroundStyle(.tertiary)
                    .frame(maxWidth: .infinity, alignment: .leading)
            } else {
                VStack(alignment: .leading, spacing: 4) {
                    ForEach(speakerIDs, id: \.self) { id in
                        row(for: id)
                    }
                }
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .glassEffect(in: RoundedRectangle(cornerRadius: 12, style: .continuous))
    }

    /// Sorted union of utterance-roster ids and cluster-DB ids.
    /// Sort is alphabetical, which for `S0N` ids is also numeric;
    /// no point overthinking ordering when the keys are zero-padded
    /// 2-digit suffixes.
    private var speakerIDs: [String] {
        var seen: Set<String> = []
        var ids: [String] = []
        for id in recorder.knownSpeakerIDs() where seen.insert(id).inserted {
            ids.append(id)
        }
        for spk in cluster.speakers where seen.insert(spk.id).inserted {
            ids.append(spk.id)
        }
        return ids.sorted()
    }

    @ViewBuilder
    private func row(for id: String) -> some View {
        HStack(spacing: 8) {
            Text(id)
                .font(.caption.monospaced())
                .foregroundStyle(speakerTint(for: id))
                .frame(width: 40, alignment: .leading)
            if let name = recorder.speakerDisplayName(forStored: id),
               !name.isEmpty {
                Text(name)
                    .font(.caption)
                    .foregroundStyle(.primary)
                    .lineLimit(1)
                    .truncationMode(.tail)
            } else {
                Text("—")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            }
            Spacer(minLength: 4)
        }
    }
}
