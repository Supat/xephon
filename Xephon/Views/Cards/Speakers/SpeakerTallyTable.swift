import SwiftUI

/// Per-speaker tally table scaffold shared by every "name +
/// N right-aligned numeric columns" section across the analytics
/// cards (mood rescue, modality disagreement, recovery time, …).
///
/// Renders a header row of column titles in a thin tertiary chrome
/// + one row per tally, each row leading with the speaker chip in
/// its assigned tint. The caller provides the per-row cell content
/// as a `@ViewBuilder` closure that emits one `Text` (or other
/// view) per column header — each cell is already given a
/// `.frame(maxWidth: .infinity, alignment: .center)` slot so the
/// caller only sets the value + tint.
struct SpeakerTallyTable<Row, ID: Hashable, Cells: View>: View {
    let rows: [Row]
    let rowID: KeyPath<Row, ID>
    let speakerID: (Row) -> String
    let columnHeaders: [String]
    @ViewBuilder let cells: (Row) -> Cells
    var labelWidth: CGFloat = 44

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 0) {
                Color.clear.frame(width: labelWidth)
                ForEach(columnHeaders, id: \.self) { header in
                    Text(header)
                        .font(.caption2.bold())
                        .foregroundStyle(.tertiary)
                        .frame(maxWidth: .infinity, alignment: .center)
                }
            }
            ForEach(rows, id: rowID) { row in
                HStack(spacing: 0) {
                    Text(speakerID(row))
                        .font(.caption2.bold())
                        .foregroundStyle(speakerTint(for: speakerID(row)))
                        .frame(width: labelWidth, alignment: .leading)
                    cells(row)
                }
            }
        }
    }
}
