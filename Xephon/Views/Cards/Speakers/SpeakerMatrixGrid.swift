import SwiftUI

/// Square row-by-column speaker matrix used by every directed-pair
/// card in the app (leadership, interruption count, response
/// latency, …). Renders a header row of column-speaker chips, a
/// leading column of row-speaker chips, and a body of cells
/// supplied by the caller via the `cell` view builder.
///
/// Horizontally scrollable so sessions with more speakers than fit
/// the pane stay readable without compressing the cells. The cell
/// builder receives the row and column speaker IDs in that order
/// and is responsible for everything inside the cell rect
/// (background tint, value text, "—" for the diagonal, etc.) so
/// each card can render its own visual encoding.
struct SpeakerMatrixGrid<Cell: View>: View {
    let speakers: [String]
    var cellSize: CGFloat = 36
    var cellSpacing: CGFloat = 2
    var labelWidth: CGFloat = 44
    @ViewBuilder let cell: (_ row: String, _ col: String) -> Cell

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            VStack(alignment: .leading, spacing: cellSpacing) {
                HStack(spacing: cellSpacing) {
                    Color.clear.frame(width: labelWidth)
                    ForEach(speakers, id: \.self) { col in
                        speakerChip(col)
                            .frame(width: cellSize, alignment: .center)
                    }
                }
                ForEach(speakers, id: \.self) { row in
                    HStack(spacing: cellSpacing) {
                        speakerChip(row)
                            .frame(width: labelWidth, alignment: .leading)
                        ForEach(speakers, id: \.self) { col in
                            cell(row, col)
                        }
                    }
                }
            }
        }
    }

    @ViewBuilder
    private func speakerChip(_ id: String) -> some View {
        Text(id)
            .font(.caption2.bold())
            .foregroundStyle(speakerTint(for: id))
            .lineLimit(1)
            .minimumScaleFactor(0.7)
    }
}
