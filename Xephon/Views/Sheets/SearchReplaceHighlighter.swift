import Foundation
import SwiftUI

/// Match URL scheme used by `SearchReplaceSheet` to make every
/// highlighted search hit individually tappable via attributed-
/// string links. The view's `OpenURLAction` decodes these URLs
/// and toggles the per-row selection set on the
/// `SearchReplaceCoordinator`.
///
/// Encoded via `URLComponents` rather than string-concat so the
/// UUID's hyphens (and any future query-item additions) stay
/// percent-safe.
enum SearchReplaceMatchURL {
    static let scheme = "xephon-match"

    static func make(utteranceID: UUID, index: Int) -> URL? {
        var components = URLComponents()
        components.scheme = scheme
        components.host = "m"
        components.queryItems = [
            URLQueryItem(name: "u", value: utteranceID.uuidString),
            URLQueryItem(name: "i", value: String(index)),
        ]
        return components.url
    }

    static func parse(_ url: URL) -> (utteranceID: UUID, index: Int)? {
        guard url.scheme == scheme,
              let components = URLComponents(
                url: url, resolvingAgainstBaseURL: false
              ),
              let uString = components.queryItems?
                .first(where: { $0.name == "u" })?.value,
              let utteranceID = UUID(uuidString: uString),
              let iString = components.queryItems?
                .first(where: { $0.name == "i" })?.value,
              let index = Int(iString) else {
            return nil
        }
        return (utteranceID, index)
    }
}

/// AttributedString assembly for `SearchReplaceSheet`'s per-row
/// transcript view. Pure function of the inputs — testable
/// without SwiftUI surface — and split out so the sheet body
/// stays a rendering layer.
enum SearchReplaceHighlighter {
    /// Build an AttributedString tinting each match by selection
    /// state. Pre-replace renders attach a `xephon-match://` link
    /// to each match so the openURL handler can toggle selection
    /// on tap; post-replace renders skip the links because the
    /// staged text is shown read-only with the inserted replace
    /// term highlighted in green.
    static func attributed(
        utteranceID: UUID,
        text: String,
        replaceTerm: String,
        matchRanges: [Range<String.Index>],
        selectedIndices: Set<Int>,
        replaced: Bool
    ) -> AttributedString {
        var attributed = AttributedString(text)
        // Post-replace path: highlight every occurrence of the
        // replace term so the user sees what landed.
        if replaced {
            let needle = replaceTerm
            guard !needle.isEmpty else { return attributed }
            let bg = Color.green.opacity(0.35)
            var cursor = text.startIndex
            while cursor < text.endIndex,
                  let r = text.range(
                    of: needle,
                    options: .caseInsensitive,
                    range: cursor..<text.endIndex
                  ) {
                if let attrRange = Range(r, in: attributed) {
                    attributed[attrRange].backgroundColor = bg
                }
                cursor = r.upperBound
            }
            return attributed
        }
        // Pre-replace path: each search-term match gets a
        // tappable link. Selected matches read green; unselected
        // stay yellow.
        for (idx, range) in matchRanges.enumerated() {
            guard let attrRange = Range(range, in: attributed) else { continue }
            let isSelected = selectedIndices.contains(idx)
            attributed[attrRange].backgroundColor = isSelected
                ? Color.green.opacity(0.55)
                : Color.yellow.opacity(0.55)
            attributed[attrRange].link = SearchReplaceMatchURL.make(
                utteranceID: utteranceID, index: idx
            )
            // Override link foreground so it doesn't paint blue
            // — we want the text to stay primary-tinted, with
            // only the background telling the user this is a
            // hot zone.
            attributed[attrRange].foregroundColor = .primary
        }
        return attributed
    }
}
