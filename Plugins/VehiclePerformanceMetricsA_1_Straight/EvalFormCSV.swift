import Foundation

/// Draft → CSV renderer: one row per sheet item plus leading
/// header-metadata rows. RFC-4180 quoting (fields containing
/// commas, quotes, or newlines are quoted; quotes doubled) so
/// Japanese comments with punctuation survive spreadsheet import.
public enum EvalFormCSV {
    public static func render(
        draft: EvalFormDraft,
        template: EvalFormTemplate
    ) -> String {
        var lines: [String] = []

        lines.append(row(["field", "value"]))
        for field in template.metadataFields {
            lines.append(row([field, draft.metadata[field] ?? ""]))
        }
        lines.append("")

        lines.append(row([
            "number", "item", "strengthScore", "strengthScoreInferred",
            "likeDislike", "comment", "evidenceRows", "evidenceRoads",
            "reviewed", "conflicts",
        ]))
        for item in template.items {
            let result = draft.items.first { $0.itemID == item.id }
            let evidence = result?.evidenceRows ?? []
            lines.append(row([
                item.number,
                item.titleJa,
                result?.strengthScore.map { String($0) } ?? "",
                result?.strengthScoreInferred.map { String($0) } ?? "",
                result?.likeDislike.map(String.init) ?? "",
                result?.comment ?? "",
                evidence.map(String.init).joined(separator: " "),
                roadPairs(evidence, roadByRow: draft.roadByRow),
                draft.reviewedItemIDs.contains(item.id) ? "yes" : "",
                (result?.conflicts ?? []).joined(separator: " / "),
            ]))
        }

        if let supplementary = draft.supplementaryComment {
            lines.append("")
            lines.append(row(["supplementaryComment", supplementary]))
            if let evidence = draft.supplementaryEvidenceRows, !evidence.isEmpty {
                lines.append(row([
                    "supplementaryEvidenceRows",
                    evidence.map(String.init).joined(separator: " "),
                ]))
            }
        }

        // Undetected block — the reviewer's to-fill list, one row
        // per scope so it filters cleanly in a spreadsheet.
        let undetected = EvalFormCoverage.undetected(
            draft: draft,
            template: template
        )
        if !undetected.isEmpty {
            lines.append("")
            lines.append(row(["undetected", "scope", "missing"]))
            if !undetected.missingHeaderFields.isEmpty {
                lines.append(row([
                    "undetected", "header",
                    undetected.missingHeaderFields.joined(separator: " "),
                ]))
            }
            for gaps in undetected.items {
                let number = template.items
                    .first { $0.id == gaps.itemID }?.number ?? gaps.itemID
                lines.append(row([
                    "undetected", number, EvalFormCoverage.gapPhrase(gaps),
                ]))
            }
        }
        return lines.joined(separator: "\n")
    }

    private static func row(_ fields: [String]) -> String {
        fields.map(escaped).joined(separator: ",")
    }

    /// "16:D路 36:D路" — road provenance for the cited rows that
    /// have one; empty when the draft carries no road map.
    private static func roadPairs(
        _ rows: [Int],
        roadByRow: [Int: String]?
    ) -> String {
        guard let roadByRow, !roadByRow.isEmpty else { return "" }
        return rows.compactMap { row in
            roadByRow[row].map { "\(row):\($0)" }
        }.joined(separator: " ")
    }

    static func escaped(_ field: String) -> String {
        guard field.contains(",") || field.contains("\"") || field.contains("\n")
        else { return field }
        return "\"\(field.replacingOccurrences(of: "\"", with: "\"\""))\""
    }
}
