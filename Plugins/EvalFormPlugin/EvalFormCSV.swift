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
            "likeDislike", "comment", "evidenceRows", "reviewed", "conflicts",
        ]))
        for item in template.items {
            let result = draft.items.first { $0.itemID == item.id }
            lines.append(row([
                item.number,
                item.titleJa,
                result?.strengthScore.map { String($0) } ?? "",
                result?.strengthScoreInferred.map { String($0) } ?? "",
                result?.likeDislike.map(String.init) ?? "",
                result?.comment ?? "",
                (result?.evidenceRows ?? []).map(String.init).joined(separator: " "),
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
        return lines.joined(separator: "\n")
    }

    private static func row(_ fields: [String]) -> String {
        fields.map(escaped).joined(separator: ",")
    }

    static func escaped(_ field: String) -> String {
        guard field.contains(",") || field.contains("\"") || field.contains("\n")
        else { return field }
        return "\"\(field.replacingOccurrences(of: "\"", with: "\"\""))\""
    }
}
