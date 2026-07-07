import Foundation
import Testing
@testable import Xephon

/// Pins the keyword export → import round-trip: text, group
/// membership (through the re-keying id remap), and — the
/// regression this file exists for — the color tag, which
/// `replaceContents` silently dropped by rebuilding keywords
/// without passing `tagColor` through.
@MainActor
@Suite("Keyword export/import round-trip")
struct KeywordRoundtripTests {

    private func makeStore() -> KeywordStore {
        KeywordStore(
            fileURL: FileManager.default.temporaryDirectory
                .appendingPathComponent("keyword-roundtrip-\(UUID().uuidString).json"),
            document: KeywordDocument(keywords: [], groups: [])
        )
    }

    @Test("replaceContents alone preserves tagColor")
    func replaceContentsKeepsTag() {
        let doc = KeywordDocument(
            keywords: [Keyword(text: "x", tagColor: .teal)],
            groups: []
        )
        let store = makeStore()
        store.replaceContents(with: doc)
        #expect(store.keywords.map(\.tagColor) == [.teal])
    }

    @Test("Color tags and groups survive export → import")
    func roundtripKeepsTagsAndGroups() throws {
        let source = makeStore()
        let groupID = try #require(source.addGroup(name: "G1"))
        source.add("資料", groupID: groupID)
        source.add("ミット")
        source.setTagColor(.teal, for: source.keywords[0].id)
        source.setTagColor(.rose, for: source.keywords[1].id)

        let data = try source.exportJSONData()
        let target = makeStore()
        try target.importJSONData(data)

        #expect(target.keywords.map(\.text) == ["資料", "ミット"])
        #expect(target.keywords.map(\.tagColor) == [.teal, .rose])
        // Group membership survives through the id remap; the ids
        // themselves are deliberately re-keyed.
        let importedGroupID = try #require(target.groups.first?.id)
        #expect(target.groups.map(\.name) == ["G1"])
        #expect(target.keywords[0].groupID == importedGroupID)
        #expect(target.keywords[1].groupID == nil)
        #expect(target.keywords[0].id != source.keywords[0].id)
    }
}
