import Foundation

/// Declarative description of one evaluation sheet — the tier-T2
/// "data pack" of docs/plugin_architecture.md. `Codable` so future
/// packs import as JSON documents; the A-1 sheet ships embedded as
/// the default (`.a1StraightRoad`).
public struct EvalFormTemplate: Codable, Sendable, Equatable {
    /// Stable template identity, stored in drafts so a draft knows
    /// which sheet it belongs to.
    public var id: String
    /// Sheet display name.
    public var name: String
    /// Relative-strength scale: −1…+1 in 0.125 steps for A-1. The
    /// parser accepts exactly these quantized values — anything
    /// else spoken is not a score.
    public var strengthScale: StrengthScale
    /// Preference scale (好き/嫌い): 1…9 for A-1.
    public var preferenceScale: PreferenceScale
    /// The evaluation rows.
    public var items: [Item]
    /// Header metadata fields the extractor asks the LLM to fill
    /// from session-start talk (vehicle, absorber specs, weather…).
    public var metadataFields: [String]
    /// Surface cues marking a row as metadata-bearing anywhere in
    /// the session (weather changes mid-drive, specs restated at a
    /// swap). Optional so pre-cue packs decode; nil falls back to
    /// session-start rows only.
    public var metadataCues: [String]?
    /// Per-road callout surfaces for section detection. When
    /// present this DEFINES the lexicon (order = match priority):
    /// `road` is the section title, matched when any of its
    /// `surfaces` appears in a row. Lets a pack accept partial
    /// forms exactly where they're unambiguous ("E3" for E3路 —
    /// but never a bare "D") and cover course-specific naming
    /// (5ヘルツ…). Optional; nil falls back to the exact labels
    /// derived from `referenceRoads`.
    public var roadCallouts: [RoadCallout]?

    public struct RoadCallout: Codable, Sendable, Equatable {
        /// Section title (the road's canonical label).
        public var road: String
        /// Transcript forms that count as this road's callout.
        public var surfaces: [String]

        public init(road: String, surfaces: [String]) {
            self.road = road
            self.surfaces = surfaces
        }
    }

    public struct StrengthScale: Codable, Sendable, Equatable {
        public var minimum: Double
        public var maximum: Double
        public var step: Double
        /// The sheet's ※ rubric: what each magnitude MEANS
        /// (0.125 = arguable … 1.0 = totally different). Injected
        /// into the extraction prompt so inferred magnitudes are
        /// calibrated the same way a human reads the sheet, and
        /// rendered as the card's footnote. Optional for packs
        /// that predate the field.
        public var anchors: [Anchor]?

        public struct Anchor: Codable, Sendable, Equatable {
            /// Unsigned magnitude the meaning applies to (±).
            public var magnitude: Double
            /// The rubric text as printed on the sheet.
            public var meaning: String

            public init(magnitude: Double, meaning: String) {
                self.magnitude = magnitude
                self.meaning = meaning
            }
        }

        /// The quantized values a stated score may take.
        public var allowedValues: [Double] {
            guard step > 0 else { return [] }
            return stride(from: minimum, through: maximum, by: step)
                .map { ($0 * 1000).rounded() / 1000 }
        }
    }

    public struct PreferenceScale: Codable, Sendable, Equatable {
        public var minimum: Int
        public var maximum: Int
    }

    public struct Item: Codable, Sendable, Equatable, Identifiable {
        /// Stable per-item id (draft results key on it).
        public var id: String
        /// The sheet's printed item number ("10", "11", …).
        public var number: String
        /// Japanese row title as printed on the sheet.
        public var titleJa: String
        /// English gloss.
        public var titleEn: String
        /// The perceptual definition column (what the item means).
        public var definition: String
        /// Reference roads / speeds this item is evaluated on.
        public var referenceRoads: [String]
        /// Surface forms that mark an utterance as being about this
        /// item — the onomatopoeia and their common variants. Used
        /// for candidate-row matching AND seeded into the keyword
        /// bank on activation.
        public var vocabulary: [String]
    }
}

extension EvalFormTemplate {
    /// The embedded default pack: 車両評価性能指標Ａ－１ (直線路走行専用).
    /// Field content transcribed from the lab's sheet — see
    /// docs/eval_form_autofill_research.md §1 for provenance. The
    /// sheet itself is confidential; this template stays in the
    /// private repo.
    public static let a1StraightRoad = EvalFormTemplate(
        id: "a1-straight-v1",
        name: "車両評価性能指標Ａ－１（直線路走行専用）",
        strengthScale: StrengthScale(
            minimum: -1.0,
            maximum: 1.0,
            step: 0.125,
            anchors: [
                .init(magnitude: 0.125, meaning: "なんとなく違うレベル（5:5で賛否両論、議論要）"),
                .init(magnitude: 0.25, meaning: "敏感な人が分かるレベル（30%の人が感じる）"),
                .init(magnitude: 0.5, meaning: "ほとんどの人が感じる（70%の人が感じる）"),
                .init(magnitude: 0.75, meaning: "明らかに差を感じられる（差は100%の人が感じる）"),
                .init(magnitude: 1.0, meaning: "まったく違う"),
            ]
        ),
        preferenceScale: PreferenceScale(minimum: 1, maximum: 9),
        items: [
            Item(
                id: "10_flat",
                number: "10",
                titleJa: "フラット感（あおり，ピッチ，ロール）",
                titleEn: "Flat feel (at body)",
                definition: "入力に対するバネ上の動きのバランス",
                referenceRoads: ["E3路 60km/h", "D路 40km/h", "G路 60km/h"],
                vocabulary: ["フラット感", "フラット", "あおり", "ピッチ", "ロール"]
            ),
            Item(
                id: "11_hyokohyoko",
                number: "11",
                titleJa: "ヒョコヒョコ（過減衰感、バネ上の動き）",
                titleEn: "Hyoko-hyoko",
                definition: "ストローク感の伴わない収まり悪い動きの有無（周波数：3-8Hz想定）",
                referenceRoads: ["E3路 60km/h", "F路 60km/h"],
                vocabulary: ["ヒョコヒョコ", "ひょこひょこ"]
            ),
            Item(
                id: "12_buruburu",
                number: "12",
                titleJa: "ブルブル（バネ下の収まり悪さ）",
                titleEn: "Buru-buru",
                definition: "減衰不足によるバタツキ（周波数：8-15Hz想定）",
                referenceRoads: ["F路 60km/h", "G路 60km/h", "段差路"],
                vocabulary: ["ブルブル", "ぶるぶる", "バタツキ", "ばたつき"]
            ),
            Item(
                id: "13_gotsugotsu",
                number: "13",
                titleJa: "ゴツゴツ",
                titleEn: "Gotsu-gotsu",
                definition: "入力の質感（周波数：15-30Hz以上想定）",
                referenceRoads: ["D路 40km/h", "G路 60km/h"],
                vocabulary: ["ゴツゴツ", "ごつごつ"]
            ),
            Item(
                id: "13_biribiri",
                number: "13b",
                titleJa: "ビリビリ",
                titleEn: "Biri-biri",
                definition: "入力の質感（周波数：30Hz以上想定）",
                referenceRoads: ["H路 60km/h"],
                vocabulary: ["ビリビリ", "びりびり"]
            ),
            Item(
                id: "14_harshness",
                number: "14",
                titleJa: "ハーシュネス（ショック・ノイズ・減衰）",
                titleEn: "Harshness (shock, noise, dumping)",
                definition: "入力Gの大きさ、角感",
                referenceRoads: ["段差路", "スペイン歩道"],
                vocabulary: ["ハーシュネス", "ハーシュ"]
            ),
        ],
        metadataFields: [
            "評価車両", "評価アイテム", "評価者",
            "基準SA仕様", "評価SA仕様", "評価位置",
            "天気", "気温", "路面状況",
        ],
        metadataCues: [
            "車両", "アブソーバー", "仕様", "SA",
            "天気", "気温", "路面", "運転席", "助手席",
        ],
        // Bare letters are acceptable callouts in Japanese-only
        // conversation ("Dに入ります") — the detector additionally
        // requires single-letter surfaces to stand alone (no Latin
        // alphanumeric neighbours), so 4WD/HD in the transcript
        // can't open a segment.
        roadCallouts: [
            .init(road: "E3路", surfaces: ["E3路", "E3"]),
            .init(road: "D路", surfaces: ["D路", "D"]),
            .init(road: "G路", surfaces: ["G路", "G"]),
            .init(road: "F路", surfaces: ["F路", "F"]),
            .init(road: "H路", surfaces: ["H路", "H"]),
            .init(road: "段差路", surfaces: ["段差路"]),
            .init(road: "スペイン歩道", surfaces: ["スペイン歩道", "スペイン"]),
        ]
    )

    /// Decode a template pack from imported JSON bytes.
    public static func decode(_ data: Data) throws -> EvalFormTemplate {
        try JSONDecoder().decode(EvalFormTemplate.self, from: data)
    }

    /// The section-detection lexicon: explicit `roadCallouts` when
    /// the pack provides them, else the exact derived labels (one
    /// surface each).
    public var effectiveRoadCallouts: [RoadCallout] {
        if let roadCallouts, !roadCallouts.isEmpty {
            return roadCallouts
        }
        return roadNames.map { RoadCallout(road: $0, surfaces: [$0]) }
    }

    /// Distinct road labels across all items' reference roads,
    /// speeds stripped ("F路 60km/h" → "F路"; "段差路" → itself),
    /// in first-appearance order. Fallback lexicon for packs
    /// without explicit `roadCallouts`.
    public var roadNames: [String] {
        var seen = Set<String>()
        var ordered: [String] = []
        for road in items.flatMap(\.referenceRoads) {
            let base = road.split(separator: " ").first.map(String.init) ?? road
            if seen.insert(base).inserted {
                ordered.append(base)
            }
        }
        return ordered
    }
}
