import Foundation
import Fusion

/// Known-answer eval harness, model-independent of any human
/// evaluator: sessions are GENERATED from a fact spec, so the
/// correct sheet is known by construction and every metric is
/// computable mechanically. What it measures is extraction
/// fidelity — "given that these things were said, does the
/// pipeline recover exactly them, cite the right rows, and invent
/// nothing" — NOT the validity of a human's ride judgment; the
/// ground-truth eval (research doc §6) remains the final gate for
/// that.
public enum EvalFormSynthetic {

    // MARK: - Spec

    /// What the synthetic evaluator "said" about one item.
    public enum ItemFact: Sendable, Equatable {
        /// A spoken quantized score (+ optional spoken 1–9
        /// preference). Expect: strengthScore == value, evidence
        /// hits the planted row.
        case statedScore(Double, preference: Int? = nil)
        /// Discussed qualitatively, no number. Expect: comment
        /// non-nil, strengthScore nil (an inferred suggestion is
        /// allowed, a stated fill is a violation).
        case qualitativeOnly
        /// Never mentioned. Expect: everything empty.
        case absent
    }

    public struct Spec: Sendable {
        /// itemID → fact. Items missing from the map are `.absent`.
        public var facts: [String: ItemFact]
        /// Header fields actually stated in the opening.
        public var metadata: [String: String]
        /// Unrelated drive-talk rows interleaved between facts.
        public var distractorCount: Int
        /// Deterministic layout seed.
        public var seed: UInt64

        public init(
            facts: [String: ItemFact],
            metadata: [String: String] = [:],
            distractorCount: Int = 12,
            seed: UInt64 = 0x5EED
        ) {
            self.facts = facts
            self.metadata = metadata
            self.distractorCount = distractorCount
            self.seed = seed
        }
    }

    /// The generated session plus its by-construction truth.
    public struct Session: Sendable {
        public let utterances: [UtteranceEstimate]
        public let expected: Expected
    }

    public struct Expected: Sendable {
        /// itemID → the stated score that must be recovered.
        public var statedScores: [String: Double] = [:]
        /// itemID → the stated preference that must be recovered.
        public var preferences: [String: Int] = [:]
        /// Items that were discussed (stated or qualitative) — a
        /// comment is expected for these.
        public var discussedItems: Set<String> = []
        /// itemID → 1-based rows where its facts were planted.
        /// Recovered evidence must intersect these.
        public var factRows: [String: [Int]] = [:]
        /// Stated header fields.
        public var metadata: [String: String] = [:]
    }

    // MARK: - Generation

    /// Deterministic (seeded) session assembly: metadata opening,
    /// then item-fact utterances shuffled among distractors. Fact
    /// phrasing includes the item's first vocabulary surface so
    /// candidate matching engages — vocabulary-free paraphrase
    /// hardness is a future spec knob, not an accident.
    public static func generate(
        template: EvalFormTemplate,
        spec: Spec
    ) -> Session {
        var rng = SplitMix64(seed: spec.seed)
        var expected = Expected(metadata: spec.metadata)

        // Body lines: (itemID?, text) — itemID marks fact rows.
        var body: [(itemID: String?, text: String)] = []
        for item in template.items {
            let fact = spec.facts[item.id] ?? .absent
            guard fact != .absent else { continue }
            let surface = item.vocabulary.first ?? item.titleJa
            expected.discussedItems.insert(item.id)
            switch fact {
            case .statedScore(let value, let preference):
                expected.statedScores[item.id] = value
                let sign = value < 0 ? "マイナス" : "プラス"
                let magnitude = String(format: "%g", abs(value))
                var text = "\(surface)は\(sign)\(magnitude)ぐらいですね"
                if let preference {
                    expected.preferences[item.id] = preference
                    text += "。これは好きですね、\(preference)点です"
                }
                body.append((item.id, text))
            case .qualitativeOnly:
                body.append((item.id, "\(surface)がかなり気になりますね、収まりがもう少し欲しいです"))
            case .absent:
                break
            }
        }
        for i in 0..<spec.distractorCount {
            body.append((nil, Self.distractors[i % Self.distractors.count]))
        }
        // Seeded shuffle (Fisher–Yates over the body block).
        for i in stride(from: body.count - 1, through: 1, by: -1) {
            let j = Int(rng.next() % UInt64(i + 1))
            body.swapAt(i, j)
        }

        // Opening: metadata statements at the session head, where
        // the metadata pass expects them.
        var lines: [(itemID: String?, text: String)] = spec.metadata.keys.sorted().map {
            (nil, "\($0)は\(spec.metadata[$0]!)です")
        }
        lines.append(contentsOf: body)

        var utterances: [UtteranceEstimate] = []
        for (idx, line) in lines.enumerated() {
            utterances.append(UtteranceEstimate(
                speakerID: line.itemID == nil ? "S02" : "S01",
                start: TimeInterval(idx) * 3,
                end: TimeInterval(idx) * 3 + 2.5,
                transcript: line.text,
                asrConfidence: 0.9,
                dimensional: nil,
                acousticCategorical: nil,
                plutchik: nil,
                fusedValence: nil,
                fusedArousal: nil,
                fusedDominance: nil,
                fusedTopLabel: nil
            ))
            if let itemID = line.itemID {
                expected.factRows[itemID, default: []].append(idx + 1)
            }
        }
        return Session(utterances: utterances, expected: expected)
    }

    /// Unrelated-but-substantive drive talk. Long enough to pass
    /// the supplementary length floor; free of item vocabulary and
    /// of anything the signed-score grammar could match.
    private static let distractors: [String] = [
        "この交差点を過ぎたら次の計測区間に入ります",
        "エアコンの風量をひとつ下げてもらえますか",
        "先ほどの区間は対向車が多かったので参考程度にします",
        "シートポジションをもう少し下げたほうが見やすいですね",
        "次の信号を右折して計測コースに戻ります",
        "今日は交通量が少なくて計測しやすいですね",
        "この後の休憩でデータを一度確認しましょう",
        "ステアリングの持ち替えが増える区間ですね",
        "後続の計測車両との間隔を保ってください",
        "この区間は路肩の工事があるので注意してください",
        "帰りは高速道路を使って戻る予定です",
        "記録用のカメラは問題なく動いています",
    ]

    /// Seeded deterministic RNG (SplitMix64) — same rationale as
    /// the test suites' seeded generators: failures reproduce.
    public struct SplitMix64 {
        private var state: UInt64
        public init(seed: UInt64) { state = seed }
        public mutating func next() -> UInt64 {
            state &+= 0x9E3779B97F4A7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
            z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
            return z ^ (z >> 31)
        }
    }

    // MARK: - Scoring

    public struct Report: Sendable {
        // Stated strength scores.
        public var scoreExact = 0
        public var scoreExpected = 0
        public var scoreMAE = 0.0
        /// Scores filled where NOTHING was stated — the cardinal
        /// sin. Counts stated fills on qualitative-only and absent
        /// items.
        public var falseScoreFills = 0
        public var missedScores = 0
        // Preferences.
        public var preferenceExact = 0
        public var preferenceExpected = 0
        // Comments.
        public var commentsPresent = 0
        public var commentsExpected = 0
        /// Comments on items that were never discussed.
        public var falseComments = 0
        // Evidence: filled items whose evidence intersects the
        /// planted fact rows.
        public var evidenceValid = 0
        public var evidenceChecked = 0
        // Metadata.
        public var metadataCorrect = 0
        public var metadataExpected = 0
        public var falseMetadata = 0

        public func render(label: String) -> String {
            let mae = scoreExpected > 0
                ? String(format: "%.4f", scoreMAE / Double(scoreExpected))
                : "—"
            return """
            [\(label)]
            stated scores : \(scoreExact)/\(scoreExpected) exact, MAE \(mae), \
            false fills \(falseScoreFills), missed \(missedScores)
            preferences   : \(preferenceExact)/\(preferenceExpected) exact
            comments      : \(commentsPresent)/\(commentsExpected) present, false \(falseComments)
            evidence      : \(evidenceValid)/\(evidenceChecked) valid
            metadata      : \(metadataCorrect)/\(metadataExpected) correct, false \(falseMetadata)
            """
        }
    }

    public static func score(
        draft: EvalFormDraft,
        expected: Expected,
        template: EvalFormTemplate
    ) -> Report {
        var report = Report()
        for item in template.items {
            let result = draft.items.first { $0.itemID == item.id }
            let expectedScore = expected.statedScores[item.id]
            let produced = result?.strengthScore

            if let expectedScore {
                report.scoreExpected += 1
                if let produced {
                    if abs(produced - expectedScore) < 0.0005 {
                        report.scoreExact += 1
                    }
                    report.scoreMAE += abs(produced - expectedScore)
                } else {
                    report.missedScores += 1
                    report.scoreMAE += abs(expectedScore)
                }
            } else if produced != nil {
                report.falseScoreFills += 1
            }

            if let expectedPreference = expected.preferences[item.id] {
                report.preferenceExpected += 1
                if result?.likeDislike == expectedPreference {
                    report.preferenceExact += 1
                }
            }

            let discussed = expected.discussedItems.contains(item.id)
            if discussed {
                report.commentsExpected += 1
                if result?.comment != nil { report.commentsPresent += 1 }
            } else if result?.comment != nil {
                report.falseComments += 1
            }

            // Evidence check on items with any filled field.
            let filled = produced != nil || result?.comment != nil
                || result?.likeDislike != nil
            if discussed, filled {
                report.evidenceChecked += 1
                let planted = Set(expected.factRows[item.id] ?? [])
                if !planted.isDisjoint(with: result?.evidenceRows ?? []) {
                    report.evidenceValid += 1
                }
            }
        }

        for (field, value) in expected.metadata {
            report.metadataExpected += 1
            if let produced = draft.metadata[field],
               produced.contains(value) || value.contains(produced) {
                report.metadataCorrect += 1
            }
        }
        for field in draft.metadata.keys
        where expected.metadata[field] == nil {
            report.falseMetadata += 1
        }
        return report
    }
}
