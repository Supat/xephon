import Foundation
import ASR
import SERAcoustic
import SERText

// One row of the canonical per-utterance JSON output (see docs/output_schema.md).
public struct UtteranceEstimate: Sendable, Hashable, Codable, Identifiable {
    /// Stable per-utterance identifier. Default-initialized to a fresh UUID
    /// so call sites that don't supply one keep working; survives JSON
    /// export so external tooling can cross-reference rows.
    public let id: UUID
    public let speakerID: String
    /// User-supplied display name for `speakerID` at export time
    /// (e.g. `"Alice"` for `"S01"`). Stamped by
    /// `RecordingController.exportJSON` from the active rename map
    /// so external tooling can read the human name without losing
    /// the canonical id. Nil for every code path that constructs an
    /// estimate from the pipeline (LateFusion, re-eval, hand-edit) —
    /// the rename layer lives one level above the fusion stage.
    /// Optional + nil by default keeps the JSON / `.xph` schema
    /// backward-compatible: pre-rename files decode cleanly with a
    /// nil here.
    public let speakerName: String?
    public let start: TimeInterval
    public let end: TimeInterval
    public let transcript: String
    public let asrConfidence: Float?

    // Acoustic side
    public let dimensional: VADScore?
    public let acousticCategorical: CategoricalEmotion?
    /// Per-utterance demographics estimate from the W2V2 age-gender
    /// model. Optional because: (1) the model is only present when
    /// the user has downloaded the weights, and (2) very short
    /// clips don't always produce a stable read. Persisted in the
    /// `.xph` bundle and the JSON export so external tooling can
    /// stratify by demographics.
    public let ageGender: AgeGenderEstimate?

    // Text side
    public let plutchik: PlutchikScore?
    /// Pre-bias Plutchik distribution — exactly what the text-SER
    /// backend returned before any glossary entries tilted it. Kept
    /// alongside `plutchik` so the user can re-apply the glossary
    /// (e.g. after editing weights in the Custom Glossary sheet)
    /// without re-running WRIME text SER / Apple FM: the controller hands
    /// `plutchikRaw` back through `LexiconBias.apply` to produce a
    /// fresh `plutchik`. Nil when text SER was skipped, when no
    /// backend was wired, or for sessions saved before this field
    /// existed (legacy rows fall back to `plutchik` only when
    /// `lexiconBiasMatched == nil`, so a stale double-bias can't
    /// happen).
    public let plutchikRaw: PlutchikScore?
    /// Identifier for the text-SER backend that produced `plutchik`
    /// (e.g. "deberta", "foundationModels"). Nil when text SER was skipped.
    public let textBackend: String?

    /// Whether the speech-boost EQ was enabled when this utterance was
    /// captured. Nil when unknown (e.g. batch processing of imported audio).
    public let speechBoost: Bool?

    /// Whether this utterance was produced (or refreshed) by a manual
    /// re-evaluate pass — offline ASR re-run with padded boundaries,
    /// then SER + fusion redone. `true` after at least one successful
    /// re-evaluation; nil for utterances that came straight from the
    /// streaming pipeline. Persisted in both the `.xph` bundle and
    /// the JSON export, so the marker survives a Save/Load round-trip
    /// and shows up alongside the affect data in external tooling.
    public let wasReevaluated: Bool?

    /// Whether the user manually edited the transcript / time range
    /// via the Edit Utterance dialog. `true` after at least one
    /// successful hand-edit; nil otherwise. A later re-evaluation
    /// clears this back to nil (the row reverts to a model-driven
    /// estimate). Persists in `.xph` and JSON so external tooling
    /// can flag rows whose transcript came from human review.
    public let wasHandEdited: Bool?

    /// Glossary terms (verbatim, in the form the user typed them)
    /// that matched this row's transcript and biased the text-SER
    /// Plutchik distribution. Nil for rows where the lexicon was
    /// disabled / empty or no entry matched; non-empty triggers the
    /// "Glossary" chip on `UtteranceRow`. Persisted so the chip
    /// survives a Save/Load round-trip and so JSON export readers
    /// can audit which bias inputs influenced a given row.
    public let lexiconBiasMatched: [String]?

    // Fused
    public let fusedValence: Float?
    public let fusedArousal: Float?
    /// Fused dominance. **Acoustic-only by design** — the text-
    /// SER models on this pipeline (WRIME-tuned text SER, Apple
    /// FoundationModels) don't estimate dominance, per CLAUDE.md.
    /// Fusion sets this to `dimensional?.dominance` directly,
    /// so a nil value means "acoustic SER didn't run or failed"
    /// — NOT "the speaker came across as neutrally dominant."
    /// Downstream analyses that aggregate dominance across a
    /// session should filter nil rows out (treat as missing)
    /// rather than imputing a midpoint default, otherwise the
    /// per-speaker dominance score gets pulled toward 0.5 by
    /// every utterance whose acoustic path was skipped (short
    /// clips, ORT errors, empty audio).
    public let fusedDominance: Float?
    public let fusedTopLabel: String?

    public init(
        id: UUID = UUID(),
        speakerID: String,
        speakerName: String? = nil,
        start: TimeInterval,
        end: TimeInterval,
        transcript: String,
        asrConfidence: Float?,
        dimensional: VADScore?,
        acousticCategorical: CategoricalEmotion?,
        ageGender: AgeGenderEstimate? = nil,
        plutchik: PlutchikScore?,
        plutchikRaw: PlutchikScore? = nil,
        textBackend: String? = nil,
        speechBoost: Bool? = nil,
        wasReevaluated: Bool? = nil,
        wasHandEdited: Bool? = nil,
        lexiconBiasMatched: [String]? = nil,
        fusedValence: Float?,
        fusedArousal: Float?,
        fusedDominance: Float?,
        fusedTopLabel: String?
    ) {
        self.id = id
        self.speakerID = speakerID
        self.speakerName = speakerName
        self.start = start
        self.end = end
        self.transcript = transcript
        self.asrConfidence = asrConfidence
        self.dimensional = dimensional
        self.acousticCategorical = acousticCategorical
        self.ageGender = ageGender
        self.plutchik = plutchik
        self.plutchikRaw = plutchikRaw
        self.textBackend = textBackend
        self.speechBoost = speechBoost
        self.wasReevaluated = wasReevaluated
        self.wasHandEdited = wasHandEdited
        self.lexiconBiasMatched = lexiconBiasMatched
        self.fusedValence = fusedValence
        self.fusedArousal = fusedArousal
        self.fusedDominance = fusedDominance
        self.fusedTopLabel = fusedTopLabel
    }

    /// Replace the utterance's `[start, end]` bounds. Used by the
    /// live-mode acoustic-SER trim path to tighten the row to the
    /// portion the diarizer placed this speaker in — only that path
    /// has authority to narrow the bounds. The new range must be
    /// inside the prior one (caller's responsibility; not clamped
    /// here because a non-subset would silently swallow audio
    /// outside the captured slice on re-evaluation).
    public func withBounds(start newStart: TimeInterval, end newEnd: TimeInterval) -> UtteranceEstimate {
        UtteranceEstimate(
            id: id,
            speakerID: speakerID,
            speakerName: speakerName,
            start: newStart,
            end: newEnd,
            transcript: transcript,
            asrConfidence: asrConfidence,
            dimensional: dimensional,
            acousticCategorical: acousticCategorical,
            ageGender: ageGender,
            plutchik: plutchik,
            plutchikRaw: plutchikRaw,
            textBackend: textBackend,
            speechBoost: speechBoost,
            wasReevaluated: wasReevaluated,
            wasHandEdited: wasHandEdited,
            lexiconBiasMatched: lexiconBiasMatched,
            fusedValence: fusedValence,
            fusedArousal: fusedArousal,
            fusedDominance: fusedDominance,
            fusedTopLabel: fusedTopLabel
        )
    }

    /// Return a copy with `transcript` replaced. Used by the
    /// transcription-review sheet to hand the in-progress inline
    /// edit off to the full Edit Utterance panel without losing
    /// what the user has already typed.
    public func withTranscript(_ newTranscript: String) -> UtteranceEstimate {
        UtteranceEstimate(
            id: id,
            speakerID: speakerID,
            speakerName: speakerName,
            start: start,
            end: end,
            transcript: newTranscript,
            asrConfidence: asrConfidence,
            dimensional: dimensional,
            acousticCategorical: acousticCategorical,
            ageGender: ageGender,
            plutchik: plutchik,
            plutchikRaw: plutchikRaw,
            textBackend: textBackend,
            speechBoost: speechBoost,
            wasReevaluated: wasReevaluated,
            wasHandEdited: wasHandEdited,
            lexiconBiasMatched: lexiconBiasMatched,
            fusedValence: fusedValence,
            fusedArousal: fusedArousal,
            fusedDominance: fusedDominance,
            fusedTopLabel: fusedTopLabel
        )
    }

    /// Stamp the per-utterance demographics estimate (age + gender)
    /// onto an already-fused row. The pipeline runs the W2V2
    /// age-gender model in parallel with the SER models and stitches
    /// its output on with this helper since `LateFusion.fuse` is
    /// affect-only.
    public func withAgeGender(_ score: AgeGenderEstimate?) -> UtteranceEstimate {
        UtteranceEstimate(
            id: id,
            speakerID: speakerID,
            speakerName: speakerName,
            start: start,
            end: end,
            transcript: transcript,
            asrConfidence: asrConfidence,
            dimensional: dimensional,
            acousticCategorical: acousticCategorical,
            ageGender: score,
            plutchik: plutchik,
            plutchikRaw: plutchikRaw,
            textBackend: textBackend,
            speechBoost: speechBoost,
            wasReevaluated: wasReevaluated,
            wasHandEdited: wasHandEdited,
            lexiconBiasMatched: lexiconBiasMatched,
            fusedValence: fusedValence,
            fusedArousal: fusedArousal,
            fusedDominance: fusedDominance,
            fusedTopLabel: fusedTopLabel
        )
    }

    public func withTextBackend(_ backend: String?) -> UtteranceEstimate {
        UtteranceEstimate(
            id: id,
            speakerID: speakerID,
            speakerName: speakerName,
            start: start,
            end: end,
            transcript: transcript,
            asrConfidence: asrConfidence,
            dimensional: dimensional,
            acousticCategorical: acousticCategorical,
            ageGender: ageGender,
            plutchik: plutchik,
            plutchikRaw: plutchikRaw,
            textBackend: backend,
            speechBoost: speechBoost,
            wasReevaluated: wasReevaluated,
            wasHandEdited: wasHandEdited,
            lexiconBiasMatched: lexiconBiasMatched,
            fusedValence: fusedValence,
            fusedArousal: fusedArousal,
            fusedDominance: fusedDominance,
            fusedTopLabel: fusedTopLabel
        )
    }

    /// Stamp the raw (pre-bias) Plutchik distribution. Called by the
    /// pipeline right after text SER returns so the controller can
    /// later replay the glossary against this snapshot when the user
    /// edits weights.
    public func withPlutchikRaw(_ raw: PlutchikScore?) -> UtteranceEstimate {
        UtteranceEstimate(
            id: id,
            speakerID: speakerID,
            speakerName: speakerName,
            start: start,
            end: end,
            transcript: transcript,
            asrConfidence: asrConfidence,
            dimensional: dimensional,
            acousticCategorical: acousticCategorical,
            ageGender: ageGender,
            plutchik: plutchik,
            plutchikRaw: raw,
            textBackend: textBackend,
            speechBoost: speechBoost,
            wasReevaluated: wasReevaluated,
            wasHandEdited: wasHandEdited,
            lexiconBiasMatched: lexiconBiasMatched,
            fusedValence: fusedValence,
            fusedArousal: fusedArousal,
            fusedDominance: fusedDominance,
            fusedTopLabel: fusedTopLabel
        )
    }

    /// Atomic re-bias swap: replace `plutchik`, `lexiconBiasMatched`,
    /// and the fused V/A/D/top-label together. Used by the controller's
    /// "Done in glossary sheet → reapply" path after `LateFusion.fuse`
    /// produces a fresh estimate from the re-biased Plutchik snapshot.
    /// Preserves `plutchikRaw` (invariant across re-applies) and
    /// every non-affect field on the row.
    public func withRebiased(
        plutchik newPlutchik: PlutchikScore?,
        matched: [String],
        fusedValence newV: Float?,
        fusedArousal newA: Float?,
        fusedDominance newD: Float?,
        fusedTopLabel newTop: String?
    ) -> UtteranceEstimate {
        UtteranceEstimate(
            id: id,
            speakerID: speakerID,
            speakerName: speakerName,
            start: start,
            end: end,
            transcript: transcript,
            asrConfidence: asrConfidence,
            dimensional: dimensional,
            acousticCategorical: acousticCategorical,
            ageGender: ageGender,
            plutchik: newPlutchik,
            plutchikRaw: plutchikRaw,
            textBackend: textBackend,
            speechBoost: speechBoost,
            wasReevaluated: wasReevaluated,
            wasHandEdited: wasHandEdited,
            lexiconBiasMatched: matched.isEmpty ? nil : matched,
            fusedValence: newV,
            fusedArousal: newA,
            fusedDominance: newD,
            fusedTopLabel: newTop
        )
    }

    /// Stamp the glossary terms that biased this row's text-SER
    /// distribution. Empty list → store nil (matching the "no chip"
    /// case) so the JSON export doesn't carry `"lexiconBiasMatched":
    /// []` noise.
    public func withLexiconBiasMatched(_ matched: [String]) -> UtteranceEstimate {
        UtteranceEstimate(
            id: id,
            speakerID: speakerID,
            speakerName: speakerName,
            start: start,
            end: end,
            transcript: transcript,
            asrConfidence: asrConfidence,
            dimensional: dimensional,
            acousticCategorical: acousticCategorical,
            ageGender: ageGender,
            plutchik: plutchik,
            plutchikRaw: plutchikRaw,
            textBackend: textBackend,
            speechBoost: speechBoost,
            wasReevaluated: wasReevaluated,
            wasHandEdited: wasHandEdited,
            lexiconBiasMatched: matched.isEmpty ? nil : matched,
            fusedValence: fusedValence,
            fusedArousal: fusedArousal,
            fusedDominance: fusedDominance,
            fusedTopLabel: fusedTopLabel
        )
    }

    public func withSpeakerID(_ id: String) -> UtteranceEstimate {
        UtteranceEstimate(
            id: self.id,
            speakerID: id,
            speakerName: speakerName,
            start: start,
            end: end,
            transcript: transcript,
            asrConfidence: asrConfidence,
            dimensional: dimensional,
            acousticCategorical: acousticCategorical,
            ageGender: ageGender,
            plutchik: plutchik,
            plutchikRaw: plutchikRaw,
            textBackend: textBackend,
            speechBoost: speechBoost,
            wasReevaluated: wasReevaluated,
            wasHandEdited: wasHandEdited,
            lexiconBiasMatched: lexiconBiasMatched,
            fusedValence: fusedValence,
            fusedArousal: fusedArousal,
            fusedDominance: fusedDominance,
            fusedTopLabel: fusedTopLabel
        )
    }

    /// Return a copy with `speakerName` set to `name`. Used by
    /// `RecordingController.exportJSON` to stamp the active rename
    /// (`speakerNameOverrides[speakerID]`) onto each row right
    /// before writing the JSON, so external tooling sees the human
    /// name alongside the canonical `speakerID`. Pass nil to clear.
    public func withSpeakerName(_ name: String?) -> UtteranceEstimate {
        UtteranceEstimate(
            id: id,
            speakerID: speakerID,
            speakerName: name,
            start: start,
            end: end,
            transcript: transcript,
            asrConfidence: asrConfidence,
            dimensional: dimensional,
            acousticCategorical: acousticCategorical,
            ageGender: ageGender,
            plutchik: plutchik,
            plutchikRaw: plutchikRaw,
            textBackend: textBackend,
            speechBoost: speechBoost,
            wasReevaluated: wasReevaluated,
            wasHandEdited: wasHandEdited,
            lexiconBiasMatched: lexiconBiasMatched,
            fusedValence: fusedValence,
            fusedArousal: fusedArousal,
            fusedDominance: fusedDominance,
            fusedTopLabel: fusedTopLabel
        )
    }

    public func withSpeechBoost(_ enabled: Bool?) -> UtteranceEstimate {
        UtteranceEstimate(
            id: id,
            speakerID: speakerID,
            speakerName: speakerName,
            start: start,
            end: end,
            transcript: transcript,
            asrConfidence: asrConfidence,
            dimensional: dimensional,
            acousticCategorical: acousticCategorical,
            ageGender: ageGender,
            plutchik: plutchik,
            plutchikRaw: plutchikRaw,
            textBackend: textBackend,
            speechBoost: enabled,
            wasReevaluated: wasReevaluated,
            wasHandEdited: wasHandEdited,
            lexiconBiasMatched: lexiconBiasMatched,
            fusedValence: fusedValence,
            fusedArousal: fusedArousal,
            fusedDominance: fusedDominance,
            fusedTopLabel: fusedTopLabel
        )
    }
}
