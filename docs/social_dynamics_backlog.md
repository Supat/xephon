# Social-dynamics analysis backlog

Candidate analyses that the existing pipeline outputs already support
but the UI doesn't yet surface. Numbered for cross-referencing in
follow-up discussion. Order within a section is roughly by "data
already there, easy" → "needs more derivation."

Existing for context: `Core/Fusion/SpeakerBehavior.swift`
(5-dim per-speaker profile), `AffectiveSynchronyCard` (lag-1
pairwise leadership + V/A synchrony), `SpeakerHeatmapCard`,
`SERAggregateCard`, `StatisticsCard`.

## Turn-taking dynamics

Uses: utterance timing + `speakerID`.

1. **Interruption / overlap rate per directed pair.** Cumulative-
   timeline segments where a new speaker starts before the previous
   one's segment ends. Asymmetric counts per `(A → B)` pair encode
   power balance. Render: directed heatmap.
2. **Floor-holding distribution.** Per-speaker histogram of
   consecutive same-speaker run lengths (in seconds or in
   utterances). Long-tail = monologue style; short = dialogue.
   Render: per-speaker small-multiples histogram.
3. **Backchannel rate.** Per-speaker count of short utterances
   matching the existing `AnalysisPipeline.fillers` set
   (うん, はい, そう, ええ, …) divided by total utterances. Reveals
   "agreer" vs. "driver" roles. Render: stat alongside talk-share.
4. **Response-latency asymmetry.** Median gap from B-finish to
   A-start, conditioned on the partner. "Speaker A responds to B
   in 0.8 s but to C in 2.3 s" encodes attention allocation.
   Render: directed-pair latency matrix.

## Directed influence + emotional contagion

Uses: fused V/A + Plutchik + ordered utterance sequence.

5. **Multi-lag leadership graph.** Extend the existing lag-1
   `AffectiveSynchrony.leadershipScores` to lag-1..lag-3 per
   directed pair. Render as a chord diagram or weighted DAG —
   "who shapes the room's emotional weather over the whole
   session." Most-recommended single addition.
6. **Mood-rescue events.** Count cases where speaker B's next
   turn has fused V above A's previous turn when A's V was below
   a threshold (e.g., 0.4). Per-speaker tally = "emotional
   caretaker" tendency. Render: stat column in the
   `SpeakerBehaviorCard`.
7. **Contagion windows.** Detect stretches where ≥3 speakers'
   fused V converge within a sliding window (e.g., 60 s) after
   one speaker's strong-valence turn. Render: highlighted
   ranges on the cumulative timeline strip with a tooltip
   identifying the "infector."

## Accommodation / cohesion

Uses: fused V/A trajectory.

8. **Accommodation over time.** Mean V/A distance between
   consecutive cross-speaker turns, plotted as a session-time
   curve. Falling = warming up; rising = splitting. A linear-fit
   slope is a single per-session number. Render: small inline
   line chart in `SessionSummary`.
9. **Group cohesion index.** Rolling-window variance (e.g.,
   90 s) of all active speakers' fused V/A. Low = cohesive;
   high = fragmenting. Surfaces phase transitions across the
   session. Render: heatband under the cumulative timeline.
10. **Drift toward group mean.** Per-speaker V/A deviation from
    the session-wide mean, plotted over time. Speakers who flatten
    out are accommodating; speakers whose deviation grows are
    anchoring. Render: per-speaker line in the heatmap card.

## Modality disagreement (sarcasm / mixed-affect proxy)

Uses: `acousticCategorical` (9-class softmax) + `plutchik`
(8-class) + `fusedTopLabel`. Data already stored per row.

11. **Per-row disagreement score.** Distance between the acoustic
    and Plutchik categorical vectors after projecting both onto a
    shared affect space (e.g., Plutchik label closest to each
    acoustic class). High = candidate sarcasm / mixed affect /
    irony. Render: as a row badge (similar to "Apple FM ✕"), and
    as a session-level "complex affect" tally per speaker. Cheap
    to compute; underused.

## Reactivity to session environment

Uses: timing + V/A trajectory + turn boundaries.

12. **Response vs. initiative classification.** Per utterance,
    label "response" (started within τ seconds of another speaker
    finishing) vs. "initiative" (started after a long pause).
    Per-speaker ratio reveals "reactive" vs. "agenda-setting"
    role. Render: chip on the row + per-speaker stat.
13. **Recovery time after negative valence.** For each speaker,
    median time from a sub-threshold V utterance back to that
    speaker's session-baseline V. Resilience-style metric.
14. **Reaction to interruption.** When speaker A's turn is cut
    short (overlap detected per #1), is A's *next* turn lower
    in V than A's session mean? Aggregate = how the speaker
    handles being talked over.

## Recommended build order

If we resume this, the highest-leverage two are:

- **#5 — Multi-lag leadership graph.** Direct extension of code
  already in `AffectiveSynchrony`, answers the question most
  researchers ask first ("who's driving"). Visualization fits the
  existing card pattern.
- **#11 — Modality disagreement.** Pure derivation from data
  already on every `UtteranceEstimate`. Surfaces the *interesting*
  rows researchers would otherwise have to scrub for.

Then in declining order of impact-per-effort: #1 (interruption
matrix), #8 (accommodation curve), #4 (latency asymmetry),
#9 (cohesion heatband), #12 (response/initiative chip).

The rest (#2, #3, #6, #7, #10, #13, #14) are cheap individual
additions but mostly variants on the above themes; build only when
a specific research question calls for them.
