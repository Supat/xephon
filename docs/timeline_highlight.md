# Cumulative-timeline highlight

How the focus stroke on `DiarizationTimelineStrip` is computed, and
why expansion-aware filtering matters.

## What the strip shows

`DiarizationTimelineStrip` (in `Xephon/Views/`) renders the per-session
diarizer timeline as a thin horizontal capsule: speaker-tinted runs of
"who's talking" computed by per-instant majority vote over the
cumulative segments, with a black-rule **focus stroke** overlaid on
whatever audio-time window the rest of the UI is currently
interested in.

The strip sits below the speaker chip bar in the transcript pane and
is reused inside `SERAggregateCard` etc. with the same `selectedRange`
contract.

## How the focus stroke is computed

The stroke's `selectedRange` comes from
`ContentView.selectedUtteranceRange`, in priority order:

1. **Explicit selection.** If `selectedUtteranceID` is non-nil — set
   by a row tap, a cluster-plot node tap, or an SER-aggregate ring
   tap — the stroke spans exactly that utterance's `[start, end]`.
2. **Visible-range fallback.** Otherwise the stroke spans the union
   of every utterance whose id is in `visibleUtteranceIDs` (the per-
   row visibility tracker writes to it on layout). Effectively, it
   reflects "what audio window am I currently scrolled to?"

When neither produces a finite range — empty list, no scrolled rows —
the function returns nil and the strip omits the stroke entirely.

## Why expanded rows are excluded from the fallback

A row in `expandedUtteranceIDs` is rendered with its detail panel
inline, which can be ~300pt tall. On a typical iPad pane that is
enough to push every other row off-screen, so `visibleUtteranceIDs`
collapses to a single id — the expanded one.

If the fallback voted that id in, the stroke would shrink to the
expanded utterance's narrow `[start, end]` window and stay parked
there for as long as the row stayed open. From the user's point of
view the stroke gets **anchored to the expanded entry**, even though
they only opened the row to read its detail, not to declare a
selection.

The fix: in the visible-range fallback, skip any utterance whose id
is in `expandedUtteranceIDs`. The resulting behavior:

| State                                                  | Stroke                                          |
|--------------------------------------------------------|-------------------------------------------------|
| Nothing selected, nothing expanded                     | Spans the visible rows (tracks scroll position) |
| Row tapped (explicit selection)                        | Spans that single row's range                   |
| Row expanded, neighbors still visible                  | Spans the neighbors (expansion is ignored)      |
| Row expanded so large it's the only thing in viewport  | No stroke (fallback returns nil)                |

The explicit-selection path is unchanged — if the user *wants* to
pin the stroke onto a single row, tapping it still works.

## Where this lives in code

- `Xephon/Views/DiarizationTimelineStrip.swift` — the strip view.
  `selectedRange` is the only input governing the stroke.
- `Xephon/ContentView.swift::selectedUtteranceRange` — the priority
  resolver. The expansion exclusion lives in this getter's fallback
  branch; it reads `expandedUtteranceIDs` directly.

## Related state

- `selectedUtteranceID: UUID?` — explicit selection, two-way bound
  into `TranscriptList(selection:)`.
- `expandedUtteranceIDs: Set<UUID>` — per-row inline-detail toggle.
  Mutated by `toggleExpansion(_:)`, which is wired to the fused-label
  chip tap on `UtteranceRow` (and the Space key when the row is the
  list selection).
- `visibleUtteranceIDs: Set<UUID>` — written by each row's onAppear /
  onDisappear, read on every scroll. Keep the work in the fallback
  branch O(utterances) — it fires per scroll frame.
