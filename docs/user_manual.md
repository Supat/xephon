# Xephon User Manual

Xephon is an iPadOS research app that listens to Japanese conversational
audio and tells you, per utterance, **who** spoke, **what** they said, and
**how they felt** — both as an emotion label (Plutchik 8) and as a point
on the valence / arousal / dominance axes. Everything runs on-device. Your
audio never leaves the iPad.

This manual walks you through setup, the main workflows, and the parts of
the UI that aren't self-explanatory.

---

## 1. Before you start

**Requirements**

- iPad Pro with an M4 or M5 chip (the 16-core Neural Engine SKU). Other
  iPads can install the app but the primary speech recognizer won't run
  on them.
- iPadOS 26 or later.
- **16 GB unified memory** (the 1 TB / 2 TB iPad Pro SKUs) if you want
  to run an on-device LLM for session summarization or transcription
  review. The MLX backends (Qwen3 8B, Llama-Swallow 8B) need ~5 GB
  resident on top of the ~4 GB analysis pipeline, which trips iOS's
  Jetsam ceiling on the 8 GB SKUs. On an 8 GB iPad the rest of the
  app — recording, transcription, diarization, emotion analysis —
  works fine; only the on-device LLM is out of reach, and you can
  still get a session summary via **Apple Foundation Models** (light
  enough to run on 8 GB) or **LM Studio** pointing at a Mac on the
  same network. See § 11.
- Microphone permission (for live recording) and Speech Recognition
  permission (for transcription). The app will prompt the first time
  each is needed.

**First launch**

On first launch the app downloads its ML models (~ 1 GB total) over
Wi-Fi. This takes a few minutes. The download is resumable — if you close
the app mid-download, it picks up where it left off on next launch.

> **Figure 1: First-launch model download progress.**
> Centered modal showing a circular progress ring at ~ 45 %, with the
> active model name ("WRIME RoBERTa · tokenizer.json") and a "1 of 9
> models" subtitle. Cancel button at the bottom.

When the ring closes you land on the main screen.

---

## 2. The main screen

Xephon's main screen is a two-pane layout: a **control pane** on the
left and a **transcript pane** on the right. Both panes stay visible
in every orientation — the split is roughly 1 / 3 vs 2 / 3, so the
control pane just gets narrower in portrait (cards stack vertically
inside it where the wider landscape pane fit them side-by-side).

> **Figure 2: Main screen, landscape, idle state.**
> Left pane: input picker ("iPad Microphone ⌄"), a large "Start
> Recording" button, an "Open Audio File" icon button next to it, then
> a swipeable card region with page-indicator dots at the bottom. Right
> pane: a "Search utterances" field at top, a "All Labels ⌄" filter
> chip on the right, a horizontal diarization strip below them, then
> the transcript list (empty state: "No utterances yet — tap Start
> Recording to begin"). Top chrome carries an editable **session
> title** field in the center and (left → right on the trailing edge)
> **Summarize**, **Review**, **Search & Replace**, **Export** buttons.

The control pane is organized as a horizontally swipeable strip of six
pages. Swipe left / right, tap the page indicator at the bottom, or
press ⌘1 – ⌘6 on an external keyboard to jump directly:

| ⌘    | Page          | What's on it                                                                                                                                                                                       |
|------|---------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ⌘1   | **Settings**  | Language picker, offline-ASR backend, text-SER backend, Speech Boost toggle, Speaker Sensitivity slider. Plus the **Pipeline** "is each stage working?" card below.                                |
| ⌘2   | **Affect**    | Session-level Summary, Statistics, the SER aggregate, and the **Fusion Legend** card (acoustic-vs-text mix sliders + the **Custom Glossary** entry point).                                          |
| ⌘3   | **Speakers**  | Speaker roster, voice-embedding cluster, speaker-behavior heatmap, turn-taking matrix, affective synchrony, and the influence / accommodation / reactivity / synchrony-arc analyses.                |
| ⌘4   | **Sections**  | Named chapter markers over the transcript with optional per-section LLM summaries. See § 13.                                                                                                        |
| ⌘5   | **Keywords**  | Personal keyword bank used as filter chips above the transcript and as anchors for the heuristic summarizer mode. See § 14.                                                                         |
| ⌘6   | **Summarizer**| Session-summary card, model picker (Apple FM / Qwen / Llama-Swallow / LM Studio), Prompts editor, Models card for download / delete.                                                                |

---

## 3. Recording from the microphone

1. Pick your input device from the picker at the top of the control
   pane. If a USB-C mic is plugged in, it shows up there alongside
   "iPad Microphone".
2. (Optional) Toggle **Speech Boost** in the Settings card if your input
   is quiet or distant. It boosts the 1–4 kHz speech band on the
   recognizer's feed only; the acoustic emotion models still see the
   original audio.
3. Tap **Start Recording**. The button turns into a "Stop" pill and an
   elapsed-time counter starts running.

> **Figure 3: Mid-recording state.**
> Left pane top: red "Stop · 00:34" pill replacing the blue Start
> button, a horizontal level meter below it pulsing. Pipeline card
> shows the six stages (Capture, ASR (Live), Diarizer, Acoustic SER,
> Text SER, Fusion) each with a green check; ASR card shows "1 snt"
> count. Right pane: diarization strip filling left-to-right with
> alternating colored bands per speaker; transcript list shows two
> finalized utterances (gray "Speaker 1" chip, transcript text,
> emotion label badge on the right) and a third volatile-text row in
> italic gray.

While you're recording, **volatile text** (the recognizer's
in-progress hypothesis) appears in italic gray at the bottom of the
list and updates 5× per second. When the recognizer commits a
sentence boundary, the row solidifies into a finalized utterance.

To stop, tap the **Stop** pill. The app finishes processing any
in-flight utterances and returns to idle.

---

## 4. Analyzing a recorded audio file

1. Tap the **Open Audio File** icon (📂) next to the Start Recording
   button.
2. Pick a file in the system file picker. Supported: anything
   AVFoundation can decode (MP3, M4A, WAV, AAC, FLAC, ...).

Analysis starts as soon as the picker hands the file back — there's
no second tap. The status line above the button switches to "File:
*filename.mp3*" and a thin progress bar tracks how far through the
file the analyzer is.

If you already have a session loaded when you pick a new file, Xephon
shows a **Discard current session?** alert first; confirming starts
the new analysis, cancelling leaves the existing session intact.

> **Figure 4: File-mode picker.**
> System document picker open over Xephon. Files browser shows two
> selectable audio files highlighted in blue.

File-mode analysis runs faster than real time (the pump reads as fast
as the analyzer can keep up). A typical 10-minute MP3 finishes in
~ 3 minutes on M4. While it runs, the same Pipeline card shows
throughput and the transcript fills in progressively.

When the file is exhausted the app automatically stops, finalizes the
last utterances, and returns to idle.

---

## 5. Reading the transcript

Each row in the transcript is one utterance.

> **Figure 5: One utterance row, annotated.**
> A single row with callouts pointing at each part:
>   • Speaker chip "S01" (tinted to that speaker's color) — bottom-left
>   • Play button (▶) — leading edge
>   • Re-evaluate button (↻) — directly below play
>   • Backend badge "WRIME" — top-right of the metadata strip
>   • Modality badge "≠ Modalities" — next to backend badge when text
>     and acoustic disagree
>   • Glossary chip "Glossary 2" (purple) — when glossary terms fired
>   • Hand-edited badge "Edited" — when the user committed a hand-edit
>   • Top-line emotion label "Happy" (color-coded) — top-right
>   • V / A readout "V +0.07  A +0.10" — second line right
>   • Transcript text — center, the largest text element

Tapping the row expands it into a detail panel:

> **Figure 6: Expanded detail panel.**
> Same row, expanded. Shows three columns:
>   • Top fused labels (bar chart, top 3 emotions with probabilities)
>   • V/A pulls scatter mini (110 × 110): a small square plot with
>     three dots — acoustic (orange), text (purple), fused (green) —
>     and translucent connecting lines
>   • Acoustic SER (emotion2vec) bars on the left side, Text SER
>     (WRIME) bars on the right
> Below the columns: ASR confidence, Acoustic V/A/D readout, Fused
> V/A/D readout, Fusion-weight summary ("Text 72% · Acoustic 28%"),
> and an Age + Gender readout when available.

### Gestures and quick actions

- **Tap the transcript text** for ~ 0.5 s → **Edit Utterance** sheet
  (see § 7).
- **Tap the speaker chip** → reassign popover (pick a different
  speaker, rename the current one, or "Teach diarizer this is
  Speaker N").
- **Tap the ▶ button** → play this utterance's audio from the source
  file. Tap again to stop. The button shows a stop icon while
  playing.
- **Tap the ↻ button** → re-evaluate (see § 8).
- **Long-press the "Edited" badge for 2 s** → revert to the
  pre-edit row.

---

## 6. Filtering the transcript

Above the list you have:

- A **search field** that does Japanese-aware substring matching across
  all transcripts. Search is normalized (hiragana/katakana
  interchangeable, full-width and half-width digits/Latin treated as
  equivalent).
- A **label filter** chip ("All Labels ⌄") for narrowing by fused
  emotion.
- A **speaker filter row** that shows every speaker as a tappable chip
  (tap to include only that speaker; "All Speakers" to clear). A
  "Mismatch" chip filters to rows where acoustic and text SER
  disagree.

> **Figure 7: Filter bar with two speakers selected.**
> Top row: search field on the left with placeholder text, label
> filter chip on the right. Second row of pill-shaped speaker chips:
> "All Speakers" unselected, "S01" and "S03" highlighted (filled),
> "S02" and "S05" outlined.

When no rows match, an empty state appears with a "Clear filters"
button.

---

## 7. Editing an utterance

Long-press the transcript text of any row to raise the **Edit
Utterance** sheet.

> **Figure 8: Edit Utterance sheet.**
> Modal sheet. Top: "Transcript" header, then a multi-line text editor
> filled with the row's text. Below: "Audio Range" header, a
> horizontal control row with a large play button on the left and a
> stacked Start/End time control on the right (each with a numeric
> field flanked by -/+ buttons). Below that: a full-width "Transcribe"
> button. Toolbar: Cancel (top-left), Commit (top-right).

You can:

- **Edit the transcript text** directly in the editor.
- **Adjust the time range** with the -/+ buttons or by typing
  `m:ss.s` into either field.
- **Play the current range** to hear what you've selected.
- **Tap Transcribe** to re-run offline ASR on the current range and
  drop the result into the text field. If you've enabled the
  glossary's ASR Hint pathway (see § 9), this run uses your hinted
  terms; otherwise it uses the default offline recognizer. The text
  editor locks while transcription is running.
- **Commit** to apply your edits. The row's transcript, range, and
  affect scores are all re-computed.

Hand-edited rows get an "Edited" badge. Long-press it for 2 s to
revert.

---

## 8. Re-evaluating a row

Tap the ↻ button on any row to re-run ASR + SER + fusion on that
utterance's audio with padded boundaries (so the recognizer has
prosodic context the streaming pass didn't see). Useful when the
streaming transcript looks cut off or wrong.

> **Figure 9: Re-evaluation in progress.**
> The target row's re-evaluate button replaced by a small spinner.
> Underneath the row, in italic gray, the offline recognizer's
> volatile text crawls left-to-right as it processes.

When the re-eval finishes the row is replaced in place with the new
result, stamped with a "Re-evaluated" indicator. Like hand-edits,
you can revert via long-press.

---

## 9. Custom Glossary

The Custom Glossary is a per-user list of terms that can do two
things:

1. **Bias** the text-SER Plutchik distribution toward a specific
   emotion when that term appears in a transcript (e.g. "残業" →
   tilt toward sadness).
2. **Hint** the SFSpeechRecognizer with a list of vocabulary terms
   on the Transcribe pathway so domain words / proper nouns are
   more likely to be transcribed correctly.

Each entry can do either, both, or neither.

Open the sheet from the **Affect page (⌘2)** — scroll to the **Fusion
Legend** card and tap the **Custom Glossary** button (a purple book
icon with a count chip showing how many entries you've added).

> **Figure 10: Custom Glossary sheet.**
> Modal sheet, top to bottom:
>   • Two master toggles, each with a hint footer:
>       - "📕 Apply Bias" (purple book icon)
>       - "🎙 ASR Hint" (mic icon)
>   • Scrollable list of entry cards. Each card shows:
>       - Top row (left → right): colored emotion chip ("Joy"),
>         spacer, purple book toggle (filled when useAsBias on),
>         blue mic toggle (filled when useAsASRHint on), red trash
>         button
>       - Term TextField below (e.g. "残業")
>       - Bottom row: Emotion picker ("Joy ⌄"), spacer, "WEIGHT"
>         label + numeric readout (e.g. "0.7"), -/+ stepper
>   • Bottom: full-width "+ Add Entry" button
> Toolbar: Done (top-left), ⋯ menu (top-right) with Import / Export.

### Per-entry toggles

- **📕 Book** — when filled (purple), this entry contributes to
  text-SER bias. When outlined, the entry is excluded from bias
  even if the master "Apply Bias" is on. The emotion label chip on
  the left greys out when this is off — that's your visual
  preview that the row won't actually tilt the distribution.
- **🎙 Mic** — when filled (blue), this entry's term is forwarded
  to SFSpeechRecognizer's `contextualStrings` on the next
  Transcribe call. When outlined, the entry is excluded.

Both per-entry toggles dim when their respective master toggle is
off, so you can see at a glance that they're inert.

### Adding entries

Tap **+ Add Entry** at the bottom. A new empty row appears at the
end of the list with the keyboard already focused on its term
field. Type your term, pick an emotion, set the weight (0.0 = no
effect, 1.0 = strong tilt). Both per-entry toggles default to on.

### Editing weights

Weight controls how strongly the bias tilts the matched class. The
math is in logit space, so weight 1.0 roughly multiplies that
class's odds-ratio by 7×; weight 0.5 by 2.7×.

> **Figure 11: Weight stepper close-up.**
> A horizontal control: small label "WEIGHT" in gray, monospaced
> numeric readout "0.6", and a circular -/+ stepper. Below: an
> implicit slider trail showing 0.0 ─── 0.6 ─── 1.0.

### Deleting entries

Tap the trash button on any card. The row is removed immediately;
there's no undo, so be deliberate.

### Import / Export

Tap **⋯ → Export…** to save your glossary as a JSON file. Tap
**⋯ → Import…** to load one. Use this to share glossaries between
devices or sessions.

### When edits take effect

Every glossary change persists immediately (write-through JSON in
Application Support; survives relaunch). Bias / hint behavior on
**new** utterances applies right away. To re-bias **existing**
utterances with the current glossary, tap **Done** to dismiss the
sheet — the app re-applies bias to every stored row and re-runs
fusion in the background. This is fast (no model calls; pure math).

> **Figure 12: Done re-apply.**
> Just after tapping Done, the transcript list shows several
> previously-purple "Glossary N" chips updating in place, with
> some fused labels visibly changing color. No spinner — it's
> sub-second on a typical session.

---

## 10. Speaker management

Xephon's diarizer identifies speakers automatically by their voice
embedding. They get default labels "S01", "S02", etc.

> **Figure 13: Diarization strip.**
> The horizontal bar at the top of the transcript pane. Solid
> colored segments per speaker stacked into a single horizontal
> band, with a thin time-cursor line showing the current playback
> position. Speaker colors match the row chips.

### Renaming a speaker

Tap any speaker chip in a row → popover → **Rename Speaker…** →
type a name (e.g. "Alice"). The rename applies everywhere
immediately.

### Reassigning a row

Tap the chip → tap another speaker's chip in the popover. That
row's speaker is changed.

### Teaching the diarizer

If you reassign a row AND want the diarizer to learn from your
correction for future utterances, flip **Teach diarizer** on in
the popover before tapping the target speaker. The new audio gets
folded into that speaker's centroid.

### Affirming a speaker

Use Affirm when the row's **current** speaker label is already
correct but you want the diarizer to lock that judgment in — for
example a row tagged "S01" that you've listened to and confirmed
really is S01, but the diarizer wasn't sure (so the row shows a
caution / mismatch glyph). Tap the chip → **Affirm Speaker**.
Xephon folds the row's audio into S01's centroid in the diarizer's
internal database the same way the Teach-diarizer path does, but
without changing the speaker label or splitting / merging anything.
It also rewrites the cumulative timeline for the row's range so
later utterances in the same window stop being flagged as
mismatched.

Use Reassign + Teach diarizer (above) instead when the current
label is **wrong** — Affirm reinforces, it doesn't correct.

Requires source audio: the row needs its waveform available so the
speaker embedding can be extracted. That means file-mode sessions
and imported `.xph` sessions only; rows captured live without a
bundled source file have nothing to fold.

### Speaker sensitivity

The **Speaker Sensitivity** slider in Settings controls how eagerly
the diarizer splits voices into distinct speakers. Drag right →
more distinct speakers; drag left → speakers merge more. Double-tap
the label to reset to the default.

> **Figure 14: Speaker Sensitivity slider.**
> A horizontal slider labeled "👥 Speaker Sensitivity   0.60" with
> "Merge" on the left end and "Split" on the right. Below: a
> single-line hint "Higher = more distinct speakers. Double-tap
> label to reset."

---

## 11. Session summary

If you've enabled the summarizer (Settings → Summarizer card), Xephon
can produce a one-page LLM summary of the whole session.

Backends, picked in the Summarizer card on the **Summarizer page
(⌘6)**:

- **Apple Foundation Models** (built into iPadOS 26, no download
  needed). Default when available.
- **Qwen3 8B** (MLX). Downloads ~ 4.6 GB on first use.
  **Requires a 16 GB iPad** — needs ~5 GB resident on top of the
  analysis pipeline; 8 GB iPads Jetsam-kill the app during prefill.
- **Llama-Swallow 8B** (MLX). Same size class as Qwen, tuned for
  Japanese; downloads on first use. **Also 16 GB-only**, same
  Jetsam constraint as Qwen.
- **LM Studio** (remote). Talks to an LM Studio server running on
  your Mac or another machine on the same network. Configure the
  base URL + model in the LM Studio Server section of the
  Summarizer card. Audio still stays on the iPad; only the
  transcript text is sent to the configured server.

Open the summary sheet from the **Summarize** button on the top
chrome (a book icon).

> **Figure 15: Session Summary sheet.**
> Modal sheet. Sections, top to bottom:
>   • Setting (e.g. "Casual conversation, two friends")
>   • Topic (e.g. "Weekend plans and a recent work incident")
>   • Overall Mood (e.g. "Mostly positive, briefly tense around
>     11:30")
>   • Per Speaker (one paragraph per speaker on their affect arc)
> Footer: "Model: Apple Foundation Models · AI Generated · May
> contain errors". Toolbar: Done (top-right), Regenerate
> (bottom-left).

The summary auto-generates the first time you open the sheet (when
the summarizer is ready). Tap **Regenerate** to redo it after
edits.

---

## 12. Transcription review

The Transcription Review sheet asks the same on-device LLM to scan
your transcript for likely transcription errors (homophones, grammar,
context).

> **Figure 16: Transcription Review sheet.**
> Modal sheet. List of issue cards, each showing:
>   • Kind chip ("Homophone" / "Contextual" / "Grammar" / "Other")
>     with a confidence percentage
>   • Inline text editor with the current transcript (editable)
>   • Reason text underneath in gray
>   • Action row: Dismiss (✕), Re-evaluate (↻), Range… (→ opens
>     Edit Utterance sheet), Commit (✓)
> Bottom action bar: "Review" / "Re-review" full-width button.

Use the inline editor for quick fixes; the Range… button hands the
in-progress edit off to the full Edit Utterance sheet when you need
to adjust the time range too.

---

## 13. Sections (chapter markers)

The **Sections** page (⌘4) lets you mark named ranges of utterances
inside a session — like chapters in a long interview. Each section
points to a **start utterance** and an **end utterance**, and can
optionally carry its own LLM summary.

> **Figure 17: Sections card.**
> The Sections page on the control pane. Stacked list of section rows,
> each showing the section title, a "Speaker · m:ss → m:ss" range, an
> optional cached summary excerpt, and a row of action buttons
> (Summarize, Edit, Delete). Below the list: a prominent "Add Section"
> button. Two quick-add buttons sit beneath it — **Add Start** and
> **Add End** — that snap the focused-utterance id into a fresh
> incomplete section so you can build markers while you read.

### Creating a section

You have three paths:

1. **Add Section** → raises the section editor sheet. Pick a title
   and the start / end utterances explicitly from a picker.
2. **Add Start** (with one utterance focused) → creates an
   incomplete section whose start is the focused row; fill in the
   end later.
3. **Add End** (with one utterance focused) → mirror of the above —
   end is set, start is empty.

### Completing a section

Each row has a small **complete-with-focus** button that stamps the
focused utterance into whichever bound (start or end) is still
missing. Useful when you marked one boundary live and want to set
the other later without opening the editor.

### Per-section summary

Tap the **Summarize** button on a section row to ask the active
summarizer backend (see § 11) for a one-paragraph summary of just
that section's utterances. The result is cached in the section,
survives a Save / Load round-trip, and renders as an excerpt under
the title.

### Sections vs. session swap

Sections reference per-session utterance UUIDs. Loading a different
session drops them; starting a new recording clears them.

---

## 14. Keywords

The **Keywords** page (⌘5) is a personal keyword bank that does
two things:

- Surfaces tappable **keyword chips** above the transcript that
  filter to utterances whose normalized transcript contains that
  keyword (case- and width-insensitive, hiragana / katakana
  interchangeable). Multiple chips OR together.
- Anchors the **heuristic** summarizer mode (no LLM call) — when
  enabled, the summary is built from sentences that match your
  selected keywords plus their neighbors.

> **Figure 18: Keywords card.**
> Vertical list of keyword rows. Each row carries the keyword text,
> an occurrence count chip ("12 utts"), a group-assignment menu
> ("Ungrouped ⌄"), and a destructive Delete button. Above the list:
> a group selector and an "Add Group" button. Bottom: a TextField
> "Add a keyword…" with a + button that commits on submit.

### Adding, deleting, organizing

- Type in the **Add a keyword…** field and submit. New keywords
  always land at the bottom of **Ungrouped** — the add field has
  no notion of a target group. Move the keyword into a group
  afterwards using its per-row group menu, or by drag-and-drop
  onto the destination group's header.
- Each row has a trash button — confirms before deleting.
- Drag a row onto a different group header to move it; drag onto a
  different row to reorder.
- The **⋯** menu offers Import / Export JSON for the whole bank
  and a destructive **Remove All**.

### Groups

Tap **Add Group** to define a named bucket (e.g. "Negative",
"Names", "Technical"). Use groups to keep large banks organized
and to scope the heuristic summarizer to a subset.

Keywords are app-global, not per-session — they persist across
launches and don't ship inside the `.xph` bundle. Use Export to
share them deliberately.

---

## 15. Saving and sharing

### Save a session

**File → Save Session…** (⌘S) raises the system save sheet. The
session is written as a `.xph` bundle (binary plist) containing the
utterance list, source audio (file mode only), diarizer state,
sections, summary, and metadata. Save anywhere — Files app, iCloud
Drive, etc.

### Open a saved session

**File → Import Session…** (⌘⇧O) raises the system file picker
filtered to `.xph` documents. Alternatively, tap a `.xph` file in
the Files app and it opens in Xephon. The session loads with full
state (including playback if the source audio was bundled).

### Export JSON

The **Export** button on the top chrome (a share icon) writes the
per-utterance rows as JSON (see `docs/output_schema.md`). The same
action is also available as **File → Export to JSON** (⌘⇧S). Use
this to bring results into external tooling.

---

## 16. Settings reference

> **Figure 19: Settings card, expanded.**
> The full Settings card on the Settings page (⌘1) with every
> control labeled:
>   • Language picker (🇯🇵 Japanese ⌄) and Offline ASR picker
>     (Apple SpeechAnalyzer / WhisperKit / Qwen3-ASR) on the same
>     row in landscape; stacked on portrait
>   • Text SER picker (WRIME ⌄), shown only when more than one
>     backend is available
>   • Speech Boost toggle (mic mode only)
>   • Speaker Sensitivity slider (👥 ─── value, double-tap to reset)

| Control               | When to change                                          |
|-----------------------|---------------------------------------------------------|
| Language              | Recording in a language other than Japanese. Locked while a session is running. |
| Offline ASR           | Pick the fallback recognizer used by file-mode analysis and Re-evaluate. Apple SpeechAnalyzer is default; switch to WhisperKit (Kotoba-Whisper) for more conservative Japanese transcripts, or Qwen3-ASR (experimental). |
| Text SER              | Pick WRIME for fastest Japanese inference, or Apple FM for richer (slower) reasoning. Auto-falls back when one is unavailable. Hidden when only one backend is installed. |
| Speech Boost          | Quiet or distant input. Affects ASR only.              |
| Speaker Sensitivity   | Diarizer is splitting one person into many speakers (drag left), or merging two people into one (drag right). Double-tap to reset. |

The **Custom Glossary** button lives on the Fusion Legend card on the
Affect page (⌘2), not on the Settings card — see § 9.

---

## 17. Pipeline card — what each indicator means

> **Figure 20: Pipeline card, all stages active.**
> Vertical list of six stages, each with an icon, label, throughput
> number, and a green-check status pill. Stages:
>   • 🎙 Capture
>   • 🔊 ASR (Live)        — "5 snt"
>   • 👥 Diarizer          — "2 spk"
>   • 🎚 Acoustic SER      — "1.84 s"
>   • 💬 Text SER          — "120 ms"
>   • ✨ Fusion            — "5 utts"
>   • 📤 Export

The status pill is green when the stage is healthy, yellow when
degraded (e.g. running on CPU because the ANE is busy), and red when
unavailable.

A diagnostics banner appears at the top of the screen if any model
fails to load. Tap the banner for details.

---

## 18. Troubleshooting

| Symptom                                              | Try                                                                                                                                                                                                          |
|------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| "Some models didn't load" banner on launch           | Settings → Models card → Manage Models → tap Re-download for the failing model. Network required.                                                                                                            |
| Live recording shows transcript but no emotions      | The acoustic SER models may have failed to load. See the Pipeline card. Re-download via Models card.                                                                                                         |
| One person is being split into multiple "speakers"   | Drag the Speaker Sensitivity slider left. Then re-evaluate the affected rows, or reassign them and toggle "Teach diarizer" so the merge sticks.                                                              |
| Two people are being merged into one speaker        | Drag the Speaker Sensitivity slider right. Reassign rows and use Teach diarizer for corrections.                                                                                                              |
| Transcribe button gives garbled output                | If glossary's ASR Hint is on, the SFSpeechRecognizer pathway may be biasing toward your hint terms incorrectly. Turn the master ASR Hint toggle off and try again to compare with the default offline path. |
| Glossary entries with weight set but no chip on rows | The master "Apply Bias" toggle is off, OR the entry's 📕 toggle is off, OR the term doesn't substring-match the transcript (case-insensitive). Check the emotion label chip on the entry — if greyed, the entry is inert. |
| "Empty audio" or missing acoustic on late rows      | The audio source ended before ASR finished. The app now re-reads from the source file for file-mode sessions, so this should be rare. Re-evaluate the affected row.                                          |
| Apple FM never produces text                        | The app is backgrounded or another app has the ANE. Foreground Xephon; it will retry the Foundation Models call on the next utterance.                                                                       |

---

## 19. Privacy

- **All audio processing is on-device.** Recording, transcription,
  diarization, acoustic and text emotion analysis all run on the
  iPad's Neural Engine / GPU / CPU. No audio, no transcript, no
  embedding ever leaves the device.
- **No analytics, no telemetry.** Xephon doesn't phone home.
- **Model weights** are downloaded from the public release the first
  time they're needed. After that, everything is local.
- **Saved sessions** (`.xph` files) live wherever you save them —
  Files app, iCloud Drive, USB, etc. Treat them like any other
  sensitive document.
- **Glossary entries** persist in the app's private Application
  Support directory. They're not bundled with `.xph` sessions, so
  sharing a session doesn't share your glossary; use the glossary
  sheet's Export to move it deliberately.

See `docs/privacy.md` for the formal privacy statement.

---

## 20. Keyboard shortcuts (external keyboard)

The shortcuts below are the ones the app actually registers with the
system menu bar. They surface in the iPadOS 26 menu strip (press and
hold ⌘) and in macOS / Designed-for-iPad on Apple Silicon menus.

**File**

| Shortcut | Action                              |
|----------|-------------------------------------|
| ⌘ O      | Open audio file                     |
| ⌘ ⇧ O    | Import a saved session (`.xph`)     |
| ⌘ S      | Save session as `.xph`              |
| ⌘ ⇧ S    | Export per-utterance JSON           |

**Edit**

| Shortcut | Action                                                |
|----------|-------------------------------------------------------|
| ⌘ Z      | Undo last edit (transcript / speaker / sections / glossary / keywords / settings) |
| ⌘ ⇧ Z    | Redo                                                  |
| ⌘ F      | Focus the utterance search field                      |

**View** — switch the control pane to a specific page

| Shortcut | Page         |
|----------|--------------|
| ⌘ 1      | Settings     |
| ⌘ 2      | Affect       |
| ⌘ 3      | Speakers     |
| ⌘ 4      | Sections     |
| ⌘ 5      | Keywords     |
| ⌘ 6      | Summarizer   |

Sheet items in the View menu (Summary, Review, Search & Replace) are
listed there as labelled buttons but ship without keyboard shortcuts —
trigger them with the matching toolbar buttons in the top chrome.

There is **no** ⌘R shortcut to start / stop recording — tap the
Record button instead.

---

## 21. Getting help

Xephon is research software — there's no support desk. The
documentation in `docs/` covers the architecture, model selection,
and known limitations. The `eval_log.md` file in particular tracks
known CER / WER / SER metrics on the project's held-out calibration
set so you can sanity-check your own results.

For bug reports or feature requests, contact the developer directly.
