# Undo/Redo Verification — Xephon

Manual on-device checklist for the unified `UndoManager` feature (commit
`f6d33ff` + grouping crash fix). See `RecordingController+Undo.swift`.

**Setup:** physical M-series iPad, hardware keyboard attached (for Cmd-Z /
Cmd-Shift-Z). Each edit below: do it → `Cmd-Z` reverses it → `Cmd-Shift-Z`
reapplies it. Also confirm the **Edit menu** reads the right action name
("Undo Edit Transcript" etc.) and greys out when the stack is empty.

## Regression guard (grouping crash)
- [ ] First edit of a fresh session does **not** crash (`_registerUndoObject...
  must begin a group`). `groupsByEvent = false` requires every step to open
  its own group — test before anything else.

## Per-surface
- [ ] **Hand-edit transcript** — edit one utterance's text → Cmd-Z restores it.
- [ ] **Hand-edit 1→N split** — edit text so it splits into multiple sentences
  → one Cmd-Z collapses all child rows back to the original single row.
- [ ] **Speaker rename** → Cmd-Z restores old name everywhere it appears.
- [ ] **Speaker reassign** (utterance → different speaker) → Cmd-Z restores
  prior speaker.
- [x] **Re-evaluate utterance** → Cmd-Z reverts the re-eval (row + side maps +
  any reorder).
- [ ] **Long-press revert-reevaluation** → Cmd-Z is itself undoable (re-applies
  the re-eval).
- [ ] **Search & Replace → Commit All** → **one** Cmd-Z reverses the *entire*
  batch (menu says "Undo Replace All"), not row-by-row.
- [ ] **Session title** — edit field, tab/return out → Cmd-Z restores prior
  title. (Per-keystroke undo *inside* the field stays UIKit-native — expected.)
- [ ] **Fusion: acoustic weight slider** — drag → one Cmd-Z restores pre-drag
  value (not per-pixel).
- [ ] **Fusion: text weight floor slider** — same.
- [ ] **Diarizer clustering threshold slider** — same; confirm value pushes back
  into the diarizer.
- [ ] **Fusion Reset button** → Cmd-Z restores the grouped pre-reset snapshot.
- [ ] **Sections** — add / rename / remove / complete-section → each Cmd-Z
  reverses cleanly.
- [ ] **Glossary** — add / remove entry, toggle enabled, toggle ASR-hint →
  Cmd-Z restores (entry UUIDs preserved).
- [ ] **Keywords** — add / remove / move (reorder) / assign-to-group / group
  add/rename/remove / Remove All → Cmd-Z restores including selection set.

## Cross-cutting
- [ ] **Redo chains** — after several undos, repeated Cmd-Shift-Z walks forward
  through the same steps.
- [ ] **Independent edits don't coalesce** — two unrelated edits in quick
  succession take **two** Cmd-Z presses (this is why `groupsByEvent = false`).
- [ ] **Stack reset on new recording** — start recording → stack is empty.
- [ ] **Stack reset on session load** — open a saved session → stack empty (old
  UUIDs invalidated).
- [ ] **Save preserves stack** — Save session → undo still works afterward.
- [ ] **levelsOfUndo = 50** — past 50 edits, oldest drops off (spot-check).

## Deliberately NOT undoable (confirm excluded, not broken)
- [ ] Speaker **correct / affirm / promote-to-new** (the three diarizer-DB-
  teaching ops) — no undo step, no crash. By design (no unteach hook).
- [ ] Per-keystroke text in title/glossary/keyword fields — native UIKit text
  undo only.
