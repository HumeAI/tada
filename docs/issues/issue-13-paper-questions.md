# Issue #13 — Paper clarification note

## Summary
This note turns the maintainer answers from issue #13 into a short reference for readers of the paper.

## Clarifications already provided in the issue thread

### 1) Lookahead shift `K`
The maintainer response states that acoustic features are shifted by **5 positions**. In practice, that means TADA uses **5 text-token lookahead** during TTS generation.

### 2) Evaluation setup for Tables 5 and 6
The maintainer response states that the evaluation is the **voice cloning setup from Table 2**, using **PPL as in Table 4**.

### 3) Cross-entropy / KD losses for the TTS setting
The maintainer response states that removing the CE and KD losses did **not significantly improve TTS performance**. The ablations in Table 6 were run at a smaller scale before the final main model, so the team does not currently report a separate "base" number for the fully removed-loss setting.

## Follow-up questions that remain open in the thread
The issue still contains follow-up questions that are not yet answered in the repository docs:
- how the text/audio pair construction handles the final lookahead positions during training
- whether text prediction remains active under the hood during inference in the TTS path

## Why this note exists
The GitHub issue already contains useful maintainer answers, but they are easy to miss if a reader only consults the repository files.
