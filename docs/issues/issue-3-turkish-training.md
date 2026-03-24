# Issue #3 — Turkish training status

## Summary
The repository currently documents multilingual generation for a fixed set of languages, but it does not publish a supported training workflow for adding Turkish.

## Current repository state
- The README documents multilingual generation with language-specific aligners.
- The listed supported languages are `ar`, `ch`, `de`, `es`, `fr`, `it`, `ja`, `pl`, and `pt`.
- Turkish is not listed in that supported-language set.
- The repository does not currently ship public training or fine-tuning scripts for adding new languages.

## What this means today
- The public repo does not currently document Turkish generation support.
- The public repo also does not currently expose a supported training path for adding Turkish.

## What would be required
Supporting Turkish through the public repository would likely need:
1. released training / fine-tuning code
2. a Turkish alignment strategy
3. data preparation guidance for Turkish text/audio pairs
4. validation for generation quality and alignment quality in Turkish

## Current safe answer
Based on the repository as it exists today, Turkish training support is not yet documented in the public release.
