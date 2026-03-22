# Issue #4 — IPA phoneme fine-tuning status

## Summary
The current repository does not document a supported fine-tuning workflow for Piper-compatible IPA phoneme training.

## Current repository state
- The public repository is inference-focused.
- The root README documents model loading, prompting, multilingual alignment, and generation.
- The repository does not currently expose end-to-end training or fine-tuning scripts.
- The codebase contains tokenization / alignment components, but there is no documented pathway here for swapping the training target to a Piper-compatible IPA phoneme representation.

## What this means today
- Fine-tuning on Piper-compatible IPA phonemes is currently undocumented in this repository.
- The repository does not provide a supported recipe for preparing IPA labels, retraining the aligner/tokenizer stack, or adapting the model for that representation.

## What would be required to support it
A real support path would likely need:
1. training or fine-tuning scripts
2. data-format guidance for phoneme-aligned supervision
3. tokenizer / aligner decisions for IPA-based inputs
4. validation examples that show the resulting generation path still works

## Practical next step
Until training code is released, the safest statement is that IPA phoneme fine-tuning is not yet documented or supported in the public repo.
