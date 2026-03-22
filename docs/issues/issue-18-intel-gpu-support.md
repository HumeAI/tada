# Issue #18 — Intel GPU support status

## Summary
The current repository does not document or implement a dedicated Intel GPU execution path.

## Current repository state
- The main README examples use `device = "cuda"`.
- Test device selection in `tada/utils/test_utils.py` falls back in this order: `cuda` -> `mps` -> `cpu`.
- The main Python package does not contain Intel-specific runtime hooks such as `torch.xpu`, Intel Extension for PyTorch (IPEX), OpenVINO, or oneAPI setup.
- The repository does include an Apple-specific MLX port under `apple/`, which shows that non-CUDA backends need explicit maintenance.

## What this means today
- NVIDIA CUDA is the primary documented acceleration path.
- Apple Silicon has a separate MLX path.
- Intel Arc / XPU support is currently undocumented and should be treated as unsupported until it is validated.

## What would likely be required for Intel GPU enablement
1. Pick a concrete backend target (for example PyTorch XPU / IPEX or an OpenVINO-based path).
2. Add backend detection and device selection that can choose Intel hardware explicitly.
3. Validate codec, aligner, and generation code paths on Intel GPU, not just generic tensor creation.
4. Add a small smoke test and setup instructions for the chosen backend.
5. Document any dtype or kernel limitations separately from CUDA and MLX behavior.

## Safe contribution starting point
A minimal first contribution would be a backend support note plus a small device-selection abstraction that can report whether Intel GPU execution is available, before claiming full support.

## Out of scope for this note
This document does not claim that Intel GPU inference works today. It only records the current repository boundaries and the likely bring-up work.
