# Issue #5 — ONNX support status

## Summary
The current repository does not ship an ONNX export or ONNX Runtime inference path.

## Current repository signals
- No ONNX or ONNX Runtime dependency is declared in the root `pyproject.toml`.
- No export script based on `torch.onnx.export(...)` is present in the repository.
- The repository currently focuses on PyTorch inference plus an Apple-specific MLX port.

## What is missing for ONNX inference support
A credible ONNX path would need at least:
1. an export script for the relevant model components
2. shape / dynamic-axis decisions for text + acoustic generation
3. runtime validation against ONNX Runtime or another ONNX backend
4. clear guidance for unsupported operators or model components that do not export cleanly

## Safe contribution starting point
A minimal first step would be a prototype export investigation for a single component, with an operator compatibility report and a small parity check against PyTorch outputs.

## Important limitation
This note should not be read as claiming ONNX support today. It only records that the repository does not currently expose that path and explains what would need to be added.
