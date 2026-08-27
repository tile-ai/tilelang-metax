---
name: maca-example-fixer
description: Use when fixing or triaging failing tests, examples, xfail markers, or skip markers under examples/maca/ in this TileLang MetaX repository. Trigger for MetaX C500 MACA backend issues including shared-memory limits, CUDA-named torch APIs routed to MACA, unsupported CUDA/PTX intrinsics, MACA codegen gaps, and C500-oriented tile or kernel parameter tuning.
---

# MACA Example Fixer

## Scope

Treat `examples/maca/` as MACA backend coverage for MetaX C500. Do not treat these files as generic CUDA examples.

The local PyTorch build is MACA-customized. `torch.cuda.*` APIs are valid here because they route to MACA, so do not rewrite them only to avoid CUDA naming.

Keep changes minimal and local. Prefer editing the failing example kernel or paired test under `examples/maca/`. Touch `tilelang/` or `src/` only after confirming a backend, lowering, codegen, or intrinsic gap.

## Workflow

Start with the exact failing pytest target:

```bash
python -m pytest examples/maca/<case>/test_*.py::<test_name> -q
```

If a case is marked `xfail`, run it directly and decide whether the marker is stale. Remove stale `xfail` markers when the case passes. Do not keep defensive `xfail` markers for passing cases.

Record the failing case, observed error, handling method, and final status when maintaining reports such as `report.csv`.

## Fix Strategy

Classify before editing:

- `shared memory / launch resource`: reduce kernel resource usage by changing tile size, thread count, head/block partitioning, or pipeline stages. Do not reduce test data volume unless explicitly requested.
- `codegen or intrinsic gap`: patch the MACA-specific lowering/template path narrowly. Keep CUDA behavior unchanged.
- `unsupported C500 feature`: keep or add `xfail` and state the concrete unsupported feature in the marker reason.
- `numerical mismatch`: debug correctness first. Shape reduction is only acceptable for smoke-test cases when explicitly allowed.

After classification, fix cases by failure class instead of mixing unrelated changes. If several cases share the same root cause and the patch is small, group that class into one commit. If a fix is large, risky, or case-specific, keep it as one commit per case. Do not combine unrelated backend, resource-tuning, and xfail-documentation changes in the same commit.

For `examples/maca/`, assume MetaX C500 unless the user says otherwise. Prefer fixed C500-oriented parameters over runtime device detection.

## Kernel Rules

For shared-memory failures, fix from the kernel implementation side when requested. Prefer smaller per-CTA shared allocations, smaller head tiles, fewer pipeline stages, fewer threads, direct fragment stores, or split staging buffers when TileLang layout inference requires distinct layouts.

When porting CUDA/PTX-dependent logic, do not preserve inline PTX. Implement MACA C or MACA-specific helper code instead.

For files already named with `_maca`, avoid adding redundant `maca` suffixes to public function names unless needed to avoid symbol collisions.

## Validation Expectations

Run the exact failing pytest target after each fix. If related cases share the same failure class, run them together before reporting.

Run formatting and lint checks before finalizing:

```bash
pre-commit run --all-files
```

After a direct fix, create commits without asking for additional permission. Use the commit granularity from the fix strategy: one commit for a compact shared root-cause fix, or one commit per case for larger case-specific fixes.

Include the validation commands and pass/fail results in the final response. Do not revert unrelated dirty files or submodules.
