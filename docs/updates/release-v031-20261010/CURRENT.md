# vLLM 0.31 release record

The qualified stack is **xpu-v1.12.0**. See the
[release notes](../../releases/xpu-v1.12.0.md) and
[machine-readable manifest](../../releases/xpu-v1.12.0.json) for exact identities,
validation, limits and rollback. Publication verification and the final host
handoff are recorded on packaging main after the immutable release snapshot.

- vLLM upstream v0.31.0: `db9527a46873454610df6dbedf79a36d6bf1a7f6`;
  fork main: `ef92dc5600af64103c01f97e3300b81c3a1083ee` (five commits).
- Kernel wheel 0.1.15.4 maps to official release-branch source
  `ddf336d86e3c8602888572a3502f951abd51df12`; fork main:
  `dbb684f4c9b38a8b3bbf52bda0c490ee47049221` (seven commits).
  No matching upstream tag or embedded native source SHA exists. Seven wheel
  Python files match; native code was rebuilt from the frozen source.
- Dedicated agents audited all 13 source changes and four Nix patches against
  the 728-commit vLLM and four-commit kernel deltas. Upstream replaces local
  ragged-speculation code; its independent regressions remain. Retiring unused
  dispatcher pruning also passed full and narrow builds. Native production
  delta shrank by 173 changed lines across four files.
- KVarN replay preserves upstream checkpoint, allocator, DCP/TP and MTP behavior
  and adapts the spec-aware block-size API. The reconciliation and audit maps
  distinguish essential functionality, optional behavior and unmeasured gains.
- Required Torch 2.14, Triton 3.8, torchvision 0.29, oneAPI 2026.1, PTI 1.0.1,
  pyzes 0.1.2, XGrammar 0.2.7 and oss-harmony 0.0.11 were built. Unrelated locks
  remain fixed. Packaging fixes reconcile the XPU build pin and provide the
  correct OpenCL ICD and oneCCL plugin discovery. Failed starts/control probes
  and superseded candidate identities remain preserved.
- 608 packaging cases passed; installed CPU cases: 660 passed and 14 expected
  CUDA-only skips. Installed native/backend cases: 724 passed, no skips. These
  suites overlap and include CPU contracts; counts are not additive or all GPU.
- All 18 calibrated W4 cases passed on both stacks with byte-identical outputs.
  The reference uses empirically isolated B70 BF16 RTZ weight conversion;
  original RNE-reference discrepancies remain documented.
- The exact Brutus package loaded the same pinned INT4 Qwen model and passed
  health, text, reasoning, two-image vision, tools, concurrency, mixed prefill,
  cancellation/reuse, 131071/262015-token retrieval, and two simultaneous
  174079-token prompts generating 512 tokens each without preemption.
  Occupied attention capacity reached 349184 of 350336 usable tokens.
- Matched performance screening retained MTP2 over MTP1. See
  [the measurements](performance-results.md) for small-sample timing results,
  output drift and limits; [the feature review](performance-review.md) explains
  applicable upstream fusion and the separate KVarN port needed for V2/DFlash2.
- The historical host foreground app was removed at dc05414a. The updated
  runbook derives the current systemd command/environment and pins permanent
  release branches, preserving matching nested HTTPS source-tag overrides.
- Raw synthetic requests, logs and build evidence remain in ignored
  `benchmark-results/release-v031-20261010/`. All owned foreground servers were
  stopped. Original production state is preserved: chat failed with MainPID=0;
  embedding and STT inactive. The user performs the host rebuild/switch.
