# Selected release pair and compatibility evidence

This is a scoped release refresh for existing eager V1 K4V2 MTP2 and vision.
Issue #17 and its V2/DFlash work remain paused. Source compatibility is a
prerequisite for qualification, not a claim that the new pair has passed it.

| Component | Frozen upstream target | Advance from current fork base |
| --- | --- | --- |
| vLLM v0.30.0 | `ced6857afa0ea7b2e3f0846a62e1394e90f15607` | Direct descendant of `51da0ca66c8065619c79e35dff97aa99aeaf5644`, 444 commits ahead |
| Kernels v0.1.15 | `1c7cbeee1cd0c1481d48f5031f679a47f3f0ef45` | Direct descendant of `e7f20cbc18d741419a84aa57178e1f1cc0b453da`, 16 commits ahead |

The manifest records verified tag objects, source recovery branches and exact
old rolling/released tips. The released tips contain fixes missing from rolling
main; all 27 commits are included in the independent audit inventory.

## Kernel wheel pin versus coordinated source build

v0.30.0 `requirements/xpu.txt` names `vllm_xpu_kernels==0.1.14.1`,
Torch 2.13.0 and Triton 3.7.2+xpu. The v0.1.14.1 source tag
`6d92b1bfbf32767ecda8e819613eb151e70030ad` predates the existing kernel
integration base. Moving back to that source tag would discard already
integrated upstream work. The selected v0.1.15 is the newer published release
on the current Torch/toolchain line.

The existing Nix package deliberately removes the kernel wheel requirement
and supplies the coordinated source package (`nix/vllm-xpu.nix`). This is an
explicit source-pair qualification, not a claim that upstream changed its exact
wheel pin. Evidence for attempting this pair:

- v0.30's `vllm/_xpu_ops.py` is byte-identical to the old upstream base
  (blob `c1f2bb6969b9d4bfe22989c9fcc258d8bcfd3adb`). Existing XPU operator
  requirements remain, including the local paired GDN extension.
- v0.1.15 retains Torch 2.13.0+xpu in `pyproject.toml`, the existing CMake
  Torch version and oneAPI 2026.0 tooling. Its oneDNN revision is
  `0e2a5bfeef1bfbffc3137464606540233086ce9b`, matching `nix/mk-kernels.nix`.
- The registered schema delta adds standard/fused/Nemotron LayerNorm and
  compact sampling-mask operators. It does not remove the deployed model's
  operator interfaces. Sampler state retains the existing CPU `[2]` mode
  while adding device `[batch, 2]` support. The raw header/schema diff is
  `benchmark-results/release-v030-20260922/kernel-api-delta.patch`.
- The entire GDN implementation is unchanged from the old kernel base.
  In particular, release v0.1.15 does **not** contain the immediately following
  upstream `da16a559` (#600). Local k06 production changes remain necessary;
  the paused snapshot audit's different disposition does not transfer.
- KVarN remains a local paired feature. Audits must preserve all schemas,
  trailing value-width arguments, layouts and lifecycle contracts together.

The pre-existing MoE `topk_hash_softplus_sqrt` keyword mismatch is outside
the deployed dense-Qwen path. Do not claim universal model compatibility or
silently import post-release kernel main to fix it. Qualification must inspect
the realized imports/schemas and exercise the actual target, MTP and vision.

## Scoped dependency and build changes

The only changed common requirements between the old vLLM base and v0.30 are
Hugging Face Hub >=1.31.0 (for its `utils.httpx` re-export) and OpenAI >=2.25.0
(namespace tools). Read-only Nix evaluation of the existing lock found
OpenAI 2.41.1, Hub 1.26.0 before the existing overlay (which supplies 1.28.0),
and FastAPI 0.139.0. Thus Hub needs a scoped 1.31.0 bump; OpenAI does not.
XGrammar >=0.2.1 is still sufficient by the declared requirement, so preserve
the existing 0.2.1 override. Do not carry the paused snapshot's 0.2.7 bump.

FastAPI's pre-existing relaxed upper bound remains a limitation requiring
actual serving/API checks; this update does not establish support for optional
model-hosting-container handler overrides. Preserve the pinned nixpkgs,
Torch, Triton, oneAPI and unrelated locks.

The split kernel factories already explicitly disable upstream's XE3P default.
Verify generated options, Brutus `bmg` AOT selection, source projections and
all four downstream patches against the final source pair before building.
No native build or host/GPU maintenance has begun.

## Resume gate

All source and packaging audits are reconciled. Source replay is published;
packaging locks match the final pair. Final source projections, architecture/AOT
options and lightweight packaging checks pass (see `packaging-checks.json`).
Use `resume.md` for the maintenance handoff, exact Brutus build and native
qualification. Stable publication requires those checks to pass; the
operational release remains xpu-v1.10.0 meanwhile.
