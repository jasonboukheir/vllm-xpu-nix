# Frozen pair and compatibility evidence

Discovery ran on Brutus on 2026-09-20 UTC. Exact commits, timestamps,
remote URLs and baseline identities are in `manifest.json`; remote advertisements
are saved separately. The user explicitly permits upstream-main snapshots.

The latest observed vLLM main is
`e378275a8f20eac9e92212e4a1fee84ec5a31dc7` (2026-09-19 23:23:29 -0700).
Its `requirements/xpu.txt` requires Torch 2.13.0, Triton 3.7.2+xpu and
vllm_xpu_kernels 0.1.14.1. `pyproject.toml` also pins Torch 2.13.0.

The latest observed kernel main is
`d7c35d281cf50564996fb0b9c41676b64f13f970`. Its ancestor
`0bfb37287b335fccf94f005141774af99e3eef33` changes the exact Torch dependency
to 2.14.0+xpu in pyproject/requirements, expects 2.14 in CMake, and changes the
wheel toolchain to oneAPI 2026.1. vLLM has not made this dependency transition.
Those heads therefore do not have a consistent declared dependency pair.

Select the newest kernel main ancestor before that boundary:
`da16a5595c105605bcd44b558c7933b934a05f85` (2026-09-16 16:33:41 +0800).
It requires Torch 2.13.0+xpu, uses the existing 2026.0 toolchain and descends
from the old kernel base and the 0.1.14.1 lineage. No Torch/Triton/oneAPI pin
change is justified for this pair. The upstream exact kernel wheel requirement
is replaced by the coordinated source-built kernel package, as in the existing
Nix packaging; the newer source's relevant interface compatibility is audited
here and must pass build/runtime qualification.

`kernel-api-delta.patch` records all binding/header changes from the prior
upstream base. Existing registered schemas are retained. Additions are standard
LayerNorm variants and compact sampling masks. The sampler preserves its CPU
shared [2] int64 RNG mode and adds device per-row [batch,2] RNG. The old/new
`flash_attn_interface.py` change generalizes block-size tiling without changing
its public signature. Relevant attention, GDN, normalization and W4A16 paths
still require paired regression tests; source inspection is not qualification.

Known out-of-profile API limitation: vLLM's `topk_hash_softplus_sqrt` wrapper
passes `bias_vl`/`image_sentinel_lo`, while this kernel snapshot lacks them.
The same mismatch exists at the old released upstream bases; neither the dense
Qwen target nor DFlash draft uses this MoE router. Kernel head #603 adds those
parameters, after the Torch2.14 transition. Do not claim full-model compatibility
or silently discard this finding. If this path becomes required, version the
candidate and resolve it explicitly rather than moving frozen heads unnoticed.

The source-compatible selection is scoped to the required dense Qwen target,
DFlash2, existing MTP and vision, with all qualification still pending. This is
not a claim that either unpatched upstream snapshot supports KVarN.

Checks completed: old-base ancestry, main-to-release ancestry, immutable object
resolution, dependency inspection, binding/header diff inspection; dry-run
application of both kernel packaging patch sets passed at the selected target.
Native build, loaded operator checks, runtime regressions and GPU qualification
remain pending. New contrary evidence must revise this record and affected
audits/tests before release.
