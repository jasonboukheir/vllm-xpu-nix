# xpu-v1.8.0

Qualification completed on Brutus with the profile below.

This release removes the artificial single-request restriction for Qwen
bundled MTP with KVarN. The scheduler's `max_num_seqs` setting controls
concurrency. Other existing KVarN configuration checks remain in place.
The context estimator now includes recurrent state for every scheduler slot,
matching the existing allocator. Previously its suggested context could still
fail allocation when more than one request was configured.

## Sources and version

| Component | Upstream base retained from integration main | Released fork commit |
| --- | --- | --- |
| vLLM | `51da0ca66c8065619c79e35dff97aa99aeaf5644` | `3b6c822e8a4f56dc3904c7fce388a76dff4c7bdd` |
| XPU kernels | `e7f20cbc18d741419a84aa57178e1f1cc0b453da` | `9bb1e2d428b8a25b262c0cd6704614bdf6ee3ad8` |

Both bases are September 7 upstream snapshots, explicitly retained for this
release. This is a minor version because these bases differ from the preceding
stable `xpu-v1.7.2` pair, despite remaining unchanged relative to integration main.
The kernel tree is unchanged from the current integration tip. vLLM also retains
the model-inspection hard-exit retirement already present on integration main.
PyTorch, Triton, oneAPI, nixpkgs and all unrelated locks are unchanged.

On September 10, official discovery found [vLLM 0.29.0](https://github.com/vllm-project/vllm/releases/tag/v0.29.0),
published September 9 at `98dff2a81d747d1dba01a47f939f48c3526d4206`.
Its merge-base with the current vLLM snapshot is the August 31 commit
`f5c3cc240bc1fc0519694fe689edb87dee8dfd92`: publication is newer, but the
release branch omits subsequent integration work. Its exact kernel dependency,
[0.1.14.1](https://github.com/vllm-project/vllm-xpu-kernels/releases/tag/0.1.14.1),
resolves to `6d92b1bfbf32767ecda8e819613eb151e70030ad`, also older than
the retained kernel base. No rebase or history rewrite was performed.

## Build and qualification

The tested package uses Brutus's `package.nix`: torchvision, BMG AOT, and its existing
attention kernel selections. The split factory explicitly disables Xe3p for
its Xe2/default native libraries and Python bindings. Upstream now enables
Xe3p by default, which otherwise adds CRI AOT targets and selects a different
SYCL-TLA architecture. The initial build was stopped before qualification to
correct that unintended expansion. The attention compile cap now also honors
lower caller limits: `--cores 4` stays at four compiler processes instead of
being raised to twelve. Its package derivation after these fixes is
`/nix/store/1pwi2dgk4dw3nz6zncavn6bcgcdjslwz-python3.12-vllm-xpu-0.28.0+unstable.2026.09.11.g3b6c822.drv`.

The cold attention build remains memory-intensive: the single
`fmha_xe2.cpp` compiler process reached approximately 78 GiB RSS on Brutus.
The parallelism fix bounds simultaneous jobs, not the memory of one compiler.

- Configuration regression suite: 57 passed, including scheduler concurrency
  with one and two MTP draft tokens. These are admission tests, not GPU results.
- Pinned Ruff 0.14.0 and applicable local Python source checks passed. Downloaded
  pre-commit executables could not run on NixOS; equivalent checks used Nix tools.
- Nix formatting and both kernel cache identity checks passed.
- Build-parallelism regression passed for requested limits of 1, 4, 12 and 24;
  the attention compile cap preserves the first three and limits 24 to 12.
- All four downstream patches applied to isolated copies of the retained kernel
  source. Each remains required: split prebuilt libraries and offline source
  forwarding, bounded compile/link memory, skipping installation of externally
  supplied libraries, and forwarding the MHC build flag. The current source
  still lacks those packaging behaviors.
- Native libraries built successfully. All 51 focused GPU regressions passed:
  34 GDN, causal-convolution and INT4 cases, plus 17 KVarN attention cases. The
  KVarN fixture initially skipped until given the explicit new library path;
  its rerun passed all 17 cases. The final Python package built successfully
  and resolves both native extensions to the same files as that GPU test run.
- Capacity estimator: 13 focused CPU regressions passed. Two new four-request
  cases failed before the fix; one-request controls passed. Pinned mypy 1.20.2
  with Python 3.10 targeting and Nix-native typos also passed. A pre-existing
  scalar/list name collision in the touched allocator was resolved for mypy.

The release workflow now distinguishes releasing current main from patching a
deployed stable line, compares release lineage before choosing a base, keeps
ignored private data out of local Nix source snapshots, requires live overlap
evidence for concurrency changes, and supports exact-ref recovery after partial
publication. Runtime requirements and the limits of qualification evidence are
documented separately.

Qualified serving profile: `RedHatAI/Qwen3.8-27B-INT4` at
`bf08f3dbd9a324e53956920aad378a1f1b6dd24a`, compressed-tensors W4A16,
BF16 compute, KVarN compact K4V4/G128, two-token MTP, four active requests,
262,144 maximum combined context, 2,048 prefill tokens per batch, 0.96 GPU
memory utilization, and up to two 448×448 images with video disabled.

Cold and cached starts both failed at 90% GPU memory utilization. A 92%
start passed the initial check but exposed its underestimated recurrent-state
reservation, leaving attention capacity of 195,968 tokens. At 96%, startup
reached HTTP readiness with four recurrent slots and 265,856 shared attention
tokens. The configured 262,144 limit is per request; simultaneous requests
share this attention pool. The 96% budget leaves approximately 1.28 GiB of
the device outside vLLM's budget, with other vLLM workers disabled.

All 22 live API checks passed on the final package, using Brutus's foreground
app on local port 18000 to isolate qualification traffic:

| Check | Result |
| --- | --- |
| Streaming text, reasoning, two images and tool-call parsing | 4 passed; complete streams and usage records |
| Three rounds of four simultaneous requests | 12 passed; measured peak of 4 running requests |
| Two-image label/color retrieval at 8,191 and 32,767 input tokens | 2 passed; exact input token counts checked |
| An 8,191-token two-image prefill arriving during three decoding requests | All 4 outputs passed their label/content checks; peak of 4 running requests |

The simultaneous-request workload recorded 2,249 MTP drafts, 4,498 draft
tokens and 3,514 accepted tokens, confirming active two-token speculation.
Its observed acceptance rate was 78.1%; this is workload-dependent and is
not a model-quality score or controlled throughput benchmark. Synthetic tool
calls were parsed and inspected without execution.

The focused GPU suite includes a ragged four-request attention oracle reaching
262,144 tokens. That kernel check is separate from a full-model request at the
configured 262,144-token ceiling; no such full-model run is claimed.

Raw test artifacts remain in the ignored, private
`benchmark-results/releases/xpu-v1.8.0/` directory.

## Handoff

All three repositories use the annotated `xpu-v1.8.0` tag and permanent
`releases/xpu-v1.8.0` branch. Both source inputs in this packaging release
reference that coordinated tag. Previous release references remain unchanged.

Pin Brutus's packaging input to `refs/tags/xpu-v1.8.0`, retain its tested
package overrides and 96% serving budget, and compare the host package
derivation with the one above before rebuilding. The foreground worker was
stopped after qualification; operator-stopped inference services remain in
maintenance until the operator activates the new NixOS generation.
