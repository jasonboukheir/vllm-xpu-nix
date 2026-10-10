# v04 — KEEP: explicit unquantized XPU cache dtypes

Audit owner: `/root/v04_explicit_kv`. Read-only source audit for
`release-v031-20261010`; only this report was written. No source, Git ref,
service, GPU, native build, or publication operation was performed.

## Frozen inputs

| Input | Value |
| --- | --- |
| Assigned commit | `984e6c3edda05c4f2dca37703534cb0e69cfa0fa` |
| Parent | `d909b25f0119419481bc378c3c0a3c656f9b76b0` |
| Subject | `fix(xpu): accept explicit unquantized KV cache dtypes` |
| Stable patch ID | `6f704ef6d098637584d969bf3fe2b27a26615957` |
| Old vLLM base | `ced6857afa0ea7b2e3f0846a62e1394e90f15607` (v0.30.0) |
| Old vLLM tip | `466b9d5abe84f13c29b38c686e4d77fc09e09aa4` |
| New vLLM base | `db9527a46873454610df6dbedf79a36d6bf1a7f6` (v0.31.0) |
| Old kernel base | `1c7cbeee1cd0c1481d48f5031f679a47f3f0ef45` |
| Old kernel tip | `193cee080d654ff67c4f8bfaf924201bc88b9cf5` |
| New kernel base | `ddf336d86e3c8602888572a3502f951abd51df12` (wheel release 0.1.15.4) |

Full local inventories and kernel-release provenance are in
[inventory.json](../inventory.json). Read both source `AGENTS.md` files,
the update skill's `references/commit-audit.md`, [fork delta](../../../fork-delta.md),
the published xpu-v1.11.0 notes, and archived v09/v15 audits and commit map
at packaging `8c72895340deafef77998a579335d078800626ab`.

## Requirement and final old behavior

The ordinary native XPU FlashAttention cache writer must accept explicitly
selected `float16` and `bfloat16` when key, value, key cache and value cache
all have that exact dtype. Its native dispatch spells this mode `auto`, so
the local patch verifies all four tensor dtypes and translates only the
native-call argument. It does not convert buffers or permit mixed input and
storage types. Changing the engine configuration to `auto` would defeat
the user's explicit choice: compressed-tensors checkpoint metadata can then
select FP8 caching.

This remains a supported optional configuration. Brutus's current
`RedHatAI/Qwen3.8-27B-INT4` profile uses BF16 compute, compact K4V2 target and
draft caches, bundled MTP2, eager V1, four scheduler slots and B70 hardware;
its ordinary K4V2 writer does not need this translation. Explicit native
FP16/BF16 caches and the retained K4V2-target/BF16-draft control do need it.
The current user request has not retired those configurations, and passing
the all-K4V2 profile alone would not establish that this patch is redundant.

The subsequent KVarN commit `466b9d5abe84f13c29b38c686e4d77fc09e09aa4`
does not modify either file touched by this commit. Thus its production
guard and tests retain their full effect at the frozen old tip. The
independent native causal-reference tests were already folded here in the
previous release; there is no remaining repair/revert pair to remove.

## Upstream coverage and changed architecture

The requirement remains uncovered by the selected pair:

- At vLLM target `db9527a46873454610df6dbedf79a36d6bf1a7f6`,
  `vllm/_custom_ops.py:2544`, `reshape_and_cache_flash`, still forwards
  `kv_cache_dtype` unchanged. Its complete function AST is identical to
  the old upstream base. No old-to-new commit changes this function's
  dtype handling.
- `FlashAttentionBackend.supported_kv_cache_dtypes` in
  `vllm/v1/attention/backends/flash_attn.py:288` admits the explicit names.
  `FlashAttentionImpl.do_kv_cache_update` at line 1507 splits the cache
  into the actual strided key/value views and passes `self.kv_cache_dtype`
  unchanged. `fa_utils.py:30` still binds the XPU writer to
  `ops.reshape_and_cache_flash`. There is no intermediate normalization.
- At kernel target `ddf336d86e3c8602888572a3502f951abd51df12`,
  `csrc/cache.cpp:955` dispatches contiguous and strided-head writers
  through `DISPATCH_BY_KV_CACHE_DTYPE` in
  `csrc/quantization/fp8/quant_utils.h:81`. That macro accepts `auto`,
  `fp8`/`fp8_e4m3`, and `fp8_e5m2`; literal FP16/BF16 names reach
  `TORCH_CHECK(false, "Unsupported data type of kv cache: ", KV_DTYPE)`.
  Both native files and `vllm_xpu_kernels/flash_attn_interface.py` are
  byte-identical between the frozen kernel bases. Last modifications of
  the native files are respectively
  `619c409849500e0f56f8da00d5f11c4aa75cd6f9` and
  `6822a3f91b83504424389eca7350abece4cf6e1e`, both before this delta.
- The existing explicit-config precedence in
  `vllm/model_executor/layers/attention/attention.py:281` still requires
  leaving engine/config state explicit. Upstream
  `852f567444cf8c206219edb7b2c42aec55fc41cf` supplied that policy;
  `53a20886752bdf8f26c4c5121458c514096da107` independently supplied native
  dtype-name acceptance to the Triton cache writer. Both were verified
  ancestors of the old v0.30 base, and neither fixes this native XPU path.

The v0.31 delta does introduce an alternate path worth distinguishing:
`e4340e41c9a32b2247cc2442b35aa91b081c424f` adds XPU batch-invariance
support and selects Triton attention by default when `use_batch_invariant`
is true. Triton's `_NATIVE_KV_CACHE_DTYPES` accepts explicit names. This
removes exposure for that particular default-selected path, not for native
FlashAttention: `XPUPlatform.get_attn_backend_cls` still honors an explicit
FlashAttention choice even under the batch-invariant policy and still
defaults ordinary FP16/BF16 attention to FlashAttention. Multimodal-prefix
and float32 fallbacks also do not retire native unquantized caching.

Other examined FlashAttention changes,
`927c87b3472a39da79a95a8df12d0a2b21e2d32e` (DCP/draft support),
`4fb767ff0a0e0aae9dd2720c47245d0202136dfd` (SM90 FA4 FP8 selection), and
`8c1557a79c539ffe82d004d2a0c8d7b5e71159ce` (docstrings), leave this writer
boundary unchanged. A wholesale backend switch would change behavior and
potential performance; it is not an equivalent upstream fix.

## Disposition and replay dependencies

**KEEP** the existing fourteen-line production guard/translation and its
focused native regression tests as one independent correctness commit.
This is already the minimum residual: no kernel schema changes, dtype
conversion, engine-level normalization, new aliases, or KVarN changes are
needed. Preserve the XPU device gate and all four dtype guards; retain
unchanged forwarding for other devices, `auto`, and FP8.

There is no local production prerequisite despite the historical parent.
The paired kernel release already provides the required `auto` mode and
the same eight-argument native schema (`csrc/torch_bindings.cpp:314`). The
KVarN feature's optional explicit-BF16 draft path depends on this behavior;
it does not replace it. No other old commit needs folding into v04.

## Validation observed and still required

Ran a CPU-only differential check using `../vllm/.venv/bin/python`, standard
library AST extraction and tensor/operator stubs, without importing Torch
or initializing XPU. **43 checks passed**:

- the old-base and new-upstream wrapper ASTs are identical;
- both explicit names normalize only in the old fork tip;
- each of the four tensors with a mismatching dtype is rejected before
  dispatch for each explicit name in the fork tip, while upstream has no
  guard;
- `auto`, FP8 and non-XPU literal names remain unchanged in all three
  implementations.

Separately compared exact Git blob contents and confirmed the native
writer, dispatch header and native-attention Python interface are unchanged
between kernel bases. These are observed source/forwarding checks, not
native numerical qualification. No new GPU result is claimed.

The v1.11.0 release notes record passing ordinary finite-storage native
references and sparse-writer controls, with six historical intentionally
unused-BF16-NaN diagnostic failures. Those old outcomes neither qualify
v0.31 nor imply that this translation fixes NaN-padding behavior.

After replay/build, run the installed candidate's
`tests/v1/attention/test_xpu_unquantized_cache.py`: four sparse-write cells
(two dtypes by contiguous/actual split-strided layout), plus 24 BF16 causal
reference cells (two layouts, two finite-padding values, batches 1/4 and
query lengths 1/3/129). The latter must retain the direct
`_fallback_varlen_attn` rejection and independently computed FP32 causal
reference, including future-value poisoning and exact replay. Confirm a
missing native policy triggers the forbidden-fallback sentinel. Check
explicit BF16 survives checkpoint FP8 metadata, all four mismatch guards,
and ordinary final-pair Brutus lifecycle tests. An upstream-alone explicit
native writer call is the expected-failure sensitivity control. GPU tests
are pending and must not be represented as passed by this audit.
