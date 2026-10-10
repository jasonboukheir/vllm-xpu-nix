# k08 — native KVarN cache and compact K4V2 attention

**Disposition: KEEP.** The selected kernel release supplies no replacement for
the native KVarN feature. Preserve the consolidated implementation and paired
ABI. The vLLM feature needs the allocator and block-size adaptations documented
in [v05](v05-kvarn.md); those changes do not require replacing the native record
format or decoder. No new runtime or performance result is claimed here.

## Frozen inputs and scope

| Input | Immutable identity |
| --- | --- |
| Assigned commit / final old kernel tip | `193cee080d654ff67c4f8bfaf924201bc88b9cf5` |
| Parent | `8b6ec9d6a31836e3f19447ec790169655ce7da6a` |
| Subject | `feat(attn): add native KVarN cache and compact K4V2 attention` |
| Stable patch ID, independently recomputed | `4ff4c3eb11770280350fa8f4739b6edee6cacff4` |
| Old kernel base | `1c7cbeee1cd0c1481d48f5031f679a47f3f0ef45` |
| Selected kernel source | `ddf336d86e3c8602888572a3502f951abd51df12`, release/0.1.15.4 |
| Old vLLM base / tip | `ced6857afa0ea7b2e3f0846a62e1394e90f15607` / `466b9d5abe84f13c29b38c686e4d77fc09e09aa4` |
| New vLLM base | `db9527a46873454610df6dbedf79a36d6bf1a7f6`, v0.31.0 |

Read the update skill's `references/commit-audit.md`, both source repository
`AGENTS.md` files, `docs/fork-delta.md`, this run's complete five/eight-commit
`inventory.json`, dependency/provenance records, v05 and the prior release
notes. The previous k08/k10/k11 audits were read from packaging archive commit
`8c72895340deafef77998a579335d078800626ab`. All implementation inspection used
frozen Git objects. Only this report was written; no source, Git state,
services or GPU state was changed.

The kernel target is the frozen official source branch associated with the
published 0.1.15.4 wheel, not a source tag. The parent's
`kernel-wheel-source-check.json` records matching pure-Python payloads;
that is not proof that the wheel embeds this native source revision.

## Requirement and old final behavior

This 31-file consolidated feature supplies native grouped K4V4/K4V2 record
writers/readers, Hadamard transforms, KV/QKV scatter, bounded materialization
and hybrid packed-history/resident-FP16 decode. D256/G128 compact records are
35,072 bytes per head for K4V4 and 26,880 for K4V2. Dropping it makes the
paired vLLM KVarN backend unavailable; reconstructing whole caches would defeat
its memory requirement.

The recorded xpu-v1.11.0 workload is Arc Pro B70, RedHatAI/Qwen3.8-27B-INT4
revision `bf08f3dbd9a324e53956920aad378a1f1b6dd24a`, BF16 compute, eager V1,
compact K4V2 target and bundled MTP draft, two speculative tokens, four
scheduler slots, 2048 batched prefill tokens and 262144 maximum context.
The recorded capacity was 346880 shared attention tokens. These facts describe
the previous qualification, not measurements of the refreshed pair.

Native decode remains guarded to variant 18, DPAS layout, Q24/KV4/D256,
G128 and batch 1–12. Both value widths, allocating/reusable scratch APIs,
natural-layout materialization/reference paths and independent GPU-test
operators remain supported. Retiring those alternatives would change support
scope; the upstream delta does not justify doing so.

## Exact upstream coverage

Only four commits lie between the old and selected kernel bases:

| Upstream commit | Actual change and relevance |
| --- | --- |
| `f4b729ae7fb3eb22c564606ec8b4d184c7c8bc4a` | Ragged speculative traversal in three GDN sources and their tests. Relevant to separate k06; neither defines KVarN nor changes its attention ABI. |
| `d09702a637cfea2a0c02909af82bd789d26ea319` | Torch 2.14, updated build images/oneAPI 2026.1 and Xe3 C++20 target setting. Preserve these build changes. |
| `885b120ffa082312b7d93c2be7a1e3b3dd79672f` | Adds paged-decode policy `16,128,64,false,true,false`; ordinary attention coverage, not the D256 KVarN record reader. |
| `ddf336d86e3c8602888572a3502f951abd51df12` | MoE `topk_softplus_sqrt` interface/implementation/tests. No KVarN replacement. |

The entire `csrc/xpu/attn/xe_2`, `csrc/flash_attn` and `csrc/cache.cpp`
subtrees/files are unchanged across those bases. Searching target `csrc`,
`tests` and `vllm_xpu_kernels` finds no `kvarn` or `k4v2` implementation.
Alternative implementations were also inspected:

- Target `flash_api.cpp::mha_varlen_fwd` and its registered `varlen_fwd`
  accept ordinary separate K/V/scales; they do not accept grouped packed
  records plus physical-block-to-resident-slot indirection.
- Ordinary Xe2 `DecodeFwdMainloop` supports FP8 KV conversion and per-tensor
  scales. These cannot interpret K4/V2 codes, grouped row/column factors,
  zero points and DPAS swizzles. Its `ReduceSplitK` does not supply the
  fused inverse-H256 output/two-FP16-rounding contract either.
- At the new vLLM target, `platforms/xpu.py` can route TurboQuant presets
  to `TurboQuantAttentionBackend`, whose `customize_spec` allocates a
  per-token/head slot and `LBNHC` layout. This is a different compression
  contract, not a native KVarN substitute. Its only old-to-new backend edit
  is accepting `kv_cache_spec=None` in the block-size method. No new
  grouped K4V2/resident-tail implementation appeared there.

Thus there is no upstream-driven source reduction inside the essential
native feature for this release.

## Minimum retained change and replay guidance

Retain the feature's operators in `csrc/flash_attn/flash_api.cpp` and the
Xe2 writer, Hadamard, scatter, decoder and dequant/materializer sources.
Retain its shared Q6 epilogue row ownership/reduction behavior and empty-split
natural-LSE sentinel initialization in `kernel/paged_decode_kernel.hpp`.
The ordinary mainloop opts out of that initialization; keep ordinary attention
dispatch and run its native controls after replay.

Keep `value_bits` ordering/defaults, record offsets/strides, exact multisplit
scratch extents and tensor lifetime handling paired with vLLM `466b9d5`'s
successor. `kvarn_decode_with_scratch_xe2` uses the first public statistics
tensor for natural-log LSE; `max_logits` is still validated, passed and
stream-tracked although its values are unused. Removing it is a possible
coordinated ABI cleanup, saving only `4 * B * 24 * splits` bytes per scratch
instance. It requires both repositories' schemas, callers, guards and tests
to change together. It is not needed for this release's minimum replay.

Reconcile with v05 as follows: its spec-aware block-size method should use
the per-layer spec, and its independent-pool allocator must preserve new
upstream checkpoint/alignment behavior. Neither permits changing native
G128/D256 geometry, widening supported decode shapes, or replacing exact
committed history with optimistic CPU length bounds. Target/draft mutable
tail ownership and accepted-history boundaries remain vLLM responsibilities;
the native kernel cannot repair premature packing of rejected draft tokens.

Build details that must survive replay:

- The feature's `pyproject.toml` hunk **only adds `subtiles` to codespell**.
  It does not own the Torch requirement. Keep the new `2.14.0+xpu` pin,
  matching requirements/CMake/Nix inputs; do not restore the old entire file.
- Its `cmake/utils.cmake` changes affect `add_xe2_kernel_library` and
  `MIXED_GRF_SIZE`. Upstream's new C++20 edit affects
  `add_xe3_kernel_library`; keep both. Xe2 already used C++20 at the old
  base. The target retains SYCL-TLA
  `87f6850680a580654b9ea2c80dbc01aeb36ad231`.
- Keep Xe2 mixed-GRF library handling: explicit decoder/reducer GRF
  properties do not prove the library-wide option redundant for its writer,
  scatter and materializer helpers. Rebuild against the parent's pinned
  Torch 2.14/oneAPI 2026.1 substrate and verify actual loaded libraries.
- The prior definition-only coordinate-loader cleanup is already absent
  from `193cee0`. Do not resurrect it by replaying the pre-consolidation
  feature. Keep the dequant and fragment-coordinate operators used by
  independent tests.

## Folded optimizations and limits of evidence

Two independent optimizations are already folded into this feature. Neither
is covered by the new upstream delta; preserve them initially with their
existing guards. They remain separately reversible if matched measurements
support a smaller implementation.

`KVarNReduceSplit16FourColumnOutputHadamardKernel` in `kvarn_decode.hpp`
is selected only for fused unrotation at sixteen splits. The earlier k10
audit documented about 3.03% geometric improvement in six short-MTP native
cells, but failed broad fresh-start repeatability and a controlled K4V4
B2/4096/split16/resident3 regression of 3.66796%. Those are historical
event-span observations including submission gaps, not v0.31 serving gains.
The generic fused reducer remains a potential replacement for that one
dispatch arm if a controlled comparison rejects the specialized reducer.

`load_tail_k_fragment`/`load_tail_v_fragment` in
`collective/kvarn_decode_mainloop.hpp` use block copies only for K4V2 and
independently 64-byte-aligned K/V bases. Unaligned contiguous views retain
the scalar path; K4V4 retains its scalar resident loader. The earlier k11
audit documented roughly 39–51% shorter native K4V2 spans at 16K/65K,
and an unresolved short K4V4 +12.3% anomaly. It also preserved the limits of
one address set, synthetic values and serving temporal effects. None is a
fresh measurement here.

The xpu-v1.11.0 release explicitly accepted operational model checks without
new controlled optimization comparisons. Do not convert its 705 installed
native/backend passes, reasonable model outputs or preserved capacity into
proof of either optimization's current speed benefit. If revisiting them,
hold resident loads fixed when comparing reducers and hold the reducer fixed
when comparing resident loads. Use bounded fresh-process ABBA with native
dispatch evidence, identical inputs/loaded-library provenance and the adverse
K4V4 cells; otherwise report the remaining uncertainty.

## Validation observed and required

Observed in this audit: frozen diff and alternate-path inspection, complete
four-commit upstream coverage, stable patch-ID recomputation, native
schema/guard/source tracing, paired caller lookup and historical audit review.
No build, installed test, raw timing reanalysis or GPU check ran here.

Parent qualification should include the existing suites rather than add
implementation-mirroring tests:

1. Compact/natural/DPAS record layout tests and
   `tests/flash_attn/test_kvarn_k4v2_xpu.py`: independent writer bytes,
   dequant/materialization oracles, both widths, padded stride, adjacent-page
   canaries, wrong-width rejection, ragged mixed packed/resident attention
   and independently offset K/V tails. The final old alignment test already
   includes split16 fused FP16/BF16 output in addition to split1/24/32.
2. `test_kvarn_decode_xpu.py`: Q6 distinct-row LSE ownership, poisoned
   inactive splits, exact scratch-extent guards, allocating/reusable parity,
   allocator pressure, fused/separate H256 and BF16-copy contract, long ragged
   contexts and replay. Preserve shape/variant/layout rejection behavior.
3. Hadamard/scatter tests and paired vLLM lifecycle/MTP tests, including
   lengths around page/flush boundaries, accepted/rejected tail overwrite,
   bounded materialization, target/draft ownership, cancellation and reuse.
   Standard finite-storage checks do not erase historical NaN-poisoned
   unused-BF16-position failures documented in xpu-v1.11.0.
4. Ordinary native attention/vision controls because shared epilogue and
   build settings are affected, followed by actual Brutus health/model,
   chat/tools/images, active MTP, mixed concurrent prefills and long-context
   checks on the exact paired package. Capacity and quality must remain
   acceptable; performance claims require matched measurements.

Map original `193cee0` to one surviving native feature commit, with any
required build/API compatibility repair folded into that owner. This audit
does not qualify Model Runner V2/DFlash, change production settings or claim
that the frozen source branch's native wheel provenance is stronger than
the recorded evidence.
