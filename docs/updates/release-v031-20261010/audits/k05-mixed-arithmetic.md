# k05 — preserve cached-decode arithmetic in ordinary mixed GDN batches

**Disposition: KEEP**, paired with vLLM `6efe5e7cc59de64ae8d3ec67bbe77379be34d2c9`.
The selected upstream pair still changes the arithmetic route of cached decode
requests when a prefill joins the batch. Keep the narrow native dispatch, its
default-false schema argument and numerical regression. Neither upstream #600
nor the v0.31 first-chunk classification fix implements this requirement.
This is a source-audit decision; new-pair runtime qualification remains required.

## Frozen inputs and instructions

| Input | Identity |
| --- | --- |
| Owner / run | `/root/k05_mixed_arithmetic` / `release-v031-20261010` |
| Assigned commit | `9a17a2c8617be2b0471db79fb301912601ee945c`, `fix(gdn): preserve cached decode arithmetic in mixed batches` |
| Verified parent | `3b5deb0773be875a328a1831c5e4a08233460d32` |
| Verified stable patch ID | `21d478c5a5c87f2f04d87acf4cb3b4a0e21e8672` |
| Kernel old base / final tip | `1c7cbeee1cd0c1481d48f5031f679a47f3f0ef45` / `193cee080d654ff67c4f8bfaf924201bc88b9cf5` |
| Kernel selected source | release/0.1.15.4, `ddf336d86e3c8602888572a3502f951abd51df12`; wheel/source provenance is owned by the parent |
| vLLM old base / final tip | `ced6857afa0ea7b2e3f0846a62e1394e90f15607` / `466b9d5abe84f13c29b38c686e4d77fc09e09aa4` |
| vLLM target | v0.31.0, `db9527a46873454610df6dbedf79a36d6bf1a7f6` |

Read `.agents/skills/vllm-xpu-update/references/commit-audit.md`, both source
repositories' `AGENTS.md`, this run's complete `inventory.json`,
`docs/fork-delta.md`, the companion `v02-gdn-scheduling.md`, and the previous
k05 audit preserved at packaging commit
`8c72895340deafef77998a579335d078800626ab`, path
`docs/updates/release-v030-20260922/audits/k05-9e2ac704b862.md`.
The published old release's actual Brutus profile was also checked in
`docs/releases/xpu-v1.11.0.md`.

## Requirement and final-old-stack contribution

For Qwen GDN on Xe2, an ordinary cached one-token decode should retain its
standalone convolution/recurrent arithmetic when an unrelated prefill shares
the batch. The focused regression uses BF16 inputs, FP32 recurrent state,
16 key heads, 48 value heads, dimensions 128, convolution width four, TP1,
a 127-token fresh prefill and separate state slots 3/1. Historical hardware
was Intel Arc Pro B70. This requirement is independent of KV-cache
quantization and does not assert whole-model exact invariance.

The patch opts eligible fused `gdn_attention` calls into two groups. It narrows
decode-first input/output/z and request metadata, rebases prefill query offsets,
and invokes the existing convolution/delta stages for decode and prefill
separately. Decode then retains the ordinary native recurrent route; prefill
retains the Xe2 chunk route. Both intermediate tensor sets stay alive until
all launches are enqueued. No third arithmetic implementation is introduced.

Native guards require both groups, no speculative decodes, no token indirection
and the necessary query/state/initial-state metadata. The companion Python
predicate additionally requires one token per decode, exact ordinary token
totals and no speculative masks. `split_mixed_non_spec=False` preserves old
callers. The split body is compiled only for Xe2.

Read-only function comparison found `gdn_attention` byte-identical between
this assigned commit and the final old kernel tip; AST comparison found its
mixed regression unchanged. Subsequent `30fda95e35666948deb3c9bd7fd72aaff9c2ad66`
changes the speculative-capacity assertion elsewhere in the interface, not
this dispatch. The later native FlashAttention tests and KVarN feature do not
supersede it. The actual eager V1 K4V2/MTP2/vision host profile needs separate
qualification: this split applies to ordinary batches with no actual
speculative metadata, even if the configured model supports speculation.

## Selected upstream coverage

All target line references below use the frozen SHAs above.

| Evidence | Result |
| --- | --- |
| Kernel `csrc/xpu/gdn_attn/gdn_attn_interface.cpp:633`, `gated_delta_rule_non_spec` | Byte-identical to the old upstream function. Lines 695–738 select `chunk_gated_delta_rule_xe2` for the entire ordinary group whenever `num_prefills > 0`; decode-only calls select `gdn::gated_delta_rule`. The arithmetic-switch mechanism remains. |
| Same file, `gdn_attention` at line 772 | Byte-identical to old upstream. Lines 864–908 chain one unsplit non-spec convolution/delta group. Separately exposed stage operators split convolution from recurrence, not decode requests from prefills. |
| Kernel `csrc/xpu/torch_bindings.cpp:246` | Fused schema still ends with `reorder_input`, without `split_mixed_non_spec`. Upstream kernels alone reject the retained Python keyword. |
| vLLM `vllm/_xpu_ops.py`, `_gdn_attention_core_xpu_impl` | AST-identical between the old upstream base and v0.31 target; no ordinary split predicate or native keyword. |
| vLLM `qwen_gdn_linear_attn.py:972`, `forward_xpu` | Still reaches `torch.ops.vllm.gdn_attention_core_xpu` at line 999. Generic CUDA/Triton mixed splitting does not replace this XPU path. |

Only four commits separate the selected kernel bases:

- `f4b729ae7fb3eb22c564606ec8b4d184c7c8bc4a` (#600) supplies actual
  ragged speculative query traversal and retained rollback capacity. Its
  interface hunk only relaxes the speculative-token capacity assertion;
  other production hunks are in the speculative convolution/recurrent
  kernels. It does not touch the two unchanged functions above. That can
  retire overlap in k06, independently of this ordinary non-spec requirement.
- `d09702a637cfea2a0c02909af82bd789d26ea319` bumps the torch toolchain;
  `885b120ffa082312b7d93c2be7a1e3b3dd79672f` updates paged-decode
  configuration; `ddf336d86e3c8602888572a3502f951abd51df12` changes
  `topk_softplus_sqrt`. None adds an equivalent GDN dispatch.

vLLM `79468c20ef23227d4e051f807e10fd54fb24e24f` (#51565) changes
`GDNAttentionMetadataBuilder.build` so a fresh one-token first chunk is a
prefill, excluding trailing zero-length padding from prefill counts. The
target implementation is at `gdn_attn.py:268–305`; cached/resumed one-token
requests still qualify as decodes. This fixes recycled-state initialization,
not the cached decode's arithmetic route when another request is a prefill.
It actually introduces a relevant small mixed case: cached decode plus fresh
one-token prefill. Keep the upstream classification and test this combination.

The target `split_decodes_and_prefills` at `backends/utils.py:799–870` still
locates a boundary in a reordered batch. Its V1 reorder helper at line 901
orders decode, short extend, long extend, then first-chunk prefill. Therefore
the new first-chunk classification preserves the decode-first native split
contract; it does not justify reclassifying fresh rows back into cached decode.
Speculative mixed batches use token indirection and are excluded by the caller.

## Minimum residual and dependencies

Retain the three production-file changes (`gdn_attn_interface.cpp`, `ops.h`,
`torch_bindings.cpp`) and the focused test in `tests/gdn_attn/test_gdn_attn.py`.
There is no demonstrated redundant production hunk to delete on this target.
Moving the grouping into Python would merely move orchestration across the
ABI and require new lifetime validation; upstream has not supplied that
replacement. Preserve the existing narrow opt-in/default-false contract.

- Hard pair: vLLM `6efe5e7cc59de64ae8d3ec67bbe77379be34d2c9` enables
  the argument. Keeping only that caller breaks the schema; keeping only the
  native patch leaves it inactive for existing callers. Removing both would
  retire an unmet arithmetic policy, outside this audit's recommendation.
- vLLM `cfbd127232d3121e185eac2d24a9a50ea8141aa0` retains live-request
  metadata trimming needed by exact native shape checks.
- Kernel `956b61f95a0b38e73eeebe9cbfeff1be782a9283` protects prefill
  convolution and `3b5deb0773be875a328a1831c5e4a08233460d32` protects
  fresh-state prefill output synchronization. Splitting the requests does
  not retire these independent requirements; preserve their reconciled fixes.
- Kernel `30fda95e35666948deb3c9bd7fd72aaff9c2ad66` is independently
  reviewed for upstream #600 overlap. Its regression's explicit false split
  keyword is an incidental schema dependency, not a reason to retain duplicate
  speculative production code or to remove this ordinary split.
- No later local repair or revert needs folding into k05. Test insertion
  context follows k04's existing tests; relocate context if needed without
  deleting the actual regression.

## Validation observed and required

Observed: immutable parent/patch-ID verification; assigned/final function and
test identity checks; old/new target function/adapter checks; selected kernel
delta and cross-repository call-path/schema/metadata inspection. Checks used
the vLLM repository virtualenv for standard-library source comparison only.
No build, pytest, GPU execution, service operation, source-worktree or ref
mutation occurred; only this report was written.

Historical logs were reread and rehashed from packaging `benchmark-results`:

- `kvarn/20260830T164500Z/direct-gdn-mixed-route/pytest.log` records
  3383 cached-decode output mismatches, maximum absolute difference 0.001953125.
  SHA-256: `94c5ad06de18e58043febec4056eb0e5909e6fb9190b19515dc62c8c3664809a`.
- `kvarn/20260830T213744Z/focused-gdn-mixed-route/pytest.log` records
  one pass. SHA-256:
  `a22c43ddc520b51f98de12bed8f134094205638acb6fb0cf00cf42baa74c4368`.
  Its `python-xpu-preflight.log` identifies Arc Pro B70 and the split schema.

These August results establish the historical failure/control, not qualification
of the v0.31 binaries. Required on the rebuilt pair:

1. Run `test_qwen38_gdn_cached_decode_is_exact_in_mixed_prefill`, comparing
   each request's output, z, convolution and recurrent state with standalone
   execution. A meaningful upstream-only comparison omits the unsupported
   keyword or calls the exposed stages; keyword rejection is not numerical
   evidence. The same patched binary with opt-in disabled is another useful
   route control.
2. Exercise the new one-token first-prefill classification beside cached
   decode, preserving dirty/reused-state masking and trailing padding. Include
   multiple decodes/prefills, ragged lengths and queued calls; verify inactive
   output/state storage remains untouched. Retain default-false, indexed and
   speculative exclusions and the paired adapter tests.
3. Qualify actual Brutus eager V1 K4V2/MTP2/vision requests, including joining
   and leaving batches, cancellation and state-slot reuse. The native exact
   arithmetic test does not establish full-model determinism.
4. If reporting performance, compare matched ordinary mixed workloads with
   identical binaries/settings apart from the split flag. Extra launches and
   the offset subtraction can cost latency; no fresh speedup or parity result
   is established by this audit.

No upstream equivalence or requirement retirement is demonstrated. Keep the
paired behavior covered while runtime qualification and release provenance
checks remain with the parent.
