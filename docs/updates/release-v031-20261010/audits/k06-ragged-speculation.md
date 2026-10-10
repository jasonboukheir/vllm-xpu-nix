# K06: shortened speculative GDN queries

Recommendation: **ADAPT to a test-only commit**. Drop all three production-file
hunks from `30fda95e35666948deb3c9bd7fd72aaff9c2ad66`; upstream #600 implements
the required behavior. Retain its focused shortened-MTP regression, which adds
controls beyond the upstream numerical-reference tests. No source worktree,
branch, service, or GPU was modified or exercised by this audit.

## Frozen inputs

- Assigned commit: `30fda95e35666948deb3c9bd7fd72aaff9c2ad66`, parent
  `9a17a2c8617be2b0471db79fb301912601ee945c`, subject
  `fix(gdn): preserve rollback for shortened speculative queries`, patch ID
  `a937990b68b5515486ac49363b7879acd876eb7e`.
- Kernels: old base `1c7cbeee1cd0c1481d48f5031f679a47f3f0ef45`, old main
  `193cee080d654ff67c4f8bfaf924201bc88b9cf5`, selected 0.1.15.4 source
  `ddf336d86e3c8602888572a3502f951abd51df12`.
- vLLM: old base `ced6857afa0ea7b2e3f0846a62e1394e90f15607`, old main
  `466b9d5abe84f13c29b38c686e4d77fc09e09aa4`, v0.31.0 target
  `db9527a46873454610df6dbedf79a36d6bf1a7f6`.
- Read both source repositories' `AGENTS.md`, the update skill's
  `references/commit-audit.md`, `docs/fork-delta.md`, and this run's
  `inventory.json`. The latter records the complete five/eight-commit pair.
- Line numbers below refer to the frozen target in the relevant repository,
  except where explicitly marked as the old fork test.

## Requirement and final old-stack behavior

Speculative verification may execute fewer queries than its retained rollback
capacity, including different lengths within one batch. Let `K` be the number
of columns in `spec_state_indices_tensor`, `L_n` the current query length, and
`A_n` the accepted count from the previous step. The required contract is:

1. Visit the compact token interval described by each request's prefix sums,
   then gather/scatter through `spec_token_indx`. Do not assume `n*K` is its
   first compact token.
2. Preserve all `K` state-index columns. `A_n` may exceed `L_n`: the initial
   SSM state is column `max(A_n-1, 0)`, independent of current query length.
3. Write only the current `L_n` SSM slots and the corresponding convolution
   history prefix, leaving capacity-only storage and output guards untouched.
4. Subsequent verification must consume the correct state at every accepted
   prefix, including after shortened execution.

The old rectangular assertion rejected such requests; rectangular traversal
would otherwise use incorrect token locations or write unused state slots.
This matters for Qwen-family native XPU GDN with MTP, including the Brutus
configuration. The retained test exercises BF16 inputs with FP32 SSM state,
both input layouts, `(k_heads, v_heads, dim)` of `(4, 8, 32)` and
`(16, 48, 128)`, one-token/two-token/ragged queries, interleaved convolution
storage, and accepted counts of 1 and 3 at capacity 3.

`git diff --exit-code 30fda95... 193cee0... -- csrc/xpu/gdn_attn tests/gdn_attn`
passed: neither later kernel commit modifies this behavior or its test.

## Upstream coverage and behavioral equivalence

The old-to-new GDN delta contains
`f4b729ae7fb3eb22c564606ec8b4d184c7c8bc4a`,
`[GDN] Fix ragged speculative token traversal (#600)`. It is an ancestor of
the frozen 0.1.15.4 source. An exact path diff from that commit to the target
is empty for `csrc/xpu/gdn_attn` and `tests/gdn_attn`. This is the selected
release-branch SHA; the previous release notes' `da16a559` was the equivalent
upstream-main reference and is not the source identity used here.

| Contract | Frozen target evidence | Why this replaces the production patch |
| --- | --- | --- |
| Compact ragged traversal | `causal_conv1d.hpp:620-625,663-669,722-725`; `gated_delta_rule.hpp:412-416` | All BA, Z, QKV and recurrence loops use `query_start_loc[n:n+2]`. Local `seq_start+t_local` and upstream `token_start..token_end` enumerate exactly the same local indices. |
| Retain full capacity | `gdn_attn_interface.cpp:89-104,119-126,183-197` | `K` still comes from state-index column count; convolution allocation must support `width-2+K` rows. The equality assertion becomes the capacity upper bound, while intermediates are allocated for actual compact tokens. |
| Accepted convolution history | `causal_conv1d.hpp:681-717` | Reads rows starting at `max(A_n-1,0)` from the full retained state allocation, independently of `L_n`. It uses explicit logical dimensions, not the leading storage stride, which may span other layers. |
| Only live convolution writes | `causal_conv1d.hpp:800-825` | Upstream writes `logical_state_len = (width-2+K)-(K-L_n) = width-2+L_n`, exactly the local patch's `state_len`. In both, `hist_rows=width-2`; the same history rows and draft inputs are copied, and the physical tail is untouched. |
| Accepted SSM slot and live writes | `gated_delta_rule.hpp:392-408,507-523` | Initial slot uses full row stride plus `max(A_n-1,0)`; destination uses the same full stride plus local token ordinal. Loop ends at `L_n`, preserving columns `L_n..K-1`. |
| Global output locations and padding | `causal_conv1d.hpp:663-669`; `gated_delta_rule.hpp:496-503`; `gdn_attn_interface.cpp:128-165,579-603` | Token remapping is unchanged, and padded tensor leading dimensions are narrowed without writing padding. The correction does not substitute compact indices for global output indices. |
| Native caller plumbing | `gdn_attn_interface.cpp:199-222,608-623`; `gated_delta_rule.hpp:740-744` | Query prefix sums reach both speculative launchers. Public op argument shapes are unchanged. |

There are small **invalid-input diagnostic differences**: the old patch also
checked total tokens `>= num_spec_decodes` and device-asserted `0<L_n<=K`;
upstream does not add those checks. They do not distinguish valid supported
vLLM requests. Neither implementation fully validates all prefix-sum values
or accepted-count bounds. The requirement is correct execution of valid live
metadata, not identical failure handling for arbitrary malformed native-op
calls. These diagnostics do not justify replaying the production patch.

## Caller compatibility and dependencies

vLLM target `vllm/v1/attention/backends/gdn_attn.py:356-394` creates compact
speculative prefix sums and retains `self.num_spec+1` state columns.
`:475-520` can stage request-padded state indices, prefix sums and acceptance
counts, with zero-length padded requests. The native interface still requires
exact live metadata row counts (`gdn_attn_interface.cpp:65-70,89-95,113-117`).
The target `vllm/_xpu_ops.py:178-222` forwards those tensors directly.

Therefore kernel #600 **does not obsolete v01**
`cfbd127232d3121e185eac2d24a9a50ea8141aa0`: preserve its trimming to live rows,
while keeping every rollback column. Do not trim columns to current query
length or manufacture padded zero-length requests to satisfy row checks.

The assigned patch follows kernel mixed-batch patch
`9a17a2c8617be2b0471db79fb301912601ee945c`, but its production speculative
behavior is independent of that patch. Its old test explicitly passes
`split_mixed_non_spec=False`. Remove that keyword when retaining the test:
it is irrelevant to pure speculative execution, upstream lacks the argument
(`csrc/xpu/torch_bindings.cpp:247-258`), and the fork's mixed-batch schema
defaults it to `False`. This lets the test qualify both upstream alone and
the final fork without creating a test dependency on k05.

No operator/schema deletion or companion production change is needed for
dropping the assigned production hunks. KVarN's state ownership must continue
to supply distinct rollback slots, as already required by the native API.

## Minimum residual and upstream test comparison

Retain only `test_gdn_shortened_mtp_preserves_rollback` added to
`tests/gdn_attn/test_gdn_attn_padded.py` (old fork lines 29-189), omitting the
unnecessary mixed-batch keyword. Do not copy its surrounding pre-existing
padded-leading-dimension test as additional delta.

Upstream `tests/gdn_attn/test_gdn_attn.py:1174-1216` supplies four ragged cases
and a two-step state-reuse case. Its independent PyTorch reference understands
ragged history (`:737-815`), and it checks untouched unused SSM slots and
second-step convolution tails (`:1000-1012,1126-1147`). State reuse includes
accepted counts greater than current lengths (`:1019-1023`). These are useful
upstream tests and should be run, not duplicated.

The retained local test still adds meaningful controls:

- Interleaved three-layer convolution storage with exact adjacent-layer
  guards; upstream allocates ordinary contiguous per-layer storage.
- Sparse global token placement and sentinel output/padding rows; upstream
  ragged tests permute a completely occupied compact output buffer.
- Bitwise equality to full-width native prefixes and subsequent continuation
  for every possible accepted prefix, for both reordered and interleaved
  inputs; upstream ragged numerical tests use only `reorder_input=True` and
  tolerance `5e-2` against a separate mathematical reference.
- Capacity-3, current-length-1 with previous acceptance 3; the upstream
  reuse case covers related acceptance/length mismatches but not this maximal
  shortening combination or these two head configurations.

Thus the smallest safe source delta is zero production lines plus this one
focused test. It may be folded with the other surviving native regression
tests into a coherent test commit, preserving the inventory mapping and
original authorship/sign-off.

## Validation observed and still required

Observed in this audit:

- Ancestor check for #600 to target: pass.
- Exact GDN code/test diff #600 to target: empty.
- Exact GDN code/test diff assigned commit to old fork main: empty.
- A host-only index-trace enumeration compared the local and upstream query
  traversal and convolution-history write formulas for `K=1..8`, `width=2..8`,
  `L=1..K`, `A=0..K`, and compact start offsets `0..31`: **53,760 cases
  matched**, including **18,816 with A>L**. It also checked the accepted
  initial convolution window remained within full `width-2+K` capacity.
  This checks indexing algebra; it is not native execution or a numerical
  correctness result.
- Source comparison confirms unchanged arithmetic within each visited token
  and identical SSM initial/destination slot indexing. No GPU tests were run
  under this read-only audit assignment.

Required parent qualification against the actual candidate native library:

```bash
.venv/bin/python -m pytest tests/gdn_attn/test_gdn_attn.py -k 'mtp_ragged' -v
.venv/bin/python -m pytest tests/gdn_attn/test_gdn_attn_padded.py -k 'shortened_mtp' -v
```

Expected focused collection: 5 upstream cases and 24 retained local cases.
Use the parent workflow's equivalent Nix/dev-shell Python if that is the
candidate environment; ensure the imported native module is built from the
candidate. Removing `split_mixed_non_spec=False` also permits the 24-case
local test to run directly on selected upstream kernels, demonstrating the
production patch is unnecessary. The former rectangular base should fail the
shortened cases at its size assertion; source proves that distinction without
rebuilding or loading the historical stack during this audit.

Also retain/run v01's metadata-boundary regression before full Brutus MTP
server qualification. The source-based disposition is strong, but an actual
candidate native test pass remains a release gate, not an observed audit result.
