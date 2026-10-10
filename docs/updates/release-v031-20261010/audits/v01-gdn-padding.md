# v01 — retain GDN request-index normalization

**Disposition: KEEP `cfbd127232d3121e185eac2d24a9a50ea8141aa0`.** The selected
vLLM and kernel releases still disagree about padded request-index extents.
Retain its five slices and existing adapter regression test unchanged. The
new upstream ragged-speculation repair handles token traversal, not this
request-padding contract. No additional source change is proposed.

Auditor: `/root/v01_gdn_padding`, 2026-10-10. Read the commit-audit runbook and
`../vllm/AGENTS.md`. Source, Git refs, locks, services and GPU state were not
modified; this report is the only written artifact.

## Frozen inputs and inventory

| Input | Identity |
| --- | --- |
| Assigned commit | `cfbd127232d3121e185eac2d24a9a50ea8141aa0` |
| Parent / vLLM old upstream base | `ced6857afa0ea7b2e3f0846a62e1394e90f15607`, v0.30.0 |
| Subject | `fix(xpu): strip graph padding from GDN index tensors` |
| Recomputed stable patch ID | `73c55934b009279f414f33eb37ce51c9ad43b846` |
| Complete old vLLM fork tip | `466b9d5abe84f13c29b38c686e4d77fc09e09aa4` |
| Selected vLLM target | v0.31.0, `db9527a46873454610df6dbedf79a36d6bf1a7f6` |
| Old kernels base | v0.1.15, `1c7cbeee1cd0c1481d48f5031f679a47f3f0ef45` |
| Complete old kernels tip | `193cee080d654ff67c4f8bfaf924201bc88b9cf5` |
| Selected kernels source | release/0.1.15.4, `ddf336d86e3c8602888572a3502f951abd51df12` |

The parent supplied the frozen kernels target after resolving the 0.1.15.4
source branch; wheel-to-source provenance verification belongs to the parent.
This audit directly inspected that SHA, not a moving branch.

Verified both complete old-base-to-old-tip histories with
`git log --reverse --format='%H %P %s' <old-base>..<old-tip>` against the full
inventory and rationale in `docs/fork-delta.md`. vLLM has exactly five commits,
in order: `cfbd127232`, `6efe5e7cc5`, `d909b25f01`, `984e6c3edd`, `466b9d5abe`.
Kernels have exactly eight: `98bdbfd`, `e1bec3b`, `956b61f`, `3b5deb0`,
`9a17a2c`, `30fda95`, `8b6ec9d`, `193cee0`. The existing published v0.30 audit
was consulted at packaging commit
`8c72895340deafef77998a579335d078800626ab`, path
`docs/updates/release-v030-20260922/audits/v02-cf0b834cd66c.md`, and rechecked
against the new frozen targets.

## Requirement and final old behavior

The native XPU GDN operator accepts exact live request counts. Its metadata
producer can expose suffix-padded request buffers, particularly for graph
capture/replay. Request count, token count and speculative rollback capacity
must remain distinct. The adapter in `vllm/_xpu_ops.py`, symbol
`_gdn_attention_core_xpu_impl`, passes these extents:

| Tensor | Required extent |
| --- | --- |
| `non_spec_query_start_loc` | `num_prefills + num_decodes + 1` |
| `non_spec_state_indices_tensor` | `num_prefills + num_decodes` |
| `spec_query_start_loc` | `num_spec_decodes + 1` |
| `spec_state_indices_tensor` | `num_spec_decodes` rows, retaining every rollback column |
| `num_accepted_tokens` | `num_spec_decodes` |

The patch adds five ordinary slicing operations with `None` guards. They keep
backing storage, values, token-index arrays, `num_actual_tokens` and the four
projected/output tensors intact. The existing non-spec `.contiguous()` operation
predates this patch. Two live speculative requests plus two padded rows can
otherwise deliver state/query/accepted extents `4/5/4` to native predicates
requiring `2/3/2`.

The old final tip retains all five slices and their 92-line parametrized test
unchanged. Later commit `6efe5e7cc59de64ae8d3ec67bbe77379be34d2c9` adds the
independent mixed-batch predicate and native keyword immediately after them;
none of the remaining local commits supersedes this normalization.

The current Brutus recipe was read at
`~/.config/nix/hosts/brutus/services/vllm-xpu/models/RedHatAI__Qwen3.8-27B-INT4.nix`:
BF16 computation, compressed-tensors INT4 weights, K4V2 target/draft caches,
MTP with two speculative tokens, eager V1, four scheduler slots and 2048-token
prefill budget. The published hardware is Intel Arc Pro B70. GDN recurrent
state is still used with K4V2 attention. Eager exact-size metadata does not
demonstrate the need for trimming on every request, but neither does it retire
the retained boundary behavior. This is a compatibility/correctness safeguard,
not a claimed throughput optimization.

## Upstream coverage and alternate implementations

At vLLM `db9527a46873454610df6dbedf79a36d6bf1a7f6`:

- `vllm/_xpu_ops.py::_gdn_attention_core_xpu_impl` still forwards all five
  request-index tensors unchanged. The old-to-new file changes add rotary
  embedding support and docstring cleanup, without changing this adapter.
- `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` still selects
  `forward_xpu` for XPU at line 406; `forward_xpu` calls
  `torch.ops.vllm.gdn_attention_core_xpu` at line 999. Other CUDA/ROCm FLA and
  fused-normalization paths do not replace this native XPU call.
- `79468c20ef23227d4e051f807e10fd54fb24e24f` (#51565) repairs stateless
  first-chunk classification. `GDNAttentionMetadataBuilder.build` now excludes
  trailing zero-length requests from its prefill counts, while still assigning
  `non_spec_state_indices_tensor = block_table_tensor[:, 0]` and
  `non_spec_query_start_loc = query_start_loc`. Reducing the live counts is not
  an equivalent tensor-extent normalization. Its new padding test checks
  counts, not native-boundary sizes.
- `dd3bc9c72faf410043b453e18e58553a1ad4d388` (#58762) reuses GDN metadata
  between MRV2 cache groups. Its `supports_update_block_table` explicitly
  requires `device.type == "cuda"`. Even there, graph state buffers are sized
  from existing metadata, and the common staging paths still use
  `batch_size = m.num_reqs`, expand query/state/acceptance arrays to that
  count, and fill inactive suffix rows. It does not provide XPU normalization.
- `1417022c3dbb633f9d0c2ac1db2171445bb91b08` (#58045) adds
  `uniform_spec_sequence_length` and CPU sequence masks for low-concurrency
  KDA work. Neither changes the native XPU adapter's extents.
- The earlier request-versus-token fix,
  `cec2ec11760f9f3beabd4c90451936078bf91533`, was already in the old baseline.
  Choosing `m.num_reqs` rather than token count does not prove it equals each
  live subgroup count when suffix request padding is present.

At kernels `ddf336d86e3c8602888572a3502f951abd51df12`,
`csrc/xpu/gdn_attn/gdn_attn_interface.cpp` preserves exact request checks:
`causal_conv1d_spec` requires query length `num_spec_decodes + 1` at line 69,
state rows `num_spec_decodes` at line 93, and accepted-token length
`num_spec_decodes` at line 116. `causal_conv1d_non_spec` still requires query
and state extents `num_prefills + num_decodes + 1` and
`num_prefills + num_decodes`. The public `gdn_attention` forwards these tensors
to these split operators without trimming them.

Only one old-base-to-target change touches that interface:
`f4b729ae7fb3eb22c564606ec8b4d184c7c8bc4a` (#600) relaxes speculative
**token** cardinality from equality with rectangular rollback capacity to
`<=` and repairs traversal in companion headers. It leaves every request-index
check above unchanged. This fixes a different dimension of the contract and
does not make request-row trimming obsolete. Remaining target commits are
`d09702a637cfea2a0c02909af82bd789d26ea319` (Torch 2.14),
`885b120ffa082312b7d93c2be7a1e3b3dd79672f` (paged-decode config) and
`ddf336d86e3c8602888572a3502f951abd51df12` (topk interface); none replaces GDN
request normalization.

## Minimum residual and dependencies

Retain the existing 14-line adapter addition (11 executable lines) and its
regression test. No new native schema, ABI, operator or cache format is needed.
This is independently useful and has no required companion fork patch.

`6efe5e7cc59de64ae8d3ec67bbe77379be34d2c9` and kernels
`9a17a2c8617be2b0471db79fb301912601ee945c` own the additional mixed-batch
keyword; do not confuse that paired interface with this patch's slicing.
Kernel `30fda95e35666948deb3c9bd7fd72aaff9c2ad66` versus upstream #600 is a
separate audit. Its likely upstream coverage does not justify dropping this
patch. Retain every speculative column because columns encode rollback
capacity even when actual query length is shorter.

No repair commit needs folding into this change. Preserve original authorship
and trailers on replay. A broader producer redesign could remove the adapter
slices only after proving all XPU call paths satisfy these native extents;
the selected release does not do that.

## Validation performed and remaining gates

Observed source checks: full histories and final-old-stack diff inspected;
stable patch ID recomputed with `git show --format= <sha> | git patch-id
--stable`; old/new producer, adapter, dispatch and native contracts inspected;
all four new kernel commits reviewed for relevance. Native interface diff
shows only the speculative token-capacity relaxation described above.

Ran a CPU-only sensitivity check using the existing `.venv/bin/python -B`
with `PYTHONDONTWRITEBYTECODE=1` and the published qualification environment's
site-packages on `PYTHONPATH`:
`/nix/store/zhv84h5afvyx99fkw0l56rpj8rldhivl-v030-brutus-qualification-env/lib/python3.12/site-packages`.
It imports Torch `2.13.0+xpu` but creates CPU tensors only. The script AST-extracts
the unchanged five-case
`test_xpu_gdn_adapter_trims_only_request_padding` from old tip
`466b9d5abe`, the new target's actual `GDNAttentionMetadata` dataclass, and
each ref's real adapter function; only module plumbing, forward context and
the native dispatch are stubbed. It asserts exact extents, all rollback
columns, data-tensor identity, storage sharing and unchanged source values.

```text
upstream-v031 ['FAIL', 'FAIL', 'FAIL', 'FAIL', 'PASS']
old-final-fork ['PASS', 'PASS', 'PASS', 'PASS', 'PASS']
PASS: upstream sensitivity and retained boundary-view contract; CPU only, native stubbed
```

The four present-metadata cases distinguish upstream from the retained fix;
both accept all-`None` metadata. An initial harness attempt used
`SimpleNamespace` instead of the dataclass and stopped for a missing default
field; replacing it with the real extracted dataclass corrected the harness.
The bare vLLM `.venv` lacks Torch, so its initial import probe failed; the
installed historical qualification environment supplied CPU dependencies.
These observations are not a full pytest run, new package import test, native
execution, graph test, GPU qualification or performance result.

Parent-owned checks still required:

1. Run the existing metadata/adapter test in the rebuilt release environment.
   Keep the upstream sensitivity control if changing the slices or producer.
2. During native qualification compare suffix-padded metadata with an unpadded
   control, including speculative rollback columns, shortened queries,
   inactive-state sentinels and mixed batches. Test actual capture/replay if
   the chosen serving configuration enables graphs; retained slices alone do
   not qualify graph execution or MRV2.
3. Qualify the actual Brutus model, health endpoint, chat/vision/tool requests,
   active MTP and state reuse on the selected final pair. No speedup claim is
   associated with this view-only normalization.

Padding must remain a suffix: this patch does not filter interior zero-length
requests or correct invalid scalar counts. No contrary upstream coverage was
found. Record the final replay SHA in the parent map.
