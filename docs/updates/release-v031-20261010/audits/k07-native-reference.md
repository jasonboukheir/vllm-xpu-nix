# k07 — native FlashAttention replay/reference coverage

**Disposition: KEEP unchanged.** This is a small, independent test requirement;
upstream reference tests do not replace its native-only, bitwise replay checks.
Removing the commit changes no serving behavior, but removes regression coverage
without an equivalent replacement. No source, branch, service or GPU changes were
made for this audit.

## Frozen assignment

- Repository: `/home/jasonbk/Projects/vllm-xpu-kernels`; read its `AGENTS.md`
  and packaging `.agents/skills/vllm-xpu-update/references/commit-audit.md`.
- Commit: `8b6ec9d6a31836e3f19447ec790169655ce7da6a`,
  `test(attn): cover native FlashAttention replay determinism`.
- Parent: `30fda95e35666948deb3c9bd7fd72aaff9c2ad66`;
  patch ID: `43a970403358d0f0303a73a99af6367d7a0e5012`.
- Kernel old base: `1c7cbeee1cd0c1481d48f5031f679a47f3f0ef45`;
  full old tip: `193cee080d654ff67c4f8bfaf924201bc88b9cf5`.
- Kernel target: released wheel `0.1.15.4`, frozen source
  `ddf336d86e3c8602888572a3502f951abd51df12` on `release/0.1.15.4`.
  There is no source release tag or embedded wheel source SHA; provenance is
  recorded by the parent in `../inventory.json`.
- Companion vLLM: old base `ced6857afa0ea7b2e3f0846a62e1394e90f15607`,
  old tip `466b9d5abe84f13c29b38c686e4d77fc09e09aa4`, target v0.31.0
  `db9527a46873454610df6dbedf79a36d6bf1a7f6`.
- Complete local inventory and retention rationale: `../inventory.json` and
  `../../../fork-delta.md`.

## Requirement and effective old behavior

The commit adds only the 223-line
`tests/flash_attn/test_flash_attn_determinism.py`. Its three test functions expand
to five native BF16 cases: dense causal prefill at lengths 104, 107 and 162,
paged causal prefill at 162, and paged decode at query length 1 / KV length 162.
All use 24 query heads, four KV heads, head dimension 256 and page size 64 where
applicable, matching the recorded model-shape probes. Each baseline is checked
against an independent CPU FP32 attention reference and for finite output. Eight
subsequent replays alternate unrelated attention data and allocator pressure,
then compare BF16 bit patterns exactly. An autouse fixture replaces
`flash_attn_interface._fallback_varlen_attn` with a failure, preventing a missing
native build policy from silently passing through reference attention.

This covers repeated native execution after unrelated work, not invariance across
different scheduling/batch shapes or whole-model outputs. The final local kernel
feature commit `193cee080d654ff67c4f8bfaf924201bc88b9cf5` leaves this file unchanged.
There is no intervening revert or production modification owned by this commit.

## Upstream coverage and call paths

The selected old-to-new kernel range contains exactly four commits:

| Upstream SHA | Relevance |
| --- | --- |
| `f4b729ae7fb3eb22c564606ec8b4d184c7c8bc4a` | Ragged speculative GDN traversal; no FlashAttention test replacement. |
| `d09702a637cfea2a0c02909af82bd789d26ea319` | Torch 2.14 / toolchain update; warrants fresh native execution, not deletion of coverage. |
| `885b120ffa082312b7d93c2be7a1e3b3dd79672f` | Adds `16,128,64,false,true,false` to `csrc/xpu/attn/kernel_configs/paged_decode_default.conf`; unrelated to this test's 24/4 heads and dimension 256. |
| `ddf336d86e3c8602888572a3502f951abd51df12` | Changes `topk_softplus_sqrt` interface; no FlashAttention test replacement. |

At target, `tests/flash_attn/test_flash_attn_varlen_func.py`,
`vllm_xpu_kernels/flash_attn_interface.py` and
`csrc/flash_attn/flash_api.cpp` are each byte-identical to the old upstream base.
The entire upstream `tests/flash_attn`, `csrc/flash_attn`, `csrc/xpu/attn` and
wrapper delta is just the configuration addition described above.

The target's `test_flash_attn_varlen_func.py:30` retains the independent
`ref_paged_attn` helper, and e.g. lines 372–400 compare a single output to a
tolerance reference and release cached allocations. Such tests have broad
numerical coverage but no repeated bitwise comparison with interleaved decoys
and allocator churn. A target-wide test search found no equivalent attention
replay/bitwise assertion or native-fallback-rejection test. Existing nonattention
determinism tests do not replace this requirement.

At `flash_attn_interface.py:423`, `flash_attn_varlen_func` still accepts all
arguments used by the local tests. The dense and 162-token paged-prefill calls
and the one-token paged decode reach `torch.ops._vllm_fa2_C.varlen_fwd`; the
small-query speculative dispatch condition (`1 < max_seqlen_q <= ...`) does not
intercept these cases. Missing-policy errors can still invoke the fallback at
line 640, so the local fixture remains useful and correctly targeted.

**Minimum retained delta:** retain this file, including all five frozen shapes,
independent FP32 checks, fallback rejection and exact replay checks. No new
production code is needed and no adaptation is justified by this release delta.
Upstreaming the coverage would remove fork maintenance without losing it.

## Dependencies

This test imports upstream's reference helper and public wrapper; it adds no
operator schema, ABI, kernel or vLLM integration. It has no dependency on the
KVarN feature despite its historical first-chunk motivation. No local repairs
need folding into it, and keeping/dropping another audited runtime patch does
not change its independent requirement. Build policies must enable its existing
BF16 attention shapes; the fallback guard intentionally fails otherwise.

## Validation and qualification plan

Observed, read-only static checks used the repository's `.venv/bin/python` and
frozen `git show` contents, without importing Torch or starting GPU work:

- AST parsing of the complete test file: PASS.
- Three test functions / five parameterized cases confirmed.
- File equality between assigned commit and final old tip: PASS.
- Exact old/target equality of wrapper, shared reference test and native binding
  files: PASS.
- AST check that every wrapper-call keyword remains accepted: PASS.
- Target fallback fixture symbol and independent reference helper exist: PASS.

The prior release notes (`docs/releases/xpu-v1.11.0.md`) record 705 installed
native/backend cases passing, with an explicit caveat that some cases are CPU
contracts. This aggregate does not independently establish this file's five
results, nor does it qualify the new toolchain. No new GPU result is claimed.

After the parent builds and activates the candidate test environment, run its
Python against this retained file from the candidate kernel checkout:

```text
<candidate-python> -m pytest tests/flash_attn/test_flash_attn_determinism.py -v
```

Require all five cases to execute and pass on Brutus with the installed candidate
native extension; retain fallback rejection and both finite/reference and exact
replay assertions. Record imported module paths, Torch and native build versions
so a stale environment cannot count as qualification. The target metadata moves
to `torch==2.14.0+xpu`; commit `d09702a637cfea2a0c02909af82bd789d26ea319` also
updates oneAPI CI setup to 2026.1 and C++20-related build handling. These changes
can affect arithmetic despite unchanged attention source, making a fresh run
necessary. Failed replay should be investigated as a candidate regression;
passing tests confirms coverage remains healthy but does not justify deleting
the tests. No performance comparison is required for a test-only change.
