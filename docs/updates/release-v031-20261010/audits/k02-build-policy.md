# K02: disabled chunk-prefill policy expansion

**Disposition: DROP, gated on successful host-profile and narrow FA2 builds.**
Retire the optional, unmeasured early dispatcher-pruning optimization under the
requested maintenance-delta reduction. This is not an upstream-merged fix.
Keep upstream's existing configuration pruning and disabled-kernel guards.
If the narrow dispatcher fails to compile or exceeds available build resources,
reconsider this decision before publishing; do not silently remove that supported
configuration.

## Frozen assignment

- Kernels commit: `e1bec3b3e9d0c02cc0e8ce9354d5d2908d51a87f`, parent
  `98bdbfde3e75f164bc5040574990db843bf888de`, subject
  `fix(attn): avoid instantiating disabled chunk-prefill policies`, patch ID
  `a67e39bf2d719c6ffb4a8a0389b3e9d3bfbb6a76`.
- Kernels old base: `1c7cbeee1cd0c1481d48f5031f679a47f3f0ef45`; old final
  tip: `193cee080d654ff67c4f8bfaf924201bc88b9cf5`; target:
  `ddf336d86e3c8602888572a3502f951abd51df12`, official release/0.1.15.4
  source recorded in the run inventory (published wheel release, no source tag).
- vLLM old base: `ced6857afa0ea7b2e3f0846a62e1394e90f15607`; old tip:
  `466b9d5abe84f13c29b38c686e4d77fc09e09aa4`; target v0.31.0:
  `db9527a46873454610df6dbedf79a36d6bf1a7f6`.
- Read the source `AGENTS.md`, update runbook `references/commit-audit.md`,
  `docs/fork-delta.md`, and complete paired inventory in `../inventory.json`.
  This audit changed only this report.

## Requirement and effective contribution

The patch wraps twelve Xe2 head-size policy calls in
`dispatch_enabled_chunk_policy`. Its outer `if constexpr` bypasses the recursive
five-boolean dispatcher for completely disabled policies. It also extracts the
existing diagnostic into `report_chunk_policy_not_compiled`. The change is
71 insertions and 39 deletions across two files; it does not change generated
kernel tuples, supported attention math, operators, or cache formats.

The commit message mentions compiling narrowed configurations without
unsupported branches. Inspection distinguishes the dispatcher expansion from
actual kernel instantiation: upstream already prevents disabled kernels from
reaching `policy_dispatch_impl` at each dispatch leaf. The remaining contribution
is avoiding the recursive host dispatcher expansion itself. The existing release
record supplies no controlled compile-time/RSS comparison and no current
compiler-error reproducer proving this wrapper necessary.

Brutus's `~/.config/nix/hosts/brutus/services/vllm-xpu/package.nix` selects
`chunk_prefill_default` plus extra tuples. The frozen target's preset includes
head sizes **64, 96, 128, 192, 256, 512**. The generator enables both standard
and b16 policies per included head size: all twelve policies are enabled, so
the patch prunes **zero** policies in this host build. Extra tuples do not
change that conclusion.

The packaging `kvarnValidationKernelConfig` in `flake.nix` is narrower: heads
96 and 256 enable four policies and disable eight. It is the important build
regression gate for dropping this patch. Its selected kernels and runtime
capabilities remain supported by upstream's existing leaf guards; increased
dispatcher compilation cost remains possible and is not measured here.

## Upstream coverage and alternative implementation

Only four commits exist in the frozen kernel upstream interval:

| Commit | Change relevant to this decision |
| --- | --- |
| `f4b729a` (#600) | Ragged speculative GDN traversal; unrelated. |
| `d09702a637cfea2a0c02909af82bd789d26ea319` (#525) | Torch 2.14; Docker toolkit image 2026.1.2; Xe3 C++20 adjustment. No Xe2 dispatcher rewrite. |
| `885b120` (#602) | Paged-decode preset changes; no chunk-prefill change. |
| `ddf336d86e3c8602888572a3502f951abd51df12` (#603) | topk_softplus_sqrt interface; unrelated. |

`git diff OLD_BASE TARGET -- csrc/xpu/attn/xe_2` is empty. Exact identical
old/new blob IDs establish no refactor was missed for the affected paths:

| Path under `csrc/xpu/attn/` | Identical blob |
| --- | --- |
| `xe_2/chunk_prefill_utils.hpp` | `f2f182e9ddcc3c7598b0e675df853931b6dc7144` |
| `xe_2/fmha_xe2.cpp` | `ca3466a4b9ed59b5b527da143a7dd5551e397bf9` |
| `xe_2/chunk_prefill_configure.cmake` | `19c330a8137b8570efe3de69d7e51888b644a2ca` |
| `kernel_configs/chunk_prefill_default.conf` | `603f6d39b2889ea455a9070e99ee30f2454845fd` |

Exact target source evidence:

- `xe_2/fmha_xe2.cpp:266-356` selects standard/b16 policies and calls
  `policy_dispatch_func` directly for twelve head-size alternatives.
- `xe_2/chunk_prefill_utils.hpp:42-130` uses `if constexpr` to reject invalid
  LSE combinations, disabled policies, and disabled tuples. Only the final
  enabled branch calls `policy_dispatch_impl`. Lines 133-146 recursively
  specialize the five runtime booleans before reaching these guards.
- `xe_2/chunk_prefill_configure.cmake:140-230` selects requested tuples;
  lines 191-200 enable both policy variants; lines 236-261 emit one selected
  tuple per source file; lines 289-321 generate policy/tuple traits.
- `xe_2/chunk_prefill_enabled_policies.hpp.in:16-28` defaults both traits to
  false and adds only generated enabled specializations.
- `xe_2/chunk_prefill.hpp:380-489` contains the expensive dtype-dependent
  `FMHAConfig` kernel path, reached only by enabled dispatcher leaves.

The smaller implementation is therefore the unmodified upstream dispatcher,
with its already-existing generated configuration selection. The new compiler
is not evidence that early pruning was fixed, became faster, or ceased to matter.
No such claim is needed for retiring an optimization with zero normal-host
pruning and no measured benefit for the remaining narrower build.

## Dependencies and minimum residual delta

There are no cross-repo API/ABI dependencies or required companion removals.
`git log e1bec3b..193cee0 --` the two touched paths is empty, and a final-old-tip
symbol search finds the helper and all its callers only in those same files.
Later KVarN/GDN/test commits do not change their behavior. Skip this commit
entirely during replay; keep zero residual code. Do not fold it into KVarN.

Update `docs/build.md`'s paragraph claiming the source fix prevents dispatcher
memory consumption of “tens of GiB.” That claim is stronger than the retained
release evidence. Suggested replacement after the build gate passes:

> Partial chunk-prefill/paged-decode configs bound the generated kernel buildout.
> Upstream's dispatcher checks generated policy and tuple availability before
> instantiating kernels. The former local early dispatcher-pruning optimization
> was retired for v0.31; it pruned no policies in the normal Brutus preset.

Also update the current fork-delta table and release map to distinguish this
scope/maintenance decision from an upstream-equivalent fix.

## Validation: observed versus required

**Observed, read-only:** complete commit/diff inspection; empty upstream Xe2
delta; four matching old/new blob pairs above; no later-local modifications to
the affected paths; helper dependency search; preset head-size enumeration;
inspection of host and narrow packaging selectors. No GPU, service, source-tree,
or branch changes occurred. No new native compilation or performance comparison
was performed by this audit.

**Parent release gates:**

1. Build the replayed Xe2 FA2 library with the actual Brutus preset/extras and
   new Torch 2.14/oneAPI 2026.1 substrate. This verifies the published host
   configuration and current compiler, including dispatcher compilation/linking.
2. Build the existing narrow KVarN validation package (heads 96 and 256). Check
   `fmha_xe2.cpp` completes without unsupported-template/link errors or resource
   exhaustion. Preserve this configuration; if it regresses, restore/adapt the
   minimal guard before releasing. No speed/RSS improvement is claimed for DROP.
3. Include head-96 noncausal vision prefill and head-256 causal/paged prefill in
   the parent's native/real-model qualification, followed by ordinary health
   and completion checks. These cover selected runtime paths; they do not
   measure compilation benefits or exhaust every optional attention tuple.

A successful build is the decisive missing evidence for viability of the
narrow configuration. There is no need to run production merely to obtain an
optimization baseline, and no serving performance benefit is attributed to
either retaining or removing this patch.
