# Patch 0005: KEEP unchanged, with build-resource validation pending

Audit date: 2026-09-22. Assigned scope: only
`nix/patches/0005-reduce-kernel-build-memory.patch`. This is a read-only source
and packaging audit; the only authored artifact is this report. No source
checkout, commit, build, service, external issue, or other report was changed.
The project update skill, its `references/commit-audit.md`, the applicable
build/qualification procedure, and the frozen kernel `AGENTS.md` were read.

**Disposition: KEEP the existing two-cache-variable implementation.** The
target release still hardcodes the exact values that packaging needs to
override. Upstream's smaller generated translation units and job estimation
do not replace configurable device-link concurrency. Keep the diagnostic
backtrace knob pending a controlled comparison; the historical assertion
that it materially lowers successful-compilation RSS is not established by
the evidence inspected here. Textual compatibility is demonstrated, but the
final replayed source and exact package remain unqualified.

This concerns the ordinary v0.30 release preserving eager V1, K4V2 target and
draft caches, bundled MTP, and vision. Paused issue17/DFlash work supplies no
authorization, acceptance gate, or performance claim for this audit.

## Frozen inputs

| Input | Exact identity |
| --- | --- |
| Packaging baseline | `4ac4375aa3d0f649fb4b6b43d9447f413a8b1cc7` |
| Kernels previous upstream base | `e7f20cbc18d741419a84aa57178e1f1cc0b453da` |
| Kernels released tip | `2c39287deec5e59cc2656e6d4a6dce51a567c411` |
| Selected kernels release | `v0.1.15`, `1c7cbeee1cd0c1481d48f5031f679a47f3f0ef45` |
| Kernel object database | `build-dev/release-v030-20260922/vllm-xpu-kernels` |
| Patch 0005 SHA-256 | `79701051bd0b04bf09074ecce35e80a81a74b4e4b257de06eae3a630e972eb98` |
| Patch 0005 Git blob at packaging baseline | `4a6b21753172c523776479974e1bdc2bde991b3f` |
| Patch 0001 SHA-256 | `0d7b31cef2a1855a67910023d6c35e0a91345376dfd2e82a4f663d55d7809ead` |
| `nix/vllm-xpu-lib.nix` SHA-256 | `4e0975eaa17604e94e0cc61cf8aa4bbb5f2ad648f88268e886f946af2b9d9063` |
| `nix/vllm-xpu-kernels.nix` SHA-256 | `a04700b23e39b9531c97b2ea9c6d25b856801ee8299e861330bed2451b303647` |
| `nix/mk-kernels.nix` SHA-256 | `bf6b026722e66168b9ad7dd66ae12b86b736f9da9b1ae9067c86d962759c286d` |
| Parent preflight JSON SHA-256 | `322d634a5e36f8fc2494ca7d902c0d90c6ece94e96dd58f323e95fa6b9674d09` |

Kernel source was read with `git show <immutable-sha>:<path>`, `git grep
<immutable-sha>`, `git ls-tree <immutable-sha>`, and explicit two-SHA diffs.
No finding depends on the moving source worktree HEAD while the parent
replays local commits.

## Requirement and effective behavior

The original requirement is a configurable resource budget for building
SYCL/CUTLASS native libraries, especially on the historical 24-core,
approximately 96-GiB builder. It is a build-host RAM requirement, not a GPU
KV-cache optimization. Current relevant deployment qualification is Brutus's
oneAPI 2026.0, Torch 2.13.0+xpu, BMG AOT package and selected attention configs;
the generic factory also supports JIT through empty AOT device lists.

The effective patch adds exactly two CMake `CACHE STRING` settings:

| Setting | Default | Consumer | Current split-factory override |
| --- | --- | --- | --- |
| `VLLM_XPU_SYCL_LINK_PARALLELISM` | `16` | `-fsycl-max-parallel-link-jobs=...` in `SYCL_DEVICE_LINK_FLAGS` | configure-time `NIX_BUILD_CORES` |
| `VLLM_XPU_CUTLASS_TEMPLATE_BACKTRACE_LIMIT` | `0` | `-ftemplate-backtrace-limit=...` in `VLLM_CUTLASS_FLAGS` | `10` |

Without overrides these reproduce the target's literal flag values. The
patch preserves `-flink-huge-device-code`, AOT targets, backend GRF choices,
optimization level, kernel selection, and all source/operator interfaces.
It does not set an absolute memory limit, alter device-code splitting, or
guarantee that any particular compiler process fits memory.

The implementation and its history must be distinguished:

- Packaging commit `e08d07a9de6fdecfb731b69e08e7534e1abaff20` introduced
  the memory patch on 2026-05-11, initially also adding
  `-fsycl-device-code-split=per_source` and `-fno-pretty-templates`, with a
  link default of four and hardcoded diagnostic limit ten.
- `6877ebadfaab8dfd7e2ea915e8cdb81aba620f38` removed the pretty-template
  flag; `9fbe0d3d08b3171d831507564d504b7364943d29` removed `per_source`.
  Neither retired workaround is in the present patch and neither should be
  restored in this release.
- `6d4e1b4814e164ef66f59d46e406eece490ec764` made the two settings
  configurable with upstream-compatible defaults. Its memory commentary
  also involved PCH, translation-unit splitting, and scheduling changes;
  it cannot isolate the effect of patch 0005.
- `ba101ce978f5fe83ef016012d03feb4527606499` fixed downstream forwarding
  into `cmakeFlagsArray`, avoiding a literal `$NIX_BUILD_CORES` in the flag.
- `1ea86f9d7350df7f165ab5da3a54aa58c5e18039` rebased patch context after
  the upstream `getMemoryInfo` link additions; the retained patch is its
  result. No further semantic adaptation is needed for the selected tag.
- Subsequent factory scheduling chains six native-library derivations and
  caps attention compilation at `min(NIX_BUILD_CORES, 12)`. The lower-limit
  fix is in `93016d4dd4c3d4cc1bea65ec5b583a227f2bea34`.

The current factory captures device-link jobs in `preConfigure`, before
`preBuild` lowers attention frontend jobs. Thus a caller budget of four gives
four link jobs and at most four attention compiler processes; a budget of
24 can give 24 link jobs while attention compilation is capped at twelve.
Patch 0005 is a control mechanism, not an unconditional reduction from 16.
The historical patch-header claim about four simultaneous library links
does not describe the current serialized split-library dependency chain.

## Upstream coverage and flag propagation

At the frozen target, `CMakeLists.txt:297-325` still hardcodes
`-fsycl-max-parallel-link-jobs=16` and derives the GPU link flags from it.
Line 414 still hardcodes `-ftemplate-backtrace-limit=0`; lines 430-432 feed
the CUTLASS flags into `SYCL_TLA_KERNELS_COMPILE_FLAGS`. A whole-tree
immutable `git grep` finds neither downstream cache variable upstream.

This is not merely patch-context evidence. `cmake/utils.cmake` connects
the settings to the actual native targets:

- `add_xe2_kernel_library`, lines 550-610, applies the shared compile flags
  at lines 583-584, copies `SYCL_DEVICE_LINK_FLAGS` at line 602, and applies
  those link options at line 610.
- `add_xe_default_kernel_library`, lines 722-776, applies the shared compile
  flags at lines 754-755 and shared device-link flags at lines 770-776.
- The attention, GDN, MQA-logits, MHC, and Xe2 grouped-GEMM target CMake
  files call the Xe2 helper; default grouped GEMM calls the default helper.
  The Xe3 helper also consumes the common flags, but current split-factory
  feature flags explicitly disable Xe3p, so Xe3 is outside this build audit.

The old upstream base and released fork tip have the identical top-level
CMake blob `9cda28a59fe06935eb813d3a7ffdabbb63eda8bf`. The old-to-new
upstream top-level CMake diff adds only
`csrc/xpu/sampler/compact_sampling_mask.cpp`, in upstream commit
`4111ba268535372314a86f1a2dc930524ba7a92c` (#582). `cmake/utils.cmake`
and the selected native target CMake files have no old-to-new upstream
changes. Consequently neither knob gained a new upstream equivalent in this
transition.

Alternative upstream resource controls were inspected:

- `dae50e2aa58301d3161c26d15bfd867dc5954c12` (#324) refactored generated
  attention buildout. It is already an ancestor of the old upstream base.
  The target generator emits each selected policy/boolean tuple into a
  separate `.cpp` (`chunk_prefill_configure.cmake:238-261`) and offers
  narrow configs. These reduce work and change per-TU characteristics;
  they do not control the link driver's internal concurrency or supply the
  two cache variables. The chunk and paged generators themselves are
  unchanged between old base and target.
- `setup.py:98-138` estimates frontend jobs using approximately 8 GiB per
  process when `MAX_JOBS` is unset. This is not device-link job control,
  and `nix/vllm-xpu-lib.nix` directly invokes CMake/Ninja rather than this
  Python build path.
- Upstream's changed `fmha_xe2.cpp` policy selection and the local disabled
  policy-instantiation repair address another source of frontend cost.
  They do not implement these knobs. Source commit
  `27a9c28f8598a531d34bcd92654ee55c291211c5` has its own audit and must
  be held fixed when attributing benefit to patch 0005.

## Dependency on the split-library sequence and local source changes

`nix/vllm-xpu-lib.nix:120-122` applies **0001 then 0005**, configures with
both overrides, and sets `ninjaFlags = [ libName ]`. No prebuilt-library
override is passed at this stage: patch 0001's ordinary `add_subdirectory`
branches still build the selected native library and therefore consume
patch 0005's flags. The flags are set before the target subdirectories.

The Python/native glue package applies 0001, 0004, and 0006, supplies the
prebuilt DSOs, and does not apply 0005. This is the existing intended scope.
Patch 0001 supplies the prebuilt import/offline build interfaces; patch 0005
supplies resource controls while creating those imported libraries. Patch
0005 is textually independent of 0001, but validation must use the actual
ordered split-library patch set. Dropping it while keeping the current
factory would leave its two `-D` settings unconsumed and restore the
upstream literal 16/0 values.

The released local KVarN commit
`9bb1e2d428b8a25b262c0cd6704614bdf6ee3ad8` changes
`cmake/utils.cmake` and attention's CMake call to add `MIXED_GRF_SIZE`.
Its inspected implementation still starts `XE2_GPU_LINK_FLAGS` from
`SYCL_DEVICE_LINK_FLAGS` and retains the shared compile flags. It only
changes the subsequent library-wide GRF backend option. This is compatible
with patch 0005, but replayed/folded versions require final-source inspection.
There is no vLLM-side schema or runtime API companion change required by
0005 itself, and it must not motivate dropping any K4V2, GDN, MTP, or vision
feature.

## Revalidation blobs

These are Git blob IDs at the frozen selected target, before downstream
source replay or packaging patches. They identify the touched file and
supporting flag paths, not an assertion that all final blobs must stay equal.

| Path | Target blob |
| --- | --- |
| `CMakeLists.txt` | `f7041ef56fc006ed7adeb4432f0a75d3ada7f191` |
| `cmake/utils.cmake` | `710f2beed2f4c8e1aedc3b00dac379fe2e7c4096` |
| `setup.py` | `e841c754d353b39fb566e7c63b31ab70a845dab0` |
| `csrc/xpu/attn/xe_2/CMakeLists.txt` | `83f1719ab6bbf044c380fef1b3b114cc15b2ab56` |
| `csrc/xpu/attn/xe_2/chunk_prefill_configure.cmake` | `19c330a8137b8570efe3de69d7e51888b644a2ca` |
| `csrc/xpu/attn/xe_2/paged_decode_configure.cmake` | `c7393a40d95d5e92ea15fa644bfeb5e4709f007c` |
| `csrc/xpu/gdn_attn/xe_2/CMakeLists.txt` | `4cce0de5acfe1edfe121339d2021de8b2eab11f0` |
| `csrc/xpu/grouped_gemm/xe_2/CMakeLists.txt` | `ff6e8269fa96bdc89f052c0a9e69e50741ac265b` |
| `csrc/xpu/grouped_gemm/xe_default/CMakeLists.txt` | `10a77e8e0c1b98f8ce9e94c097fb920d2d21ee3f` |
| `csrc/xpu/mhc/xe_2/CMakeLists.txt` | `3d93bb89506d75d4a2ef285a8868ef56b92fe17b` |
| `csrc/xpu/mqa_logits/xe_2/CMakeLists.txt` | `09a2b8f487d6615bf9b1a511b0ec7a153294513a` |

For comparison, the released tip's locally changed helper blob is
`b8eab47e0e82a0f96126cdf89c015cba66be60d8`; its attention CMake blob is
`3dc0a504763c3225b5397cd8e88c825dac8d7e27`.

## Commands and observed results

The commands below summarize the exact read-only evidence queries. Run
kernel commands in the object-database directory listed above and packaging
commands at the packaging root. `T`, `O`, and `R` mean the exact selected
target, old base, and released tip recorded in the input table.

| Command/query | Observed result |
| --- | --- |
| `git rev-parse T:CMakeLists.txt O:CMakeLists.txt R:CMakeLists.txt` | Target `f7041ef...`; both historical inputs `9cda28a...`. |
| `git show T:CMakeLists.txt`; `git show T:cmake/utils.cmake`; native target CMake reads | Literal 16/0 values and concrete target flag propagation described above. |
| `git grep -n 'VLLM_XPU_SYCL_LINK_PARALLELISM\|VLLM_XPU_CUTLASS_TEMPLATE_BACKTRACE_LIMIT' T -- .` | No matching upstream definition or alternative consumer. |
| `git diff O T -- CMakeLists.txt` and `git log --format='%H %s' O..T -- CMakeLists.txt cmake` | Only compact sampling-mask source addition; commit #582 above. |
| `git diff --name-only O T -- csrc/xpu/attn cmake` | Attention changes are preset, extern-header, dispatcher, and decode utility; no helper or generator change. |
| `git show T:setup.py`; `git show T:csrc/xpu/attn/xe_2/chunk_prefill_configure.cmake` | Frontend job estimation and per-tuple source emission, not replacements for 0005. |
| `git merge-base --is-ancestor dae50e2aa58301d3161c26d15bfd867dc5954c12 O` | Success: template-generation refactor predates this transition. |
| `git diff O R -- CMakeLists.txt cmake csrc/xpu/attn/xe_2/CMakeLists.txt` | Only the described mixed-GRF helper/call changes in these paths. |
| `git log --follow --format='%H %ad %s' --date=iso -- nix/patches/0005-reduce-kernel-build-memory.patch` and historical `git show` | Five patch-history commits, with reverted experiments and current cache-knob form traced above. |
| Factory, composition, build documentation, xpu-v1.8.0 release note, and parallelism-check reads | Current override/scheduling contract and historical evidence limits described here. |
| `sha256sum` on inputs and `git ls-tree T` on the listed source paths | Exact identities in the input and blob tables. |

The parent's persisted
`benchmark-results/release-v030-20260922/upstream-patch-preflight.json`
records return code zero for sequential 0001 and 0005 application to a
disposable exact-tag archive. The parent reported `--fuzz=0`; the JSON
contains patch output rather than a full command transcript. For 0005 it
records hunk 1 at line 295 (offset 30), hunk 2 at line 304 (offset 30), and
hunk 3 at line 414 (offset 36). No fuzz is reported. I inspected this artifact
and did not rerun patch application or configure/build commands. This proves
textual application to upstream, not final-source flag consumption or memory
benefit.

## Historical evidence limits and remaining gates

The inspected original patch/header, commit messages, factory comments,
`docs/build.md`, and release notes do not supply a controlled 0005-only
RSS/time comparison. The header's claims about retained template diagnostic
chains, approximately 200 TUs, and a particular linker-memory mechanism are
historical rationale, not measurements or compiler-internals proof established
by this audit. In particular, changing diagnostic depth alone has not been
shown here to lower RSS of a successful oneAPI 2026.0 compile.

`docs/releases/xpu-v1.8.0.md` records successful native builds and a single
`fmha_xe2.cpp` process reaching approximately 78 GiB RSS. That demonstrates
that substantial frontend pressure survived the existing knobs; it does not
measure their benefit, establish a new-pair upper bound, or substitute for
the selected release's build. Historical runtime test passes do not validate
resource savings. No production baseline was rerun for this audit.

Before release acceptance, the parent must retain these gates:

1. **Final-source revalidation:** freeze the reconciled source SHA, inspect
   changed blobs against the table, and reapply 0001 then 0005 to a disposable
   exact-final-source archive with `--fuzz=0`. Inspect the resulting CMake and
   helper paths, preserving KVarN mixed-GRF behavior. Record patched file
   hashes. Repeat both kernel cache-identity checks and the existing
   build-parallelism check on final packaging. The latter tests only frontend
   caps for 1, 4, 12, and 24; it does not test link-command flags or actual RSS.
2. **Configure/command validation:** for actual split targets, inspect
   `CMakeCache.txt`, generated compile commands, and Ninja link commands.
   Verify one effective backtrace limit of ten and the requested numeric
   link-job value, without a surviving overriding literal 16 or literal
   shell variable. Verify default 16/0 behavior when the knobs are omitted.
   Confirm that prebuilt glue imports the resulting libraries and that
   Xe3p stays disabled for the intended Xe2/default package.
3. **Controlled compile/link evidence:** on identical final source,
   toolchain/SYCL-TLA, generated configs, AOT devices, and scheduling, compare
   diagnostic limits 0 versus 10 for representative template-heavy TUs,
   including `fmha_xe2.cpp`. Hold link settings and the separately audited
   policy-instantiation fix fixed. Separately compare link jobs 16 versus a
   bounded value such as four using identical object inputs. Record repeated
   cold wall time, peak per-process and aggregate build RSS, actual concurrent
   subprocesses, swap/OOM/failure events, output identity/size, and cache
   policy. A compiler-process maximum alone misses simultaneous child RSS.
   Cache hits are not cold-compile evidence. Keep narrow and actual Brutus
   configurations distinct. Do not claim diagnostic-limit memory savings
   without reproducible evidence; if none appears, reconcile that knob and
   its rationale separately without discarding useful link-job control.
4. **Actual build and paired qualification:** after the workflow's required
   maintenance authorization/state verification, build the exact final
   Brutus package with bounded outer and inner jobs (the documented starting
   point is `--max-jobs 1 --cores 4`). Observe compiler/link concurrency and
   memory, confirm selected native outputs and loaded DSO identities, then
   complete the existing native correctness and V1/K4V2/MTP/vision serving
   qualification. Do not infer final success from archive patching, Nix
   evaluation, historical binaries, or the controlled single-TU experiment.

The minimum retained source delta is the current two cache declarations and
the corresponding flag substitutions. No additional memory workaround or
runtime change is justified by this audit. Final-source and measured build
results may refine the diagnostic knob's disposition; until then the
requirement remains covered and the unmeasured benefit remains explicit.
