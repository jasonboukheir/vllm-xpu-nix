# P05 — kernel build concurrency and diagnostic controls

Disposition: **MODIFY the inaccurate explanation; KEEP the existing link-parallelism control.** Upstream has no replacement at the frozen target. The template-backtrace setting is a diagnostic-output control with no established successful-build memory benefit; do not present it as a measured optimization. Removing that independent knob is a possible further reduction, but this audit has not established a resource comparison that justifies claiming its removal is performance-neutral.

## Inputs and scope

- Packaging patch: `nix/patches/0005-reduce-kernel-build-memory.patch`, SHA-256 `79701051bd0b04bf09074ecce35e80a81a74b4e4b257de06eae3a630e972eb98`.
- Old kernel base: `1c7cbeee1cd0c1481d48f5031f679a47f3f0ef45`; old fork tip: `193cee080d654ff67c4f8bfaf924201bc88b9cf5`.
- Frozen new kernel base: `ddf336d86e3c8602888572a3502f951abd51df12`, source of the selected released `0.1.15.4` wheel.
- Candidate substrate inspected: Torch `2.14.0+xpu`, oneAPI `2026.1.0.192`.
- Read workflow stage 5, the commit-audit contract, `docs/fork-delta.md`, and source `AGENTS.md`; verified execution host `brutus`.
- Read-only investigation used frozen Git blobs and a private `/tmp/p05-build-memory.WCAAJa` copy. No shared source, configuration, Git refs, services or GPU state changed.

## Requirement and upstream coverage

The useful requirement is control over **inner SYCL device-image post-link/AOT parallelism**, which is separate from the outer number of Ninja compile/link commands. Lowering outer jobs alone does not alter the inner device-image fan-out of one link command.

At the frozen target:

- `CMakeLists.txt:304–306` still supplies `-fsycl-max-parallel-link-jobs=16` unconditionally.
- `CMakeLists.txt:414` still supplies `-ftemplate-backtrace-limit=0` unconditionally.
- `cmake/utils.cmake:601–610` copies `SYCL_DEVICE_LINK_FLAGS` into `XE2_GPU_LINK_FLAGS` and passes them to the split target through `target_link_options`.
- `cmake/utils.cmake:525–526` also consumes supplied link flags for the Python extension targets.
- The complete old-base/new-base delta for `CMakeLists.txt` only updates expected Torch `2.13.0` to `2.14.0` in `d09702a637cfea2a0c02909af82bd789d26ea319` (#525). Its `cmake/utils.cmake` change gives the Xe3 target C++20; it does not alter concurrency. There are no `setup.py` changes in this delta. The other three selected commits concern ragged GDN, paged-decode configuration and top-k behavior.

Thus the release update does not obsolete the parallelism knob or replace it with memory-aware native link scheduling. There is no runtime operator/schema dependency, and no dependency on any of the retained feature commits.

The installed candidate compiler's own `icpx --help` distinguishes the two options:

- `-fsycl-max-parallel-link-jobs` controls maximum parallelism of device-code post-link actions, including SPIR-V image generation and AOT compilation per image.
- `-ftemplate-backtrace-limit` controls how many entries are **printed** in a template-instantiation backtrace; zero means unlimited.

This directly supports a bounded-link-parallelism purpose. It does **not** establish the patch header's claims that the diagnostic option makes the frontend retain an entire instantiation chain, dominates peak compilation RAM, or recovers successful-build RSS. No controlled memory or build-time comparison was located or run. The header's claim about up to four libraries linking concurrently also does not describe the current serialized split-library dependency chain.

## What `--cores` actually controls here

Current packaging differs from the old summary in `docs/fork-delta.md`: split-library device links are **not fixed at four jobs**.

| Build path | Outer build jobs | Inner device-image link jobs | Template backtrace |
| --- | --- | --- | --- |
| Split attention library | `min(NIX_BUILD_CORES, 12)` | Original `NIX_BUILD_CORES`, captured before the frontend cap | 10 |
| Other split libraries | `NIX_BUILD_CORES` | `NIX_BUILD_CORES` | 10 |
| Python glue / FA2 extension packages | `MAX_JOBS=NIX_BUILD_CORES` | Upstream hardcoded 16 | Upstream 0 where CUTLASS flags apply |

Evidence:

1. `nix/vllm-xpu-lib.nix:205–207` appends the already expanded `NIX_BUILD_CORES` to `cmakeFlagsArray` as `-DVLLM_XPU_SYCL_LINK_PARALLELISM=...`. This is a real CMake cache override, not a literal shell-variable string in compiler flags.
2. `nix/vllm-xpu-lib.nix:210–215` caps `NIX_BUILD_CORES` only in `preBuild`, after configure has captured the original value for device links. `nix/mk-kernels.nix:253–260` sets `compileJobs = 12` only for attention.
3. `nix/vllm-xpu-lib.nix:228` sets the diagnostic cache variable to 10.
4. `nix/mk-kernels.nix:223–256` serializes the memory-heavy split derivations using predecessor build dependencies. This bounds competing split-library compiler farms independently of inner link parallelism.
5. `nix/vllm-xpu-kernels.nix:125–129` does **not** apply P05 to Python extension builds; `:170` exports `MAX_JOBS` from `NIX_BUILD_CORES`. Frozen `setup.py:98–104` accepts that value, `:235–242` configures the Ninja compile pool with it, and `:282–291` passes it as `cmake --build -j=...`.
6. The upstream automatic estimate of roughly 8 GiB per compile process at `setup.py:105–134` is bypassed when `MAX_JOBS` is explicitly set, as packaging does.

For example, `--cores 4` gives split attention outer/inner limits 4/4 and glue outer/inner limits 4/16. At `--cores 24`, attention gets 12/24 and glue gets 24/16. Read-only `nix config show` during this audit reported `cores = 24`, `max-jobs = 12`; these settings are overridable per invocation. The patch permits a lower limit, but the current default invocation can actually use more inner jobs than upstream's fixed 16. None of these settings is a global cap on every process/thread in the build graph.

The comment mentioning `cmakeBuildHook` at `nix/vllm-xpu-lib.nix:211` is imprecise: with Ninja enabled, the Ninja build phase reads `NIX_BUILD_CORES` after `preBuild`. A locally installed nixpkgs Ninja setup hook was inspected for that ordering; no full candidate derivation build was run in this audit.

## Minimum justified change

Keep the CMake link cache variable and its Nix wiring. Removing the whole patch would silently restore fixed inner parallelism 16 and defeat the operator's ability to bound split-library device linking with `--cores`.

Rewrite the patch header and the template-variable help text to state their demonstrated purposes: configurable device-image concurrency and bounded diagnostic output. Remove unmeasured statements about dominant peak RAM, retained instantiation chains and successful-build RSS recovery. Correct the current-delta documentation to describe the actual table above. No new CMake architecture, ABI or runtime changes are needed.

If the release owner explicitly retires diagnostic tuning, the smallest source reduction is to delete the template-backtrace hunk from P05 **and** `-DVLLM_XPU_CUTLASS_TEMPLATE_BACKTRACE_LIMIT=10` from `nix/vllm-xpu-lib.nix`, restoring upstream's zero setting. That is independent of the link knob. A matched oneAPI 2026.1 compile/RSS comparison would be needed to claim a memory or build-time result; passing runtime tests alone would not supply that evidence. Merely adding P05 to glue would also be insufficient to honor `--cores` there: setup.py would need to forward the cache value explicitly. Extending glue plumbing is outside the minimum retained patch.

## Validation performed and limits

- Applied P01 to a private copy of frozen upstream build files, then ran P05 `patch --dry-run` and actual application in that copy. All three P05 hunks succeeded without fuzz (line offsets +30, +30 and +36).
- Evaluated the actual patched link/cache flag statements with installed CMake 4.3.4 using `cmake -P`; all three cases passed: defaults `16/0`, overrides `4/10`, and overrides `24/10`. Assertions inspected resulting link and CUTLASS flag lists.
- Inspected oneAPI 2026.1 `icpx --help` to verify both option semantics without compiling or using a GPU.
- An optional `nix eval` to resolve the nixpkgs source path failed because the default fetcher-lock cache under `~/.cache/nix` is read-only in the audit sandbox. It made no packaging/source changes and is not recorded as a successful derivation evaluation.
- Existing `nix/tests/kernel-build-parallelism.nix` checks the attention frontend cap for 1, 4, 12 and 24 cores; this audit read it but did not run its derivation. It does not validate device-link flags or glue behavior.
- No native compilation, fresh maximum-RSS measurements, runtime/performance comparison, or claim that template tuning lowers memory. Parent qualification must inspect effective link flags for the selected `--cores` value and complete its normal native build and paired server tests.
