# P06: forward the MHC feature flag

Decision: **KEEP `0006-forward-mhc-feature-flag.patch` unchanged.** Its
single added list entry remains the smallest correction for the separate FA2
binding build. No new source-fork commit is needed.

## Inputs and scope

- Packaging patch: `nix/patches/0006-forward-mhc-feature-flag.patch`, SHA-256
  `8f2cba38bf161f7d557f3f1fce0b584c6beda21cb315a4b17ee39a4c5411de98`.
- Introduced by packaging commit
  `ec8c035b8deda3606e2c4960a416f64a8584dd32`,
  `packaging: split reusable kernel glue from FA2 binding`.
- Old upstream kernel base:
  `1c7cbeee1cd0c1481d48f5031f679a47f3f0ef45`.
- Selected kernel release source:
  `ddf336d86e3c8602888572a3502f951abd51df12` (`release/0.1.15.4`, frozen
  by the parent release workflow).
- Replayed kernel candidate inspected:
  `dbb684f4c9b38a8b3bbf52bda0c490ee47049221`.
- Read the update skill, workflow, commit-audit procedure, and old/new kernel
  `AGENTS.md`. This audit modifies only this report; application checks use
  disposable copies under `/tmp`.

## Requirement and call path

The Nix package composes separately built base glue and FA2 bindings. In
`nix/mk-kernels.nix`, base glue enables `MHC_KERNELS_ENABLED`; the FA2 binding
sets it to `false`, sets `withMhcLibrary = false`, and uses a source projection
that excludes `csrc/xpu/mhc`. The feature booleans become `ON`/`OFF`
environment attributes through `featureEnvAttrs` in
`nix/vllm-xpu-kernels.nix`. Both components build through upstream `setup.py`.

At the frozen upstream target, `setup.py::cmake_build_ext.configure` forwards
other kernel feature options to CMake but omits `MHC_KERNELS_ENABLED` from
`_kernel_options`. Its later `additional_libraries` selection does read that
environment variable. Thus an environment setting of `OFF` skips Python's
additional-library installation entry but does not disable CMake's MHC build.

`CMakeLists.txt:76` defaults the MHC option to `ON`. The conditional at line
462 enters `csrc/xpu/mhc/xe_2` when SYCL-TLA and Xe2 are enabled, which they
are for the FA2 component. The projected FA2 source excludes that directory
and does not supply a prebuilt MHC library. The missing forwarded option would
therefore make this configure path attempt to enter an absent directory. The
option also controls `VLLM_MHC_ENABLED` at line 518. There is no alternate
environment-to-CMake mapping elsewhere in the frozen upstream tree.

The patch adds only `"MHC_KERNELS_ENABLED"` to the existing forwarding list.
It preserves upstream's default-enabled behavior and makes explicit disabling
consistent with the other kernel features. This is a build-configuration
requirement for the actual release package, not a claim that the Brutus model
uses MHC at runtime.

## Upstream coverage

`setup.py` has the same Git blob at both upstream bases:
`e841c754d353b39fb566e7c63b31ab70a845dab0`. The four upstream commits between
the bases cover ragged GDN speculation
(`f4b729ae7fb3eb22c564606ec8b4d184c7c8bc4a`), Torch 2.14 and Xe3 language
standard changes (`d09702a637cfea2a0c02909af82bd789d26ea319`), an attention
preset (`885b120ffa082312b7d93c2be7a1e3b3dd79672f`), and the MoE top-k
interface (`ddf336d86e3c8602888572a3502f951abd51df12`). None changes this
configuration path. The only root `CMakeLists.txt` delta is the supported
Torch version, 2.13.0 to 2.14.0; MHC option defaults and gates are unchanged.
The candidate's replayed native features do not supply the missing mapping.

There is no upstream replacement to adopt and no residual simplification
beyond the existing one-line patch. Dropping MHC from the overall package or
collapsing the source projections would change unrelated supported behavior
and discard the existing reusable build components.

## Dependencies

- `0001-split-kernel-libs.patch` supplies imported prebuilt-library handling.
  That handling does not replace the explicit MHC disable needed by FA2.
- `0004-skip-prebuilt-additional-libs.patch` controls installation of supplied
  libraries, separately from configure-time feature selection.
- All three are already applied in order by `nix/vllm-xpu-kernels.nix`.
- No vLLM Python API, native operator schema, runtime ABI, or KVarN change is
  required. No source-fork commit should be dropped or modified for P06.
- Existing Nix Xe3P disabling and explicit Brutus `bmg` AOT selection are
  unaffected; P06 introduces no architecture defaults.

## Checks performed

Using `/home/jasonbk/Projects/vllm/.venv/bin/python` and disposable copies:

1. Applied `patch --dry-run` and then the actual patch to frozen upstream
   `setup.py` and the replayed candidate's `setup.py`: both passed without
   fuzz or offsets for P06.
2. Applied the production packaging sequence P01, P04, P06 to frozen upstream
   `CMakeLists.txt` and `setup.py`: all applied. P01/P04 had context offsets;
   P06 applied directly.
3. Executed the actual AST-extracted `_is_enabled` and
   `cmake_build_ext.configure` code with compiler discovery and the subprocess
   boundary stubbed. Checked eight environment values in four variants:
   unpatched upstream, upstream plus P06, candidate plus P06, and the full
   packaging patch sequence. All **32 captured-argument checks passed**.
   Unpatched upstream omitted the MHC `-D` argument in all eight controls.
   Each patched variant emitted `OFF` for `OFF`, `0`, `false`, and `NO`, and
   `ON` for an unset variable, `ON`, `1`, and `TRUE`.

These checks establish patch compatibility and the actual Python-to-CMake
argument behavior without importing Torch, running a compiler, or using the
GPU. They are not native-build or runtime qualification. The parent's exact
Brutus package build must still configure and build both glue components,
including the FA2 component with MHC disabled and base glue with MHC enabled.
No separate MHC performance experiment is required for this forwarding fix.
