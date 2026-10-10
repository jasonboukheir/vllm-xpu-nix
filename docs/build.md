# Build something locally

```bash
nix build .#vllm-xpu-kernels             # upstream pin
nix build .#vllm-xpu-kernels-unstable    # maintained fork pin

nix build .#torch-xpu
nix build .#triton-xpu
nix build .#intel-oneapi
```

Each kernel `.so` is also a top-level package, so you can rebuild a single
kernel without re-doing the full target:

```bash
nix build .#attn-kernels-xe-2
nix build .#gdn-attn-kernels-xe-2
nix build .#grouped-gemm-xe-2
```

Upstream's generated attention buildout now emits one translation unit per
kernel variant, replacing the former local `0007`/`0008` split patches. A
partial chunk-prefill/paged-decode config bounds the generated kernel buildout.
Upstream's dispatcher checks generated policy and tuple availability before
instantiating kernels. The former local early dispatcher-pruning optimization
was retired for v0.31; it pruned no policies in the normal Brutus preset.

Split libraries use `NIX_BUILD_CORES` for compiler and device-link parallelism;
attention also caps concurrent compiler processes at 12. On memory-constrained
hosts use `nix build --cores N` (or `cores = N` in nix.conf). The Python extension
and FA2 binding honor that compile limit but retain upstream's inner device-link
limit of 16. These settings cannot cap the memory of an individual frontend.
The template-backtrace limit controls diagnostic length; a successful-build
memory reduction has not been measured.

The v0.31 XPU closure uses Torch 2.14, Triton 3.8 and oneAPI 2026.1.
Its Python build reconciles the common CUDA Torch build pin with the exact XPU
requirement. The executable wrapper supplies the pinned OpenCL ICD directory
for oneDNN discovery and the toolkit's `ccl/latest/lib` loader path for oneCCL's
runtime plugins. An import-only build check does not exercise either startup
path; qualification includes native primitives and a real XCCL process group.

## Compiler launcher cache

Both `vllm-xpu-lib.nix` (the six kernel SHARED libs — attn, gdn-attn, mhc,
mqa-logits, grouped-gemm-xe-2, grouped-gemm-xe-default) and
`vllm-xpu-kernels.nix` (the Python extension module + the libs not
covered by the prebuilt-lib split: basic_kernels, xpu_specific,
xpumem_allocator) run cmake + ninja through **ccache** as the
compiler launcher. ninja invokes `ccache icpx ...` per TU; warm
rebuilds pull each `.o` from `/var/cache/ccache` when the
preprocessed-source hash matches.

The other build phases (`vllm-xpu`, `torch-xpu`, `triton-xpu`,
`intel-pti`, `flash-linear-attention`, `auto-round-xpu`) ship as
wheels or as pure-Python and produce no native code, so ccache has
nothing to do there.

Why the `CCACHE_*` env vars are set as **derivation attrs** rather
than `impureEnvVars` / nix.conf `impure-env`: both of those
mechanisms gate on `!isSandboxed()` and skip CA / input-addressed
builds (these kernel drvs set `__contentAddressed = true`). Top-level
mkDerivation attrs are the only mechanism that crosses the sandbox
unconditionally.

### Host setup

One-time, NixOS:

```nix
systemd.tmpfiles.rules = [ "d /var/cache/ccache 0770 root nixbld - -" ];
nix.settings.extra-sandbox-paths = [ "/var/cache/ccache" ];
```

Inspect / manage the cache from outside the sandbox:

```bash
ccache --show-stats --dir /var/cache/ccache
ccache --zero-stats --dir /var/cache/ccache
ccache --max-size=200G --dir /var/cache/ccache
```

### Disabling ccache for a single build

Default is `useCcache = true`. To skip the launcher for one rebuild
(useful on a CI host that hasn't bind-mounted `/var/cache/ccache`):

```bash
nix build '.#vllm-xpu-kernels.withCcache false'
```

```bash
# one-shot for a single rebuild without touching daemon config
nix build .#vllm-xpu-unstable -j 4
```
