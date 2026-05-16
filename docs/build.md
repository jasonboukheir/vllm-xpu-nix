# Build something locally

```bash
nix build .#vllm-xpu-kernels             # upstream pin
nix build .#vllm-xpu-kernels-unstable    # jasonboukheir/vllm-xpu-kernels main

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

After `0007-fa2-dtype-split.patch` the worst-case kernel_template TU
peaks at ~7 GiB RSS (down from ~40 GiB pre-patch); after
`0008-fa2-dispatcher-split.patch` the chunk_prefill / paged_decode
dispatcher TUs — the residual OOM hotspot at `-j$(nproc)` — split into
per-policy shards that also sit below that ceiling. Each lib drv runs
`ninja -j$NIX_BUILD_CORES` at the daemon default (`nproc`). On
memory-constrained hosts cap parallelism with `nix build --cores N`
(or `cores = N` in nix.conf); the same value feeds
`-fsycl-max-parallel-link-jobs`.

## Compiler launcher cache

Both `vllm-xpu-lib.nix` (the five kernel SHARED libs — attn, gdn-attn,
mqa-logits, grouped-gemm-xe-2, grouped-gemm-xe-default) and
`vllm-xpu-kernels.nix` (the Python extension module + the libs not
covered by the prebuilt-lib split: basic_kernels, xpu_specific,
xpumem_allocator) run cmake + ninja through **ccache** as the
compiler launcher. ninja invokes `ccache icpx ...` per TU; warm
rebuilds pull each `.o` from `/var/cache/ccache` when the
preprocessed-source hash matches.

The other build phases (`vllm-xpu`, `torch-xpu`, `triton-xpu`,
`intel-pti`, `oneccl-bmg`, `flash-linear-attention`,
`auto-round-xpu`) ship as wheels or as pure-Python and produce no
native code, so ccache has nothing to do there.

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
