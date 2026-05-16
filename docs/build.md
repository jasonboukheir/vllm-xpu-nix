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

Each lib drv runs cmake + ninja through `sccache` as the compiler
launcher. With `SCCACHE_BUCKET` (or `SCCACHE_REDIS`) configured on the
host, warm rebuilds get .o cache hits from the shared backend across
machines — set the env vars before invoking `nix build`; they're
inherited into the sandbox via `impureEnvVars`.

```bash
# one-shot for a single rebuild without touching daemon config
nix build .#vllm-xpu-unstable -j 4
```
