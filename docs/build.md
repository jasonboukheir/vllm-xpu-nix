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

The full kernels target rebuilds in roughly 90 minutes from cold on a
24-core box; individual kernel `.so`s are 8-25 minutes each. After the
`0007-fa2-dtype-split.patch` rework, per-TU peak RSS sits around ~7 GiB
on the worst-case `q16_h256_p128` attn TU (down from ~40 GiB pre-patch).
The intra-drv ninja cap is fixed at `-j2` (override with `cores=`
per-builder if you have headroom) so multiple kernel `.so` drvs can run
concurrently under the outer Nix scheduler without overrunning a 96 GiB
box.

Each lib drv runs cmake + ninja through `sccache` as the compiler
launcher. With `SCCACHE_BUCKET` (or `SCCACHE_REDIS`) configured on the
host, warm rebuilds get .o cache hits from the shared backend across
machines — set the env vars before invoking `nix build`; they're
inherited into the sandbox via `impureEnvVars`.

```bash
# one-shot for a single rebuild without touching daemon config
nix build .#vllm-xpu-unstable -j 4
```
