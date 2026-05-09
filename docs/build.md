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
24-core box; individual kernel `.so`s are 8-25 minutes each, with each
icpx process holding ~5 GiB of resident memory in the steady state and
~40 GiB on the heaviest head-dim/policy template combos. The intra-drv
ninja cap is fixed at `-j2` (override with `cores=` per-builder if you
have headroom) so multiple kernel `.so` drvs can run concurrently under
the outer Nix scheduler without overrunning a 96 GiB box.

`attn-kernels-xe-2` uses a dynamic-derivations build (~600 single-TU
drvs that link into one `.so`); each TU is its own Nix derivation, so
the **outer** `max-jobs` setting is what gates per-TU concurrency. On a
96 GiB host, keep `max-jobs ≤ 4` (or provision swap/zram) for the
duration of an `attn-kernels-xe-2` rebuild — the default `auto`
schedules 24 concurrent icpx processes and reliably OOMs when several
heavy template TUs overlap.

```bash
# one-shot for a single rebuild without touching daemon config
nix build .#vllm-xpu-unstable -j 4
```
