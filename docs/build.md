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
icpx process holding ~40 GiB of resident memory during SYCL-TLA template
instantiation. `MAX_JOBS` in the kernels build is pinned to
`NIX_BUILD_CORES`, but the per-`.so` derivations are independent, so two
kernels can run in parallel as long as you have ~80 GiB of RAM free.
