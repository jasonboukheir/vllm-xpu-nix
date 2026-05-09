# vllm-xpu-nix

[![build](https://github.com/jasonboukheir/vllm-xpu-nix/actions/workflows/build.yml/badge.svg)](https://github.com/jasonboukheir/vllm-xpu-nix/actions/workflows/build.yml)

Nix-native Intel XPU substrate for [vLLM](https://github.com/vllm-project/vllm):
`torch+xpu`, `triton-xpu`, `vllm-xpu-kernels`, and (in progress) `vllm`
itself, all packaged as nix-store derivations rather than baked into a
container image.

The aim is to let a NixOS host run `vllm serve` as a native systemd unit,
with the SYCL toolchain, Level Zero loader, oneCCL, MKL, and the AOT-compiled
SYCL-TLA kernel `.so`s all referenced from `/nix/store`. No `intel/vllm`
container, no host-managed `~/.local/lib/python*/site-packages`, no
`/opt/intel/oneapi` write directories.

## Status

| Output | What it builds | Source |
| --- | --- | --- |
| `intel-oneapi` | DPC++ compiler + MKL + DPL (oneAPI base toolkit, narrowed) | nixpkgs `intel-oneapi.base` |
| `intel-pti` | Intel Profiling Tools Interface | `nix/intel-pti.nix` |
| `oneccl-bmg` | oneCCL with the single-card BMG runtime workarounds | `nix/oneccl-bmg.nix` |
| `torch-xpu` | `torch==2.11.0+xpu` wheel, auto-patchelf'd against the nix-store oneAPI closure | `nix/torch-xpu.nix` |
| `triton-xpu` | `triton-xpu==3.7.0` (Intel's Triton port) | `nix/triton-xpu.nix` |
| `vllm-xpu-kernels` | upstream `vllm-project/vllm-xpu-kernels` with split-kernel-libs patch | `nix/vllm-xpu-kernels.nix` |
| `vllm-xpu-kernels-unstable` | same, but pinning the `jasonboukheir/vllm-xpu-kernels` fork's `main` | same factory |
| `attn-kernels-xe-2`, `gdn-attn-kernels-xe-2`, `mqa-logits-kernels-xe-2`, `grouped-gemm-xe-{2,default}` | individual SYCL-TLA `.so`s, built in parallel and stitched into `vllm-xpu-kernels` | `nix/vllm-xpu-lib.nix`, `nix/vllm-xpu-attn-dyndrv.nix` |

`vllm-xpu` itself (a source build of vLLM linked against the above) is
tracked in [#5][i5]; the `services.vllm-xpu` NixOS module is tracked in
[#1][i1]. Until those land, you can already build the kernel + runtime
substrate against your own checkout of vLLM via `--override-input` (see
[Iterating against a local checkout](#iterating-against-a-local-checkout)
below).

[i1]: https://github.com/jasonboukheir/vllm-xpu-nix/issues/1
[i5]: https://github.com/jasonboukheir/vllm-xpu-nix/issues/5

## Build something locally

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

## Using on a NixOS server (overlay pattern)

The intended consumption story is "add an input + overlay + import the
NixOS module to your existing system flake" — not "vendor this whole
flake into your config".

### 1. Add the input

```nix
# ~/.config/nix/flake.nix (or wherever your system flake lives)
{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    vllm-xpu-nix = {
      url = "github:jasonboukheir/vllm-xpu-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };
}
```

`inputs.nixpkgs.follows = "nixpkgs"` keeps a single `glibc`/`libstdc++`
across the closure. This flake is developed against `nixos-unstable`; any
consumer pin recent enough to expose `pkgs.intel-oneapi.base` (added Feb
2026) is fine.

### 2. Wire packages via an overlay

```nix
# ~/.config/nix/modules/nixpkgs/overlays/vllm-xpu-nix.nix
{ inputs, ... }: final: prev: {
  inherit (inputs.vllm-xpu-nix.packages.${prev.system})
    intel-oneapi intel-pti oneccl-bmg
    torch-xpu triton-xpu
    vllm-xpu-kernels vllm-xpu-kernels-unstable;
}
```

Register the overlay through whichever overlay-loader your flake uses
(`nixpkgs.overlays = [ … ]`, flake-parts' `perSystem.nixpkgs.overlays`,
etc.).

### 3. Import the NixOS module

Once [#1][i1] lands:

```nix
# host config
{ pkgs, inputs, ... }: {
  imports = [ inputs.vllm-xpu-nix.nixosModules.vllm-xpu ];

  services.vllm-xpu = {
    enable      = true;
    package     = pkgs.vllm-xpu-unstable;
    model       = "palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4";
    servedName  = "qwen3.6-35b-a3b";
    dtype       = "bfloat16";
    quantization = "inc";
  };
}
```

### 4. allowUnfree

`intel-oneapi-base-toolkit` is unfree. Scope the predicate narrowly
rather than flipping `allowUnfree = true` globally:

```nix
nixpkgs.config.allowUnfreePredicate = pkg:
  builtins.elem (pkgs.lib.getName pkg) [
    "intel-oneapi-base-toolkit"
  ];
```

## Hardware prerequisites

The packaging side is solved (auto-patchelf, nix-store paths). The GPU
still has to be reachable through the kernel + ICD config so apps can
load Level Zero or OpenCL contexts.

### NixOS hosts

```nix
{ pkgs, ... }: {
  hardware.graphics = {
    enable = true;
    extraPackages = with pkgs; [
      intel-compute-runtime
      level-zero
      intel-graphics-compiler
    ];
  };
}
```

This populates `/run/opengl-driver/lib/` and
`/run/opengl-driver/etc/OpenCL/vendors/intel.icd`. Apps that read
`OCL_ICD_VENDORS` and pick up `LD_LIBRARY_PATH` from the OpenGL wrapper
find the driver at runtime. The `services.vllm-xpu` unit shipped by
[#1][i1] inherits this automatically.

The service user needs to be in `render` (and ideally `video`) — the
NixOS module adds these as `SupplementaryGroups`. The host kernel must
have `i915` or `xe` loaded with a supported Intel GPU.

### Non-NixOS hosts

For Ubuntu / Fedora / Arch consumers running this flake directly:

- Install the distro's OpenCL ICD + Level Zero loader via the package
  manager.
- Wrap `vllm` so that `OCL_ICD_VENDORS=…/etc/OpenCL/vendors` and
  `LD_LIBRARY_PATH=…/lib` are exported before exec.
- The nixpkgs `intel-compute-runtime` (currently 26.14) is newer than
  what the official `intel/vllm:0.17.0-xpu` container ships (25.48).
  The driver/runtime ABI is generally backward-compatible, but flag this
  if you're on an older host kernel.

### Verification

```bash
nix shell .#intel-oneapi -c sycl-ls
# Expected output includes:
#   [level_zero:gpu:0] Intel(R) Battlemage / Arc / ...

nix shell --impure --expr '
  with import <nixpkgs> { config.allowUnfree = true; };
  (python3.withPackages (ps: [ (import ./. {}).torch-xpu ])).env
' -c python -c 'import torch; print(torch.xpu.is_available(), torch.xpu.device_count())'
```

If `torch.xpu.is_available()` is `False` inside `nix shell` but `True`
outside, the wrapper isn't propagating `LD_LIBRARY_PATH` /
`OCL_ICD_VENDORS` to the SYCL runtime.

## Iterating against a local checkout

The `*-unstable` flake inputs pin a fork's `main` branch, but during
active kernel development you want to build against your working tree
without committing+pushing first. Nix's `--override-input` does this;
the recipe is just non-obvious.

### Build against a local kernels checkout

```bash
nix build .#vllm-xpu-kernels-unstable \
  --override-input vllm-xpu-kernels-unstable-src \
  path:/home/me/Projects/vllm-xpu-kernels
```

`path:` follows the working tree as-is, so make sure the submodules are
populated:

```bash
cd /home/me/Projects/vllm-xpu-kernels
git submodule update --init --recursive
```

If `third_party/oneDNN/` (or any other submodule) isn't on disk, the
build trips at CMake's `git submodule update` invocation.

### Build vLLM itself against a local checkout

Once [#5][i5] / [#6][i6] land:

```bash
nix build .#vllm-xpu-unstable \
  --override-input vllm-xpu-unstable-src \
  path:/home/me/Projects/vllm
```

[i6]: https://github.com/jasonboukheir/vllm-xpu-nix/issues/6

### Combine overrides

```bash
nix build .#vllm-xpu-unstable \
  --override-input vllm-xpu-kernels-unstable-src path:/home/me/Projects/vllm-xpu-kernels \
  --override-input vllm-xpu-unstable-src         path:/home/me/Projects/vllm
```

### NixOS rebuild against a local source

The `flake-input/sub-input` syntax is what overrides a transitive input
from a consuming flake:

```bash
sudo nixos-rebuild switch \
  --override-input vllm-xpu-nix/vllm-xpu-unstable-src path:/home/me/Projects/vllm
```

### Fast in-tree iteration on the attn kernels

For tight edit-compile-test loops on the SYCL-TLA attn kernel
specifically, skip the `nix build` round-trip and use the `attn-dev`
shell:

```bash
nix develop .#attn-dev
make dev-attn KERNELS_SRC=/path/to/vllm-xpu-kernels
export VLLM_XPU_DEV_LIB_DIR=$PWD/build-dev/csrc/xpu/attn/xe_2
python -c 'import vllm_xpu_kernels'
```

The 0002-dev-lib-override.patch reads `VLLM_XPU_DEV_LIB_DIR` at import
time and prefers a dev-built `.so` over the nix-store one, so your
edit lands in the running interpreter without rebuilding the
`vllm-xpu-kernels` derivation.

### Editable install of vllm-xpu-kernels (`kernels-dev`)

For a full editable install (all kernels, not just attn), use the
`kernels-dev` shell — it pulls the full kernels closure plus
`torch-xpu` / `triton-xpu` and the SYCL toolchain:

```bash
cd /path/to/vllm-xpu-kernels
nix develop /path/to/vllm-xpu-nix#kernels-dev
git submodule update --init --recursive
pip install -e . --no-build-isolation
ninja -C build/temp.*/release install     # incremental rebuild after a .cpp edit
pytest tests/
```

`MAX_JOBS=2` by default — each SYCL-TLA template instantiation holds
~40 GiB during icpx; raise only if your box has >100 GiB of free RAM.

## Continuous integration

`.github/workflows/build.yml` runs two jobs:

1. **flake-check** — `nix flake check --no-build` on `ubuntu-latest`.
   Eval-only sanity, runs on every push and PR.
2. **build** — `nix build .#<each-package>` across a matrix of all
   substrate outputs, on a self-hosted runner. Gated behind the repo
   variable `SELF_HOSTED_RUNNER_AVAILABLE=true` so the workflow stays
   green for forks that don't have a runner attached.

The matrix needs a self-hosted runner because each SYCL-TLA kernel
icpx process holds ~40 GiB of resident memory during template
instantiation (and `MAX_JOBS=2` means ~80 GiB working set on the full
target). GitHub-hosted `ubuntu-latest` caps at 16 GiB RAM and ~14 GiB
disk; the torch+xpu wheel alone is 793 MiB.

To enable the matrix build:

1. Provision a self-hosted runner with labels
   `[self-hosted, x86_64-linux, vllm-xpu-nix]`, ≥96 GiB RAM, ≥150 GiB
   disk, and a Nix install with flakes enabled.
2. Configure a binary cache (Cachix or a self-hosted `nix serve`) on
   the runner host as a substituter, otherwise every PR rebuilds the
   90-minute kernels closure from scratch.
3. Set the repo variable `SELF_HOSTED_RUNNER_AVAILABLE=true` to
   un-gate the `build` job.

## Roadmap

The non-trivial roadmap lives in the issue tracker:

- [#1][i1] — `services.vllm-xpu` NixOS module
- [#5][i5] — `vllm-xpu` source-build derivation
- [#6][i6] — `vllm-xpu-unstable` fork variant
- [#7][i7] — `flash-linear-attention` (qwen3_5_moe gated-delta-rule)
- [#8][i8] — CI: `nix flake check` + per-package build verification
- [#10][i10] — `auto-round-xpu` (replaces the auto-round Containerfile)
- [#11][i11] — `quantize` / `kl-eval` flake apps
- [#12][i12] — editable `kernels-dev` / `vllm-dev` shells

[i7]: https://github.com/jasonboukheir/vllm-xpu-nix/issues/7
[i8]: https://github.com/jasonboukheir/vllm-xpu-nix/issues/8
[i10]: https://github.com/jasonboukheir/vllm-xpu-nix/issues/10
[i11]: https://github.com/jasonboukheir/vllm-xpu-nix/issues/11
[i12]: https://github.com/jasonboukheir/vllm-xpu-nix/issues/12

## License

Apache-2.0 for the kernel code and Nix glue authored here. Upstream
components retain their own licenses (PyTorch BSD-3, Triton MIT, oneAPI
under Intel's End User License Agreement).
