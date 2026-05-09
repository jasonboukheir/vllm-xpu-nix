# Iterating against a local checkout

The `*-unstable` flake inputs pin a fork's `main` branch, but during
active kernel development you want to build against your working tree
without committing+pushing first. Nix's `--override-input` does this;
the recipe is just non-obvious.

## Build against a local kernels checkout

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

## Build vLLM itself against a local checkout

Once [#5](https://github.com/jasonboukheir/vllm-xpu-nix/issues/5) /
[#6](https://github.com/jasonboukheir/vllm-xpu-nix/issues/6) land:

```bash
nix build .#vllm-xpu-unstable \
  --override-input vllm-xpu-unstable-src \
  path:/home/me/Projects/vllm
```

## Combine overrides

```bash
nix build .#vllm-xpu-unstable \
  --override-input vllm-xpu-kernels-unstable-src path:/home/me/Projects/vllm-xpu-kernels \
  --override-input vllm-xpu-unstable-src         path:/home/me/Projects/vllm
```

## NixOS rebuild against a local source

The `flake-input/sub-input` syntax is what overrides a transitive input
from a consuming flake:

```bash
sudo nixos-rebuild switch \
  --override-input vllm-xpu-nix/vllm-xpu-unstable-src path:/home/me/Projects/vllm
```

## Fast in-tree iteration on the attn kernels

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

## Editable install of vllm-xpu-kernels (`kernels-dev`)

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
