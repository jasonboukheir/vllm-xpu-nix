# Iterating against a local checkout

The `*-unstable` flake inputs pin a fork's `main` branch, but during active
kernel development you want to build against your working tree without
committing+pushing first. Nix's `--override-input` does this; the recipe is just
non-obvious.

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

If `third_party/oneDNN/` (or any other submodule) isn't on disk, the build trips
at CMake's `git submodule update` invocation.

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

The `flake-input/sub-input` syntax is what overrides a transitive input from a
consuming flake:

```bash
sudo nixos-rebuild switch \
  --override-input vllm-xpu-nix/vllm-xpu-unstable-src path:/home/me/Projects/vllm
```

## Develop the kernels (editable install, `kernels-dev`)

To edit, build, and test `vllm-xpu-kernels` in tree, use the `kernels-dev` shell
— it carries the SYCL toolchain plus the full kernels closure (`torch-xpu` /
`triton-xpu`) as build _inputs_, so you build the kernels yourself rather than
pulling a prebuilt copy:

```bash
cd /path/to/vllm-xpu-kernels
nix develop /path/to/vllm-xpu-nix#kernels-dev
git submodule update --init --recursive
pip install -e . --no-build-isolation
ninja -C build/temp.*/release install     # incremental rebuild after a .cpp edit
pytest tests/                             # e.g. tests/gdn_attn for the GDN/conv1d work
```

For a cache-shared cold build of a single kernel lib (reuses
`/var/cache/ccache`, no editable install), build the narrowed package target
against your checkout instead:

```bash
nix build .#gdn-attn-kernels-xe-2 \
  --override-input vllm-xpu-kernels-unstable-src path:/path/to/vllm-xpu-kernels
```

## Editable install of vllm (`vllm-dev`)

For iterating on a local `vllm` checkout, use the `vllm-dev` shell — it pulls
the full `vllm-xpu` closure (`torch-xpu`, `triton-xpu`, `vllm-xpu-kernels`,
runtime python deps), pre-sets `VLLM_TARGET_DEVICE=xpu`, and bakes in the BMG
single-card oneCCL env:

```bash
cd /path/to/vllm
nix develop /path/to/vllm-xpu-nix#vllm-dev
pip install -e . --no-build-isolation --no-deps
python -c 'import vllm; print(vllm.__version__)'
vllm serve <model> --enforce-eager
```

The `--no-deps` is load-bearing — without it pip would try to resolve torch from
PyPI, which would either pull in stock CPU torch (and fight the XPU build) or
fail outright. The shell already supplies torch-xpu via
`inputsFrom = [ vllm-xpu ]`.
