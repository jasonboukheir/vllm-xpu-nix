# Hardware prerequisites

The packaging side is solved (auto-patchelf, nix-store paths). The GPU
still has to be reachable through the kernel + ICD config so apps can
load Level Zero or OpenCL contexts.

## NixOS hosts

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
[#1](https://github.com/jasonboukheir/vllm-xpu-nix/issues/1) inherits
this automatically.

The service user needs to be in `render` (and ideally `video`) — the
NixOS module adds these as `SupplementaryGroups`. The host kernel must
have `i915` or `xe` loaded with a supported Intel GPU.

## Non-NixOS hosts

For Ubuntu / Fedora / Arch consumers running this flake directly:

- Install the distro's OpenCL ICD + Level Zero loader via the package
  manager.
- Wrap `vllm` so that `OCL_ICD_VENDORS=…/etc/OpenCL/vendors` and
  `LD_LIBRARY_PATH=…/lib` are exported before exec.
- The nixpkgs `intel-compute-runtime` (currently 26.14) is newer than
  what the official `intel/vllm:0.17.0-xpu` container ships (25.48).
  The driver/runtime ABI is generally backward-compatible, but flag this
  if you're on an older host kernel.

## Verification

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
