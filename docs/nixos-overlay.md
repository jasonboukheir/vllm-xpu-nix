# Using on a NixOS server (overlay pattern)

The intended consumption story is "add an input + overlay + import the
NixOS module to your existing system flake" — not "vendor this whole
flake into your config".

## 1. Add the input

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

## 2. Wire packages via an overlay

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

## 3. Import the NixOS module

```nix
# host config
{ pkgs, inputs, ... }: {
  imports = [ inputs.vllm-xpu-nix.nixosModules.vllm-xpu ];

  services.vllm-xpu = {
    enable       = true;
    package      = pkgs.vllm-xpu-unstable;
    model        = "palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4";
    servedName   = "qwen3.6-35b-a3b";
    dtype        = "bfloat16";
    quantization = "inc";
  };
}
```

The module brings in a `vllm-xpu` systemd unit that runs `vllm serve`
under a dedicated `vllm` user, with `render`/`video` supplementary groups
for `/dev/dri` access and `HF_HOME=/var/lib/vllm` for the model cache.
The default `cclEnv` is the BMG single-card oneCCL configuration; override
per host if you have a multi-card setup.

## 4. allowUnfree

`intel-oneapi-base-toolkit` is unfree. Scope the predicate narrowly
rather than flipping `allowUnfree = true` globally:

```nix
nixpkgs.config.allowUnfreePredicate = pkg:
  builtins.elem (pkgs.lib.getName pkg) [
    "intel-oneapi-base-toolkit"
  ];
```
