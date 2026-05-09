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
    vllm-xpu-kernels vllm-xpu-kernels-unstable
    vllm-xpu vllm-xpu-unstable;
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
    package = pkgs.vllm-xpu-unstable;
    instances.chat = {
      enable               = true;
      model                = "palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4";
      servedName           = "qwen3.6-35b-a3b";
      dtype                = "bfloat16";
      quantization         = "inc";
      kvCacheDtype         = "turboquant_k3v4_nc";
      maxModelLen          = 65536;
      maxNumSeqs           = 32;
      gpuMemoryUtilization = 0.85;
      enableXpuGraph       = true;
      cudagraphCaptureSizes = [ 1 4 ];
      reasoningParser      = "qwen3";
      enableAutoToolChoice = true;
      toolCallParser       = "qwen3_coder";
      languageModelOnly    = true;
    };
    instances.embedding = {
      enable               = true;
      port                 = 8001;
      runner               = "pooling";
      model                = "Qwen/Qwen3-Embedding-0.6B";
      servedName           = "qwen3-embedding-0.6b";
      maxModelLen          = 8192;
      maxNumSeqs           = 8;
      gpuMemoryUtilization = 0.07;
      enforceEager         = true;
    };
  };
}
```

Each enabled `instances.<name>` becomes a `vllm-xpu-<name>.service`
systemd unit running under a shared `vllm` user with `render`/`video`
supplementary groups for `/dev/dri` access. Per-instance state lives at
`/var/lib/vllm-xpu/<name>` (used as `HF_HOME` and `VLLM_CACHE_ROOT`).
The default `cclEnv` is the BMG single-card oneCCL configuration;
override per host for multi-card setups.

## 4. allowUnfree

`intel-oneapi-base-toolkit` is unfree. Scope the predicate narrowly
rather than flipping `allowUnfree = true` globally:

```nix
nixpkgs.config.allowUnfreePredicate = pkg:
  builtins.elem (pkgs.lib.getName pkg) [
    "intel-oneapi-base-toolkit"
  ];
```
