# Using on a NixOS server

The intended consumption story is "add an input + import the NixOS
module to your existing system flake" — not "vendor this whole flake
into your config".

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
across the closure. This flake is developed against `nixos-unstable`;
any consumer pin recent enough to expose `pkgs.intel-oneapi.base`
(added Feb 2026) is fine.

## 2. Import the NixOS module

```nix
# host config
{ pkgs, inputs, ... }: {
  imports = [ inputs.vllm-xpu-nix.nixosModules.default ];

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

`nixosModules.default` does two things in one import:

1. Adds `overlays.default` to `nixpkgs.overlays`, so `pkgs.vllm-xpu`,
   `pkgs.vllm-xpu-unstable`, `pkgs.torch-xpu`, etc. are visible without
   the consumer writing their own overlay file.
2. Imports the option module that defines `services.vllm-xpu`.

Each enabled `instances.<name>` becomes a `vllm-xpu-<name>.service`
systemd unit running under a shared `vllm` user with `render`/`video`
supplementary groups for `/dev/dri` access. Per-instance state lives
at `/var/lib/vllm-xpu/<name>` (used as `HF_HOME` and
`VLLM_CACHE_ROOT`). The default `cclEnv` is the BMG single-card
oneCCL configuration; override per host for multi-card setups.

### BYO-overlay variant

If you already manage overlays explicitly (e.g. you want to scope the
XPU package set to certain hosts only) use the bare option module and
apply the overlay yourself:

```nix
{
  imports = [ inputs.vllm-xpu-nix.nixosModules.vllm-xpu ];
  nixpkgs.overlays = [ inputs.vllm-xpu-nix.overlays.default ];
}
```

## 3. allowUnfree

`intel-oneapi-base-toolkit` is unfree. Scope the predicate narrowly
rather than flipping `allowUnfree = true` globally:

```nix
nixpkgs.config.allowUnfreePredicate = pkg:
  builtins.elem (pkgs.lib.getName pkg) [
    "intel-oneapi-base-toolkit"
  ];
```
