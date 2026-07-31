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
      revision             = "d1fef185160f938fca00c3c664f21250dd544d63";
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
at `/var/lib/vllm-xpu/<name>`. Models default to the shared
`/var/cache/huggingface` `HF_HOME`; package-keyed compilation caches remain
per-instance. The default `cclEnv` is the BMG single-card oneCCL configuration;
override per host for multi-card setups.

## Hugging Face cache garbage collection

Set an immutable `revision` on each instance that downloads a Hugging Face
model. The module passes it to `vllm serve --revision` and contributes it to
`/etc/huggingface/cache-roots.json`. Because that manifest is part of the system
closure, every retained NixOS generation keeps its own cache roots.

After switching models and rebuilding, inspect stale cache revisions with:

```console
sudo hf-cache-gc
```

The command is dry-run-only unless `--delete` is explicit:

```console
sudo hf-cache-gc --delete
```

It scans every `/nix/var/nix/profiles/system-*-link` generation, unions their
cache roots, and removes only Hugging Face revisions absent from all retained
generations. An instance without a `revision` roots its entire model repository
for backward compatibility.

The underlying `nixosModules.hf-cache` module is independent of vLLM and also
manages datasets and Spaces. Other services can contribute roots declaratively:

```nix
services.hf-cache.roots = [
  {
    type = "dataset";
    repo = "openai/gsm8k";
    revision = "0123456789abcdef0123456789abcdef01234567";
    source = "evaluation-suite";
  }
  {
    type = "space";
    repo = "owner/demo";
    # Null revision conservatively roots every cached revision.
    revision = null;
  }
];
```

For migration safety, collection is refused while any retained system
generation predates cache manifests. Once the newly built generation is known
good, remove those older generations normally, then rerun the collector:

```console
sudo nix-env --profile /nix/var/nix/profiles/system --delete-generations old
sudo hf-cache-gc --delete
```

`nix-collect-garbage` still manages Nix store paths. `hf-cache-gc` is the
corresponding generation-aware collector for runtime Hugging Face downloads.

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
