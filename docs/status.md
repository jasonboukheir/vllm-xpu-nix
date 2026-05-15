# Status

| Output | What it builds | Source |
| --- | --- | --- |
| `intel-oneapi` | DPC++ compiler + MKL + DPL (oneAPI base toolkit, narrowed) | nixpkgs `intel-oneapi.base` |
| `intel-pti` | Intel Profiling Tools Interface | `nix/intel-pti.nix` |
| `oneccl-bmg` | oneCCL with the single-card BMG runtime workarounds | `nix/oneccl-bmg.nix` |
| `torch-xpu` | `torch==2.11.0+xpu` wheel, auto-patchelf'd against the nix-store oneAPI closure | `nix/torch-xpu.nix` |
| `triton-xpu` | `triton-xpu==3.7.0` (Intel's Triton port) | `nix/triton-xpu.nix` |
| `vllm-xpu-kernels` | upstream `vllm-project/vllm-xpu-kernels` with split-kernel-libs patch | `nix/vllm-xpu-kernels.nix` |
| `vllm-xpu-kernels-unstable` | same, but pinning the `jasonboukheir/vllm-xpu-kernels` fork's `main` | same factory |
| `attn-kernels-xe-2`, `gdn-attn-kernels-xe-2`, `mqa-logits-kernels-xe-2`, `grouped-gemm-xe-{2,default}` | individual SYCL-TLA `.so`s, built in parallel and stitched into `vllm-xpu-kernels` | `nix/vllm-xpu-lib.nix` |
| `vllm-xpu` | source build of vLLM with `VLLM_TARGET_DEVICE=xpu`, linked against the above | `nix/vllm-xpu.nix` |
| `vllm-xpu-unstable` | same, but pinning the `jasonboukheir/vllm` fork's `main` | same factory |
| `flash-linear-attention` | qwen3_5_moe gated-delta-rule kernels (Triton, fla-org) | `nix/flash-linear-attention.nix` |
| `auto-round-xpu` | AutoRound + transformers ≥5.2.0 + FLA, no IPEX, no causal-conv1d | `nix/auto-round-xpu.nix` |
| `quantize`, `kl-eval` | flake apps wrapping the AutoRound recipe + KL-eval workflow | `nix/{quantize,kl-eval}.nix` |
| `nixosModules.vllm-xpu` | systemd unit running `vllm serve` natively (no container) | `nix/modules/vllm-xpu.nix` |
