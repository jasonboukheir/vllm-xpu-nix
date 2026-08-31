# Stable vs unstable

The `vllm-xpu` and `vllm-xpu-kernels` outputs both come in stable and
`-unstable` variants. The unstable variants pin
`jasonboukheir/{vllm,vllm-xpu-kernels}`. On this experimental release
branch, both inputs follow the coordinated `releases/xpu-v1.4-kvarn`
branches so the validated Kvarn source pair cannot drift independently.
They remain consumer-side opt-in.

Bumping a fork pin:

```bash
nix flake update vllm-xpu-unstable-src         # for vllm
nix flake update vllm-xpu-kernels-unstable-src # for kernels
git commit flake.lock -m "bump <input> pin to <short rev>"
```
