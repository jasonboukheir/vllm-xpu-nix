# Stable vs unstable

The `vllm-xpu` and `vllm-xpu-kernels` outputs both come in stable and
`-unstable` variants. The unstable variants pin
`jasonboukheir/{vllm,vllm-xpu-kernels}` `main`, where in-flight patches
land before they make it upstream — they will rebase, may break, and
should be considered consumer-side opt-in.

Bumping a fork pin:

```bash
nix flake update vllm-xpu-unstable-src         # for vllm
nix flake update vllm-xpu-kernels-unstable-src # for kernels
git commit flake.lock -m "bump <input> pin to <short rev>"
```
