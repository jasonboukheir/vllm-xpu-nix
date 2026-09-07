# Stable vs unstable

The `vllm-xpu` and `vllm-xpu-kernels` outputs both come in stable and
`-unstable` variants. The unstable variants use the project forks and remain
consumer-side opt-in.

See `flake.nix` for source URLs and release refs, and `flake.lock` for resolved
revisions. Update the vLLM and kernel sources as a compatible pair. To change a
tag-pinned release, change its input ref before updating the lock file.

```bash
nix flake update vllm-xpu-unstable-src vllm-xpu-kernels-unstable-src
```
