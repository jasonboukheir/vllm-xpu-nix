# Continuous integration

`.github/workflows/build.yml` runs two jobs:

1. **flake-check** — `nix flake check --no-build` on `ubuntu-latest`.
   Eval-only sanity, runs on every push and PR.
2. **build** — `nix build .#<each-package>` across a matrix of all
   substrate outputs, on a self-hosted runner. Gated behind the repo
   variable `SELF_HOSTED_RUNNER_AVAILABLE=true` so the workflow stays
   green for forks that don't have a runner attached.

The matrix needs a self-hosted runner because each SYCL-TLA kernel
icpx process holds ~40 GiB of resident memory during template
instantiation (and `MAX_JOBS=2` means ~80 GiB working set on the full
target). GitHub-hosted `ubuntu-latest` caps at 16 GiB RAM and ~14 GiB
disk; the torch+xpu wheel alone is 793 MiB.

To enable the matrix build:

1. Provision a self-hosted runner with labels
   `[self-hosted, x86_64-linux, vllm-xpu-nix]`, ≥96 GiB RAM, ≥150 GiB
   disk, and a Nix install with flakes enabled.
2. Configure a binary cache (Cachix or a self-hosted `nix serve`) on
   the runner host as a substituter, otherwise every PR rebuilds the
   90-minute kernels closure from scratch.
3. Set the repo variable `SELF_HOSTED_RUNNER_AVAILABLE=true` to
   un-gate the `build` job.
