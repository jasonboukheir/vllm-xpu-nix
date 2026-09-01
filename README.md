# vllm-xpu-nix

[![build](https://github.com/jasonboukheir/vllm-xpu-nix/actions/workflows/build.yml/badge.svg)](https://github.com/jasonboukheir/vllm-xpu-nix/actions/workflows/build.yml)

Nix-native Intel XPU substrate for [vLLM](https://github.com/vllm-project/vllm):
`torch+xpu`, `triton-xpu`, `vllm-xpu-kernels`, and (in progress) `vllm` itself,
all packaged as nix-store derivations rather than baked into a container image.

The aim is to let a NixOS host run `vllm serve` as a native systemd unit, with
the SYCL toolchain, Level Zero loader, oneCCL, MKL, and the AOT-compiled
SYCL-TLA kernel `.so`s all referenced from `/nix/store`. No `intel/vllm`
container, no host-managed `~/.local/lib/python*/site-packages`, no
`/opt/intel/oneapi` write directories.

## Documentation

- [Build something locally](docs/build.md)
- [Using on a NixOS server (overlay pattern)](docs/nixos-overlay.md)
- [Hardware prerequisites](docs/hardware.md)
- [Iterating against a local checkout](docs/iterate.md)
- [Quantize / eval](docs/quantize.md)
- [Stable vs unstable](docs/stable-vs-unstable.md)
- [KVarN XPU beta](docs/kvarn-beta.md)
- [Historical KVarN acceptance runbook](docs/kvarn-brutus-runbook.md)

The non-trivial roadmap lives in the
[issue tracker](https://github.com/jasonboukheir/vllm-xpu-nix/issues).

## License

Apache-2.0 — see [LICENSE.md](LICENSE.md). Upstream components retain their own
licenses (PyTorch BSD-3, Triton MIT, oneAPI under Intel's End User License
Agreement); the LICENSE file lists the third-party notices.
