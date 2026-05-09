{
  lib,
  fetchurl,
  python3Packages,
  torch-xpu,
  triton-xpu,
  flash-linear-attention,
}:

# Auto-round runs calibration forward passes on Intel GPUs through
# torch+xpu and emits GPTQ/AWQ/etc-formatted weights. Mirrors the recipe
# from intellm/nix/auto-round/Containerfile (currently a podman wrapper
# off intel/vllm:0.17.0-xpu) — but cuts out the container by leaning on
# torch-xpu + triton-xpu + flash-linear-attention from this flake.
#
# Skipped vs the Containerfile:
# - causal-conv1d: CUDA-only (its setup.py looks up nvcc bare_metal_version
#   and crashes on XPU). qwen3_5_moe falls back to F.silu(self.conv1d(...))
#   when causal_conv1d_fn is None, so correctness is preserved; expect a
#   "fast path is not available" warning at startup.
# - IPEX: only loaded by auto_round_extension/ipex/qlinear_ipex_{gptq,awq}.py
#   for IPEX-runtime export. GPTQ export (the format vLLM's `inc` quant path
#   consumes) uses native torch.xpu and never imports IPEX.

python3Packages.buildPythonPackage rec {
  pname = "auto-round";
  version = "0.12.3";
  pyproject = true;

  src = fetchurl {
    url = "https://files.pythonhosted.org/packages/d6/8e/2723a5ccee074c6e823e24ebb198cbcfebce86097d6076e3a5ecf4d8b4e9/auto_round-${version}.tar.gz";
    hash = "sha256-vDNw98PavK411/UT4L6joe5gu5v+B2b/JeTMcf3xjzE=";
  };

  build-system = with python3Packages; [
    setuptools
    wheel
  ];

  propagatedBuildInputs = with python3Packages; [
    torch-xpu
    triton-xpu
    flash-linear-attention
    accelerate
    datasets
    numpy
    py-cpuinfo
    pydantic
    safetensors
    tqdm
    transformers
    huggingface-hub
  ];

  pythonRelaxDeps = true;
  dontCheckRuntimeDeps = true;

  # nixpkgs `accelerate` propagates the stock CUDA-flavored torch, which
  # conflicts with torch-xpu in the closure. The nix python catch-conflicts
  # hook flags this even though sys.path ordering ensures torch-xpu wins at
  # import time. Same pattern used in vllm-xpu.
  dontUsePythonCatchConflicts = true;

  pythonImportsCheck = [ "auto_round" ];

  meta = {
    description = "AutoRound: low-bit LLM/VLM quantization (sign-gradient descent)";
    homepage = "https://github.com/intel/auto-round";
    license = lib.licenses.asl20;
    platforms = [ "x86_64-linux" ];
    mainProgram = "auto-round";
  };
}
