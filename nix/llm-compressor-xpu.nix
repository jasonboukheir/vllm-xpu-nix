{
  lib,
  fetchurl,
  python3Packages,
  torch-xpu,
  auto-round-xpu,
}: let
  compressed-tensors = python3Packages.buildPythonPackage rec {
    pname = "compressed-tensors";
    version = "0.18.0";
    pyproject = true;
    src = fetchurl {
      url = "https://files.pythonhosted.org/packages/source/c/compressed-tensors/compressed_tensors-${version}.tar.gz";
      hash = "sha256-VrkClWN81AJ7UgQWdvbE5suTPhyhDxFLNF/d3gDBa4s=";
    };
    postPatch = ''substituteInPlace pyproject.toml --replace-fail "setuptools_scm==8.2.0" "setuptools_scm>=8.2.0"'';
    build-system = with python3Packages; [
      setuptools
      wheel
      setuptools-scm
    ];
    propagatedBuildInputs = with python3Packages; [
      torch-xpu
      transformers
      pydantic
      loguru
      psutil
    ];
    pythonRelaxDeps = ["torch"];
    dontUsePythonCatchConflicts = true;
    doCheck = false;
    pythonImportsCheck = ["compressed_tensors"];
  };
in
  python3Packages.buildPythonPackage rec {
    pname = "llmcompressor";
    version = "0.13.0";
    pyproject = true;
    src = fetchurl {
      url = "https://files.pythonhosted.org/packages/source/l/llmcompressor/llmcompressor-${version}.tar.gz";
      hash = "sha256-/n6gag+6r2kZXrUVg5hcbwY+s5JBPHeq6TqgJCX8D8Y=";
    };
    postPatch = ''substituteInPlace pyproject.toml --replace-fail "setuptools_scm==8.2.0" "setuptools_scm>=8.2.0"'';
    build-system = with python3Packages; [
      setuptools
      wheel
      setuptools-scm
    ];
    propagatedBuildInputs = with python3Packages; [
      torch-xpu
      transformers
      datasets
      auto-round-xpu
      accelerate
      nvidia-ml-py
      pillow
      compressed-tensors
      loguru
      pyyaml
      numpy
      requests
      tqdm
    ];
    pythonRelaxDeps = [
      "torch"
      "datasets"
      "compressed-tensors"
    ];
    dontUsePythonCatchConflicts = true;
    dontCheckRuntimeDeps = true;
    doCheck = false;
    pythonImportsCheck = [
      "llmcompressor"
      "llmcompressor.modifiers.autoround"
    ];
    meta = {
      description = "vLLM model compression with the Intel XPU Python substrate";
      homepage = "https://github.com/vllm-project/llm-compressor";
      license = lib.licenses.asl20;
      platforms = ["x86_64-linux"];
    };
  }
