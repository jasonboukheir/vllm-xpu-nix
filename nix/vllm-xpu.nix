{
  lib,
  src,
  version,
  python3Packages,
  cmake,
  ninja,
  which,
  stdenv,
  intel-oneapi-base,
  intel-pti,
  oneccl-bmg,
  torch-xpu,
  triton-xpu,
  flash-linear-attention,
  vllm-xpu-kernels,
}:

let
  syclHome = "${intel-oneapi-base}/compiler/latest";
in
python3Packages.buildPythonPackage {
  pname = "vllm-xpu";
  inherit version src;
  format = "pyproject";

  nativeBuildInputs = [
    cmake
    ninja
    which
    python3Packages.setuptools
    python3Packages.setuptools-scm
    python3Packages.wheel
    python3Packages.packaging
    python3Packages.jinja2
    python3Packages.cmake
    python3Packages.ninja
  ];

  buildInputs = [
    stdenv.cc.cc.lib
    intel-oneapi-base
    intel-pti
    oneccl-bmg
  ];

  propagatedBuildInputs = with python3Packages; [
    # Core XPU stack
    torch-xpu
    triton-xpu
    vllm-xpu-kernels
    flash-linear-attention

    # Runtime deps — common.txt + xpu.txt (minus torchaudio/torchvision which
    # would drag CUDA-linked torch into the env).
    aiohttp
    anthropic
    blake3
    cachetools
    cbor2
    cloudpickle
    compressed-tensors
    datasets
    depyf
    diskcache
    einops
    fastapi
    filelock
    gguf
    ijson
    jinja2
    lark
    llguidance
    lm-format-enforcer
    mcp
    mistral-common
    pycountry
    model-hosting-container-standards
    msgspec
    numba
    numpy
    openai
    openai-harmony
    opencv-python-headless
    opentelemetry-api
    opentelemetry-exporter-otlp
    opentelemetry-sdk
    outlines-core
    partial-json-parser
    pillow
    prometheus-client
    prometheus-fastapi-instrumentator
    protobuf
    psutil
    py-cpuinfo
    pybase64
    pydantic
    python-json-logger
    pyyaml
    pyzmq
    ray
    regex
    requests
    sentencepiece
    setproctitle
    six
    tiktoken
    tokenizers
    tqdm
    transformers
    typing-extensions
    watchfiles
    xgrammar
  ] ++ fastapi.optional-dependencies.standard;

  # vLLM's XPU build path produces no native ext_modules
  # (_build_custom_ops()=False on XPU; see setup.py:845-846). The whole
  # derivation is effectively a pure-Python install — torch/SYCL env is
  # only needed for `import torch` during setup.py's auto-detect.
  dontUseCmakeConfigure = true;

  postPatch = ''
    # Drop the strict torch and setuptools pins from build-system.requires;
    # nix provides them via build-system.
    substituteInPlace pyproject.toml \
      --replace-fail 'torch == 2.11.0' 'torch' \
      --replace-fail 'setuptools>=77.0.3,<81.0.0' 'setuptools'

    # Strip the wheel-URL pin and the torch+xpu local-version pin from
    # xpu.txt. vllm_xpu_kernels and torch are provided by the nix store.
    # Also drop torchaudio/torchvision: nixpkgs builds them against its own
    # torch (CUDA-flavored), which would conflict with torch-xpu in the
    # python env. vLLM core doesn't need them; downstream apps that do can
    # add them explicitly.
    substituteInPlace requirements/xpu.txt \
      --replace-fail 'torch==2.11.0+xpu' 'torch' \
      --replace-fail 'torchaudio' '# torchaudio (provided by app, not vllm)' \
      --replace-fail 'torchvision' '# torchvision (provided by app, not vllm)'
    sed -i '/^vllm_xpu_kernels @ /d' requirements/xpu.txt
  '';

  preBuild = ''
    # icpx + SYCL toolchain — only needed because setup.py imports torch,
    # which on XPU pulls in the SYCL runtime. No kernel compilation runs.
    export SYCL_HOME=${syclHome}
    export CMPLR_ROOT=${syclHome}
    export PATH=${syclHome}/bin:$PATH

    export VLLM_TARGET_DEVICE=xpu
    export VLLM_VERSION_OVERRIDE=${version}

    export MAX_JOBS=''${NIX_BUILD_CORES:-1}
  '';

  pythonRelaxDeps = true;

  # nixpkgs python deps (e.g. transformers, datasets) propagate the stock
  # CUDA-flavored torch, which conflicts in the closure with torch-xpu.
  # sys.path ordering ensures torch-xpu wins at import time.
  dontUsePythonCatchConflicts = true;

  dontCheckRuntimeDeps = true;
  dontStrip = true;

  pythonImportsCheck = [ "vllm" ];

  meta = {
    description = "vLLM with VLLM_TARGET_DEVICE=xpu (Intel Arc / Battlemage / PVC)";
    homepage = "https://github.com/vllm-project/vllm";
    license = lib.licenses.asl20;
    platforms = [ "x86_64-linux" ];
    mainProgram = "vllm";
  };
}
