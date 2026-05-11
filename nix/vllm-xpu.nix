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
  level-zero,
  intel-graphics-compiler,
  intel-compute-runtime,
  # torchvision is only loaded by VL architectures (e.g. Qwen3.5/3.6's qwen3_vl
  # sibling, transformers' Qwen2VLImageProcessor); plain text models don't need
  # it. Default off to keep the closure lean — opt in with
  # `vllm-xpu.override { withTorchvision = true; }` or `vllm-xpu.withTorchvision`
  # passthru when serving VL models.
  withTorchvision ? false,
  # torchaudio is only loaded by a few niche audio architectures (FunASR,
  # MiDashengLM, MiMoV2Omni, Cohere ASR processor); Whisper, Qwen2-Audio, and
  # Voxtral don't import it. Same opt-in pattern as withTorchvision.
  withTorchaudio ? false,
}:

let
  syclHome = "${intel-oneapi-base}/compiler/latest";

  pythonDeps = with python3Packages; [
    # Core XPU stack
    torch-xpu
    triton-xpu
    vllm-xpu-kernels
    flash-linear-attention

    # Runtime deps — common.txt + xpu.txt. torchvision and torchaudio are
    # gated on the corresponding `with*` toggles (see argument comments).
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
  ]
  ++ lib.optional withTorchvision python3Packages.torchvision
  ++ lib.optional withTorchaudio python3Packages.torchaudio
  ++ python3Packages.fastapi.optional-dependencies.standard;
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

  # SYCL runtime dlopen()s libze_loader, libze_intel_gpu, and libigc at load
  # time. None are captured by libtorch_xpu's DT_NEEDED, so autoPatchelfHook
  # can't add them to RUNPATH. Inject them into bin/vllm's LD_LIBRARY_PATH so
  # the wrapper works on any host (NixOS or otherwise) without relying on the
  # host's /run/opengl-driver/lib being populated with the right packages.
  # Note: intel-compute-runtime ships libze_intel_gpu.so.1 in its `drivers`
  # output, separate from the default `out`.
  #
  # vLLM's model registry shells out via `subprocess.run([sys.executable, "-m",
  # "vllm.model_executor.models.registry"])` — that subprocess inherits env but
  # bypasses the inline `site.addsitedir(...)` block in `.vllm-wrapped`. Without
  # PYTHONPATH set, the subprocess can't find `vllm` and every model load fails
  # at engine init. Match the sitedirs the inline block already injects: vllm
  # itself plus the propagated deps.
  makeWrapperArgs = [
    "--prefix LD_LIBRARY_PATH : ${lib.makeLibraryPath [
      level-zero
      intel-graphics-compiler
      intel-compute-runtime
      intel-compute-runtime.drivers
    ]}"
    "--prefix PYTHONPATH : ${placeholder "out"}/${python3Packages.python.sitePackages}:${python3Packages.makePythonPath pythonDeps}"
    # Triton's Intel backend (triton/backends/intel/driver.py:find_sycl) needs
    # to locate libsycl.so + sycl headers at JIT-compile time. With no icpx on
    # PATH and no intel-sycl-rt wheel installed, it falls through to ONEAPI_ROOT
    # and looks under <root>/compiler/latest/{include,include/sycl,lib} — which
    # the nixpkgs toolkit layout already provides. --set-default leaves it
    # overridable for users pointing at a system install.
    "--set-default ONEAPI_ROOT ${intel-oneapi-base}"
    # Triton's spirv_utils JIT (triton/backends/intel/driver.py:_compute_compilation_options_lazy)
    # needs the level_zero SDK at JIT-compile time, separately from ONEAPI_ROOT:
    # the intel-oneapi-base toolkit ships SYCL headers under
    # compiler/latest/include but does NOT include `level_zero/ze_api.h` or
    # `libze_loader.so`. Triton looks up the level_zero SDK via
    # `LEVEL_ZERO_V1_SDK_PATH` (falling back to `ZE_PATH`, default `/usr/local`)
    # and appends `<root>/include` to its include search list. Point that at
    # the `level-zero` package already in the closure as an LD_LIBRARY_PATH
    # dep — without this, `#include <level_zero/ze_api.h>` in
    # `triton/backends/intel/include/sycl_functions.h` fails and EngineCore
    # exits during torch._inductor's first XPU compile pass.
    "--set-default LEVEL_ZERO_V1_SDK_PATH ${level-zero}"
    # The same JIT links `-lze_loader` (triton driver.py `libraries` list).
    # Triton's Linux library-search path does NOT include
    # `LEVEL_ZERO_V1_SDK_PATH/lib` (the `os.name == "nt"` branch only); GCC
    # consults `LIBRARY_PATH` at link time to extend `-L` search, so injecting
    # level-zero/lib here lets the linker resolve `libze_loader.so` without
    # patching the upstream triton driver.
    "--prefix LIBRARY_PATH : ${level-zero}/lib"
    # Triton's first-touch of the Intel XPU driver JIT-compiles
    # triton/backends/intel/driver.py's bundled `driver.c` into a `spirv_utils`
    # CPython extension via triton/runtime/build.py:_build. That helper consults
    # $CC (and $CXX), then falls back to clang/gcc on PATH. A bare systemd unit
    # ships only the systemd-default PATH, so without CC set the import path
    # under torch._inductor (triton_backend → XPUUtils.__init__) raises
    # `RuntimeError: Failed to find C compiler` and EngineCore exits during the
    # first compile pass after weight load. Pin CC/CXX from this package's own
    # stdenv.cc so every consumer (systemd unit, dev shell, manual run) gets a
    # working toolchain regardless of host PATH. --set-default keeps it
    # overridable via CC=... in the unit env or shell.
    "--set-default CC ${stdenv.cc}/bin/cc"
    "--set-default CXX ${stdenv.cc}/bin/c++"
  ];

  propagatedBuildInputs = pythonDeps;

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
    # xpu.txt. vllm_xpu_kernels and torch are nix-provided. torchvision and
    # torchaudio are gated on the `with*` toggles, so don't claim them as
    # always-required in the wheel metadata; consumers opt in via override.
    substituteInPlace requirements/xpu.txt \
      --replace-fail 'torch==2.11.0+xpu' 'torch' \
      --replace-fail 'torchaudio' '# torchaudio (opt-in via withTorchaudio)' \
      --replace-fail 'torchvision' '# torchvision (opt-in via withTorchvision)'
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
