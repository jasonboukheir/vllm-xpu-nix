{
  lib,
  src,
  version,
  python3Packages,
  cmake,
  ninja,
  openssl,
  which,
  stdenv,
  intel-oneapi-base,
  intel-pti,
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
  # `vllm-xpu.override { withTorchvision = true; }` when serving VL models.
  withTorchvision ? false,
  # Audio decoders for /v1/audio/transcriptions and any model that calls
  # vllm/multimodal/media/audio.py:load_audio (Whisper, Qwen2-Audio, Voxtral).
  # soundfile (libsndfile) handles wav/flac/ogg; pyav (FFmpeg) handles the rest
  # — mp3/mp4/webm/m4a/aac/opus. The loader tries soundfile first and falls
  # back to pyav, so OpenAI-style clients sending arbitrary container formats
  # need pyav present. Off by default to keep libsndfile and ffmpeg out of the
  # closure for text-only deployments.
  withAudio ? false,
}:

let
  syclHome = "${intel-oneapi-base}/compiler/latest";

  # triton-xpu's spirv_utils init_devices() (driver.c:383) hard-requires
  # *either* an OpenCL SYCL platform *or* `ocloc` on PATH; without either it
  # returns NULL from the C extension without setting a Python exception, so
  # CPython segfaults dereferencing the result. Triton's check uses the literal
  # filename "ocloc" (driver.c:369), so expose intel-compute-runtime's bin
  # directory in the wrapper PATH. Fires during the first triton-XPU
  # driver init — e.g. importing `vllm.model_executor.layers.fla.ops` (any
  # GDN-attention model: Qwen3-Next, Qwen3.5/3.6).

  pythonDeps = with python3Packages; [
    # Core XPU stack
    torch-xpu
    triton-xpu
    vllm-xpu-kernels
    flash-linear-attention

    # Runtime deps — common.txt + xpu.txt. torchvision is gated on
    # `withTorchvision` (see argument comment). torchaudio is omitted: no
    # torch-2.12-compatible +xpu audio wheel is published yet (stable caps
    # at torchaudio 2.9.1+xpu, ABI-bound to torch 2.9).
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
    ijson
    jinja2
    jsonschema
    lark
    llguidance
    lm-format-enforcer
    mcp
    mistral-common
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
    opentelemetry-semantic-conventions-ai
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
    safetensors
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
  ++ lib.optionals withAudio [ python3Packages.soundfile python3Packages.av ]
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
    # torch.utils.cpp_extension probes SYCL_HOME directly before falling back
    # to the intel-sycl-rt wheel. The toolkit is already in this package's
    # closure, so expose its compiler root explicitly and avoid a misleading
    # "intel-sycl-rt package ... is not installed" warning at every process
    # start. CMPLR_ROOT is the equivalent oneAPI compiler convention.
    "--set-default SYCL_HOME ${syclHome}"
    "--set-default CMPLR_ROOT ${syclHome}"
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
    # Inductor invokes openssl when hashing generated compile artifacts.
    "--prefix PATH : ${openssl}/bin"
    "--prefix PATH : ${intel-compute-runtime}/bin"
  ];

  propagatedBuildInputs = pythonDeps;

  # vLLM's XPU build path produces no native ext_modules
  # (_build_custom_ops()=False on XPU; see setup.py:845-846). The whole
  # derivation is effectively a pure-Python install — torch/SYCL env is
  # only needed for `import torch` during setup.py's auto-detect.
  dontUseCmakeConfigure = true;

  postPatch = ''
    # Drop the strict torch and setuptools pins from build-system.requires;
    # nix provides them via build-system. Also drop setuptools-rust from
    # build-system since the Rust frontend (PR #40848) is stripped below.
    # Version-agnostic torch substitution so an upstream pin bump (e.g.
    # 2.12 -> 2.13) does not silently no-op and leave the strict pin in
    # place.
    sed -i -E 's/torch == [0-9.]+/torch/' pyproject.toml
    substituteInPlace pyproject.toml \
      --replace-fail 'setuptools>=77.0.3,<81.0.0' 'setuptools' \
      --replace-fail '"setuptools-rust>=1.9.0",' ""

    # Strip the wheel-URL pin and the torch version pin from xpu.txt.
    # vllm_xpu_kernels and torch are nix-provided. torchvision is gated on
    # the `withTorchvision` toggle; torchaudio was dropped (no
    # torch-2.12-compatible +xpu wheel published) so strip it unconditionally.
    # The `+xpu` local-version suffix is optional: upstream now pins plain
    # `torch==2.12.0`, but keep matching the older `torch==X.Y.Z+xpu` form
    # so a future re-suffix does not silently no-op.
    sed -i -E 's/^torch==[0-9.]+(\+xpu)?/torch/' requirements/xpu.txt
    substituteInPlace requirements/xpu.txt \
      --replace-fail 'torchaudio' '# torchaudio (no torch-2.12 +xpu wheel published)' \
      --replace-fail 'torchvision' '# torchvision (opt-in via withTorchvision)'
    # These optional integrations are not part of the vLLM runtime closure:
    # torchcodec is only needed by its video-decoding backend, while AutoRound
    # is provided separately by this flake. Match the package name rather than
    # an exact version pin so routine upstream pin bumps do not break patchPhase.
    sed -i -E \
      -e '/^torchcodec([[:space:]<>=!~].*)?$/d' \
      -e '/^auto_round_lib([[:space:]<>=!~].*)?$/d' \
      requirements/xpu.txt
    sed -i '/^vllm_xpu_kernels @ /d' requirements/xpu.txt

    # Strip the Rust frontend (PR #40848: vllm-rs). setuptools-rust would
    # demand rustPlatform.cargoSetupHook with a vendored Cargo.lock; the Rust
    # CLI is unused by the Python inference API, so neutralize the imports +
    # extension wiring instead of vendoring the dep tree. Upstream moved the
    # `setuptools_rust import Binding, RustExtension` into tools/build_rust.py
    # (which setup.py loads as a module), so both files need patching. The
    # build_rust.py stub keeps rust_extensions()/rust_py_extension_module_names()
    # callable but returning no PyO3 modules, so the build skips the Rust step.
    substituteInPlace setup.py \
      --replace-fail 'from setuptools_rust.build import build_rust' \
        'build_rust = type("build_rust", (object,), {"run": lambda self: None})' \
      --replace-fail 'rust_extensions=rust_extensions,' ""
    substituteInPlace tools/build_rust.py \
      --replace-fail 'from setuptools_rust import Binding, RustExtension' \
        'Binding = type("Binding", (), {"Exec": object(), "PyO3": object()}); RustExtension = lambda *a, **kw: type("RustExtension", (), {"binding": kw.get("binding"), "target": {}})()'
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
