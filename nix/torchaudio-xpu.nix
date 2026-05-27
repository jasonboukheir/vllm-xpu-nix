{
  lib,
  fetchurl,
  python3Packages,
  autoPatchelfHook,
  stdenv,
  torch-xpu,
}:

python3Packages.buildPythonPackage rec {
  pname = "torchaudio";
  # Match torch-xpu's 2026-05-21 nightly (first wheel built against oneAPI
  # 2026.0 ABI). See note in nix/torch-xpu.nix.
  version = "2.11.0.dev20260521+xpu";
  format = "wheel";

  src = fetchurl {
    url = "https://download.pytorch.org/whl/nightly/xpu/torchaudio-2.11.0.dev20260521%2Bxpu-cp312-cp312-manylinux_2_28_x86_64.whl";
    hash = "sha256-XONx0UqabHZbh24Dhve2FiyVJiI3DAusXXXHjzK+JUY=";
  };

  nativeBuildInputs = [
    autoPatchelfHook
  ];

  buildInputs = [
    stdenv.cc.cc.lib
    torch-xpu
  ];

  propagatedBuildInputs = [
    torch-xpu
  ];

  # Same story as torchvision-xpu: libtorch sits under torch-xpu's
  # site-packages/torch/lib, outside autoPatchelfHook's default search path.
  appendRunpaths = [
    "${torch-xpu}/${python3Packages.python.sitePackages}/torch/lib"
  ];
  autoPatchelfIgnoreMissingDeps = [
    "libc10.so"
    "libtorch.so"
    "libtorch_cpu.so"
    "libtorch_python.so"
  ];

  dontCheckRuntimeDeps = true;
  dontStrip = true;

  pythonImportsCheck = [ "torchaudio" ];

  meta = {
    description = "PyTorch audio (Intel XPU build, torchaudio-${version} wheel)";
    homepage = "https://pytorch.org";
    license = lib.licenses.bsd2;
    platforms = [ "x86_64-linux" ];
  };
}
