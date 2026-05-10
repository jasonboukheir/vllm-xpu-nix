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
  version = "2.11.0+xpu";
  format = "wheel";

  src = fetchurl {
    url = "https://download.pytorch.org/whl/xpu/torchaudio-2.11.0%2Bxpu-cp312-cp312-manylinux_2_28_x86_64.whl";
    hash = "sha256-VE99BDqIzK6mzzTUP5FFoPYLFOkWPAIYQduiaQaciHk=";
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
    description = "PyTorch audio (Intel XPU build, torchaudio-2.11.0+xpu wheel)";
    homepage = "https://pytorch.org";
    license = lib.licenses.bsd2;
    platforms = [ "x86_64-linux" ];
  };
}
