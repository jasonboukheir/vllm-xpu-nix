{
  lib,
  fetchurl,
  python3Packages,
  autoPatchelfHook,
  stdenv,
  torch-xpu,
  zlib,
}:

python3Packages.buildPythonPackage rec {
  pname = "torchvision";
  # 2026-05-31 XPU nightly, intentionally paired with torch-xpu's
  # 2026-05-24 nightly pin (nearest available torchvision XPU nightly to
  # that date; the dates need not match exactly — torchvision only needs
  # the same torch 2.13-dev ABI). See the pin rationale in
  # nix/torch-xpu.nix.
  version = "0.28.0.dev20260531+xpu";
  format = "wheel";

  src = fetchurl {
    url = "https://download.pytorch.org/whl/nightly/xpu/torchvision-0.28.0.dev20260531%2Bxpu-cp312-cp312-manylinux_2_28_x86_64.whl";
    hash = "sha256-dar/x1j3PbvY3HXusOvn9OK7ytDft6OfzFd7VFyTp5E=";
  };

  nativeBuildInputs = [
    autoPatchelfHook
  ];

  buildInputs = [
    stdenv.cc.cc.lib
    torch-xpu
    zlib
  ];

  propagatedBuildInputs = [
    torch-xpu
  ] ++ (with python3Packages; [
    numpy
    pillow
  ]);

  # libtorch lives under torch-xpu's site-packages/torch/lib, which
  # autoPatchelfHook doesn't search by default. Pin RPATH there and tell the
  # missing-dep check to ignore those libs (they resolve via the appended
  # RPATH at dlopen time).
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

  pythonImportsCheck = [ "torchvision" ];

  meta = {
    description = "PyTorch vision (Intel XPU build, torchvision-${version} wheel)";
    homepage = "https://pytorch.org";
    license = lib.licenses.bsd3;
    platforms = [ "x86_64-linux" ];
  };
}
