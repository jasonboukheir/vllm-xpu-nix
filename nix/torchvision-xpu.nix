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
  # Stable XPU wheel paired with torch 2.13.0+xpu.
  version = "0.28.0+xpu";
  format = "wheel";

  src = fetchurl {
    url = "https://download-r2.pytorch.org/whl/xpu/torchvision-0.28.0%2Bxpu-cp312-cp312-manylinux_2_28_x86_64.whl";
    hash = "sha256-qln/qRxELe65BBnSG/kMRtdQDADCYQ0WX0USO7E6GAc=";
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
  ]
  ++ (with python3Packages; [
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
