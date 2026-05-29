{
  lib,
  stdenv,
  fetchurl,
  unzip,
  autoPatchelfHook,
  intel-oneapi-base,
  level-zero,
}:

stdenv.mkDerivation rec {
  pname = "intel-pti";
  version = "0.16.0";

  src = fetchurl {
    url = "https://files.pythonhosted.org/packages/af/85/dee48118c530d9574f683f8cf3a7ad576a23f060a520335c9284ff6ba65b/intel_pti-${version}-py2.py3-none-manylinux_2_28_x86_64.whl";
    hash = "sha256-RS5ER5pXILP+GfAPoBveopWGoaJaVXJ4hXqwo9EceFI=";
  };

  nativeBuildInputs = [
    unzip
    autoPatchelfHook
  ];

  buildInputs = [
    stdenv.cc.cc.lib
    intel-oneapi-base
    level-zero
  ];

  unpackPhase = ''
    runHook preUnpack
    unzip -q $src -d wheel
    runHook postUnpack
  '';

  installPhase = ''
    runHook preInstall
    mkdir -p $out/lib $out/etc $out/share
    cp -a wheel/intel_pti-${version}.data/data/lib/* $out/lib/
    cp -a wheel/intel_pti-${version}.data/data/etc/* $out/etc/
    cp -a wheel/intel_pti-${version}.data/data/share/* $out/share/
    runHook postInstall
  '';

  dontStrip = true;

  meta = {
    description = "Intel Profiling Tools Interface (PTI) shared libraries";
    homepage = "https://github.com/intel/pti-gpu";
    license = with lib.licenses; [
      mit
      asl20
    ];
    platforms = [ "x86_64-linux" ];
  };
}
