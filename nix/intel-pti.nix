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
  version = "1.0.1";

  src = fetchurl {
    url = "https://files.pythonhosted.org/packages/46/d4/48737239235707852fef380005f53405740b08dc5f3700c72e5b43946452/intel_pti-${version}-py2.py3-none-manylinux_2_28_x86_64.whl";
    hash = "sha256-TnrZ6VXS2tjkHNldIuirHkdrri5uNR8Oi0ZT/nerLgw=";
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
