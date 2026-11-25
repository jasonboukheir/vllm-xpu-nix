{
  lib,
  torch215 ? false,
  stdenv,
  fetchurl,
  unzip,
  autoPatchelfHook,
  intel-oneapi-base,
  level-zero,
}:

stdenv.mkDerivation rec {
  pname = "intel-pti";
  version = if torch215 then "1.1.0" else "1.0.1";

  src = fetchurl {
    url =
      if torch215 then
        "https://files.pythonhosted.org/packages/80/60/a94fd058dfed6c774ab376279a38c1e8822b9b7ab4a9c8e0f5b960f3a8e3/intel_pti-1.1.0-py2.py3-none-manylinux_2_28_x86_64.whl"
      else
        "https://files.pythonhosted.org/packages/46/d4/48737239235707852fef380005f53405740b08dc5f3700c72e5b43946452/intel_pti-${version}-py2.py3-none-manylinux_2_28_x86_64.whl";
    hash =
      if torch215 then
        "sha256-Yrs3QGjV5DI523p10Z5nPiG+aGAwkt5iOFIsrU6oyco="
      else
        "sha256-TnrZ6VXS2tjkHNldIuirHkdrri5uNR8Oi0ZT/nerLgw=";
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
