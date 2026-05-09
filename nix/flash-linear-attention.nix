{
  lib,
  fetchurl,
  python3Packages,
  torch-xpu,
  triton-xpu,
}:

let
  fla-core = python3Packages.buildPythonPackage rec {
    pname = "fla-core";
    version = "0.5.0";
    pyproject = true;

    src = fetchurl {
      url = "https://files.pythonhosted.org/packages/03/14/2aabd37839b9f3c6a67fbc5678f906d04d0c242c603ac234eefe02df99a6/fla_core-${version}.tar.gz";
      hash = "sha256-R23ZRxFwKvgcxIJwENkgn2BT2M3OrI5D08hJcHHweoE=";
    };

    build-system = with python3Packages; [
      setuptools
      wheel
    ];

    propagatedBuildInputs = [
      torch-xpu
      triton-xpu
      python3Packages.einops
    ];

    pythonImportsCheck = [ "fla" ];

    meta = {
      description = "Core implementation of Flash Linear Attention (Triton kernels)";
      homepage = "https://github.com/fla-org/fla-core";
      license = lib.licenses.mit;
      platforms = [ "x86_64-linux" ];
    };
  };
in
python3Packages.buildPythonPackage rec {
  pname = "flash-linear-attention";
  version = "0.5.0";
  pyproject = true;

  src = fetchurl {
    url = "https://files.pythonhosted.org/packages/79/5c/1db76cc829c951117a3112f306d50333bd71399d2e35807fe7c99ffc2007/flash_linear_attention-${version}.tar.gz";
    hash = "sha256-IreJpH8Hc4tDguzfd117tA4NgDxGfDT44uzWodx4CTg=";
  };

  build-system = with python3Packages; [
    setuptools
    wheel
  ];

  propagatedBuildInputs = [
    fla-core
    python3Packages.transformers
  ];

  passthru = { inherit fla-core; };

  pythonImportsCheck = [ "fla" ];

  meta = {
    description = "Flash Linear Attention — Triton kernels for gated-delta-rule and friends";
    homepage = "https://github.com/fla-org/flash-linear-attention";
    license = lib.licenses.mit;
    platforms = [ "x86_64-linux" ];
  };
}
