{
  writeShellApplication,
  python3Packages,
  auto-round-xpu,
  llm-compressor-xpu,
}: let
  autoRoundEnv = python3Packages.python.withPackages (_: [
    auto-round-xpu
    llm-compressor-xpu
    python3Packages.hf-xet
  ]);
in
  writeShellApplication {
    name = "quantize";
    runtimeInputs = [autoRoundEnv];
    text = ''
      export QUANTIZE_LLMCOMPRESSOR_SCRIPT=${../scripts/llmcompressor_quantize.py}
      exec ${autoRoundEnv}/bin/python ${../scripts/quantize.py} "$@"
    '';
  }
