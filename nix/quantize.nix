{
  writeShellApplication,
  python3Packages,
  auto-round-xpu,
}:

let
  autoRoundEnv = python3Packages.python.withPackages (_: [ auto-round-xpu ]);
in
writeShellApplication {
  name = "quantize";
  runtimeInputs = [ autoRoundEnv ];
  text = builtins.readFile ../scripts/quantize.sh;
}
