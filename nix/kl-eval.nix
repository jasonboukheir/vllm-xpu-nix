{
  writeShellApplication,
  python3Packages,
  auto-round-xpu,
}:

let
  evalEnv = python3Packages.python.withPackages (ps: [
    auto-round-xpu
    ps.numpy
  ]);
  klEvalScript = ../scripts/kl_eval.py;
in
writeShellApplication {
  name = "kl-eval";
  runtimeInputs = [ evalEnv ];
  text = ''
    exec ${evalEnv}/bin/python ${klEvalScript} "$@"
  '';
}
