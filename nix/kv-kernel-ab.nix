{
  writeShellApplication,
  python3,
}:

let
  script = ../scripts/kv_kernel_ab.py;
in
writeShellApplication {
  name = "kv-kernel-ab";
  runtimeInputs = [ python3 ];
  text = ''
    exec ${python3}/bin/python ${script} "$@"
  '';
}
