{
  writeShellApplication,
  python3,
  vllm-xpu-chat,
}:

let
  script = ../scripts/kv_kernel_ab.py;
in
writeShellApplication {
  name = "kv-kernel-ab";
  runtimeInputs = [ python3 vllm-xpu-chat ];
  text = ''
    exec ${python3}/bin/python ${script} \
      --vllm ${vllm-xpu-chat}/bin/vllm "$@"
  '';
}
