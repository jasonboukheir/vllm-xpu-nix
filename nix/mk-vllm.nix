# mkVllm pairs a vllm source pin with the matching kernels build: the
# upstream stable variant gets vllm-xpu-kernels (vllm-project), the unstable
# variant gets vllm-xpu-kernels-unstable (jasonboukheir fork). Pre-release
# version stamp is fine — VLLM_VERSION_OVERRIDE in vllm-xpu.nix forwards to
# setuptools-scm's PRETEND_VERSION, so setuptools-scm doesn't need a .git in
# the unpacked store path.
#
# The result is makeOverridable, so consumers tune it with the standard
# nixpkgs `.override` mechanism — all flags compose in one call:
#
#   pkgs.vllm-xpu-unstable.override {
#     withTorchvision = true;      # +xpu torchvision wheel (VL model families)
#     withAudio = true;            # soundfile+pyav (/v1/audio transcription)
#     aotDevices = [ "bmg" ];      # SYCL AOT targets for the paired kernels
#     kernelConfig = { ... };      # narrowed attn-kernel set, see mk-kernels.nix
#   }
#
# `aotDevices` / `kernelConfig` cascade into the paired kernels package via
# its own `.override`; null (the default) leaves the kernels' setting as-is.
#
# `withTorchaudio` is intentionally not exposed: no consumer in this
# project's stack needs torchaudio, so we don't carry the extra +xpu wheel
# pin. Re-introduce if a model family that depends on it lands.
{
  pkgs,
  intel-oneapi,
  intel-pti,
  torch-xpu,
  triton-xpu,
  flash-linear-attention,
  python3Packages,
}: let
  inherit (pkgs) lib;
  mkVllm = lib.makeOverridable ({
    src,
    version,
    kernels,
    withTorchvision ? false,
    withAudio ? false,
    aotDevices ? null,
    kernelConfig ? null,
  }: let
    kernelOverrides =
      lib.optionalAttrs (aotDevices != null) {inherit aotDevices;}
      // lib.optionalAttrs (kernelConfig != null) {inherit kernelConfig;};
    kernels' =
      if kernelOverrides != {} && kernels ? override
      then kernels.override kernelOverrides
      else kernels;
  in
    pkgs.callPackage ./vllm-xpu.nix {
      intel-oneapi-base = intel-oneapi;
      inherit intel-pti torch-xpu triton-xpu flash-linear-attention;
      inherit python3Packages;
      vllm-xpu-kernels = kernels';
      inherit src version withTorchvision withAudio;
      inherit (pkgs) level-zero intel-graphics-compiler intel-compute-runtime;
    });
in
  mkVllm
