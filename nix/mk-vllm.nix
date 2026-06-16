# mkVllm pairs a vllm source pin with the matching kernels build: the
# upstream stable variant gets vllm-xpu-kernels (vllm-project), the unstable
# variant gets vllm-xpu-kernels-unstable (jasonboukheir fork). Pre-release
# version stamp is fine — VLLM_VERSION_OVERRIDE in vllm-xpu.nix forwards to
# setuptools-scm's PRETEND_VERSION, so setuptools-scm doesn't need a .git in
# the unpacked store path.
#
# Like mkVllmXpuKernels, the result exposes `withAotDevices` / `withJIT` /
# `withAOT` passthrus that cascade through the kernels package. Also exposes
# `withTorchvision` and `withAudio` passthrus so consumers can opt into the
# +xpu torchvision wheel (for VL model families) or soundfile+pyav audio
# decoders (for /v1/audio transcription endpoints) without spelling out a
# full `.override`. All passthrus compose:
# `pkgs.vllm-xpu-unstable.withAOT |> .withTorchvision true |> .withAudio true`.
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
  mkVllm = {
    src,
    version,
    kernels,
    withTorchvision ? false,
    withAudio ? false,
  }: let
    base = pkgs.callPackage ./vllm-xpu.nix {
      intel-oneapi-base = intel-oneapi;
      inherit intel-pti torch-xpu triton-xpu flash-linear-attention;
      inherit python3Packages;
      vllm-xpu-kernels = kernels;
      inherit src version withTorchvision withAudio;
      inherit (pkgs) level-zero intel-graphics-compiler intel-compute-runtime;
    };
  in
    base.overrideAttrs (old: {
      passthru =
        (old.passthru or {})
        // {
          withAotDevices = ds:
            mkVllm {
              inherit src version withTorchvision withAudio;
              kernels =
                if kernels ? withAotDevices
                then kernels.withAotDevices ds
                else kernels;
            };
          withJIT = mkVllm {
            inherit src version withTorchvision withAudio;
            kernels =
              if kernels ? withJIT
              then kernels.withJIT
              else kernels;
          };
          withAOT = mkVllm {
            inherit src version withTorchvision withAudio;
            kernels =
              if kernels ? withAOT
              then kernels.withAOT
              else kernels;
          };
          withTorchvision = b:
            mkVllm {
              inherit src version kernels withAudio;
              withTorchvision = b;
            };
          withAudio = b:
            mkVllm {
              inherit src version kernels withTorchvision;
              withAudio = b;
            };
          # Rebuild the paired kernels package against a narrowed attn-kernel
          # set. See mkVllmXpuKernels.withKernelConfig.
          withKernelConfig = cfg:
            mkVllm {
              inherit src version withTorchvision withAudio;
              kernels =
                if kernels ? withKernelConfig
                then kernels.withKernelConfig cfg
                else kernels;
            };
        };
    });
in
  mkVllm
