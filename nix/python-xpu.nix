# The python312Packages set with the XPU-specific swaps vllm needs layered
# on top of nixpkgs' python312Packages.
{
  pkgs,
  torch-xpu,
  torchvision-xpu,
}: let
  # nixpkgs ships mistral-common 1.8.8; vllm 0.20.x imports NamedToolChoice
  # from mistral_common.protocol.instruct.tool_calls, which only exists from
  # 1.11+. Bump to 1.11.2 (vllm's pin). overridePythonAttrs preserves the
  # nixpkgs build recipe and just swaps version+src, keeping the package
  # self-contained against whatever nixpkgs revision is in flake.lock.
  mistral-common-1_11 = pkgs.python312Packages.mistral-common.overridePythonAttrs (oldAttrs: rec {
    version = "1.11.2";
    src = pkgs.fetchFromGitHub {
      owner = "mistralai";
      repo = "mistral-common";
      rev = "v${version}";
      hash = "sha256-EXdZcBR61GNye8LqwIqRO8lP1lK6fqPJufWFO9XkkYQ=";
    };
    pythonRelaxDeps = (oldAttrs.pythonRelaxDeps or []) ++ ["numpy"];
    doCheck = false;
  });
in
  pkgs.python312Packages
  // {
    # accelerate's nixpkgs definition propagates stock `torch`, which
    # collides with torch-xpu on functorch/*.pyc when both end up in a
    # python.withPackages buildEnv. Rebuild accelerate with its `torch` arg
    # pointing at torch-xpu so the buildEnv merge sees only one torch.
    # Surgical override (vs. python set-wide packageOverrides) avoids
    # re-evaluating unrelated python packages whose passthru references
    # attrs torch-xpu doesn't carry.
    accelerate = pkgs.python312Packages.accelerate.override {
      torch = torch-xpu;
    };
    mistral-common = mistral-common-1_11;
    torchvision = torchvision-xpu;
    # vllm's compressed-tensors dep otherwise propagates stock `torch` (same
    # functorch/*.pyc buildEnv collision as accelerate, and a second torch in
    # the closure). Point it at torch-xpu so the whole stack shares one torch.
    compressed-tensors =
      (pkgs.python312Packages.compressed-tensors.override {
        torch = torch-xpu;
      }).overrideAttrs (oldAttrs: {
        # This quantization round-trip produces NaNs, so the test's
        # `torch.all(a == b)` fails (NaN != NaN) regardless of torch build.
        # TODO: drop once upstream nixpkgs disables it.
        disabledTests =
          (oldAttrs.disabledTests or [])
          ++ ["test_quantization_enabled_disabled"];
      });
  }
