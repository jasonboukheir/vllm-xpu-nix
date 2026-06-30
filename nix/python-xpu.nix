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
    # compressed-tensors propagates stock `torch`, leaving a second (unused)
    # torch in the closure. Harmless here: vllm-xpu wires its runtime env via a
    # PYTHONPATH wrapper (makePythonPath, see vllm-xpu.nix), not a
    # python.withPackages buildEnv, and torch-xpu sorts first on PYTHONPATH so
    # it wins at `import torch`. So keep stock torch and only skip the one
    # flaky test: test_quantization_enabled_disabled calibrates W8A8 activation
    # quant from a single random sample whose min/max can collapse the scale to
    # ~0, yielding NaNs that trip `torch.all(a == b)` (fails on stock torch too).
    compressed-tensors =
      pkgs.python312Packages.compressed-tensors.overrideAttrs (oldAttrs: {
        disabledTests =
          (oldAttrs.disabledTests or [])
          ++ ["test_quantization_enabled_disabled"];
      });
    # prometheus-fastapi-instrumentator 7.1.0 pins starlette<1.0.0, but nixpkgs
    # now ships starlette 1.1.0, tripping pythonRuntimeDepsCheckHook. starlette
    # 1.x keeps the middleware/request APIs this package uses, so relax the pin.
    # TODO: drop once upstream loosens the bound
    # (https://github.com/trallnag/prometheus-fastapi-instrumentator/issues).
    prometheus-fastapi-instrumentator =
      pkgs.python312Packages.prometheus-fastapi-instrumentator.overridePythonAttrs (oldAttrs: {
        pythonRelaxDeps = (oldAttrs.pythonRelaxDeps or []) ++ ["starlette"];
      });
  }
