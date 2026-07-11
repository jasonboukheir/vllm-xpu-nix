# The python312Packages set with the XPU-specific swaps vllm needs layered
# on top of nixpkgs' python312Packages.
{
  pkgs,
  torch-xpu,
  torchvision-xpu,
}:
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
  }
