# The python312Packages set with the XPU-specific swaps vllm needs layered
# on top of nixpkgs' python312Packages.
{
  pkgs,
  torch-xpu,
  torchvision-xpu,
}:
  # Recompute the scoped Python package set with the XPU Torch packages at
  # its fixed point. A shallow `python312Packages // { torch = ...; }` only
  # changes direct lookups: packages already instantiated by nixpkgs retain
  # stock CPU torch in their propagated dependencies, causing an unnecessary
  # multi-hour PyTorch source build and two Torch implementations in vLLM's
  # closure.
  pkgs.python312Packages.overrideScope (_pyFinal: pyPrev: {
    torch = torch-xpu;
    torchvision = torchvision-xpu;

    # PyTorch 2.13 Inductor invokes openssl while these packages exercise
    # torch.compile in their test suites. Keep the tool local to the consumers
    # so Torch's store path—and the expensive AOT kernel cache—stays stable.
    depyf = pyPrev.depyf.overridePythonAttrs (oldAttrs: {
      nativeCheckInputs =
        (oldAttrs.nativeCheckInputs or [])
        ++ [pkgs.openssl];
    });
    llguidance = pyPrev.llguidance.overridePythonAttrs (oldAttrs: {
      nativeCheckInputs =
        (oldAttrs.nativeCheckInputs or [])
        ++ [pkgs.openssl];
    });
    accelerate = pyPrev.accelerate.overridePythonAttrs (oldAttrs: {
      nativeCheckInputs =
        (oldAttrs.nativeCheckInputs or [])
        ++ [pkgs.openssl];
    });

    # test_quantization_enabled_disabled calibrates W8A8 activation quant
    # from a single random sample whose min/max can collapse the scale to ~0,
    # yielding NaNs that trip `torch.all(a == b)` (fails on stock torch too).
    compressed-tensors =
      (pyPrev.compressed-tensors.override {
        torch = torch-xpu;
      }).overridePythonAttrs (oldAttrs: {
        nativeCheckInputs =
          (oldAttrs.nativeCheckInputs or [])
          ++ [pkgs.openssl];
        disabledTests =
          (oldAttrs.disabledTests or [])
          ++ ["test_quantization_enabled_disabled"];
      });
  })
