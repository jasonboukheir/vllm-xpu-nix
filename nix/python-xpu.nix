# The python312Packages set with the XPU-specific swaps vllm needs layered
# on top of nixpkgs' python312Packages.
{
  pkgs,
  torch-xpu,
  triton-xpu,
  torchvision-xpu,
}:
# Recompute the scoped Python package set with the XPU Torch and Triton
# packages at its fixed point. A shallow package-set merge only changes direct
# lookups: packages already instantiated by nixpkgs retain stock accelerator
# dependencies, causing unnecessary builds and duplicate implementations in
# vLLM's closure.
pkgs.python312Packages.overrideScope (
  pyFinal: pyPrev: {
    torch = torch-xpu;
    triton = triton-xpu;
    torchvision = torchvision-xpu;

    # vLLM 0.28 uses the MCP 2.0 split packages. nixpkgs still carries the
    # monolithic MCP 1.x SDK, so keep v2 under explicit sibling attributes: old
    # vLLM releases retain pyFinal.mcp while new callers select pyFinal.mcp-v2.
    mcp-types-v2 = pyFinal.buildPythonPackage rec {
      pname = "mcp-types";
      version = "2.0.0";
      format = "wheel";
      src = pkgs.fetchurl {
        url = "https://files.pythonhosted.org/packages/f5/4c/c78d78c3d52b0ac594ad7cc8ef5972adfe070e3597a8a4c6ce0cd39196ea/mcp_types-${version}-py3-none-any.whl";
        hash = "sha256-ay3nl8onl/Vot5Up4bJZSONN5RG8wL2C/vEDmm0bjrA=";
      };
      dependencies = with pyFinal; [
        pydantic
        typing-extensions
      ];
      pythonImportsCheck = [ "mcp_types" ];
    };

    mcp-v2 = pyFinal.buildPythonPackage rec {
      pname = "mcp";
      version = "2.0.0";
      format = "wheel";
      src = pkgs.fetchurl {
        url = "https://files.pythonhosted.org/packages/67/72/7d7897418912c1d12e87556630dfb7bf0eac71160e9bef8b447960804ee3/mcp-${version}-py3-none-any.whl";
        hash = "sha256-HLTHXS0se4wddWNV5dgqOfKCLMfxPiKiBR18o1kjSdY=";
      };
      dependencies = with pyFinal; [
        anyio
        httpx2
        jsonschema
        mcp-types-v2
        opentelemetry-api
        pydantic
        pyjwt
        cryptography
        python-multipart
        sse-starlette
        starlette
        typing-extensions
        typing-inspection
        uvicorn
      ];
      pythonImportsCheck = [
        "mcp"
        "mcp.client.sse"
        "mcp.types"
      ];
    };

    # vLLM's structural tool parser imports normalize_tool_choice, which was
    # added in xgrammar 0.2.1. The pinned nixpkgs still packages 0.1.33.
    xgrammar = pyPrev.xgrammar.overridePythonAttrs (oldAttrs: rec {
      version = "0.2.1";
      src = pkgs.fetchFromGitHub {
        owner = "mlc-ai";
        repo = "xgrammar";
        tag = "v${version}";
        fetchSubmodules = true;
        hash = "sha256-h9ovM/HbbkrxHGlJNn8eEisD5fnfRGCwoSOwc6HgpVQ=";
      };
      patches = [ ];
      build-system = (oldAttrs.build-system or [ ]) ++ [ pyPrev.apache-tvm-ffi ];
      dependencies = (oldAttrs.dependencies or [ ]) ++ [ pyPrev.apache-tvm-ffi ];
      # nixpkgs 0.1.33's disabled test paths no longer exist in 0.2.1.
      # Keep the import check while upstream's renamed suite is repackaged.
      doCheck = false;
    });

    # PyTorch 2.13 Inductor invokes openssl while these packages exercise
    # torch.compile in their test suites. Keep the tool local to the consumers
    # so Torch's store path—and the expensive AOT kernel cache—stays stable.
    depyf = pyPrev.depyf.overridePythonAttrs (oldAttrs: {
      nativeCheckInputs = (oldAttrs.nativeCheckInputs or [ ]) ++ [ pkgs.openssl ];
    });
    llguidance = pyPrev.llguidance.overridePythonAttrs (oldAttrs: {
      nativeCheckInputs = (oldAttrs.nativeCheckInputs or [ ]) ++ [ pkgs.openssl ];
    });
    accelerate = pyPrev.accelerate.overridePythonAttrs (oldAttrs: {
      nativeCheckInputs = (oldAttrs.nativeCheckInputs or [ ]) ++ [ pkgs.openssl ];
    });

    # test_quantization_enabled_disabled calibrates W8A8 activation quant
    # from a single random sample whose min/max can collapse the scale to ~0,
    # yielding NaNs that trip `torch.all(a == b)` (fails on stock torch too).
    compressed-tensors =
      (pyPrev.compressed-tensors.override {
        torch = torch-xpu;
      }).overridePythonAttrs
        (oldAttrs: {
          nativeCheckInputs = (oldAttrs.nativeCheckInputs or [ ]) ++ [ pkgs.openssl ];
          disabledTests = (oldAttrs.disabledTests or [ ]) ++ [ "test_quantization_enabled_disabled" ];
        });
  }
)
