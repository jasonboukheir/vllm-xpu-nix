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
    # Rewrap the same vision wheel with this scope's Pillow and NumPy.
    torchvision = torchvision-xpu.override { python3Packages = pyFinal; };

    # Torch retains this dependency-free core. Rebuilding it only because
    # HTTP test inputs changed creates a duplicate in downstream closures.
    # Keep optional extras scoped so HTTP callers still get the new aiohttp.
    fsspec =
      let
        stock = pkgs.python312Packages.fsspec;
      in
      assert pkgs.lib.assertMsg (
        stock.version == pyPrev.fsspec.version
        && toString stock.src == toString pyPrev.fsspec.src
        && (stock.dependencies or [ ]) == [ ]
        &&
          map toString (stock.propagatedBuildInputs or [ ]) == [
            (toString pkgs.python312Packages.python)
          ]
      ) "Review fsspec reuse: its source or unconditional runtime dependencies changed";
      stock.overrideAttrs (old: {
        passthru = (old.passthru or { }) // {
          optional-dependencies = pyPrev.fsspec.optional-dependencies;
        };
      });

    fastapi = pyPrev.fastapi.overridePythonAttrs {
      version = "0.136.3";
      src = pkgs.fetchFromGitHub {
        owner = "fastapi";
        repo = "fastapi";
        tag = "0.136.3";
        hash = "sha256-lfmk8ZveKPukEEfwWq2mKtWmOHAtVzGuE5BsOskDzh0=";
      };
    };

    aiohttp =
      assert pkgs.lib.assertMsg (pkgs.lib.versionOlder pyPrev.aiohttp.version "3.14.4")
        "nixpkgs aiohttp has caught up; review the vLLM 0.33 override";
      pyPrev.aiohttp.overridePythonAttrs (oldAttrs: {
        version = "3.14.4";
        src = pkgs.fetchFromGitHub {
          owner = "aio-libs";
          repo = "aiohttp";
          tag = "v3.14.4";
          hash = "sha256-ArlEDpuUQJ6Ueup7tL9H1Nd3/TgajMaHtg4pJiSGOy8=";
        };
        postPatch = ''
          rm -r vendor
          patchShebangs tools
          touch .git
          substituteInPlace Makefile \
            --replace-fail "cythonize: .install-cython" "cythonize:"
        '';
        nativeCheckInputs = (oldAttrs.nativeCheckInputs or [ ]) ++ [ pyPrev.coverage ];
      });

    # aiohttp 3.14.4 requires yarl >= 1.25.1.
    yarl =
      assert pkgs.lib.assertMsg (pkgs.lib.versionOlder pyPrev.yarl.version "1.25.1")
        "nixpkgs yarl has caught up; review the vLLM 0.33 override";
      pyPrev.yarl.overridePythonAttrs {
        version = "1.25.1";
        src = pkgs.fetchFromGitHub {
          owner = "aio-libs";
          repo = "yarl";
          tag = "v1.25.1";
          hash = "sha256-cQsVglP3OqaQW95OYPlbWCFTs5p6L5wOzx7QCIqqJ9U=";
        };
      };

    multidict =
      assert pkgs.lib.assertMsg (pkgs.lib.versionOlder pyPrev.multidict.version "6.9.1")
        "nixpkgs multidict has caught up; review the vLLM 0.33 override";
      pyPrev.multidict.overridePythonAttrs (oldAttrs: {
        version = "6.9.1";
        src = pkgs.fetchFromGitHub {
          owner = "aio-libs";
          repo = "multidict";
          tag = "v6.9.1";
          hash = "sha256-Xe8n3Y8x5fOS+xme5pKJuiiU6y+oXosZLoEA1Rkp+Qw=";
        };
        nativeCheckInputs = (oldAttrs.nativeCheckInputs or [ ]) ++ [ pyPrev.coverage ];
      });

    pillow =
      assert pkgs.lib.assertMsg (pkgs.lib.versionOlder pyPrev.pillow.version "13.0.0")
        "nixpkgs pillow has caught up; review the vLLM 0.33 override";
      pyPrev.pillow.overridePythonAttrs {
        version = "13.0.0";
        src = pkgs.fetchFromGitHub {
          owner = "python-pillow";
          repo = "pillow";
          tag = "13.0.0";
          hash = "sha256-7FHwvgfT40qiHmvcJFZN/FlOECE0YiqOLBEYCMugk+4=";
        };
      };

    certifi =
      assert pkgs.lib.assertMsg (pkgs.lib.versionOlder pyPrev.certifi.version "2026.10.19")
        "nixpkgs certifi has caught up; review the vLLM 0.33 override";
      pyPrev.certifi.overridePythonAttrs {
        version = "2026.10.19";
        src = pkgs.fetchFromGitHub {
          owner = "certifi";
          repo = "python-certifi";
          tag = "2026.10.19";
          hash = "sha256-eiNK0yRcaOr0HW/kCc+0oOUXOl0jh+2rT41qBdDrVK8=";
        };
        # Keep this release's root removals; the inherited postPatch replaces
        # its bundle with the older pinned NSS bundle. env.patch still honors
        # an explicit NIX_SSL_CERT_FILE trust store.
        postPatch = "";
      };

    # vLLM 0.32 requires these versions ahead of the pinned nixpkgs set.
    transformers =
      assert pkgs.lib.assertMsg (pkgs.lib.versionOlder pyPrev.transformers.version "5.16.1")
        "nixpkgs transformers has caught up; review the vLLM 0.32 override";
      pyPrev.transformers.overridePythonAttrs (oldAttrs: rec {
        version = "5.16.1";
        src = pkgs.fetchPypi {
          pname = "transformers";
          inherit version;
          hash = "sha256-F7DqxybdxV6ErFiUYGPgxtN/0ADEVrWB8FDqD06CKGk=";
        };
        doCheck = false;
      });

    tokenizers =
      assert pkgs.lib.assertMsg (pkgs.lib.versionOlder pyPrev.tokenizers.version "0.23.2")
        "nixpkgs tokenizers has caught up; review the vLLM 0.32 override";
      pyFinal.buildPythonPackage {
        pname = "tokenizers";
        version = "0.23.2";
        format = "wheel";
        src = pkgs.fetchurl {
          url = "https://files.pythonhosted.org/packages/2c/ca/ca6b93c7820df123b2662a9469e8facc826ccc94e98fdd0d615f6431e73a/tokenizers-0.23.2-cp310-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl";
          hash = "sha256-QcL4TRckSbTa25zcUI4+NkB2YTw1sW527P5Hpg0eMwU=";
        };
        nativeBuildInputs = [ pkgs.autoPatchelfHook ];
        buildInputs = [ pkgs.stdenv.cc.cc.lib ];
        dependencies = [ pyFinal.huggingface-hub ];
        pythonImportsCheck = [ "tokenizers" ];
        meta.platforms = [ "x86_64-linux" ];
      };

    oss-harmony = pyFinal.buildPythonPackage rec {
      pname = "oss-harmony";
      version = "0.0.11";
      format = "wheel";
      src = pkgs.fetchurl {
        url = "https://files.pythonhosted.org/packages/89/68/001d7429a692b2db0c03bd49b756f834716b64afc0bbbece96824e8f0af8/oss_harmony-${version}-cp38-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl";
        hash = "sha256-2HfGtSbNvfVQaLro8FxgOY6SnJ4pbH9podUzM5cIvoA=";
      };
      nativeBuildInputs = [ pkgs.autoPatchelfHook ];
      buildInputs = [ pkgs.stdenv.cc.cc.lib ];
      dependencies = [ pyFinal.pydantic ];
      pythonImportsCheck = [ "openai_harmony" ];
    };

    # vLLM 0.28 uses the MCP 2.0 split packages. nixpkgs still carries the
    # monolithic MCP 1.x SDK, so keep v2 under explicit sibling attributes: old
    # vLLM releases retain pyFinal.mcp while new callers select pyFinal.mcp-v2.
    mcp-types-v2 = pyFinal.buildPythonPackage rec {
      pname = "mcp-types";
      version = "2.2.0";
      format = "wheel";
      src = pkgs.fetchurl {
        url = "https://files.pythonhosted.org/packages/8f/d7/6ffba5d8cd5dd9b8a19478875c50e04945314ba5074e84d749283f27f62d/mcp_types-${version}-py3-none-any.whl";
        hash = "sha256-6kdrc+6GcJq1q8lFI4XtNswFkH5YI1ViLilFlcmgTxM=";
      };
      dependencies = with pyFinal; [
        pydantic
        typing-extensions
      ];
      pythonImportsCheck = [ "mcp_types" ];
    };

    mcp-v2 = pyFinal.buildPythonPackage rec {
      pname = "mcp";
      version = "2.2.0";
      format = "wheel";
      src = pkgs.fetchurl {
        url = "https://files.pythonhosted.org/packages/1b/ff/8e7eade68b8a28f7da0ed1085544341b51f9c935dbf6b95c76b7edfea6a0/mcp-${version}-py3-none-any.whl";
        hash = "sha256-vemCWJRzoGCuFF40BumlMz/lOMlyKbqEH1p/kr4AT4E=";
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

    # vLLM 0.32 pins xgrammar 0.2.8; nixpkgs still packages 0.1.33.
    xgrammar =
      assert pkgs.lib.assertMsg (pkgs.lib.versionOlder pyPrev.xgrammar.version "0.2.8")
        "nixpkgs xgrammar has caught up; review the vLLM 0.32 override";
      pyPrev.xgrammar.overridePythonAttrs (oldAttrs: rec {
        version = "0.2.8";
        src = pkgs.fetchFromGitHub {
          owner = "mlc-ai";
          repo = "xgrammar";
          tag = "v${version}";
          fetchSubmodules = true;
          hash = "sha256-Ff/CLdSjq8ammfci+Eu9hY4433e7csf8EeGu2X9B2rI=";
        };
        patches = [ ];
        build-system = (oldAttrs.build-system or [ ]) ++ [
          pyPrev.apache-tvm-ffi
          pyPrev.setuptools-scm
        ];
        dependencies = (oldAttrs.dependencies or [ ]) ++ [ pyPrev.apache-tvm-ffi ];
        # nixpkgs 0.1.33's disabled test paths no longer exist in 0.2.8.
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
