# Dependencies of the 0.34 release. Older packages retain the original Python scope.
{ pkgs }:
final: prev:
let
  otelBetaVersion = "0.66b1";
  exporter =
    pname: dependencies:
    final.buildPythonPackage {
      inherit pname dependencies;
      version = otelBetaVersion;
      inherit (final.opentelemetry-api) src;
      sourceRoot = "${final.opentelemetry-api.src.name}/exporter/${pname}";
      pyproject = true;
      build-system = [ final.hatchling ];
    };
in
{
  # These test tools share filelock in-process. virtualenv 21.6.1 requires
  # filelock <4, so keep tox and virtualenv in the same pinned tool scope.
  inherit (pkgs.python312Packages) virtualenv tox;
  # NumPy has no Python runtime dependency. Reuse the already-qualified package
  # in Torch rather than introduce a second NumPy through pytest's filelock.
  numpy =
    let
      stock = pkgs.python312Packages.numpy;
    in
    assert
      stock.version == prev.numpy.version
      && toString stock.src == toString prev.numpy.src
      && (stock.dependencies or [ ]) == [ ];
    stock;
  click = final.buildPythonPackage {
    pname = "click";
    version = "8.4.2";
    format = "wheel";
    src = pkgs.fetchurl {
      url = "https://files.pythonhosted.org/packages/fb/e2/79c688af8b210d232694e31e59da9f6ec747bae31c3f5946e4e9b98860d5/click-8.4.2-py3-none-any.whl";
      hash = "sha256-5vn2YTbIFnRbnWWBfakdYdlX+xbgLk3NBVJVPFoZe3Y=";
    };
    pythonImportsCheck = [ "click" ];
  };
  # Typer 0.25's command suggestions are incompatible with Click 8.4.
  # The 0.26 series vendors its Click implementation upstream.
  typer = prev.typer.overridePythonAttrs (old: {
    version = "0.26.8";
    src = pkgs.fetchFromGitHub {
      owner = "fastapi";
      repo = "typer";
      tag = "0.26.8";
      hash = "sha256-VkqvlWLzmtbQPaplx/YRdSNN1xq/UMRl8EZVIEm97Lk=";
    };
    dependencies = with final; [
      annotated-doc
      rich
      shellingham
    ];
    # test_binary_stderr starts a new Python process which imports tests.utils.
    preCheck = (old.preCheck or "") + ''
      export PYTHONPATH="$PYTHONPATH:$PWD"
    '';
  });
  # Initialize the per-tensor scale in the upstream FP8 round-trip test.
  compressed-tensors = prev.compressed-tensors.overridePythonAttrs (old: {
    patches = (old.patches or [ ]) ++ [
      (pkgs.fetchpatch {
        url = "https://github.com/vllm-project/compressed-tensors/commit/ac8e2ba82f4e0d5eaa40069f8fc642738f124cd4.patch";
        includes = [ "tests/test_compressors/test_fp8_quant.py" ];
        hash = "sha256-YJ1Bp9GN+Mjm2IxzYy0BPPeOuRfk7jWzgTcrIUHO7T8=";
      })
    ];
  });
  # Backport FastAPI's translation-test fixture for Typer's updated CliRunner.
  fastapi = prev.fastapi.overridePythonAttrs (old: {
    patches = (old.patches or [ ]) ++ [
      (pkgs.fetchurl {
        url = "https://github.com/fastapi/fastapi/commit/57535ef85b97d95862ec3de443c648e1c6aee139.patch";
        hash = "sha256-0+PzT+sw6rVoBzXctgGdE7+9Sqlk1aKATHX1LPTZjwU=";
      })
    ];
  });
  filelock = final.buildPythonPackage {
    pname = "filelock";
    version = "4.2.4";
    format = "wheel";
    src = pkgs.fetchurl {
      url = "https://files.pythonhosted.org/packages/e0/1b/5e74289a1878e666bb4b65fe5805a04cafcac2aa5f519ef1ead67ef91c56/filelock-4.2.4-py3-none-any.whl";
      hash = "sha256-MBJBQ4xXk275P0JDqU8lWQIiaL/PmzdamnLrD424NY4=";
    };
    pythonImportsCheck = [ "filelock" ];
  };
  huggingface-hub = prev.huggingface-hub.overridePythonAttrs (old: rec {
    version = "2.0.0";
    src = pkgs.fetchPypi {
      pname = "huggingface_hub";
      inherit version;
      hash = "sha256-N15a01yzUF778ZwM3X4/vx4jBLEs0Oa3/I11xvVIRhk=";
    };
    dependencies = with final; [
      click
      filelock
      fsspec
      hf-xet
      httpx2
      packaging
      pyyaml
      tqdm
      typing-extensions
    ];
  });
  # datasets 4.x still pins huggingface-hub <2.
  datasets = prev.datasets.overridePythonAttrs {
    version = "5.2.0";
    src = pkgs.fetchFromGitHub {
      owner = "huggingface";
      repo = "datasets";
      tag = "5.2.0";
      hash = "sha256-Qv6pFttmxfpgAfZky+dAjpZgHZLr1n1ySdrWEQnVqgM=";
    };
  };
  transformers = prev.transformers.overridePythonAttrs {
    version = "5.20.1";
    src = pkgs.fetchPypi {
      pname = "transformers";
      version = "5.20.1";
      hash = "sha256-JbMIeeTCQ/AGMMEHkXHt5tT2v3bmj5eKzUdiimQLyKA=";
    };
  };
  tokenizers = prev.tokenizers.overridePythonAttrs {
    version = "0.23.3";
    src = pkgs.fetchurl {
      url = "https://files.pythonhosted.org/packages/3a/40/6706b82693715581457c6d5423eaa7faae576bb0526c5738a57085eb4449/tokenizers-0.23.3-cp310-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl";
      hash = "sha256-N2hR0ivPnWUKXDCQu4Pmz56JX78FlTafpM1DwfabX4c=";
    };
  };
  mistral-common = prev.mistral-common.overridePythonAttrs {
    version = "1.13.1";
    src = pkgs.fetchFromGitHub {
      owner = "mistralai";
      repo = "mistral-common";
      tag = "v1.13.1";
      hash = "sha256-BUX9ZkitJnqaFipMozFPOYvxhuxj9eOmemYyrYUD+Hg=";
    };
  };
  # nixpkgs builds the SDK, protocol, and exporters from the API source tree.
  opentelemetry-api = prev.opentelemetry-api.overridePythonAttrs {
    version = "1.45.1";
    src = pkgs.fetchFromGitHub {
      owner = "open-telemetry";
      repo = "opentelemetry-python";
      tag = "v1.45.1";
      hash = "sha256-/WP+0J5Bd/XBpT69YsZaoSfARZwc2pAMVT3KeIjap40=";
    };
  };
  opentelemetry-semantic-conventions = prev.opentelemetry-semantic-conventions.overridePythonAttrs {
    version = otelBetaVersion;
  };
  opentelemetry-test-utils = prev.opentelemetry-test-utils.overridePythonAttrs {
    version = otelBetaVersion;
  };
  opentelemetry-exporter-http-transport = exporter "opentelemetry-exporter-http-transport" [
    final.opentelemetry-api
    final.requests
  ];
  opentelemetry-exporter-otlp-common = exporter "opentelemetry-exporter-otlp-common" [
    final.opentelemetry-sdk
  ];
  opentelemetry-exporter-otlp-proto-http =
    prev.opentelemetry-exporter-otlp-proto-http.overridePythonAttrs
      (old: {
        nativeCheckInputs = (old.nativeCheckInputs or [ ]) ++ [ pkgs.python312Packages.mocket ];
        dependencies = old.dependencies ++ [
          final.opentelemetry-exporter-http-transport
          final.opentelemetry-exporter-otlp-common
        ];
      });
  opentelemetry-exporter-otlp-proto-grpc =
    prev.opentelemetry-exporter-otlp-proto-grpc.overridePythonAttrs
      (old: {
        dependencies = old.dependencies ++ [
          final.opentelemetry-sdk
          final.opentelemetry-exporter-otlp-common
        ];
      });
}
