# Dependencies of the 0.34 release. Older packages retain the original Python scope.
{ pkgs }:
final: prev:
let
  exporter =
    pname: dependencies:
    final.buildPythonPackage {
      inherit pname dependencies;
      version = "0.66b1";
      inherit (final.opentelemetry-api) src;
      sourceRoot = "${final.opentelemetry-api.src.name}/exporter/${pname}";
      pyproject = true;
      build-system = [ final.hatchling ];
    };
in
{
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
    version = "0.66b1";
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
