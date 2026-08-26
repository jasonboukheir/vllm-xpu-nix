# Workarounds for python packages that fail to build against this flake's
# pinned nixpkgs — flaky/stale tests or stale dependency bounds that the
# newer nixpkgs package set trips. Each entry documents why it fails and
# when to drop it. This mirrors what nixpkgs' own vllm package does
# wholesale with `pythonRelaxDeps = true`.
_final: prev: {
  pythonPackagesExtensions = (prev.pythonPackagesExtensions or [ ]) ++ [
    (_pyFinal: pyPrev: {
      # vLLM main requires huggingface-hub >=1.28.0. The pinned nixpkgs
      # substrate is intentionally held stable for the XPU runtime and only
      # carries 1.26.0, so update this one pure-Python dependency in place.
      huggingface-hub = pyPrev.huggingface-hub.overridePythonAttrs (old: rec {
        version = "1.28.0";
        src = prev.fetchPypi {
          pname = "huggingface_hub";
          inherit version;
          hash = "sha256-RqLpUMCSNN5UCT1YfRZ1OC8NCNvWANn7WZtZMvWyxss=";
        };
        meta = (old.meta or { }) // {
          changelog = "https://github.com/huggingface/huggingface_hub/releases/tag/v${version}";
        };
      });

      # inline-snapshot (check dep of anthropic/openai/fastapi/mcp)
      # golden-diffs its documentation code blocks against freshly
      # black-formatted output in tests/test_docs.py. Upstream generates
      # those docs with a dev-pinned black==25.1.0, but nixpkgs ships black
      # 26, whose 2026 stable style keeps multiline strings hugging the
      # opening paren (`multiline_string_handling`) — three doc tests fail
      # on pure formatting. The library is unaffected.
      # TODO: drop once upstream regenerates its docs against black 26
      # (dev pin: https://github.com/15r10nk/inline-snapshot/blob/main/pyproject.toml)
      inline-snapshot = pyPrev.inline-snapshot.overridePythonAttrs (old: {
        disabledTestPaths = (old.disabledTestPaths or [ ]) ++ [ "tests/test_docs.py" ];
      });

      # django (check dep via einops' jupyter test stack: django ->
      # openapi-core -> jupyterlab-server -> jupyterlab -> notebook ->
      # jupyter -> einops -> fla-core -> flash-linear-attention) has a
      # crafted-XML DoS regression test that asserts a wall-clock scaling
      # factor; it flakes whenever the builder is loaded, which is a given
      # while torch/vllm compile on all cores. django runs its own
      # runtests.py (not pytest), so rename the test away instead of using
      # disabledTests.
      # TODO: drop when the test is made load-robust upstream
      # (known flaky: https://github.com/NixOS/nixpkgs/issues/475149)
      django_5 = pyPrev.django_5.overridePythonAttrs (old: {
        preCheck = (old.preCheck or "") + ''
          substituteInPlace tests/serializers/test_deserialization.py \
            --replace-fail "def test_crafted_xml_performance" \
                           "def skipped_flaky_test_crafted_xml_performance"
        '';
      });

      # scipy 1.18's property-based Normal-distribution L-moment check can
      # generate a value 1.03e-11 beyond its absolute tolerance: the
      # reference is 2.0102755e-9 while the symmetric closed-form result is
      # exactly zero and the test requires atol=2e-9. This is a test
      # tolerance edge, not a package failure.
      # TODO: drop once scipy widens the tolerance or makes the reference
      # comparison scale-aware.
      scipy = pyPrev.scipy.overridePythonAttrs (old: {
        pytestFlags = (old.pytestFlags or [ ]) ++ [
          "--deselect=lib/python3.12/site-packages/scipy/stats/tests/test_continuous.py::TestDistributions::test_support_moments_sample[Normal]"
        ];
      });

      # model-hosting-container-standards' handler-override integration
      # tests assume import-time route registration that no longer occurs
      # with this package set, while its SageMaker LoRA integration test
      # constructs a vLLM request stub missing the newer adapter_config
      # attribute. These are compatibility tests for optional integrations;
      # the remaining 697 tests cover the packaged runtime.
      # TODO: drop when upstream tests support the current FastAPI/vLLM
      # interfaces in nixpkgs.
      model-hosting-container-standards =
        pyPrev.model-hosting-container-standards.overridePythonAttrs
          (old: {
            disabledTestPaths = (old.disabledTestPaths or [ ]) ++ [
              "tests/integration/test_handler_override_integration.py"
              "tests/integration/test_sagemaker_lora_integration.py"
            ];
          });

      # mistral-common (runtime dep of vllm) needs two repairs:
      # 1. It bounds numpy<2.4 on python<=3.12 — python-version-scoped
      #    legacy caution, already unbounded on 3.13+ — while the pinned
      #    nixpkgs ships numpy 2.5, so pythonRuntimeDepsCheckHook refuses
      #    the wheel. vllm upstream requires mistral_common with
      #    unbounded numpy.
      # 2. Upstream hard-depends on pydantic-extra-types[pycountry], but
      #    nixpkgs packages it with plain pydantic-extra-types, dropping
      #    the extra; importing mistral_common.protocol.instruct.validator
      #    (which transformers' MistralCommonBackend does at vllm startup)
      #    then dies in pydantic_extra_types.language_code with
      #    "requires pycountry". The hook can't catch missing extras.
      # TODO: drop 1. when mistral-common lifts the <=3.12 numpy bound and
      # 2. when nixpkgs adds the pycountry extra
      # (https://github.com/mistralai/mistral-common/blob/main/pyproject.toml)
      mistral-common = pyPrev.mistral-common.overridePythonAttrs (old: {
        pythonRelaxDeps = (old.pythonRelaxDeps or [ ]) ++ [ "numpy" ];
        dependencies = (old.dependencies or [ ]) ++ [ pyPrev.pycountry ];
      });
    })
  ];
}
