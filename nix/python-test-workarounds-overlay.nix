# Workarounds for python packages that fail to build against this flake's
# pinned nixpkgs — flaky/stale tests or stale dependency bounds that the
# newer nixpkgs package set trips. Each entry documents why it fails and
# when to drop it. This mirrors what nixpkgs' own vllm package does
# wholesale with `pythonRelaxDeps = true`.
_final: prev: {
  pythonPackagesExtensions =
    (prev.pythonPackagesExtensions or [])
    ++ [
      (_pyFinal: pyPrev: {
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
          disabledTestPaths = (old.disabledTestPaths or []) ++ ["tests/test_docs.py"];
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
          preCheck =
            (old.preCheck or "")
            + ''
              substituteInPlace tests/serializers/test_deserialization.py \
                --replace-fail "def test_crafted_xml_performance" \
                               "def skipped_flaky_test_crafted_xml_performance"
            '';
        });

        # mistral-common (runtime dep of vllm) bounds numpy<2.4 on
        # python<=3.12 — the bound is python-version-scoped legacy caution,
        # already unbounded on 3.13+ — while the pinned nixpkgs ships numpy
        # 2.5, so pythonRuntimeDepsCheckHook refuses the wheel. vllm
        # upstream requires mistral_common with unbounded numpy.
        # TODO: drop when mistral-common lifts the <=3.12 numpy bound
        # (https://github.com/mistralai/mistral-common/blob/main/pyproject.toml)
        mistral-common = pyPrev.mistral-common.overridePythonAttrs (old: {
          pythonRelaxDeps = (old.pythonRelaxDeps or []) ++ ["numpy"];
        });
      })
    ];
}
