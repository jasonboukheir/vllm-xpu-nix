# Workarounds for python-package tests that fail against this flake's pinned
# nixpkgs. These packages are only in vllm's graph as (transitive) check
# dependencies; each entry documents why it fails and when to drop it.
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
      })
    ];
}
