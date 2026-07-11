# inline-snapshot (a check dependency of anthropic/openai/fastapi/mcp in
# vllm's dependency closure) golden-diffs its documentation code blocks
# against freshly black-formatted output in tests/test_docs.py. Upstream
# generates those docs with a dev-pinned black==25.1.0, but the nixpkgs this
# flake pins ships black 26, whose 2026 stable style keeps multiline strings
# hugging the opening paren (`multiline_string_handling`) — so three doc
# tests fail on pure formatting. The library is unaffected; skip only the
# doc-golden tests.
# TODO: drop once upstream regenerates its docs against black 26
# (dev pin: https://github.com/15r10nk/inline-snapshot/blob/main/pyproject.toml)
_final: prev: {
  pythonPackagesExtensions =
    (prev.pythonPackagesExtensions or [])
    ++ [
      (_pyFinal: pyPrev: {
        inline-snapshot = pyPrev.inline-snapshot.overridePythonAttrs (old: {
          disabledTestPaths = (old.disabledTestPaths or []) ++ ["tests/test_docs.py"];
        });
      })
    ];
}
