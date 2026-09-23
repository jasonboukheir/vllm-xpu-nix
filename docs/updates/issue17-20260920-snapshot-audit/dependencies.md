# Required dependency updates

Packaging commit: `880e74ed488a795963e631b46be5295a9389c323` (local;
publication and final-stack qualification remain pending).

The frozen vLLM snapshot requires Hugging Face Hub >=1.31.0 and XGrammar
==0.2.7. These supersede the packaging baseline's overrides of 1.28.0 and
0.2.1. `requirements/common.txt` identifies the Hub `httpx` re-export as the
reason for its lower bound; the XGrammar pin accompanies the structural
parser changes. Torch 2.13, Triton 3.7.2, oneAPI and the nixpkgs lock remain
unchanged. Source locks await the audited replay.

| Dependency | Version | Source hash | Realized output |
| --- | --- | --- | --- |
| huggingface-hub | 1.31.0 | `sha256-+OnnEKIQYT+l0PJrum2gXvSu+fuloPI/UI9axNCLb5A=` | `/nix/store/8r5mamgga19933dnpz7ww41k0nbab3ap-python3.12-huggingface-hub-1.31.0` |
| xgrammar | 0.2.7 | `sha256-1+xL0S/AbrKi+6/pJQvh2b03R5HM04mzazVe7k4jxVg=` | `/nix/store/06qlvgpqsz8mpd4vkxr492p2m7skr2w4-python3.12-xgrammar-0.2.7` |

XGrammar uses upstream tag `v0.2.7`, commit
`82505d0d987c36a4209fb3d8571cf6b0f28b5acd`, including submodules. Its
dependencies retain TVM FFI and explicitly include typing-extensions.
The original branch-style prefetch failed; fetching `refs/tags/v0.2.7`
succeeded. The actual package build accepted the recorded fixed-output hash.

Both derivations built successfully with `--max-jobs 1 --cores 4`. Nix
formatting and whitespace checks passed. A CPU-only offline API smoke check
confirmed both versions, the Hub `httpx` import, and native XGrammar JSON
schema conversion. No GPU workload or service was started. XGrammar's
inherited disabled upstream test suite remains disabled; this is not a claim
that its full suite passed. Final vLLM tool/structured-output checks remain
required.

Artifacts under `benchmark-results/issue17-20260920/`:

- `required-python-dependencies-eval.json`: exact derivation/output identities.
- `required-python-dependencies-build.json` and `.log`: completed build.
- `required-python-dependencies-smoke.log`: API smoke output.
- `xgrammar-0.2.7-prefetch.json`: immutable source identity.

An existing unrelated dependency relaxation remains visible: nixpkgs FastAPI
0.139.0 exceeds vLLM's <0.137.0 bound for optional model-hosting-container
handler overrides. This predates the selected snapshot and the packaging
overlay documents the affected disabled integration tests. Do not present
that optional integration as qualified. Preserve the pinned substrate unless
new required-runtime evidence warrants a focused change; actual serving API
and tools still require final-package validation.
