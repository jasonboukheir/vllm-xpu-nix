# xpu-v1.7.2

This patch release preserves the current Brutus XPU stack before the next
upstream refresh. It removes the artificial 128K context ceiling for KVarN
bundled MTP and records the reorganized, feature-based fork histories.

## Sources

- vLLM: `9bfcf123f675c6b668391cb536ba06949174afd4`
- XPU kernels: `23d8103a97982c169986f65bba593885c91ad444`
- Both source inputs use the immutable `xpu-v1.7.2` tag.
- PyTorch, Triton, oneAPI, nixpkgs, and other dependency pins are unchanged.

The vLLM source tree is identical to previously deployed commit
`a29af3cc5c5d93e201485f3d2a6af1d2162bdc42`. The kernel source tree is identical
to `5ecfe045fc88bde64908961eecd815785cc332af`. Only their histories changed.

## Validation

On Brutus, 2026-09-07, the source-equivalent deployed package served
`sunny-chat` with 262,144 combined context tokens, KVarN K4V4/G128,
two-token bundled MTP, and the existing vision configuration. The model
registry advertised 65,536 output tokens.

- Health endpoint passed.
- `/v1/models` reported `max_model_len: 262144`.
- A short deterministic chat completion returned the requested `READY` answer.
- Exact Git-tree and Nix source NAR-hash equality verified that the history
  rewrite changed no source; every unrelated lock node is unchanged.
- Both release package derivations evaluated successfully. Source revision
  metadata changes their version/store paths, so this is source equivalence,
  not a claim that the newly stamped artifacts have been built or run.
- This release check did not exercise a full 256K request or rerun GPU model
  evaluations. The running artifact reports the pre-rewrite source version.

## Deployment

Pin the packaging repository to `refs/tags/xpu-v1.7.2`; its lock records the
coordinated source tags. Keep rolling `main` rebases separate from this stable
release. Do not move or reuse the release tags.
