# xpu-v1.7 — KVarN + Qwen images + two-token bundled MTP

Coordinated release for `jasonbk/vllm`, `jasonbk/vllm-xpu-kernels` and
`jasonbk/vllm-xpu-nix`. The tested MTP implementation is promoted to main;
frozen xpu-v1.5/v1.6 refs remain unchanged. Publishing does not deploy or
restart the host service. KVarN and MTP remain opt-in and narrowly qualified.

## Accepted implementation and cleanup

- Native causal verification for two/three query rows reads packed history
  and resident tails directly, removing repeated full-history materialization
  and the non-paged prefill attention route from eligible B70 MTP verification.
- Preserve authoritative committed lengths, rejected-tail handling and the
  decode history-window policy. Support guards allow one or two bundled drafts.
- Two tokens are the measured recommendation; no native/materialized trial
  toggle or additional performance ablation is retained in the serving path.
- Remove stale instructions for retired experiment switches. Keep only the
  qualified runtime changes and reusable correctness/performance harness;
  historical evidence and unrelated worktrees are preserved, not erased.
- Native kernels are unchanged from xpu-v1.6. No second kernel optimization
  is bundled into this release. Existing shape/platform correctness fallbacks
  and diagnostic tooling are not alternate B70 MTP implementations.

## Fixed unprofiled service result

One warmup and three measured repetitions per workload,256 output tokens.
Median decode tokens/s, excluding TTFT and the first output token:

| Input workload | KVarN off | KVarN1 final | KVarN2 final | Auto2 |
| --- | ---: | ---: | ---: | ---: |
| Short image222 |31.030|39.741|46.202|48.793|
| Text4096 |31.392|41.109|44.952|47.117|
| Image history6143 |31.228|39.552|43.734|47.880|

Two drafts improve decode40–49% over off and9–16% over one draft, reaching
91–95% of auto2 on this matrix. The single verification fix improves long-image
decode30.3% with one draft and25.9% with two. Two-draft TTFT medians are
0.262/2.697/4.084s; total request medians5.781/8.362/9.917s.

Matched one-draft profiles confirm materializer7.273ms/cycle and non-paged
prefill attention10.905ms/cycle disappear. GPU busy interval union falls
55.296→38.775ms versus auto37.535ms. Kernel sums/scope unions are not additive
wall time; profiles have substantial overhead and are not throughput gates.
No new MTP hardware-counter or occupancy/bandwidth/stall claim is made.

## Correctness and envelope

Pinned AEON Qwen checkpoint revision6b0622f4354481d5d04577d48ba0db844efc1330,
BF16/W4A16, B1, TP1/PP1, V1/eager, no prefix cache, max length8192,
prefill budget2048, budget0.90, up to two448x448 images, video0.
Both draft counts pass all16 exact-token image/text fixtures versus qualified
KVarN-off. Maximum shared-prefix top-5 logprob delta0.3125 is not full-logit
equivalence or a general vision-quality benchmark.

Qualification includes233 CPU tests, six real-B70 causal/rejected-tail tests,
129 reusable harness tests, and the unchanged native GDN recurrent-state oracle
at model dimensions with accepted counts0/1/2/3. Release-source runtime ASTs
match the qualified candidate; cleanup changes only comments/documentation.

Two-draft resident VRAM sampled peak27.98GiB includes preallocated cache/scratch;
not an instantaneous allocator peak or isolated MTP-weight delta. Historical
rebuild/outlier caveats are retained; later quiet long-image brackets reconfirm
the improvement. One correctness run's output-handler shutdown race occurred
after completed requests and successful XPU teardown and is documented explicitly.

## Usage and reproduction

Follow [the current KVarN guide](../kvarn-beta.md). In the qualified envelope,
use `--kv-cache-dtype kvarn_k4v4_g128_compact` and
`--speculative-config '{"method":"mtp","num_speculative_tokens":2}'`.
Remove historical experiment overrides. Omit speculative config for MTP-off;
use `auto` for cache rollback. No graph/V2/prefix/DFlash/broader concurrency
enablement is implied.

```sh
nix build .#vllm-xpu-kvarn-validation-env -o result
./result/bin/python scripts/kvarn_vision_run.py --service-env ./result \
  --cache-dtype kvarn_k4v4_g128_compact --draft-tokens 2 \
  --qualify --logprob-evidence --output /tmp/kvarn-mtp-correctness-new
./result/bin/python scripts/kvarn_vision_run.py --service-env ./result \
  --cache-dtype kvarn_k4v4_g128_compact --draft-tokens 2 \
  --perf-suite --output /tmp/kvarn-mtp-performance-new
```

Use fresh output directories and only one GPU run at a time. Model and runtime
identities, raw SSE, source snapshots and counter deltas are captured per run.

For `--profile-workload long-image`, use the qualified PTI wrapper, not the
bare runtime (whose bundled PTI failed to emit XPU traces on this host):

```sh
nix build --impure --expr 'import ./nix/profile-env.nix {
  candidateEnv = "/nix/store/<resolved-validation-env>";
  ptiDir = "/nix/store/24ys5hq6k0nmqmc6nwsiii89vq0n2mn2-intel-pti-0.17.0";
  enableMetrics = true; }' -o profile-result
./profile-result/bin/python scripts/kvarn_vision_run.py \
  --service-env ./profile-result --cache-dtype kvarn_k4v4_g128_compact \
  --draft-tokens 1 --profile-workload long-image --output /tmp/mtp-profile-new
```

Replace the validation-environment placeholder with the immutable path from
`readlink -f result`; the shown PTI path identifies the qualified Brutus tool.
`scripts/kvarn_xpu_trace.py` provides correlation-aware analysis; actual MTP
full-cycle guard/scope analysis and exact original commands are in the attached
evidence. Hardware-counter libraries are isolated in this wrapper; this does
not change host permissions or claim new MTP counter samples.

The packaging release attachment preserves the full bounded MTP qualification
report, commands, raw traces/results and immutable source archive; its SHA256
is provided alongside it. Local originals remain under
`benchmark-results/kvarn/mtp-performance-20260906/`.
Release-specific checks are under `benchmark-results/kvarn/release-xpu-v1.7/`.
[Megaissue #5](https://git.sunnycareboo.com/jasonbk/vllm-xpu-nix/issues/5)
records the results and remaining host/metadata, attention and shared-GEMM
opportunities. Those hypotheses are not additional fixes in this release.

## Final release-source gate

The remote-pinned release runtime
`/nix/store/mnq3m2rrfmn5vinqw7pasxwb8hf3gbsr-vllm-xpu-kvarn-validation-env`
passes all16 two-draft image/text fixtures with identical generated tokens
and shared top-5 logprobs to the qualified candidate, with no service errors.
It also passes233 CPU tests and all six real-B70 causal/rejected-tail oracles.
The complete packaging Nix check passes494 tests; formatting and both kernel
cache-identity checks pass. The source audit verifies identical native/process
dependencies and computational ASTs; only version metadata and stale comments
differ from the candidate. Existing one-draft and unprofiled performance evidence
therefore remains applicable without a new timing claim.

Source pins: vLLM `f39814bbdf7bbd94bfeb453e2332d78f4518cc22`, kernels
`767dc3ddf3a614f765e34b566ce788cb79bb2798` (unchanged).
Release audit and exact commands are included in the evidence attachment.
