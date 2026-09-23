# xpu-v1.11.0 — v0.30 on the existing Brutus profile

This coordinated release updates the source forks to upstream vLLM v0.30.0 and
XPU kernels v0.1.15 while retaining Brutus's eager V1, K4V2 target and draft
caches, bundled MTP2 and vision. The exact native package produced reasonable
output under the actual Brutus configuration, including concurrent and long
context requests. The user selected those operational checks as release acceptance.

All 27 original fork commits received individual audits, followed by parent
reconciliation and replay into five vLLM and eight kernel commits. Four packaging
patches were independently audited. Recovery refs and the complete commit map
are preserved in [the update record](../updates/release-v030-20260922/CURRENT.md).

| Component | Upstream base | Released fork commit |
| --- | --- | --- |
| vLLM | v0.30.0, `ced6857afa0ea7b2e3f0846a62e1394e90f15607` | `466b9d5abe84f13c29b38c686e4d77fc09e09aa4` |
| XPU kernels | v0.1.15, `1c7cbeee1cd0c1481d48f5031f679a47f3f0ef45` | `193cee080d654ff67c4f8bfaf924201bc88b9cf5` |

Every repository uses the new annotated `xpu-v1.11.0` tag and permanent
`releases/xpu-v1.11.0` branch. Packaging pins both source tags. Source revisions,
content hashes and timestamps are identical to the tested rolling candidate;
its Brutus package and runtime derivations are unchanged. The
[machine-readable manifest](xpu-v1.11.0.json) records exact identities and evidence.

## Qualified configuration and results

The model is `RedHatAI/Qwen3.8-27B-INT4`, revision
`bf08f3dbd9a324e53956920aad378a1f1b6dd24a`, on Intel Arc Pro B70. It uses W4A16
compressed-tensors weights, BF16 compute, eager Model Runner V1,
`kvarn_k4v2_g128_compact` for both target and MTP draft, and two speculative tokens.
The normal settings are unchanged: 262144 maximum context, four scheduler slots,
2048 batched prefill tokens, 0.96 GPU budget, prefix caching off, two 448-pixel
images and no video. Qwen3 reasoning and Qwen3 XML tools remain enabled.

- Native build completed in 1857 seconds with max-jobs=1 and cores=4. All four
  extension modules and six split libraries load; eleven native schemas match.
- 608 packaging tests, 705 installed native/backend cases and 14 supplemental
  GDN independent-reference cases passed. Some native/backend cases are CPU
  contracts; these counts are not all GPU executions.
- Actual Brutus foreground checks passed chat, reasoning, two-image vision,
  tool formatting, 12 concurrent requests with four running, active MTP,
  mixed prefill, cancellation past a cache flush and clean reuse.
- Retrieval passed at 131071 and 262015 input tokens. Two concurrent 172415-token
  prompts each generated 512 tokens with correct labels/arithmetic, active MTP
  and no preemption. Owned DRM sampling covered these checks without errors.

| Native-budget capacity | xpu-v1.10.0 | xpu-v1.11.0 |
| --- | ---: | ---: |
| Usable shared attention tokens | 346880 | 346880 |

The candidate exercised at least 345728 occupied tokens. Capacity is shared
across requests; four scheduler slots do not imply four full 262144-token
contexts. The capacity check is not a controlled speed comparison.

## Limits of this qualification

No new throughput, latency, optimization benefit or performance non-regression
claim is made. Prepared comparative kernel/build controls and matched performance
experiments were not run under the user's selected operational acceptance scope.
Their protocols and original audit recommendations remain preserved.

Optional AEON exact whole-model concurrent token invariance failed on both the
candidate and published baseline with the same two complete output trajectories
for the retained duplicate fixture. Direct functional RMSNorm tests passed.
An additional candidate-only mixed-long diagnostic also differed from isolated
execution; no matching baseline diagnostic was run. These observations do not
establish a causal mechanism or a new regression, and the strict invariant is
not claimed to pass. Brutus's RedHat profile does not activate the narrow AEON
projection/RMSNorm policy.

The intentional unused-BF16-NaN diagnostic retains six historical failures; its
two sparse-writer controls pass. Standard finite-storage/native-reference checks
passed. CUDA-only source checks were unavailable. The pre-existing FastAPI 0.139
relaxation exceeds upstream's declared <0.137 bound; ordinary Brutus API checks
passed. Optional configurations outside the tested dense Qwen profile are not
newly qualified by this release.

Hugging Face Hub advances from 1.28.0 to required 1.31.0. Unrelated locks,
Torch 2.13.0+xpu, Triton 3.7.2+xpu, oneAPI 2026.0, XGrammar 0.2.1 and OpenAI 2.41.1
remain unchanged. Split libraries use four link jobs; base glue/FA2 retain the
upstream sixteen. The observed build is not a cold comparative resource study.

## Deployment and rollback

This release does not repin or activate Brutus. The host remains on
`xpu-v1.10.0`, packaging commit `4ac4375aa3d0f649fb4b6b43d9447f413a8b1cc7`, and
user-stopped services remain stopped. Deployment requires its separate host
handoff. To roll back after a future deployment, select packaging
`refs/tags/xpu-v1.10.0`, perform the scoped host lock update, verify the previous
package derivation and use the operator activation workflow. Existing release
and recovery refs are unchanged.

Issue #17 remains paused. Its [checkpoint PR #18](https://git.sunnycareboo.com/jasonbk/vllm-xpu-nix/pulls/18)
is unrelated to this release. No DFlash2 speedup or performance ceiling has been
established, and no V2/DFlash phase has resumed.
