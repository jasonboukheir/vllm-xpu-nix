# xpu-v1.12.0 — vLLM 0.31 on Brutus

This coordinated release rebases the forks onto upstream vLLM v0.31.0 and
the official XPU kernel source corresponding to wheel 0.1.15.4. The release
retains Brutus's Qwen INT4 model, compact K4V2 target and draft caches, eager
Model Runner V1, bundled MTP2, vision and 262144-token context ceiling.

| Component | Upstream base | Fork commit |
| --- | --- | --- |
| vLLM | v0.31.0, `db9527a46873454610df6dbedf79a36d6bf1a7f6` | `ef92dc5600af64103c01f97e3300b81c3a1083ee` |
| XPU kernels | release/0.1.15.4, `ddf336d86e3c8602888572a3502f951abd51df12` | `dbb684f4c9b38a8b3bbf52bda0c490ee47049221` |

The kernel release has no matching tag. All seven Python files in the official
wheel match the selected source, but the wheel embeds no native source SHA.
The release rebuilds its native code from the frozen source. The
[manifest](xpu-v1.12.0.json) records exact package and source identities.

## Maintained delta

Dedicated audits covered every logical source change (five vLLM and eight
kernel changes) and all four additional Nix patches, against the actual upstream
delta: 728 vLLM commits and four kernel commits. Parent reconciliation produced
five vLLM and seven kernel commits.

Upstream kernel #600 replaces the local ragged-speculation implementation;
independent shortened-query rollback/storage regressions remain. An unused
dispatcher-pruning optimization was removed after both full and narrow native
builds passed. The native production delta shrank by 173 changed lines across
four files. KVarN was adapted to upstream allocator, checkpoint, DCP/TP and MTP
interfaces. The [audit map](../updates/release-v031-20261010/reconciliation.md)
and [fork delta](../fork-delta.md) explain retained changes and removal tradeoffs.

Required dependencies advance to Torch 2.14, Triton 3.8, torchvision 0.29,
oneAPI 2026.1, PTI 1.0.1, pyzes 0.1.2, XGrammar 0.2.7 and oss-harmony 0.0.11.
The nixpkgs pin and unrelated flake inputs remain fixed. Packaging reconciles
the common CUDA build-system Torch pin with upstream's exact XPU requirement,
defaults OpenCL discovery to the packaged Intel provider, and supplies the
oneCCL 2 plugin search directory. Actual startup exposed the last issue; a
rank-one XCCL control reproduced it, and the packaged fix passes without external
loader overrides. Original failures and control evidence are preserved.

## Qualification

The tested model is `RedHatAI/Qwen3.8-27B-INT4`, revision
`bf08f3dbd9a324e53956920aad378a1f1b6dd24a`, on Intel Arc Pro B70. The launcher
derives its command and environment from the evaluated host service. It changes
the executable, test port, writable cache location and request logging. The
host's package override remains unchanged.

- The full native build and a narrow attention-policy build passed.
- 608 packaging tests passed. Installed CPU qualification passed 660 cases,
  with 14 expected CUDA-only skips. Installed native/backend qualification
  passed 724 cases without skips; some are CPU contracts and overlap other
  suites, so these counts are not additive or all GPU executions.
- All 18 calibrated W4 reference cases passed on both stacks with byte-identical
  native outputs. One-hot controls identify BF16 RTZ weight conversion for this
  B70 path. Original RNE-reference discrepancies remain recorded; this is not
  a universal oneDNN rounding guarantee.
- HTTP health/model readiness, text, reasoning, two-image vision, tool formatting,
  12 concurrent requests with four running, active MTP, mixed prefill,
  cancellation after 3074 generated tokens and clean reuse passed.
- Retrieval with two images passed at 131071 and 262015 input tokens.

Two concurrent 174079-token prompts each generated 512 tokens with correct
labels/arithmetic, active MTP and no preemption. Usable shared attention capacity
was 350336 tokens; the run exercised at least 349184 occupied tokens (99.67%).
The capacity is shared: four scheduler slots do not imply four full 262K contexts.
The [foreground record](../updates/release-v031-20261010/foreground-qualification-review.json)
retains detailed evidence and sampling scope.

## Performance and limits

A matched short-text screen measured candidate median decode rates of 46.4–48.7
tokens/s for one request and 45.3–47.4 per request with two overlapping requests,
versus 42.4 and 41.3 for the previous release. That is an observed 9–15% gain on
these prompts, with 4.6–4.9% variation between candidate runs, one old-release
arm and output-token drift. First-token latency was similar. This measures the
whole stack, not an isolated fusion or a general speed guarantee.

MTP1 reduced median decode throughput by 8.6%/13.7% at concurrency one/two. Its
17–22 ms first-token improvement did not offset longer responses, so MTP2 is
retained. See [the performance record](../updates/release-v031-20261010/performance-results.md)
for exact workloads, timing, acceptance, output hashes and limits.

Upstream's Qwen QK-normalization/RoPE/gate fusion applies automatically to this
model on XPU. The [feature review](../updates/release-v031-20261010/performance-review.md)
explains Model Runner V2 and the likely Inco DFlash2 drafter. Current KVarN needs
a V2 allocation/history/ownership port; removing its guard would not implement
that support. A separate shorter-context BF16 prototype could assess whether
the port is worthwhile. This release does not enable V2, DFlash2, graphs or
prefix caching.

This qualification covers the actual dense Qwen service. It does not establish
optional AEON whole-model invariance or resolve historical unused-BF16-NaN
diagnostics from xpu-v1.11.0. The pre-existing FastAPI 0.139 relaxation exceeds
upstream's declared <0.137 bound; actual API checks passed. Controlled benefits
of retained reducer/load/barrier optimizations and compile-memory knobs were
not remeasured. CUDA and other model/backend configurations are not newly
qualified.

## Publication and operator handoff

All three repositories record the permanent `releases/xpu-v1.12.0` branch and
annotated `xpu-v1.12.0` tag. Packaging pins both source tags. Exact source SHAs,
hashes and timestamps match the tested candidate, and its package, kernel and
runtime derivations remain unchanged. The final packaging-main publication
record includes remote verification and the Brutus handoff: permanent packaging
release branch plus matching HTTPS source-tag overrides, scoped lock update and
exact equality with the tested host package derivation.

The prior `xpu-v1.11.0` branches and tags remain rollback checkpoints. No system
switch or production service restart is part of this release task. The user
performs the host rebuild after the configuration handoff. The
[release runbook](../../.agents/skills/vllm-xpu-update/SKILL.md) now documents
the current service-derived foreground workflow and permanent host release pins.
