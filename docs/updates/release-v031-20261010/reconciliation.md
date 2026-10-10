# Reconciled source audit

All 13 original commits have dedicated independent reports. Parent reconciled paired GDN flags, native KVarN ABI, metadata ownership and new upstream allocator interfaces before replay.

| Repo / old commit | Decision | Report | Rationale |
|---|---|---|---|
| vllm `cfbd127232` | KEEP | [audits/v01-gdn-padding.md](audits/v01-gdn-padding.md) | Requirement remains uncovered by the selected upstream pair. |
| vllm `6efe5e7cc5` | KEEP | [audits/v02-gdn-scheduling.md](audits/v02-gdn-scheduling.md) | Requirement remains uncovered by the selected upstream pair. |
| vllm `d909b25f01` | KEEP | [audits/v03-rmsnorm.md](audits/v03-rmsnorm.md) | Requirement remains uncovered by the selected upstream pair. |
| vllm `984e6c3edd` | KEEP | [audits/v04-explicit-kv.md](audits/v04-explicit-kv.md) | Requirement remains uncovered by the selected upstream pair. |
| vllm `466b9d5abe` | ADAPT | [audits/v05-kvarn.md](audits/v05-kvarn.md) | Preserve upstream allocator/checkpoint/DCP/alignment behavior; forward prefill_end and use spec-aware KVarN block-size API. |
| vllm-xpu-kernels `98bdbfde3e` | KEEP | [audits/k01-deterministic-matmul.md](audits/k01-deterministic-matmul.md) | Requirement remains uncovered by the selected upstream pair. |
| vllm-xpu-kernels `e1bec3b3e9` | DROP | [audits/k02-build-policy.md](audits/k02-build-policy.md) | Retire unmeasured early template-dispatch pruning; no effect for host profile; gate on host and narrow FA2 builds. |
| vllm-xpu-kernels `956b61f95a` | KEEP | [audits/k03-untiled-conv.md](audits/k03-untiled-conv.md) | Requirement remains uncovered by the selected upstream pair. |
| vllm-xpu-kernels `3b5deb0773` | KEEP | [audits/k04-gdn-barrier.md](audits/k04-gdn-barrier.md) | Requirement remains uncovered by the selected upstream pair. |
| vllm-xpu-kernels `9a17a2c861` | KEEP | [audits/k05-mixed-arithmetic.md](audits/k05-mixed-arithmetic.md) | Requirement remains uncovered by the selected upstream pair. |
| vllm-xpu-kernels `30fda95e35` | ADAPT | [audits/k06-ragged-speculation.md](audits/k06-ragged-speculation.md) | Upstream f4b729a replaces native ragged-speculation code. Keep test-only storage/rollback guards; remove unnecessary split_mixed_non_spec argument. |
| vllm-xpu-kernels `8b6ec9d6a3` | KEEP | [audits/k07-native-reference.md](audits/k07-native-reference.md) | Requirement remains uncovered by the selected upstream pair. |
| vllm-xpu-kernels `193cee080d` | KEEP | [audits/k08-kvarn-native.md](audits/k08-kvarn-native.md) | Requirement remains uncovered by the selected upstream pair. |

The KEEP decisions preserve supported optional configurations as well as the actual Brutus profile. They do not claim measured optimization gains. The two k02 build gates and installed native regressions passed. Exact host-profile foreground qualification passed, including 262K retrieval and concurrent near-full-cache requests. Upstream global XPU invariance excludes the quantized model/cache configuration and does not replace the scoped helper. V2/DFlash requires separate integration, as documented in the performance review.
