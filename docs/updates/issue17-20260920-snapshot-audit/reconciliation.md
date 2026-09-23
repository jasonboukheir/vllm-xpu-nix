# Parent reconciliation — historical September decisions; paused

All 27 dedicated reports, including the v05 follow-up, are reviewed and
reconciled. Inventory, unique agent ownership and report hashes passed the
integrity gate. `commit-map.json` is the authoritative per-commit map.
The decisions below supported the historical candidate replay; they do not
authorize further work while paused. Only k01 was replayed, as
`c66a48486dcd6b76336ec8a8b984a4700f89e1d9`. Reassess all decisions against the
future pair after explicit user resume. Runtime and performance retention gates
remain pending; audit completion is not release qualification.

## Requirement ownership

| Owner | Old commits | Candidate residual | Required validation beyond source audit |
| --- | --- | --- | --- |
| Safe model inspection | v01, v06 | Drop the exact net-zero workaround/revert pair; fixed Torch finalizer supplies upstream coverage. | Cold inspection, honest child errors, normal teardown on final package. |
| GDN live-row boundary | v02 | Retain request-leading-dimension trimming; correct the stale token-count comment. | Padded requests, retained rollback columns, V2 scheduling. |
| Qwen mixed arithmetic | v03 + k05 | Retain scoped prefill alignment and optional native mixed decode/prefill split, with matching schema/caller. | Standalone-versus-mixed output/state, ragged requests, guards and model regression. |
| XPU residual RMSNorm | v04 | Adapt fixed-tree functional arithmetic to XPU only; preserve upstream CUDA behavior/tests. | XPU numerical/alias tests around the native 256-row dispatch boundary. |
| KVarN integration | v05, v07–v08, v10–v16 | Fold final repairs into the feature, preserving format-specific policy, physical pools and exclusive mutable ownership; adapt changed upstream APIs and V2. Retain only v16's surviving V1 bound update and reuse upstream independent V2 bounds. | Full port and qualification gates in issue17; preserve legacy admission without a BF16 serving milestone. |
| Explicit native KV dtype | v09 | Retain narrow guarded FP16/BF16 translation at the native cache writer. | Sparse/strided writes, mismatched dtype rejection, small attention references. |
| Deterministic oneDNN | k01 | Retain the deterministic attribute and meaningful repeatability/separation coverage. | Final native W4A16 path; this has no speed claim. |
| Disabled build policy pruning | k02 | Provisionally retain early host recursion pruning for narrow build configurations. | Cold compile time/RSS with and without only this patch; revise if no practical benefit. |
| Stable GDN convolution | k03 | Retain untiled prefill guard and replay regression; no frozen upstream replacement. | Replay/reference checks, fresh/dirty state and threshold boundaries. |
| GDN fresh-state ordering | k04 | Adapt to the one missing fresh-state O2 barrier; existing upstream barriers cover U and next-chunk S. | Controlled barrier ablation and reference/asynchronous replay before accepting the reduction. |
| Shortened GDN rollback | k06 | Retain additional regression invariants only; upstream da16a559 supplies production behavior. | Short/full continuation, full-capacity acceptance, interleaved state and output guards, plus wider DFlash verification. |
| Native attention replay | k07 | Retain eager replay coverage, reject fallback directly, add independent numerical reference. | Forced-fallback negative control and actual DSO-backed cases; separate split-K/window tests. |
| Native KVarN | k08–k11 | Fold compact K4V2 and justified reducer/reader changes into the native feature. | Paired ABI/layout checks, refreshed performance retention controls including adverse short K4V4 observations, new D128/window work. |

## Preserved rules for future authorized replay

Preserve independently useful fixes and original attribution/sign-off. Fold
follow-up repairs into their owning feature rather than recreating temporary
broken states in the final published history. Keep ordinary attention and
HiSparse behavior while adapting independent physical pools. No deletion may
strand a paired operator or hide an unqualified dtype/geometry fallback.

The v05/v10 audits identify bounded cleanup candidates: obsolete trusted
inline-plan branches with no producer, stale lifecycle selector scaffolding,
the unused native-record constant and the test-only four-bit packer wrapper.
Verify callers again in the final replay before removing them. The active
bound native plan is unrelated to Model Runner V2 despite its internal name;
it must not disappear through a naming-based cleanup. Do not retire existing
AEON/request-stable behavior solely because the new primary profile differs.

Two source-level reductions require particular care: k04's smaller barrier
placement needs GPU equivalence tests, and k06's production removal relies on
upstream supported-input semantics, not identical malformed-input diagnostics.
Retain those assumptions in tests and the final map.

Performance evidence is qualified, not universal. k10's short native ABBA
gain is 3.03%, while broader repeatability and one K4V4 regression gate failed.
The native feature candidate may retain it for controlled comparison; final
retention requires resolving that evidence on the refreshed pair. Do not
attribute k11's larger block-load gains to k10 or describe native-kernel
timings as serving throughput. No new performance result exists yet.

## Future completion of A — currently suspended

After explicit user resume and v0.32/upstream applicability reassessment, select
and freeze a new candidate rather than blindly using the September pair. Complete
the per-commit final SHA/drop map, inspect full diffs and range-diffs, run
available source/build checks, and assign every remaining port/runtime gate.
Then update this document from candidate decisions to observed results.
Neither 27 reports nor a clean replay alone completes the release goal.
