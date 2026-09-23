# Paused at user request

**Issue #17 and its long-running goal are PAUSED. Resume only on the user's explicit request.** The supported goal control reports `paused`; this is neither completion nor a measured performance limit. No implementation, rebasing, optimization, qualification or publication is authorized while paused. A release appearing upstream does not automatically resume work. This section takes precedence over the suspended requirements and historical instructions below.

## Planned reassessment: upstream v0.32

A **published upstream v0.32-series release** is the planned reassessment and candidate-rebase point. No particular patch version is reserved, and publication does not establish readiness for this configuration. Upstream targets **Model Runner V1 removal** in v0.32; that is not a promised stabilization deadline for XPU, quantized DFlash2 or our custom KVarN integration. This concerns the model runner, not removal of the entire `vllm/v1` namespace. [Upstream v0.29.0 deprecation notice](https://github.com/vllm-project/vllm/releases/tag/v0.29.0).

Wait for and verify these upstream conditions before resuming implementation; they are pending criteria, not assertions that upstream already satisfies them:

1. A published v0.32-series release, with a compatible XPU-kernels pair and Torch/Triton/oneAPI dependencies that can be packaged reproducibly.
2. Clear XPU support and regression evidence for our pinned Qwen target, bundled MTP and vision on Model Runner V2. General V2 support or V1 removal alone is insufficient.
3. Quantized DFlash loading, fused-module mapping and context-projection fixes, including [#53122](https://github.com/vllm-project/vllm/pull/53122) or an equivalent, merged and applicable to `syvai/Qwen3.8-27B-DFlash2-W4A16` at `4d30ec736ffc6b8688dc2ae2b502d9b48bdec279`. Verify actual XPU dispatch and checkpoint compatibility.
4. Resolution or a supported, understood workaround for applicable XPU DFlash compilation problems, including [#56787](https://github.com/vllm-project/vllm/issues/56787). Compilation need not be mandatory: supported eager/compiled combinations, execution-mode limitations, latency and memory costs must be explicit and measured when qualification resumes.
5. Sufficient V2 metadata/cache API stability to port and maintain our custom KVarN integration, supported by inspection and regression evidence rather than a version number alone.

These conditions do **not** solve the local work: exact committed-history handling under asynchronous acceptance/rejection; separate target/draft ownership and MTP prefill metadata; D128/Q32/KV8 noncausal 2048-token sliding-window K4V2; concurrent draft-window allocation for every admitted scheduler slot; and cache lifecycle/reset correctness. Waiting also does not finish the minimized source replay, profiling, measurements, correctness matrix or release.

## Completed and preserved checkpoint

State was inspected on Brutus at **2026-09-20 07:38 UTC**. No goal-owned background process remained; all visible audit agents had completed. Builds/setup commands had exited, including the Node 24 realization. No process needed termination, and no host service or activation was changed by this pause.

- **27/27 individual commit audits completed:** 16 vLLM and 11 kernel commits, each assigned a distinct dedicated read-only audit agent (report writing only). This includes eight vLLM and three kernel release-only commits absent from the rolling sibling branches. All 27 report hashes, unique assignments and parent reconciliations were reverified. The v05 follow-up is included.
- Parent reconciliation recorded evidence-backed KEEP/DROP/ADAPT decisions, dependencies, feature folds and remaining validation gates. The final minimized effective diff and qualified old-to-new map are **not complete**. Candidate decisions are historical evidence for the September snapshots and require applicability review on a future pair.
- Independent audit clones and frozen upstream refs were created. Local recovery refs preserve the original local/remote main and released tips; both remote `recovery/issue17-20260920-main` branches were published and reverified. Existing release refs remain unchanged.
- **Exactly one source commit was replayed:** kernel k01, deterministic oneDNN W4A16, old `8530de9d4298b199436079c46233acb33efb6fc2` → candidate `c66a48486dcd6b76336ec8a8b984a4700f89e1d9`, parent `da16a5595c105605bcd44b558c7933b934a05f85`. Attribution/sign-off was retained and commit hooks passed. No native build or GPU test qualified this replay. No vLLM source commit was replayed.
- Packaging dependency update `880e74ed488a795963e631b46be5295a9389c323` is committed locally and unpublished: Hugging Face Hub 1.31.0 and XGrammar 0.2.7 were built, with formatting/whitespace and offline import/JSON-grammar API smoke checks passing. This was not a candidate vLLM/native-stack build or serving qualification. Torch, Triton, oneAPI and the nixpkgs lock were unchanged.
- Local lint tooling was partly prepared in ignored environments/caches. Kernel hooks worked; vLLM hook setup remains incomplete after generic Node/stub-loader and Nix Node/npm failures. Node 24.19.0 subsequently finished realizing, but the vLLM setup was not repaired or rerun before the pause. Failed logs and cache adjustments remain preserved.

**No DFlash2 speedup, performance ceiling or remaining-gain bound has been established.** There are no new DFlash2 serving runs, profiles or speed/capacity measurements in this run. Historical native-kernel measurements in the audits are not DFlash2 serving evidence. Integration, profiling validation, optimization, the two adversarial performance reviews, bounded alternative-quant comparisons, full correctness/lifecycle/memory/regression checks and final-package qualification/publication remain undone.

## Exact source, packaging and host identities

All paths below are on Brutus. Packaging root: `/home/jasonbk/Projects/vllm-xpu-nix`.

| Checkout / ref | Exact commit | Working state at pause |
| --- | --- | --- |
| Packaging `main` | `880e74ed488a795963e631b46be5295a9389c323` | No tracked/staged modifications; `?? docs/updates/` contains preserved uncommitted audit/pause records. One unpublished dependency commit above released main. |
| `build-dev/issue17/vllm`, `issue17-20260920-replay` | `e378275a8f20eac9e92212e4a1fee84ec5a31dc7` | Clean; frozen upstream target; zero replay commits. |
| `build-dev/issue17/vllm-xpu-kernels`, `issue17-20260920-replay` | `c66a48486dcd6b76336ec8a8b984a4700f89e1d9` | Clean; one k01 replay above frozen `da16a5595c105605bcd44b558c7933b934a05f85`. |
| Original `/home/jasonbk/Projects/vllm`, `main` | `3b6c822e8a4f56dc3904c7fce388a76dff4c7bdd` | Clean and unchanged; also remote main and remote recovery branch. |
| Original `/home/jasonbk/Projects/vllm-xpu-kernels`, `main` | `9bb1e2d428b8a25b262c0cd6704614bdf6ee3ad8` | Clean and unchanged; also remote main and remote recovery branch. |

There is no unfinished cherry-pick, merge or rebase in these five checkouts. Ignored `.venv`, caches, local hook adapters, build logs and artifacts remain present; “clean” does not mean those artifacts were removed.

The **current published and deployed operational baseline remains `xpu-v1.10.0`**: packaging `4ac4375aa3d0f649fb4b6b43d9447f413a8b1cc7`, vLLM `2fdd9d71f6895379b0940a8f755b4c9095840558`, kernels `2c39287deec5e59cc2656e6d4a6dce51a567c411`. Remote immutable release tags and packaging main were reverified; the Forgejo release-object list is incomplete for newer coordinated Git-tag releases and is not the source of truth here. No new coordinated release was published by this goal.

`vllm-xpu-chat.service` was **active/running, MainPID 1638887**, using `/nix/store/v8jzs75kb7w0wy0542a99kl6njxsd1kp-python3.12-vllm-xpu-0.28.0+unstable.2026.09.15.g2fdd9d7/bin/vllm`: the pinned RedHat target, bundled MTP2, target/draft K4V2, eager execution and four scheduler slots. The initial inactive observation is historical and superseded by this read-only inspection. Current system: `/nix/store/11y2n32j1k01p7h1hbrfigjp9b88wjh6-nixos-system-brutus-26.05.20260914.c3eea5b`. Leave host services and activation under the separate operator handoff.

The frozen September pair is an **archived source-compatible selection, not a qualified release candidate**: vLLM `e378275a8f20eac9e92212e4a1fee84ec5a31dc7` and kernels `da16a5595c105605bcd44b558c7933b934a05f85`. Observed kernel main was `d7c35d281cf50564996fb0b9c41676b64f13f970`; the older compatible ancestor was chosen at the Torch/oneAPI dependency boundary. Do not blindly replay this pair when resuming.

## Durable records, remaining problems and resume

Records under `docs/updates/issue17-20260920-snapshot-audit/`: `CURRENT.md`, `RESUME.md`, `status.json`, `pause-checkpoint.json`, `manifest.json`, `commit-map.{json,md}`, `reconciliation.md`, `audits/` (all 27 reports), `port-map.md`, `pair-compatibility.md`, `dependencies.md`, `packaging-delta.*`, `diff-metrics.json`, `draft-identities.json`, baseline lock and remote-ref snapshots. `issue17.md` is the synchronized current body; `issue17-before-pause.md` and the original `issue17.json` preserve the prior requirements/evidence.

Ignored artifacts under `benchmark-results/issue17-20260920/`: synchronized `CURRENT.md`/`RESUME.md`/`status.json`; `pause-inspection.json`; original `audit-integrity.json`; pre-edit records archive `pause-preedit-records.tar.gz`; dependency evaluation/build/smoke logs; Node build logs; failed vLLM hook logs; `precommit-nix-elf-repair.json`; checkpoint metadata and upstream PR evidence. Issue API before/after bodies and verification hashes are retained here. `pause-artifacts.json` will index retained artifact hashes after the body update is verified. Source clones, `.venv`, hook adapters and caches remain under `build-dev/issue17/`. No cleanup is authorized.

Unresolved local details are recorded in the audits and port map: V2 CPU upper bounds cannot stand in for exact accepted history; target initialization needs correct layer ownership and MTP needs draft-owned prefill metadata; DFlash reservation/insertion/flush/commit order must be explicit; D128 natural records do not make the D256-only compact native factory compatible; per-slot recurrent state and bounded draft windows need independent budgeting alongside the shared target KV pool; physical window pages differ from logical global block-table width. Preserve HiSparse placement/registry roles, null pages, pool callbacks, worker projection fields and lifecycle/reset behavior. Cold-compile pruning, minimal GDN barriers and native attention optimizations still have controlled validation/retention gates. Upstream quantized projection applicability and local Nix hook setup also remain unresolved.

**Resume procedure, only after the user's explicit request:** read this issue and `CURRENT.md`/`RESUME.md`/status first; recheck live processes/services, all checkout/remote/recovery/release identities and retained artifact hashes; check the published v0.32-series release and all five upstream criteria; reassess the paired dependency/API/ABI constraints and all audit decisions against the then-selected source. Preserve these frozen refs and reports as history. Inventory added/changed commits and give each new fork commit its own audit agent; revalidate existing reports instead of treating their old conclusions as approval to replay. Version the new candidate only after that reassessment. Do not start either proposed phase without explicit resume authorization and resolution of the scope proposal. GPU maintenance and host activation remain separate handoffs; no automatic activation or service restart follows a future release.

## Proposed future scope split — awaiting user approval

1. First qualify **Model Runner V2 with the existing K4V2 bundled MTP option and vision**, including ownership, exact-history, lifecycle and regression checks.
2. Then undertake **pinned quantized syvai DFlash2 with K4V2 on both caches**, and its profiling, performance and capacity investigation.

This is a proposal only. Neither phase has started or is authorized by this pause update. It does not silently replace the original direct-DFlash2 end-to-end requirements or create an intermediate BF16 serving milestone. On explicit resume, obtain the user's decision about the split before executing either phase. All original final obligations remain: minimized clean forks, reproducible coordinated publication, profiling before tuning, a serious measured MTP-relative speedup attempt, 60 tok/s only as a stretch target, two independent adversarial reviews before accepting a limit, bounded quant comparisons, full qualification, rollback and the stock-MTP versus DFlash2 speed/usable-KV-capacity comparison. Missing essential data during a future investigation must remain a documented blocker; **this checkpoint is paused by request, not blocked on a claimed performance limit**.
