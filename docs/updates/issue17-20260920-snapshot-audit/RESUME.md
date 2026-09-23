# Issue #17 — resume only on explicit user request

Status: **PAUSED AT USER REQUEST**. Do not implement, rebase, optimize, qualify,
publish, repair hook environments or start GPU work from this checklist while
paused. Upstream v0.32 is a planned reassessment point, not an automatic trigger.

After explicit resume authorization:

1. Read the live [issue body](https://git.sunnycareboo.com/jasonbk/vllm-xpu-nix/issues/17),
   `CURRENT.md`, `PAUSED.md`, `status.json`, `pause-checkpoint.json`, `manifest.json`,
   `commit-map.json`, `reconciliation.md`, `port-map.md` and the linked audits.
   The 27 completed audits apply to the September frozen snapshots. One kernel
   commit was replayed; the final minimized pair was never qualified.
2. Reinspect the five checkouts and compare their state to `pause-checkpoint.json`:

   ```sh
   hostname -s
   git status --short
   git rev-parse HEAD
   git -C build-dev/issue17/vllm status --short
   git -C build-dev/issue17/vllm rev-parse HEAD
   git -C build-dev/issue17/vllm-xpu-kernels status --short
   git -C build-dev/issue17/vllm-xpu-kernels rev-parse HEAD
   git -C ../vllm status --short
   git -C ../vllm rev-parse HEAD
   git -C ../vllm-xpu-kernels status --short
   git -C ../vllm-xpu-kernels rev-parse HEAD
   git -C build-dev/issue17/vllm for-each-ref refs/recovery/ refs/snapshots/
   git -C build-dev/issue17/vllm-xpu-kernels for-each-ref refs/recovery/ refs/snapshots/
   git ls-remote origin refs/heads/main 'refs/tags/xpu-v*'
   git -C build-dev/issue17/vllm ls-remote origin refs/heads/main refs/heads/recovery/issue17-20260920-main 'refs/tags/xpu-v*'
   git -C build-dev/issue17/vllm-xpu-kernels ls-remote origin refs/heads/main refs/heads/recovery/issue17-20260920-main 'refs/tags/xpu-v*'
   systemctl show vllm-xpu-chat.service -p ActiveState -p SubState -p MainPID -p ExecStart
   readlink -f /run/current-system
   ```

   Check goal/agent status and only goal-owned background processes. Do not use
   broad process termination. Preserve recovery/release refs and unrelated work.
   Reverify report and artifact hashes against `commit-map.json` and
   `benchmark-results/issue17-20260920/pause-artifacts.json`.
3. Recheck upstream at that time: a published v0.32-series release; compatible XPU
   kernels/Torch/Triton/oneAPI; XPU Qwen/MTP/vision V2 regression evidence; merged,
   checkpoint-applicable quantized loading/mapping/context-projection fixes
   (#53122 or equivalent); #56787 resolution or an understood supported execution
   workaround; and sufficiently stable metadata/cache APIs. Do not reserve a patch
   version or treat V1 removal as a stabilization promise. If the criteria are
   insufficient, report the concrete gaps before implementation.
4. Resolve the proposed scope split with the user: V2 existing K4V2 MTP+vision
   first, then quantized DFlash2/performance. It is not yet approved. Preserve the
   original end-to-end obligations and no intermediate BF16 serving milestone.
5. Reconcile all audits against the new pair; retain applicable evidence and
   re-audit changed requirements. Inventory any new fork commits individually.
   Preserve the September refs and partial replay; create a separately identified
   candidate only when authorized to proceed. Do not blindly resume k02 or replay
   the September map unchanged. Record exact dependency/source identities and
   remaining local ownership/history/window/allocation/reset work before edits.
6. When later authorized work reaches builds/tests, inspect the retained Nix hook
   failures and use the source repositories' current AGENTS.md instructions.
   Ignored lint `.venv` environments are not installed GPU runtime environments.
   Kernel k01 hooks passed; vLLM hook repair remains incomplete. The realized
   Node 24 output alone does not complete that repair.
7. Any GPU maintenance run needs fresh ownership checks and the issue's separate
   maintenance handoff. Trials remain serial. Published release xpu-v1.10.0 is
   the preserved rollback/control. Host activation remains a separate operator
   handoff, including after a future coordinated release.

Actual host consumer lock at pause:
`~/.config/nix/modules/flake/nixos/server/flake.lock`; host package override:
`~/.config/nix/hosts/brutus/services/vllm-xpu/package.nix`. Inspect their current
state rather than reusing old root-flake activation commands.

No DFlash2 speedup, performance ceiling or remaining-headroom bound is established.
The historical native-kernel data in audit reports does not provide one. Future
optimization must first validate profiling and measurements and retain the two
independent adversarial reviews and all original final qualification gates.
