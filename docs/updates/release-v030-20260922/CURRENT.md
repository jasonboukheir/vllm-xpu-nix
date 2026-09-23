# v0.30 release: actual Brutus qualification accepted

The user selected Brutus's real settings and reasonable output as release
acceptance. Those checks passed; see `release-acceptance.json`. Extra experimental
comparisons are closed for this release, with findings and unmeasured claims
preserved. Issue #17 remains paused; checkpoint draft PR #18 is unrelated.

All 27 source commit audits and four packaging patch audits were reconciled.
The clean source stacks have five vLLM and eight kernel commits. Published mains:
- vLLM `466b9d5abe84f13c29b38c686e4d77fc09e09aa4` on v0.30.0
  `ced6857afa0ea7b2e3f0846a62e1394e90f15607`.
- Kernels `193cee080d654ff67c4f8bfaf924201bc88b9cf5` on v0.1.15
  `1c7cbeee1cd0c1481d48f5031f679a47f3f0ef45`.

The exact Brutus native build passed, as did 608 packaging tests, 705 installed
native/backend cases and 14 supplemental GDN reference cases. Actual eager V1
K4V2 target/draft MTP2 with vision passed text, reasoning, two images, tools,
four-way concurrent requests, mixed prefill, cancellation/reuse and retrieval
at 131071/262015 input tokens. Two concurrent 172415-token prompts plus 512 outputs
each completed correctly without preemption. Both candidate and published
baseline provided 346880 usable attention tokens under the normal 0.96 budget.

Optional AEON exact concurrent output invariance fails on both released and
candidate stacks for the retained duplicate fixture. Candidate-only mixed-long
diagnostics also diverge; they have no matched baseline attribution. Historical
unused-BF16-NaN diagnostics retain six failures and two writer passes. None of
these are represented as passes. No controlled performance comparison was run;
no new speedup, performance non-regression or DFlash ceiling is claimed.

All qualification workers exited. Chat is failed with MainPID/ControlPID 0,
empty cgroup and no tasks. Preserve the user-stopped state; no host repin or
activation was requested. xpu-v1.10.0 remains the host's operational baseline.

Source xpu-v1.11.0 annotated tags and release branches are now published and
verified. Packaging pins those tags; Brutus package/runtime and generic/kernel
derivations are identical to qualification. Final flake evaluation and all four
lightweight packaging checks passed. Next: commit and publish packaging main,
release branch and annotated tag, then record remote verification. Read `resume.md`;
never retarget an existing release reference.

Private raw artifacts: `benchmark-results/release-v030-20260922/`. Pre-decision
status documents are retained under `before-user-scope-decision/` there. Original
audit report hashes remain verified. Two untracked issue17 report directories and
pre-existing host lock changes remain outside this release.
