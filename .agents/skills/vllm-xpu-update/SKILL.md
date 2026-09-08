---
name: vllm-xpu-update
description: Advance paired vLLM and XPU-kernel main branches onto compatible upstream releases, minimize fork drift with one audit agent per commit, and qualify the Nix stack for coordinated releases. Invoke explicitly for release checks, upgrades, patches, or an existing release handoff in vllm-xpu-nix.
---

# Update the vLLM XPU stack

This skill belongs to the `vllm-xpu-nix` project and is explicitly invoked as
`$vllm-xpu-update`. Its invocation policy disables automatic selection. Do not
install it globally or add it to another repository's default instructions.

## Choose the requested operation

- **Default invocation / update if newer**: discover the newest eligible stable
  upstream pair relative to the current integration bases. If newer, execute
  the complete source refresh: preserve recovery refs, assign one audit agent
  per fork commit in both repos, reconcile KEEP/DROP/ADAPT decisions, rebuild
  the minimal patch stacks, validate and publish both source `main` branches,
  then prepare and qualify the paired Nix candidate. Resume an existing matching
  candidate rather than repeating its rebase. If there is no newer pair and no
  unfinished candidate, report the comparison and make no changes.
- **Check only**: report integration, deployed, and available compatible pairs;
  make no repository or service changes. This explicit mode overrides the default.
- **Clean up current main**: run the same per-commit audit/replay workflow on
  the recorded current bases, even without a newer release. Collapse superseded
  changes and follow-up repairs into a minimal, reviewable stack.
- **Patch a release**: keep its upstream bases and unrelated dependency pins;
  apply the requested fix and qualify a new patch release. Apply a relevant fix
  to integration main too without including its newer baseline in the patch.
- **Validate a candidate or model**: use the packaging and foreground-testing
  stages for the requested configuration.
- **Release / deploy / resume**: inspect completed stages and actual remote refs,
  then continue the requested handoff. Publish a stable release only after its
  qualification passes; deploy only when requested. Never retarget a release.

For example:

> $vllm-xpu-update

Runs the conditional main refresh and patch-cleanup workflow above. To include
stable publication and the Brutus handoff:

> $vllm-xpu-update Update to the latest compatible stable pair, clean up every
> fork patch, and when it passes validation, cut a release and pin Brutus.

The default refresh includes publishing the audited source `main` histories,
including a backed-up rebase with exact leases. It does not implicitly request
stable tags or deployment. Follow the existing maintenance/operator handoffs
for GPU testing and host rebuilds; retain authorization already given rather
than asking again. A check-only request remains read-only.

## Read the relevant procedure

[references/workflow.md](references/workflow.md) contains the maintained
procedure and commands. Read the execution-host and repository guidance, then
only the stages needed for the current operation:

- Discovery: section 0, including release comparison and pair compatibility.
- Source updates: sections 1–4, including baseline preservation, focused fork
  audits, isolated replay, and publication with backups and exact leases. Read
  [references/commit-audit.md](references/commit-audit.md) before dispatching
  agents; it defines their evidence, ownership, and acceptance requirements.
- Packaging and GPU qualification: sections 5–6. Use the paired package with
  Brutus's overrides; inspect real HTTP readiness and generation.
- Version selection, stable patches and publication: section 7. Read it before
  choosing or creating release refs, then finish the section 6 Brutus handoff.
- Interrupted work: Recovery and Completion record.

## Preserve these invariants

- Run host, repository and GPU operations locally on Brutus. The packaging repo
  is `~/Projects/vllm-xpu-nix`, source forks are its siblings, and the consuming
  host configuration is `~/.config/nix`.
- Select published stable upstream releases, pin their tags and exact commits,
  and verify the vLLM/kernel dependency pair. Do not target upstream `main`,
  prereleases or nightlies unless the user explicitly requests that exception.
- Query both repositories. Prefer the newest eligible vLLM release and a
  compatible released kernel version; respect exact upstream dependency pins.
  Distinguish no update available from failed discovery or an incompatible pair.
- Use independent coordinated `xpu-vMAJOR.MINOR.PATCH` stack versions: major
  for incompatible changes requiring migration, minor for a new upstream base
  in either fork or compatible features, patch for compatible fixes on the same
  pair of upstream bases. Keep package versions tied to their upstream source.
- Minimize the effective upstream delta while preserving our correctness fixes,
  supported features, and measured performance improvements. Audit every commit,
  including follow-ups/reverts and commits in an unchanged companion fork;
  reconcile dependencies across both repos before dropping anything.
- Preserve immutable releases and unrelated locks. Build and test the paired
  candidate before publishing stable packaging or repinning Brutus. Source-only
  checks and old binary tests are not qualification of a newly built stack.
- Carry the existing maintenance and operator rebuild handoffs forward. If the
  user has already confirmed maintenance, verify it and continue. Prepare all
  authorized work before asking for any still-required operator action.
- Keep release metadata, source SHAs, commands/results and remaining limitations
  in the packaging release notes. Use `tea` for requested Forgejo issue context,
  and `gh` or official upstream metadata for GitHub releases. Do not post issue
  comments or other messages unless the user requests them.
