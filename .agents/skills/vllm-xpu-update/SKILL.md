---
name: vllm-xpu-update
description: Refresh paired vLLM and XPU-kernel forks, audit and minimize patches, qualify Brutus, publish a coordinated release, and pin the host for an operator rebuild. Invoke explicitly for upgrades, release checks, patches, or an existing release handoff in vllm-xpu-nix.
---

# Update the vLLM XPU stack

This skill belongs to the `vllm-xpu-nix` project and is explicitly invoked as
`$vllm-xpu-update`. Its invocation policy disables automatic selection. Do not
install it globally or add it to another repository's default instructions.

An explicit update invocation, including `$vllm-xpu-update` alone or
`$vllm-xpu-update update to 0.32`, requests the full workflow through release
publication and the committed Brutus pin. No separate publication or host-pin
request is needed. Explicit narrower requests override this default. The
operator performs the actual system rebuild/switch.

## Choose the requested operation

- **Default invocation / update if newer**: discover the newest eligible stable
  upstream pair relative to the current integration bases. If newer, execute
  the complete source refresh: preserve recovery refs, assign one audit agent
  per fork commit in both repos, reconcile KEEP/DROP/ADAPT decisions, rebuild
  the minimal patch stacks, validate and publish both source `main` branches,
  then build and qualify the paired Nix candidate with Brutus's actual model and
  service settings. After qualification passes, push packaging main and new
  coordinated release branches/tags in all three forks, update and commit the
  Brutus release pin, verify the exact tested package, and hand off the rebuild
  to the operator. Honor a named upstream version; otherwise select the newest
  eligible stable pair. Resume an existing matching candidate or unfinished
  release handoff rather than repeating completed work. If there is no newer
  pair and no unfinished work, report the comparison and make no changes.
- **Check only**: report integration, deployed, and available compatible pairs;
  make no repository or service changes. This explicit mode overrides the default.
- **Source only / qualify only**: stop at the explicitly requested boundary.
  Source-only work ends after the audited source mains are published;
  qualification-only work records the tested candidate without creating release
  refs or changing the host pin.
- **Clean up current main**: run the same per-commit audit/replay workflow on
  the recorded current bases, even without a newer release. Collapse superseded
  changes and follow-up repairs into a minimal, reviewable stack.
- **Patch a release**: keep its upstream bases and unrelated dependency pins;
  apply the requested fix and qualify a new patch release. Apply a relevant fix
  to integration main too without including its newer baseline in the patch.
- **Release current main / keep the current base**: freeze both integration
  tips and their actual upstream bases, including an existing snapshot when the
  user chooses it. Apply the requested fix without a rebase or full-stack
  cleanup. Compare with the preceding stable release to choose the version;
  unchanged integration bases can still require a minor bump relative to that
  release. Qualify the rebuilt pair before creating stable references.
- **Validate a candidate or model**: use the packaging and foreground-testing
  stages for the requested configuration.
- **Release / deploy / resume**: inspect completed stages and actual remote refs,
  then continue the requested handoff. Publish a stable release only after its
  qualification passes. The operator performs the system rebuild/switch
  unless explicitly delegated to the agent. Never retarget a release.

For example:

> $vllm-xpu-update update to 0.32

This requests patch audits, minimization, the actual Brutus foreground test and
repair loop, performance review, publication of all three mains and permanent
release branches/tags, and the committed host release pin. Bare invocation uses
the same scope when a newer eligible stable pair exists. Review new features
for the current one/two-user setup; use a dedicated performance agent and
matched measurements before promoting a serving change.

Publishing source mains includes backed-up rebases with exact leases. Follow
the existing maintenance handoff for GPU testing and preserve service state;
retain authorization already given rather than asking again. Complete the
release and host-pin work before handing off the operator rebuild. Check-only
remains read-only; an explicit source-only or qualification-only request stops
at that boundary.

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
- Compare release lineage before proposing an update. A later publication date
  or version label does not make an older release-branch cut a forward update
  from an integration snapshot. Retain that snapshot when requested; treat a
  move to an older release base as a separate, explicit migration.
- Query both repositories. Prefer the newest eligible vLLM release and a
  compatible released kernel version; respect exact upstream dependency pins.
  Distinguish no update available from failed discovery or an incompatible pair.
- Use independent coordinated `xpu-vMAJOR.MINOR.PATCH` stack versions: major
  for incompatible changes requiring migration, minor for a new upstream base
  in either fork or compatible features, patch for compatible fixes on the same
  pair of upstream bases. Keep package versions tied to their upstream source.
- Minimize the effective upstream delta while preserving our correctness fixes,
  supported features, and measured performance improvements. For an upstream
  refresh or full-stack cleanup, audit every commit, including follow-ups/reverts
  and commits in an unchanged companion fork; reconcile dependencies across both
  repos before dropping anything. A targeted release on retained bases uses
  focused review and qualification of the changes since the preceding release.
- Preserve immutable releases and unrelated locks. Build and test the paired
  candidate before publishing stable packaging or repinning Brutus. Source-only
  checks and old binary tests are not qualification of a newly built stack.
- Carry the existing maintenance and operator rebuild handoffs forward. If the
  user has already confirmed maintenance, verify it and continue. Prepare all
  authorized work before asking for any still-required operator action.
- Put update summaries, release rationale, validation outcomes and host handoffs
  in descriptive commit messages. Do not create Markdown status logs, changelogs
  or per-release narrative summaries. Preserve machine-readable release
  manifests and validation evidence; maintain the skill/runbook, technical audit
  reports and useful operating documentation.
- Use `tea` for requested Forgejo issue context, and `gh` or official upstream
  metadata for GitHub releases. Do not post issue comments or other messages
  unless the user requests them.
