# vLLM XPU update procedure

Detailed reference for [$vllm-xpu-update](../SKILL.md). Read only the stages
needed for the requested operation. Paths in commands refer to the Brutus
checkouts; the skill itself lives in the packaging repository.

## Contents

- Execution host and repository policy
- 0: Discover a newer compatible released pair
- 1–4: Preserve, audit, replay and publish source changes
- 5–6: Prepare packaging, build, qualify and hand off Brutus
- 7: Select a version and publish immutable coordinated releases
- Recovery and completion record

## Execution host

Run this workflow from **Brutus itself**. The repositories under `~/Projects`,
the host configuration under `~/.config/nix`, systemd, and the loopback-only
vLLM APIs are all expected to be local to Brutus. Do not SSH to `brutus` from
inside the workflow: first verify the current host instead.

```bash
test "$(hostname -s)" = brutus
```

Stop if that check fails. Move the session to Brutus before fetching, editing,
rebuilding, inspecting services, or testing `127.0.0.1:8000`/`:8001`.

## Repositories and update policy

| Repository           | Local path                    | Update policy                                                   |
| -------------------- | ----------------------------- | --------------------------------------------------------------- |
| vLLM fork            | `~/Projects/vllm`             | Rebase fork commits onto a selected stable upstream release tag's commit |
| XPU kernels fork     | `~/Projects/vllm-xpu-kernels` | Select a compatible released kernel version; audit and rebase when it changes |
| Nix packaging        | `~/Projects/vllm-xpu-nix`     | Refresh inputs, validate, commit, and push                      |
| Brutus configuration | `~/.config/nix`               | Update only the `vllm-xpu-release` input, then rebuild and test |

The `origin` remote is the Sunnycareboo fork and `upstream` is the corresponding
GitHub project. Stop if that is not true.

`main` is the rolling integration branch in the two source forks. Its history
may be rebased and published with `--force-with-lease`, so it is not a durable
record of deployed stacks. Preserve working releases with coordinated release
branches and tags as described in [Publish a coordinated stack
release](#7-publish-a-coordinated-stack-release).

The upstream baseline is a **specific published release**, resolved to its exact
commit. Upstream `main`, nightly builds, release-candidate tags, and moving
release branches are not automatic update targets. Our fork's branch name
`main` does not change this policy. An explicit request for an experimental
snapshot is a separate workflow and must be recorded as such.

The Nix packaging declares an exact nixpkgs revision in `flake.nix`. Do not add
a downstream `nixpkgs.follows`: the PyTorch/XPU closure must remain on the
package set tested by `vllm-xpu-nix`.

## Workflow stages

1. **Resolve the operation and eligible pair.** Bare invocation runs the
   conditional refresh defined in SKILL.md. Run section 0 first; resume an
   existing matching candidate without replaying completed work. With no newer
   pair or unfinished candidate, stop without mutations. Explicit check-only
   stops after discovery; an explicit same-base cleanup or patch uses its
   recorded bases instead of requiring a newer release.
1. **Preserve the working baseline.** Record the current source tips and bases,
   validate their relationship to the deployed stack, preserve its immutable
   release, and back up both integration tips before rewriting either fork.
   Integration main can contain unqualified work; a backup preserves that work
   without declaring it stable. Preserve unrelated dependency locks.
1. **Audit upstream changes.** Freeze the selected release targets, dispatch one
   focused subagent for each fork commit in bounded batches, and reconcile every
   KEEP/DROP/ADAPT decision before applying changes.
1. **Rebase rolling main.** Preserve one feature or fix per commit, validate both
   source stacks together, and publish using the previously recorded remote tips
   as exact leases. Existing release references remain unchanged.
1. **Integrate the new upstream stack separately.** Refresh packaging only after
   the source rebase, then build and qualify the paired candidate through the
   maintenance and
   foreground-test workflow. The default ends with published source mains and
   a qualified candidate, or an explicit record of remaining validation.
   Stable publication and the operator rebuild handoff run when requested.
   A successful rebase alone does not authorize declaring the
   new upstream stack stable. A newly adopted upstream baseline starts a new
   minor release line, unless compatibility requires a major bump. If the
   request ends at source publication, stop there with the baseline release and
   host pins preserved.

## 0. Discover a newer eligible upstream release pair

For an explicit same-base cleanup or stable patch, recover and verify the
selected line's existing bases instead of requiring a newer upstream release.
An existing snapshot base may be retained for that operation; this is not
permission to adopt another untagged snapshot. Check-only never progresses
beyond read-only discovery, even when a newer pair is available.

Read Brutus's pinned packaging release and its manifest/notes to identify the
**currently deployed upstream vLLM release**, kernel release, exact upstream
base SHAs, and fork tips. Do not compare against our `xpu-v*` version or assume
that the local checkout's `main` is deployed. Inspect any already prepared newer
candidate separately so a repeated request can resume it instead of duplicating
work or retargeting published refs. Record the current integration pair from
both source mains and their audit records as well. Use that pair for deciding
whether main can advance; the deployed pair is the recovery/consumer baseline.
A stale host pin must not trigger another rebase of already updated mains. If
one main is ahead of the other, recover the recorded paired candidate before
starting a new refresh. Never regress either integration component silently.

Query official upstream release metadata. For example, these read-only commands
list published, non-prerelease GitHub releases:

```bash
for upstream_repo in vllm-project/vllm vllm-project/vllm-xpu-kernels; do
  gh api --paginate "repos/$upstream_repo/releases" \
    --jq '.[] | select(.draft == false and .prerelease == false) | [.tag_name, .published_at, .html_url] | @tsv'
done
```

Compare parsed release versions using the project's version conventions, not
lexical tag order, publication dates, or the most recent Git commit. Exclude
drafts, prereleases and nightlies unless explicitly requested. If the kernel
project distributes final releases through tags or wheels without a GitHub
release entry, verify that version against its official release documentation
and resolve the matching source tag; do not substitute a moving branch.

Check both repositories on every discovery run. For each prospective vLLM
release, inspect its XPU dependency requirements and release notes alongside the
kernel release's supported vLLM versions, operator schemas and toolchain needs.
Prefer the newest eligible vLLM release, then the newest kernel release
allowed by its compatibility constraints, with neither component older than its
integration version (or the explicitly selected release line for a patch).
Honor an exact upstream kernel pin unless a different pair has explicit
compatibility evidence. Map any required kernel wheel version to its actual
source release/tag and record both when their labels differ.

A newer compatible kernel release can trigger an update even when vLLM's release
is unchanged, and vice versa. Keep the unchanged component at its existing exact
base commit when possible, but still audit its local commits for redundancy
or changed cross-repository requirements. If the newest release in one repo
requires an unreleased
counterpart, exclude that pair and consider the next newer compatible pair.
If none exists, retain the current pair. Do not substitute either repository's
`main` or silently downgrade the other component. Report excluded newer versions
and the compatibility reason.

Record the current and proposed version pair, official release URLs, selected
tag names, tag-object IDs where applicable, and peeled commit SHAs. Inspect the
selected release tags' requirements before deciding which Torch, Triton, oneAPI
or packaging changes are necessary. Resolve compatibility gaps before any
publication; do not silently fall back to upstream `main`.

- **No newer compatible pair:** report current, latest available, and eligible
  versions for both repositories and stop unless the requested operation is
  resuming an unfinished candidate or explicitly cleaning up current main.
  Check-only always stops here. Do not create a patch release, refresh
  dependencies, rebase, or interrupt services just because the check ran.
- **Newer compatible pair:** continue the authorized update through the
  staged workflow using the frozen pair, including the existing maintenance and
  operator rebuild handoffs. Discovery alone does not qualify it as stable.
- **Lookup failure, ambiguous version, or incompatible required dependency:**
  report what could not be established and preserve the deployed stack. A failed
  lookup is not evidence that the installation is up to date.

For older manifests that record an untagged upstream snapshot, recover the exact
baseline from the recorded source audit. Do not reinterpret the package's
nearest-release version label as proof that it was based on that release tag.
Establish whether the selected release is a forward update or a deliberate
migration from the snapshot before proceeding; never silently downgrade. Release
branches need not descend from the old upstream-main snapshot, so inspect the
actual source delta and release lineage rather than relying on ancestry alone.

## 1. Establish and preserve a safe starting point

Check every worktree before fetching or changing history:

```bash
for repo in vllm vllm-xpu-kernels vllm-xpu-nix; do
  git -C "$HOME/Projects/$repo" status --short --branch
  git -C "$HOME/Projects/$repo" remote -v
done
```

Preserve unrelated changes. Do not rebase, merge, or update locks in a dirty
worktree without first understanding and protecting those changes.

Read each repository's `AGENTS.md` or equivalent instructions before working
there.

For an authorized refresh, same-base cleanup, or resume, fetch the fork branch refs:

```bash
for repo in vllm vllm-xpu-kernels; do
  git -C "$HOME/Projects/$repo" fetch --no-tags origin \
    +refs/heads/main:refs/remotes/origin/main
done
```

For a same-base cleanup, use the already verified exact base objects; do not
require a new tag or substitute another target. For an upstream refresh,
in each source repository, fetch only its selected upstream release tag into a
dedicated namespace. `upstream_tag` is the exact tag recorded in section 0, not
a branch name. `recorded_tag_object` is the object advertised by
`refs/tags/...`; `recorded_target` is its peeled commit SHA (the same SHA for a
lightweight tag):

```bash
: "${upstream_tag:?Set the selected upstream release tag}"
: "${recorded_tag_object:?Set its recorded tag object ID}"
: "${recorded_target:?Set its recorded peeled commit SHA}"
release_ref="refs/upstream-releases/$upstream_tag"
if git show-ref --verify --quiet "$release_ref"; then
  test "$(git rev-parse "$release_ref")" = "$recorded_tag_object" || exit 1
fi
git fetch --no-tags upstream "refs/tags/$upstream_tag:$release_ref"
test "$(git rev-parse "$release_ref")" = "$recorded_tag_object" || exit 1
upstream_target=$(git rev-parse "$release_ref^{commit}")
test "$upstream_target" = "$recorded_target" || exit 1
```

Keep the tag-object ID as well when the tag is annotated. Reject a moved tag or
mismatched object; do not force-update the recorded release ref. This namespace
also avoids collisions with the fork's own immutable `xpu-v*` tags.

For each source repository, record the local `main` tip, remote `origin/main`
tip, **existing upstream base**, and fetched upstream target in a durable audit
record. Determine the existing base from the known fork stack and previous
release/cleanup records. Do not derive the old base from a merge-base with the
new release: upstream releases can be cut on a different branch. Verify that the
recorded old base is an ancestor of the existing fork and that the range above
it contains the expected fork changes. Stop and investigate ambiguous ancestry
or unexpected local/remote divergence.

```bash
old_main=$(git rev-parse main)
expected_remote=$(git rev-parse refs/remotes/origin/main)
: "${old_base:?Set the previous upstream base SHA from the manifest or audit}"
git merge-base --is-ancestor "$old_base" "$old_main" || exit 1
git log --reverse --format='%H %s' "$old_base".."$old_main"
```

Keep separate values for each repository; the variables above illustrate the
recording step in one checkout. Use full immutable SHAs from the record during
audits and rebases, even if upstream moves again. Create uniquely dated recovery
branches for both local and published tips before editing either history; do not
reuse an existing backup name. Push backups explicitly so recovery does not rely
on one worktree or reflog:

```bash
stamp=$(date +%Y%m%dT%H%M%S)
git branch "backup/main-pre-rebase-$stamp" "$old_main"
git branch "backup/origin-main-pre-rebase-$stamp" "$expected_remote"
git push origin \
  "refs/heads/backup/main-pre-rebase-$stamp:refs/heads/backup/main-pre-rebase-$stamp" \
  "refs/heads/backup/origin-main-pre-rebase-$stamp:refs/heads/backup/origin-main-pre-rebase-$stamp"
```

Before rewriting, verify that the deployed stable stack is preserved by its
existing coordinated release refs and exact manifest. Reuse those refs; do not
require the current experimental main tips to match a stable tag. Dated remote
backups preserve all additional integration work. Do not create a release or
repin Brutus merely to clean up main. If a working deployed stack has no durable
release, preserve its exact inputs and qualification evidence and prepare the
missing release handoff; never label untested integration work stable.

The September 2026 cleanup was preserved as `xpu-v1.7.2`; this is historical
context, not a hard-coded current release. A history-only rewrite can reuse
validation only when source trees and packaging/build inputs are proven
equivalent; commit-derived versions can change the derivation even when trees
match. Record that evidence and distinguish it from a new integration test.
Any behavior or build-input change needs appropriate validation. If recovery
refs cannot be preserved remotely, continue read-only audits and isolated
candidate validation while leaving published mains unchanged.

## 2. Audit every commit in both source forks

Read [the commit-audit contract](commit-audit.md). Use one focused read-only
subagent per fork-only commit in **both** repositories, including an unchanged
companion fork, follow-ups and reverts. Enumerate from each recorded old base
to its current main tip; do not use the deployed stable patch list as a substitute.
Dispatch independent audits in batches that fit the available concurrency; one
agent can complete its bounded audit before a new agent takes the next commit.
Maintain one distinct assignment and
artifact per commit, with a coverage check against both complete inventories.
If delegation is unavailable, record that limitation; do not claim the requested
one-agent-per-commit audit completed through an undifferentiated local review.
Continue useful read-only investigation and preparation, but defer replay and
publication until every required agent audit is complete or the user explicitly
changes the requested audit method.
Each receives the exact old base, fork commit and stack tip, frozen upstream
target, repository guidance, and relevant cross-repository dependencies. Audits
must not edit branches or resolve conflicts concurrently in a shared worktree.
One operator reconciles all findings before rewriting history.

The objective is the smallest necessary delta from upstream, including when
upstream solved the same problem through a different implementation. Do not
allocate an audit per upstream commit. Use affected paths, symbols, behavior,
tests, and related commit or PR subjects to narrow the upstream delta first.
Merge conflicts and matching patch IDs are evidence, not verdicts.

Each audit must record:

1. The fork commit SHA, purpose, affected paths, patch ID, and dependencies on
   other fork commits or kernel/operator APIs.
1. Relevant upstream commits and current implementation paths at the frozen
   target. Investigate new features, alternate implementations, removals,
   refactors, and fixes that might invalidate the original premise.
1. Whether the deployed XPU behavior is equivalent, including fallback paths,
   speculative decoding, cache sizing, and teardown where relevant. For a
   performance patch, identify the workload, hardware, numerical expectations
   and baseline metrics needed to preserve its benefit; functional equivalence
   alone is insufficient evidence to drop it. Verify the
   behavior rather than requiring upstream to have the same textual patch.
1. A **KEEP**, **DROP**, or **ADAPT** recommendation with supporting evidence.
   DROP requires equivalent behavior or a documented removal of the requirement;
   absence of a conflict is not a reason to KEEP. For partial overlap, ADAPT only
   the residual missing behavior and identify which portions upstream replaces.
1. Targeted checks or existing tests that support the verdict, unresolved risks,
   and cross-repository ABI/operator-schema or packaging implications. Clearly
   distinguish checks actually run from proposed XPU runtime validation.

Useful commands, using the recorded immutable values for that repository:

```bash
git show --stat <fork-commit>
git show <fork-commit> | git patch-id --stable
git cherry -v "$upstream_target" "$old_main"
git log --oneline "$old_base".."$upstream_target" -- <affected-path>
git log "$old_base".."$upstream_target" --oneline --grep='<subject keyword>'
git grep -n '<changed symbol>' "$upstream_target" -- <affected-path>
```

Reconcile all audits into one ordered disposition table per repository before
rebasing. Resolve disagreements, dependencies, and partial overlaps explicitly.
Inspect `pyproject.toml`, `setup.py`, `CMakeLists.txt`, toolchain requirements,
operator schemas, and source layout across both targets. A vLLM-side DROP must
not leave a kernel-side dependency unexplained, or vice versa. Preserve audit
evidence with the completion record.

## 3. Rebase both source stacks

Work from the recorded source tips and target SHAs, preferably in isolated
worktrees so ordinary checkouts stay usable. Use an explicit old base:

```bash
git rebase -i --onto "$upstream_target" "$old_base" main
```

Apply the reconciled KEEP/DROP/ADAPT decisions, stopping to inspect conflicts
against the intended behavior. Revise an audit if the actual port reveals new
evidence. Keep one upstream-reviewable feature or fix per commit. Remove net-zero
patch/revert pairs after confirming their combined effect, fold later repairs
into their owning patch, and remove dead feature-specific plumbing together
across repositories. Map every old commit to a retained/adapted new commit or
an evidence-backed drop; never replay a retired workaround simply because it
appears earlier in history. All KVarN
changes belong in one coherent commit **per repository**; unrelated fixes remain
separate. Squash temporary conflict-resolution or follow-up repairs into their
owning feature commit, preserving authorship and repository sign-off rules.

If a fork has no remaining local changes, its resulting `main` should point to
the frozen upstream target; fast-forward without rewriting when ancestry allows.
Use the same selected release commits throughout the workflow. Do not switch to
upstream `main`, a moving release branch, or a newly published release halfway
through the update. A newer discovery belongs to a subsequent update.

Review the new stack against both the audit and the old stack:

```bash
git merge-base --is-ancestor "$upstream_target" main
git log --reverse --format='%H %s' "$upstream_target"..main
git range-diff "$old_base".."$old_main" "$upstream_target"..main
git diff --check "$upstream_target"..main
```

Run meaningful targeted checks for the retained or adapted behavior, plus cheap
import/build-configuration checks where feasible. Validate cross-repository
interfaces together. Record unavailable checks rather than treating a clean
rebase or static validation as an XPU runtime pass. Native build and deployment
validation follow the separate integration stage below.

## 4. Publish audited rolling main branches

Publish only after both resulting histories have been reviewed and their
cross-repository dependencies reconciled. Compare the actual remote tip with the
recorded expected tip; an unexpected movement requires investigation, not a new
lease copied blindly from the remote.

Use a normal push for a fast-forward update, including an ordinary fix on
the current baseline. When the authorized audited rebase rewrites history, use
that repository's previously recorded full SHA as the exact lease:

```bash
git ls-remote origin refs/heads/main
git push --force-with-lease="refs/heads/main:$expected_remote" \
  origin refs/heads/main:refs/heads/main
```

Never use unconditional `--force` or rely on an implicit tracking-ref lease,
which a background fetch can change. Verify the published tips after pushing.
Persist the paired audit and old-to-new commit maps under packaging
`docs/updates/<unique-run-id>/` with the selected bases, backup refs and
validation status, so a subsequent invocation can resume without `/tmp` or
conversation history. When publication is in scope, commit only these intended
artifacts; preserve unrelated working changes. Record partial publication if
only one push succeeds; keep stable packaging and
Brutus pinned to the preserved baseline while resolving it. Do not move any
release branch or tag to the rebased commits, and do not create a stable release
for rebased `main` until its integration checks pass.

## 5. Refresh `vllm-xpu-nix`

This is the **paired packaging integration stage**, after section 4. It also
applies to same-base cleanup when source behavior or build inputs change.
Reuse prior validation only with the equivalence evidence required in section
1. Skip it when merely preserving the unchanged baseline in section 1.

For the default main refresh, change only the selected source pins and
dependencies required by the chosen pair. Preserve unrelated nixpkgs, Torch,
Triton and oneAPI pins. Checking for newer source releases does not imply a
broad refresh of every movable dependency.

For a requested full stack refresh, update the exact nixpkgs pin by resolving the
current `nixos-unstable` revision:

```bash
nix flake metadata github:NixOS/nixpkgs/nixos-unstable
```

Copy the full 40-character revision into the `nixpkgs.url` in
`~/Projects/vllm-xpu-nix/flake.nix`. This deliberate source edit is what keeps
downstream host-wide updates from silently moving PyTorch.

In the local integration candidate, replace the previous stable source tag refs
with the newly published rolling source refs, locking them to the exact audited
source SHAs. Leave existing release manifests unchanged. For the default scoped
update, refresh only the paired source inputs and any explicitly required
dependency inputs:

```bash
cd ~/Projects/vllm-xpu-nix
nix flake update vllm-xpu-unstable-src vllm-xpu-kernels-unstable-src
git diff --check
git diff -- flake.nix flake.lock
```

Only a requested full stack refresh uses unscoped `nix flake update`. For either
mode, verify both resolved source SHAs equal the audited pair before building;
never allow a concurrent push to either fork to change the candidate silently.

Audit every downstream source patch under `nix/patches/` against the newly
locked sources before committing. Use one bounded read-only audit per patch when
parallel help is available. For each patch, identify its target and build-time
purpose, inspect the current upstream/fork implementation and affected build
files, and classify it as keep, drop, or modify. A patch applying without a
conflict does not prove that it remains necessary; conversely, a conflict may
only reflect surrounding refactoring. Cheap checks such as `patch --dry-run`, a
standalone CMake configure/parser test, or derivation evaluation are preferred
here. Defer native compilation to Brutus.

Confirm that the unstable source locks point to the fork commits just pushed.
Set each `base` used by `mkInputVersion` in `flake.nix` to that component's
selected upstream release version. Verify that the recorded release commit is
an ancestor of the corresponding fork tip. Do not infer a base version from
moving upstream `main` or conflate the kernel wheel label with its source label.

Evaluate representative leaves without starting the long builds:

```bash
nix eval --raw .#torch-xpu.drvPath
nix eval --raw .#vllm-xpu-kernels-unstable.drvPath
nix eval --raw .#vllm-xpu-unstable.drvPath
nix eval --raw .#vllm-xpu-unstable.version
```

Keep `vllm-xpu-nix` deployment-neutral. It publishes the generic,
overridable `vllm-xpu-unstable` package; Brutus owns its torchvision, BMG AOT,
and narrowed-kernel overrides in
`hosts/brutus/services/vllm-xpu/package.nix`. Do not copy Brutus's model profile
or serving arguments into this flake.

All native kernel variants use the same `/var/cache/ccache` namespace. ccache
validates the compiler, flags, and preprocessed source, but the directory is not
partitioned by derivation or build-input set. A Brutus specialization therefore
reuses every compatible translation unit from generic and experimental builds.

Review the diff and preserve the candidate locally. Continue to the foreground
integration loop in section 6 before publishing packaging or release references.
When a release was requested and the candidate passes, section 7 commits and
publishes its release manifest. Otherwise preserve the reproducible candidate
and its qualification record for a later release invocation.

The packaging `main` branch carries integration work. Stable deployment uses
the coordinated immutable packaging tag published in section 7, whose source
inputs also reference the coordinated immutable source tags.

## 6. Qualify on Brutus and perform the requested release handoff

Re-run the execution-host preflight before host evaluation or live testing:

```bash
test "$(hostname -s)" = brutus
```

Use a staged commit-first loop. Full source builds are intentionally deferred to
Brutus because they are slow and the live XPU environment is the authoritative
integration test. The final three steps apply only to the requested release
and deployment handoff; a default main refresh ends after qualification:

1. Complete patch audits and make the source changes that are justified by
   static review and cheap targeted tests.
1. Commit and publish every changed source repository.
1. Refresh and evaluate the local `vllm-xpu-nix` candidate, but do not publish
   its release or change the Brutus lock yet.
1. Enter the maintenance handoff below, build the candidate, and run its
   Brutus-derived foreground app on Brutus.
1. Exercise the local APIs and repeat the source-fix, lock-refresh, build, and
   foreground-app loop until it passes.
1. Commit and publish the passing `vllm-xpu-nix` candidate and coordinated
   release references.
1. Update only the Brutus `vllm-xpu-release` lock input and prove the host package
   derivation is identical to the candidate already tested and realized.
1. Commit the host lock change, then ask the operator for one final rebuild.

Source commits may be published as the foreground loop advances, but do not
publish the packaging release candidate or create coordinated release
references until the foreground stack passes.

### Qualification of the minimized patch stack

Use the reconciled audit matrix from [commit-audit.md](commit-audit.md) to
validate both correctness and the benefits of retained performance patches.
Run the paired rebuilt sources, representative inference, and focused
regressions for every DROP/ADAPT decision. Compare applicable throughput,
latency and memory measurements under the same model, quantization, context,
batching, speculation, cache state and hardware settings against the recorded
working baseline. Investigate material regressions against existing project
budgets; do not invent a permissive threshold to pass a candidate. If no budget
exists, report measurements and justify equivalence or retain the known fix.
Startup and a short successful response alone do not qualify attention,
determinism, long-context, or throughput changes.

If integration reveals a missing behavior or performance loss, return the
evidence to the owning commit's audit agent, revise the disposition, and fold
the repair into that patch. Recheck dependent patches and the pair, update the
commit map/locks, and rerun affected checks. Do not leave temporary repair or
revert commits in the final stack. Preserve the previously recorded remote tip
before each authorized rewrite; never mask someone else's intervening push.

For a default invocation, complete source main publication and prepare/qualify
the packaging candidate, then record its exact refs, derivation and outcomes.
Sections 6–7's stable publication, host repin and rebuild actions apply only
when included in the user's operation. An unavailable GPU check is pending
qualification, not success. A passing candidate is not a published stable
release until all three repositories' release refs have been verified.

### Maintenance mode and exact-package prebuild

Record which workers are active before maintenance. When a qualification-only
run finishes or pauses after failed qualification, stop its foreground processes
and restore precisely the workers
that this run stopped, using the previous deployed generation. If the user had
already stopped workers or wants maintenance to continue, preserve that state.
If restoration needs operator credentials, give that concrete handoff and
report maintenance as pending; do not leave services silently stopped. A release
without deployment has the same restoration handoff. Verify HTTP readiness of
the restored workers and report any outstanding failure.

Before the expensive build, ask the operator to enter vLLM maintenance mode on
Brutus if the live services need to be stopped and maintenance is not already
authorized. If the user has stopped them or authorized maintenance, verify their
state and continue without asking again. Otherwise wait for confirmation:

```bash
sudo systemctl stop vllm-xpu-chat.service vllm-xpu-embedding.service
systemctl is-active vllm-xpu-chat.service vllm-xpu-embedding.service
```

Both units should report `inactive`. This releases their GPU allocations and
most of their RAM before C++/SYCL compilation. If maintenance must be aborted,
restore the last deployed services with:

```bash
sudo systemctl start vllm-xpu-embedding.service vllm-xpu-chat.service
```

Build the local candidate with Brutus's downstream package overrides. This
keeps the generic packaging checkout free of Brutus-specific settings while
still producing the exact package that the host will use after its input is
updated. Use bounded parallelism so compiler memory pressure does not starve
unrelated services; increase the values only when Brutus has enough headroom:

```bash
git -C ~/Projects/vllm-xpu-nix status --short --branch

candidate_package=$(nix build --impure --no-link --print-out-paths \
  --max-jobs 1 --cores 4 --expr '
    let
      candidate = builtins.getFlake "path:/home/jasonbk/Projects/vllm-xpu-nix";
    in import /home/jasonbk/.config/nix/hosts/brutus/services/vllm-xpu/package.nix {
      vllm-xpu-unstable = candidate.packages.x86_64-linux.vllm-xpu-unstable;
    }
  ')
```

Brutus already bind-mounts `/var/cache/ccache` into Nix build sandboxes, and
`vllm-xpu-nix` gives every native kernel build the same `CCACHE_DIR`. Compatible
translation units from this checkout and later `nixos-rebuild` builds therefore
share compiler intermediates automatically. Do not put mutable ccache contents
in the Nix store and do not change `CCACHE_DIR` between the prebuild and
deployment.

Run that package with the chat environment and serve arguments derived from the
Brutus NixOS configuration:

```bash
cd ~/.config/nix
nix run .#vllm-xpu-brutus -- "$candidate_package"
```

The app forces the chat instance on for testing even if deployment is
temporarily disabled, substitutes the candidate package, and otherwise reuses
the generated `vllm-xpu-chat.service` command and environment. It also uses the
same shared `HF_HOME`; only its writable runtime compilation directory is kept
under the invoking user's cache. Leave it running while testing
`127.0.0.1:8000`, and stop it with Ctrl-C before the next candidate build or
the final NixOS rebuild. Any other enabled model workers should be started
separately so the foreground test reproduces their GPU residency.

On a fresh runtime compilation cache, chat can retain enough non-Torch compiler
or driver memory to fail its first KV-cache profile even after compilation has
finished. A second foreground start should compile quickly and recover the
expected KV capacity. Do not work around this by lowering `maxModelLen` or
hard-coding `--kv-cache-memory` unless repeated cached starts still fail.

Failures in this foreground app do not require a host rebuild. Fix the relevant
source fork, commit and publish that source commit, update the local packaging
lock, rebuild, and run the app again. Do not move a coordinated release branch
or the Brutus lock until this loop passes.

After updating the host lock, compare derivations before rebuilding:

```bash
candidate_drv=$(nix eval --raw \
  --impure --expr '
    let
      candidate = builtins.getFlake "path:/home/jasonbk/Projects/vllm-xpu-nix";
    in (import /home/jasonbk/.config/nix/hosts/brutus/services/vllm-xpu/package.nix {
      vllm-xpu-unstable = candidate.packages.x86_64-linux.vllm-xpu-unstable;
    }).drvPath
  ')
host_drv=$(nix eval --raw \
  ~/.config/nix#packages.x86_64-linux.vllm-xpu-brutus.drvPath)
test "$candidate_drv" = "$host_drv"
```

Stop if they differ: a host-side package override or mismatched input would
cause deployment to build something other than the tested candidate. When they
match, the expensive output is already in Brutus's Nix store and the later
system rebuild reuses it without recompiling the stack.

Do not point Brutus at packaging `main` during candidate testing; the foreground
app is the integration environment. After section 7 publishes the passing
coordinated release, change `vllm-xpu-release.ref` in
`modules/flake/nixos/server/flake.nix` directly to the new immutable
`refs/tags/xpu-vMAJOR.MINOR.PATCH`. Remove any temporary nested vLLM or
kernel source-input overrides at the same time; otherwise they replace the
release manifest's coordinated source refs with their default branches. Then
update only the release input:

```bash
cd ~/.config/nix/modules/flake/nixos/server
nix flake update vllm-xpu-release
```

Confirm the lock diff changes `vllm-xpu-release` and its intended transitive nodes,
but does not add a follow from its nixpkgs to the host's nixpkgs.

Verify that the resolved revisions and `vllm-xpu-brutus.drvPath` are identical
to the foreground-tested candidate. Commit that repin as the release handoff.

Evaluate Brutus from the repository root:

```bash
cd ~/.config/nix
nix eval --raw .#nixosConfigurations.brutus.config.system.build.toplevel.drvPath
```

Rebuild on Brutus with the repository dev-shell `rebuild` command or directly:

```bash
sudo nixos-rebuild switch --flake ~/.config/nix#brutus
```

The switch starts the enabled vLLM units again. If a failed switch leaves them
stopped, restore the previous generation or start the units explicitly before
ending maintenance.

The operator performs the rebuild and notifies the agent when it finishes. The
agent then tests each enabled `vllm-xpu-*` service and its HTTP health, model,
and representative inference endpoint.

Run service and API checks locally on Brutus; do not prefix them with
`ssh brutus`. A unit can be `active` while vLLM is still loading weights,
compiling, and capturing graphs, so wait for the HTTP endpoint rather than
treating systemd state alone as readiness:

```bash
systemctl is-active vllm-xpu-chat.service vllm-xpu-embedding.service
journalctl -u vllm-xpu-chat.service -n 200 --no-pager

until curl --fail --silent --show-error \
  http://127.0.0.1:8000/v1/models >/dev/null; do
  sleep 2
done

curl --fail-with-body --silent --show-error \
  http://127.0.0.1:8001/v1/models
```

Use a bounded wait during unattended automation and inspect the journal if the
API does not become ready; first-time torch compilation can take several
minutes.

When a build, startup, or API failure occurs:

1. Fix it in the appropriate local project under `~/Projects`.
1. Commit and push that project.
1. Refresh and validate `vllm-xpu-nix`, then repeat the foreground test loop.
1. Publish a new coordinated version after it passes, selected using section 7.
   A compatible fix on the same upstream baseline is a patch; adopting a new
   upstream baseline requires a minor or major bump. Never modify an existing
   release reference.
1. Refresh the scoped Brutus lock to that new tag and rebuild again.

Commit each repair before asking for the next rebuild so every tested candidate
is reproducible. Repeat the handoff, rebuild, live API test, and repair cycle
until the operator and agent agree that the deployed services pass. Record
failures and the exact candidate revisions so a failed iteration is not mistaken
for a release.

Do not bypass the chain with local path overrides in the committed host lock;
the tested deployment should match what Sunnycareboo serves.

## 7. Publish a coordinated stack release

Use coordinated **stack release versions** `xpu-vMAJOR.MINOR.PATCH` in all three
project repositories. This version identifies the tested combination of vLLM,
XPU kernels, and Nix packaging independently of upstream vLLM's version.

### Choose the release version

Compare the candidate with the latest published release on the intended release
line. Record the selected upstream release tags and frozen base commits for
**both** vLLM and XPU kernels; together they identify that line's upstream
source baseline. Apply the first matching rule:

| Bump | When | Example |
| --- | --- | --- |
| **MAJOR** | An incompatible stack interface, configuration, or deployment change requires consumers to migrate. | `1.8.2` → `2.0.0` |
| **MINOR** | Adopt a new upstream base in either source fork, or introduce compatible functionality on the existing baseline. | `1.7.2` → `1.8.0` |
| **PATCH** | Apply compatible fixes or packaging corrections while keeping both upstream source bases unchanged. | `1.8.0` → `1.8.1` |

A new upstream baseline requires a minor bump even if Git can fast-forward, or
the upstream vLLM release number has not changed. Divergent history is not the
criterion: replaying or reorganizing fork commits on the **same** upstream base
does not by itself require a minor bump. A large internal architecture change
can be minor if it preserves compatibility; size alone does not require major.
Reset PATCH to zero for a minor bump and both MINOR and PATCH for a major bump.
These compatibility rules follow [SemVer](https://semver.org/), with an explicit
minor-release boundary for each newly adopted upstream baseline.

Select the bump from **all changes since the preceding release**, not only the
last fix. For example, after `xpu-v1.7.2`, publishing a newly qualified upstream
release pair together with the V1 inspection-workaround removal is `xpu-v1.8.0`.
Applying only that removal to the preserved `1.7.2` baseline would be
`xpu-v1.7.3`. A subsequent fix on the released `1.8.0` baseline becomes
`xpu-v1.8.1`. These are selection examples, not claims of published releases;
always check existing refs before choosing the next unused version. Historical
rolling candidates based on untagged upstream snapshots are not automatically
eligible under the release-tag policy; migrate and qualify them first.

### Keep stack and package versions separate

Keep the upstream release versions, exact upstream base SHAs, fork tip SHAs,
dependency locks, and tested package derivation in the packaging release
manifest/notes. Several upstream refreshes may share one upstream version label;
the source SHAs distinguish them. Existing `xpu-v1.7`, `xpu-v1.7.1`, and
`xpu-v1.7.2` references remain unchanged.

Individual Python/Nix package versions continue to describe their upstream
source and build revision using the packaging's existing version-stamping
helper. Python permits downstream local versions such as
`0.28.0+nix.1.8.0` ([Python version specification](https://packaging.python.org/en/latest/specifications/version-specifiers/#local-version-identifiers));
that is an optional package label, not the coordinated stack tag or a reason to
change the existing package-version scheme during a release.

Do not encode stable stack releases as `20.2.1-nix.3`: SemVer interprets that as
a prerelease of `20.2.1`. SemVer also ignores `+nix.3` build metadata for version
ordering, so that suffix cannot replace an ordered stack release version. The
rolling `main` branches and immutable stable tags describe release channels;
Brutus follows a specific qualified stable tag.

### Publish immutable references

Every release gets **new**, permanent references in every repository:

- `releases/xpu-vMAJOR.MINOR.PATCH`: a readable branch preserving that release.
- `xpu-vMAJOR.MINOR.PATCH`: an annotated immutable tag for the same commit.

Never retarget, advance, overwrite, or delete an existing release branch or tag.
A compatible fix to a published release on the same upstream baseline needs a
new patch version. For other changes, use the selection rules above. An unchanged
dependency can receive the new coordinated tag at its existing commit; do not
manufacture an empty source commit. Continue rebasing only rolling source
`main` branches.

Normally publication follows successful foreground integration testing. When
preserving an already working baseline before a rebase, use the recorded runtime
validation and exact tree/build-input equivalence from section 1. Explicitly
record the scope of that evidence; release publication does not prove untested
behavior.

Choose the next unused version. Check **both local and remote** references in
all three repositories before creating anything; stop on lookup failures or any
existing branch/tag instead of reusing a partially published version silently.

```bash
: "${version:?Set version to the unused xpu-vMAJOR.MINOR.PATCH selected above}"
for repo in vllm vllm-xpu-kernels vllm-xpu-nix; do
  git -C "$HOME/Projects/$repo" show-ref \
    --verify --quiet "refs/heads/releases/$version" && exit 1
  git -C "$HOME/Projects/$repo" show-ref \
    --verify --quiet "refs/tags/$version" && exit 1
  git -C "$HOME/Projects/$repo" ls-remote origin \
    "refs/heads/releases/$version" "refs/tags/$version" || exit 1
done
```

The remote lookups must return no matching references. Record the exact tested
source revisions from the candidate packaging lock. For a history-only baseline
rewrite, first update only its source locks to the current atomic tips and prove
the recorded source-tree equivalence; preserve every unrelated dependency lock.

```bash
cd ~/Projects/vllm-xpu-nix
vllm_rev=$(nix flake metadata --json \
  | jq -r '.locks.nodes["vllm-xpu-unstable-src"].locked.rev')
kernels_rev=$(nix flake metadata --json \
  | jq -r '.locks.nodes["vllm-xpu-kernels-unstable-src"].locked.rev')
git -C ~/Projects/vllm cat-file -e "$vllm_rev^{commit}"
git -C ~/Projects/vllm-xpu-kernels cat-file -e "$kernels_rev^{commit}"
```

Create and publish source references at those exact commits, using explicit
refspecs and no force option:

```bash
git -C ~/Projects/vllm branch "releases/$version" "$vllm_rev"
git -C ~/Projects/vllm tag -a "$version" "$vllm_rev" \
  -m "Sunnycareboo XPU stack $version"
git -C ~/Projects/vllm-xpu-kernels \
  branch "releases/$version" "$kernels_rev"
git -C ~/Projects/vllm-xpu-kernels tag -a "$version" "$kernels_rev" \
  -m "Sunnycareboo XPU stack $version"
for repo in vllm vllm-xpu-kernels; do
  git -C "$HOME/Projects/$repo" show --no-patch \
    "releases/$version" "$version"
  git -C "$HOME/Projects/$repo" push --atomic origin \
    "refs/heads/releases/$version:refs/heads/releases/$version" \
    "refs/tags/$version:refs/tags/$version"
done
```

The packaging repository is the release manifest. Change **both source input
refs** in its `flake.nix` to `refs/tags/$version` (substitute the literal chosen
version), then update only those two source inputs:

```bash
cd ~/Projects/vllm-xpu-nix
nix flake update vllm-xpu-unstable-src vllm-xpu-kernels-unstable-src
git diff --check
git diff -- flake.nix flake.lock
nix eval --raw .#vllm-xpu-kernels-unstable.drvPath
nix eval --raw .#vllm-xpu-unstable.drvPath
```

Confirm that the source revisions still equal the tested `vllm_rev` and
`kernels_rev`, that unrelated dependency locks did not move, and that the package
matches the tested candidate. If changing refs affects derived versions or the
build derivation, explain and validate the difference before claiming exact
package equivalence. Commit the manifest, preserving the repository's sign-off
requirements, and push packaging `main` normally. Record its committed SHA;
never tag a dirty, uncommitted manifest.

```bash
packaging_rev=$(git rev-parse HEAD)
git branch "releases/$version" "$packaging_rev"
git tag -a "$version" "$packaging_rev" \
  -m "Sunnycareboo XPU stack $version"
git show --no-patch "releases/$version" "$version"
git push --atomic origin \
  "refs/heads/releases/$version:refs/heads/releases/$version" \
  "refs/tags/$version:refs/tags/$version"
```

Verify the remote annotated tags' peeled commit SHAs and matching release branch
SHAs in all three repositories. Record partial publication if any operation
fails; never force or retarget existing references to repair it. Configure
Forgejo protection for `releases/xpu-v*` and `xpu-v*` when practical.

When deployment is requested, pin Brutus's `vllm-xpu-release.ref` to the **packaging tag**
`refs/tags/$version`, remove temporary nested source overrides, and perform the
scoped lock update, evaluation, derivation comparison, and operator rebuild
handoff from section 6. Do not deploy packaging `main` or a movable source branch
as the stable release.

### Patch an existing coordinated release

Select the next unused patch version on the intended release line, apply the
targeted fix to its preserved source base, and validate the resulting stack.
Keep both upstream base SHAs and unrelated dependency locks unchanged. If
rolling `main` has already rebased onto newer upstream, do not incorporate that rebase accidentally; use a
new working branch from the preserved release when appropriate. Apply a relevant
fix to rolling `main` as well, but treat publishing that newer baseline as its
own minor or major release. Follow the full publication sequence above, tagging
unchanged source commits with the same new coordinated version, then update
packaging's source tag refs. Update Brutus's packaging tag ref only when its
repin/deployment is requested. All previous release branches, tags, and manifests remain
unchanged.

## Recovery

- A failed rebase can be stopped with `git rebase --abort`.
- The dated backup branches in both source forks preserve the previous local
  and published histories.
- Use `git reflog` to locate pre-rewrite commits if necessary.
- If a source push succeeded but packaging failed, leave the host lock on its
  last working `vllm-xpu-nix` revision until the packaging fix is published.
- Do not delete recovery branches until the refreshed services have passed the
  Brutus integration test.
- Coordinated `releases/xpu-v*` branches and `xpu-v*` tags are permanent
  recovery points. Never delete or move them; select a new version using section
  7 for corrections or baseline changes.

## Completion record

Store the completed release manifest and validation notes in the packaging
repository's [docs/releases](../../../../docs/releases) directory. Keep the
selected upstream release pair, exact base and fork SHAs, and package derivation
there so later discovery does not depend on temporary worktrees or this skill.

Record these revisions in the commit or maintenance notes:

```text
baseline stable version and validation evidence:
selected stack version and reason for major/minor/patch bump:
upstream vLLM/kernel release labels and exact baseline SHAs (previous -> candidate):
discovery result for both repos, release URLs, tag IDs and pair compatibility evidence:
vLLM old base / old main / expected remote / frozen upstream target:
kernels old base / old main / expected remote / frozen upstream target:
recovery branch refs:
per-commit audit assignments, coverage, dispositions and evidence:
old-to-new commit map, including canceled pairs and folded repairs:
correctness and performance comparison to the working baseline:
rebased vLLM tip and retained commit list:
rebased kernels tip and retained commit list:
checks run / deferred runtime checks:
vllm-xpu-nix tip and source tag refs:
Brutus system derivation:
services tested:
coordinated stack release:
```
