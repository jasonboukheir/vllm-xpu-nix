# One agent per fork commit

Read this when auditing either a new upstream pair or an explicitly requested
cleanup on the current bases. The goal is the smallest maintainable delta that
preserves our correctness fixes, useful features and performance improvements.
Commit count alone is not the objective.

## Inventory and assignment

For each repository, enumerate every commit in `old_base..old_main` using frozen
SHAs, in order. Include repairs, reverts and commits in a companion repo whose
base is unchanged. Verify the old base from previous audit/release records;
an upstream merge or unexpected divergence must be understood before treating
all commits in the range as local patches. Record a unique run ID and persist
the inventory, inputs and results under packaging `docs/updates/<run-id>/`.

Assign a separate read-only agent to each commit in bounded batches, respecting
the available concurrency. Each agent owns that commit's investigation and
artifact, not a shared source worktree. Agents may inspect the entire paired
stack and coordinate dependencies; only the parent applies the reconciled
changes. Reuse the owning agent for follow-up questions. If an agent is no longer
available, give its replacement the saved evidence and exact assignment.

Give each agent this concrete task, populated with immutable values:

```text
Audit this one fork commit for a paired upstream refresh. Do not edit source
worktrees, branches, locks, services, or external issues. Write your findings
only to the assigned audit file.

Repository and applicable repository instructions:
Assigned commit SHA, parent SHA, subject and patch ID:
Old upstream base and full fork main tip:
Selected upstream release tag and exact target SHA:
Companion repository old/new bases, tip and local commit inventory:
Full local commit inventory and prior rationale/issue/test evidence:
Assigned output path:

Determine the problem or performance benefit this commit supplies in the final
old stack. Investigate the delta from old upstream base to selected release,
and the target implementation: equivalent upstream solutions can look entirely
different. Account for follow-up repairs, reverts, supersession and dependencies
on other local patches or companion operators. Recommend KEEP, DROP or ADAPT
with exact evidence, the minimum residual change, and relevant validation.
Separate observed results from proposed tests and unresolved assumptions.
```

## Required evidence

Each report must contain:

- **Requirement:** original failure, feature or speed/memory benefit; supported
  workload and relevant hardware/configuration; behavior after later local
  commits. A reverted workaround may already have no effective contribution.
- **Upstream coverage:** relevant old-to-new upstream changes, exact SHAs,
  symbols/paths at the frozen target, and why they cover all, some or none of
  the requirement. Inspect alternate implementations and changed call paths;
  patch-ID equivalence, conflict status and PR descriptions alone are insufficient.
- **Disposition:** KEEP for an unmet requirement still correctly implemented;
  DROP for demonstrated upstream coverage, supersession, or a requirement
  explicitly retired within scope; ADAPT for only the residual missing behavior.
  Lack of evidence does not justify dropping a fix. State unresolved gaps and
  keep the requirement covered until they are resolved.
- **Dependencies:** prerequisite/follow-up commit SHAs, cross-repo API/ABI and
  schema implications, paired changes needed to drop an operator or feature,
  and which repairs should fold into which surviving commit.
- **Validation:** meaningful regression checks run and their results, required
  GPU checks, and performance comparisons for optimizations. Explain how checks
  distinguish upstream alone from a still-needed patch where feasible. Use the
  working stack's recorded results or a controlled comparison; do not rerun or
  interrupt production merely to obtain a baseline without authorization.

## Parent reconciliation and completion

Maintain one row per original commit:

| Repo / old SHA | Agent / report | Requirement | KEEP / DROP / ADAPT | Upstream evidence | Dependencies / folds | New SHA or drop reason | Validation |
| --- | --- | --- | --- | --- | --- | --- | --- |

Require complete inventory coverage before replay. Resolve disagreements and
dependency cycles across both repositories. Agents advise; the parent owns
the final evidence-backed disposition. A vLLM-side deletion must not strand a
kernel call/schema or discard a performance benefit without investigation.

Drop patch/revert pairs only after proving their net effect and accounting for
intervening users. Fold later fixes into their owning feature so the final stack
has one coherent commit per feature/fix in each repo, with no redundant or
temporary repair commits. Preserve distinct independently useful fixes and
authorship/sign-off. A new base in only one repo still requires considering how
it changes the other repo's patches.

After replay, inspect the complete effective diff to each target and the
old/new range-diff. Every original commit must map to a final commit (many-to-one
for folded repairs) or a justified drop; every resulting commit must have an
explained requirement. Verify target ancestry, paired interfaces, and that the
retired workaround did not reappear through replay. Revision-dependent build
inputs must be checked even for identical source trees.

Run focused correctness tests and matched performance comparisons relevant to
the changed behavior, followed by paired Nix/GPU qualification in workflow.md.
Revise decisions when evidence disagrees; fold repairs and update the map.
Keep rolling-main, candidate qualification, published stable release, and host
deployment statuses separate in the completion record. Report incomplete
evidence explicitly and never turn an untested candidate into a stable tag.
