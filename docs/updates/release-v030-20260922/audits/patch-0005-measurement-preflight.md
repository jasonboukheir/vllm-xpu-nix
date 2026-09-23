# Patch 0005 controlled-measurement preflight

Read-only review completed 2026-09-23, before the first controlled resource
capture. Scope: ordinary v0.30 eager V1/K4V2/MTP/vision release; issue17 stays
paused. The parent subsequently authorized repair of only the two private
harness files discussed below, with backups before edits. No build or GPU
execution is part of this review or repair.

**Do not execute the initial harness unchanged.** The comparison design is
useful, but command-shape validation, failure retention, and process cleanup
need the bounded repairs below. The actual generated compiler/link commands
have not yet been inspected: the successful candidate build directory was
cleaned, and searches found no retained `compile_commands.json`/`build.ninja`.
This is a missing preflight, not evidence that the current selectors match.

| Reviewed input | SHA-256 before repair |
| --- | --- |
| `build-controls.nix` | `aa763807b10a88ba29f173108a90ad1b14561aabf6c9bb3c6e01cfbe88c329ed` |
| `measure-native-build.py` | `a04a82e7611f6fa2449df91cf77d494126c245e6f3fbba67dcd76d0697fc1962` |
| `qualification-protocol.json` | `b52b934ff03b451d5772a442c277c9403c97eba6534d327a1536ba8d9de3c327` |
| `native-build-review.json` | `a6fb590e796d81d70bff8b72516d298d39b4fafe1886a50f729d86f7fb3135b7` |
| `qualification-packages.nix` | `53a35fd1ef9d846f9a91f0b8d1edc97177eea16e53e5748d9aae94a3b24fd6f7` |
| `build-control-derivations.json` | `c71b5c556d6928d4829f9308b064e4ecb199cc6a28bf2940a74b8524e68594bc` |
| `build-native-controls.py` | `a31868f076cbda75e89ea5cc10cd96582fa0153c5ba5270aaeecd9ffb8df89f3` |

All paths above are under `benchmark-results/release-v030-20260922/`.
Candidate sources are vLLM `466b9d5abe84f13c29b38c686e4d77fc09e09aa4`
and kernels `193cee080d654ff67c4f8bfaf924201bc88b9cf5`.

## Concrete findings and remediation

1. **Validate both actual command shapes before expensive work.** The initial
   script assumes one `/fmha_xe2.cpp` compile entry with a `command` string,
   one separate `-o` argument and one inline backtrace flag. Its link selector
   requires `-shared` and the literal text `-o libattn_kernels_xe_2.so`; it
   runs only after four potentially expensive recompiles. Neither the build
   review nor the nonverbose build log proves these shapes. Add a
   configure-time, no-execution mode which captures the original compile
   entry, raw `ninja -t commands` output, selected link command, and generated
   CMake/Ninja files; validate unique flag/output substitutions immediately.
   Detect response-file-only flags or unsupported shell shape and stop before
   the full build, rather than guessing. Revalidate the captured commands
   before measurement. Require a nonempty output file before marking a
   measured command successful.
2. **Align resource ownership and cleanup.** Initial sampling correctly uses
   `/proc/PID/stat` field 6 (session ID), but cancellation signals only
   `killpg(child.pid)` and waits only for the leader. Other groups in that
   session are not covered; the leader exiting after TERM also skips KILL
   even if workers remain. No signal/exception cleanup protects an interrupted
   inner monitor. Record PID plus start ticks, terminate only verified owned
   session members on every exit path, wait for the session to drain, and
   escalate surviving verified processes after the grace period. Retain
   incomplete/error samples; do not report success when workers remain.
3. **Preserve failed measurements outside `$out`.** Initially all detailed
   logs and samples are created directly in `$out/share/build-measurements`.
   A failed derivation does not provide a valid output artifact. The new outer
   orchestrator uses `--keep-failed`, which preserves the build directory;
   write measurements there first and copy to `$out` only after success.
   Record the retained directory and copy it to the durable run directory
   after failure. Outer stderr and two-second host observations do not
   replace the detailed inner process samples.
4. **The initial full library build is outside the inner guard.** The resource
   script runs in `postBuild`, so even a cold no-k02 dispatcher compile must
   finish before its 4-GiB/1800-second bounds start. The initial outer script
   observes compiler names but imposes neither bound and owns only the Nix
   client session; daemon build workers are not necessarily its descendants.
   The parent owns remediation in `build-native-controls.py`: guard the
   initial build, cancel the owned Nix request safely, verify its workers are
   gone, retain failure evidence, and do not confuse host-name sampling with
   build ownership. Unregister completed client identities and clean up on
   exceptions. Initial cache-enabled build timing is separate evidence.
5. **Retain enough provenance to interpret the result.** The original script
   records only a Ninja hash, PCH-looking argument strings, and relative
   elapsed times. Preserve actual Ninja command text/files and relevant
   config/generated/PCH identities, command/session identities, and UTC start
   and end times. Stream large hashes rather than allocating entire objects
   in the monitor. Object hashes before/after link ABBA are useful; explicitly
   verify referenced objects/response inputs and keep the original library
   untouched. Sampled per-process HWM can miss short-lived workers; aggregate
   RSS double-counts shared pages and is not physical memory consumption.

## Design elements that are sound

The backtrace order `[10, 0, 0, 10]` holds source and other flags fixed within
each profile; link order `[4, 16, 16, 4]` uses the already built object set.
`CCACHE_DISABLE=1`, fresh measured output paths, and explicit retention of
OS/PCH caches distinguish recompilation from a cache hit without claiming a
cold filesystem. Keep these controls and the existing acceptance criteria.

Direct reads of the four frozen `.drv` files confirm identical CMake config
paths within each with/without-k02 pair. Both narrow arms use
`/nix/store/3d3xb0xp4i1j8skvj3887hg55xsvdf34-brutus-kvarn-chunk-prefill.conf`
and `/nix/store/fzs19iks07fgmvadbzcxha678767kvmq-brutus-auto-paged-decode.conf`.
Both Brutus arms use the standard presets; the source factory supplies the
existing Brutus extras. `mk-vllm.nix` cascades the narrow override through the
overridden kernels package, so the suspected override mismatch was ruled
out. Frozen source-variant records retain the intended k02-only reversal.

The candidate build review correctly separates four split-library link jobs
from sixteen glue link jobs and explicitly labels its cache-enabled build
and compiler-name sampling as noncausal evidence. That build, packaging test
passes, and subsequent GPU passes do not satisfy the still-pending resource
comparison. The parent must serialize these controls after the current API
capacity/GPU activity ends, retain all failures, and re-freeze the repaired
harness derivations before the first capture. Two repetitions per setting
support a bounded screen; do not infer a small benefit from noise or claim
diagnostic-depth memory savings without repeatable evidence.

Evidence commands: read the named files with `cat`/`nl`; inspect the native
build log with bounded `rg`; search the run, build-dev, and `/tmp` for retained
generated command files; inspect `mk-vllm.nix`/`mk-kernels.nix`; extract
`cmakeFlags` directly from the four `.drv` files; run `sha256sum` on inputs.
`nix derivation show` was unavailable in this sandbox (daemon socket denied;
`jq` absent), so direct `.drv` reads provided the configuration check. No
escalation, build, native compiler command, GPU command, or service action ran.

## Authorized private harness repair

After completing the review, the parent authorized edits only to
`measure-native-build.py` and `build-controls.nix`. Their original contents
were preserved under
`benchmark-results/release-v030-20260922/preflight-before/patch0005-resource-harness/`;
the backup hashes match the table above. No source, tracked packaging,
performance criterion, orchestration script, ref, or service was changed.

| Repaired private file | SHA-256 |
| --- | --- |
| `measure-native-build.py` | `fdd958ef31b41d28cc9365d88ccf3eef0200e578cb8c0adf790848a24150f7f8` |
| `build-controls.nix` | `41dee9d548ab28e511b80ef9147a6aadfb12b21c2ee92059063e3b6749b89e89` |

The repair adds a configure-time command preflight and captures original
compile entries, Ninja commands/rules, and CMake inputs before compilation.
It rejects unsupported shell/response-file shapes, verifies unique output
and numeric flag replacements, then revalidates the same inputs after the
full build. Measurements remain under `$PWD/build-measurements` for
`--keep-failed`; only successful measurement completion copies them into
`$out/share/build-measurements`. It hashes actual referenced objects and
selected source/config/generated/explicit PCH files, uses streaming hashes,
and records UTC and process start identities. Cleanup re-enumerates live
members of the owned session, verifies PID/start-tick identity before each
signal, and escalates surviving workers even after the session leader exits.
Failure records survive interrupted commands. The existing ABBA orders,
numeric guards, cache policy, and profiles are unchanged.

Validation used the existing exact Python 3.12 interpreter and temporary
synthetic fixtures, which were removed afterward. Python in-memory syntax
checking and `nix-instantiate --parse build-controls.nix` passed. Pure parser
tests covered argument preservation, quoted output paths, duplicate/missing
outputs, and unsupported shell/response files. A synthetic configure-time
preflight passed before object files existed; an unsupported response file
failed while retaining its original command. Short CPU-only subprocess tests
verified successful/failed measurement records, SIGTERM interruption cleanup,
and TERM-to-KILL cleanup of a resistant worker in a different process group
within the owned session; an unrelated-session canary remained running until
the test explicitly cleaned it up. No native compilation, linker execution,
Ninja build, or GPU operation ran.

The actual generated command preflight remains pending the parent's first
configured control build. The parent owns outer initial-build monitoring,
request cancellation/worker verification, durable failed-build artifact
export, and re-evaluation/re-freezing of the now-superseded derivations. This
repair is harness validation, not a resource measurement or release pass.
