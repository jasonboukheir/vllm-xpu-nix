# Stamp a derivation's version from a flake input's lock metadata rather
# than a hand-bumped literal. `base` is the upstream release the pin
# descends from; if omitted, it's parsed from the input's `original.ref`
# in flake.lock (e.g. "release/v0.1.9.1", "refs/heads/releases/v0.22.0"),
# so release-tracking pins need zero ceremony. `unstable=true` is for
# main-tracking pins where every lock bump moves the source — the lock
# date goes into the local-version suffix so the store path shifts in
# lockstep.
#
# Reading from flake.lock is necessary because inputs reaching `outputs`
# only carry the resolved metadata (rev, narHash, lastModified, ...) — the
# `original` spec with the ref is not exposed. The lock file is part of the
# flake's source, so this is an eval-time file read, not IFD.
#
# Output is PEP 440-clean (`+local` with `[a-zA-Z0-9.]` payload): vllm
# forwards this to VLLM_VERSION_OVERRIDE -> SETUPTOOLS_SCM_PRETEND_VERSION,
# which rejects '-'-separated local tags. Kernels are Nix-label-only, but
# using one format keeps the helper trivial.
{ lockFile }: let
  flakeLock = builtins.fromJSON (builtins.readFile lockFile);
in {
  name,
  input,
  base ? null,
  unstable ? false,
}: let
  ref = flakeLock.nodes.${name}.original.ref or "";
  matched = builtins.match ".*v([0-9]+(\\.[0-9]+)*)" ref;
  effectiveBase =
    if base != null
    then base
    else if matched != null
    then builtins.head matched
    else throw "mkInputVersion: no `base` given and could not parse version from flake.lock ref of ${name} = ${toString ref}";
  rev = input.shortRev or "dirty";
  d = input.lastModifiedDate or "00000000000000";
  ymd = "${builtins.substring 0 4 d}.${builtins.substring 4 2 d}.${builtins.substring 6 2 d}";
in
  if unstable
  then "${effectiveBase}+unstable.${ymd}.g${rev}"
  else "${effectiveBase}+g${rev}"
