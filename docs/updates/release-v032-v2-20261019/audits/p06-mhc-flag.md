# Forward the MHC build selection

Disposition: KEEP `nix/patches/0006-forward-mhc-feature-flag.patch`.

The selected kernel source base is unchanged; at fork
`dbb684f4c9b38a8b3bbf52bda0c490ee47049221`, `setup.py` still omits
`MHC_KERNELS_ENABLED` from its CMake option forwarding list. The patch makes
the glue configuration honor the same MHC selection as the split libraries.
Removing it can produce bindings inconsistent with the packaged library set.
The new Python runner does not alter that build-time requirement.

The single setup hunk applied without offset in the isolated source archive.
No new architecture or native feature is enabled by retaining it. Native and
installed-package validation remain separate from this application check.
