# Prebuilt library installation

Disposition: KEEP `nix/patches/0004-skip-prebuilt-additional-libs.patch`.

At selected kernel fork `dbb684f4c9b38a8b3bbf52bda0c490ee47049221`,
`setup.py` still attempts CMake installation for every additional library.
Libraries imported by patch 0001 have no local CMake installation directory;
skipping their installation preserves their independently realized Nix outputs.
Removing this patch while retaining split imported libraries breaks that
contract. The v0.32 Python source update does not replace it.

Applied successfully after patch 0001 in an isolated exact-source archive
(12-line offset). This is packaging evidence, not native runtime validation.
