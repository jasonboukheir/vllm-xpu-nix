# Split native libraries

Disposition: KEEP `nix/patches/0001-split-kernel-libs.patch`.

The selected kernel base remains `ddf336d86e3c8602888572a3502f951abd51df12`
and its fork remains `dbb684f4c9b38a8b3bbf52bda0c490ee47049221`.
The source `CMakeLists.txt` unconditionally adds enabled native library
subdirectories; `setup.py` does not forward the prebuilt-library paths.
The patch supplies imported targets and forwards those paths so the Nix glue
derivation can link the independently built libraries. vLLM's Python runner
change supplies no equivalent packaging mechanism.

Applied successfully against the exact kernel fork in an isolated archive.
The CMake hunks moved by 22 lines and the setup hunk by seven; no semantic
conflict. The effective target/source composition is unchanged from 1.12.2.
Final qualification must verify the selected DFlash attention backend and
whether it requires additional native presets before claiming closure reuse.

No native build or GPU pass is claimed by this patch check.
