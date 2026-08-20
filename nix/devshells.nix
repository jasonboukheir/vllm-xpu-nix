# Dev shells: the default tooling shell plus the kernels-dev / vllm-dev
# in-tree development environments.
#
# Variant-neutral args: the flake wires in whichever vllm/kernels variant
# development targets (currently the unstable fork — the deployed one), via
#   vllmPkg     - the vllm-xpu python package (closure for vllm-dev)
#   vllmKernels - the matching vllm-xpu-kernels python package
#   kernelLibs  - the mkKernelLibs set for the same kernels src
#
# Note: `git` is deliberately not in any shell's `packages`. Consumers'
# ambient git (e.g. a wrapper that sets user config) stays on PATH instead
# of being shadowed by a nixpkgs git.
{
  pkgs,
  syclToolchainShellHook,
  kernelLibs,
  vllmPkg,
  lint,
  quantize,
  torch-xpu,
  triton-xpu,
  vllmKernels,
}:
{
  default = pkgs.mkShell {
    name = "vllm-xpu-nix-dev";
    packages = with pkgs; [
      nix-tree
      nix-diff
      nixfmt
      nil
      patchelf
      file
      skopeo
      pre-commit
      lint
      quantize
    ];
    shellHook = ''
      cat <<'EOF'
      vllm-xpu-nix dev shell.

      Build packages:
        nix build .#intel-oneapi
        nix build .#torch-xpu
        nix build .#triton-xpu
        nix build .#vllm-xpu-kernels             # upstream vllm-project (stable)
        nix build .#vllm-xpu-kernels-unstable    # jasonboukheir fork (work-in-progress)
        nix build .#vllm-xpu                     # upstream vllm-project (stable)
        nix build .#vllm-xpu-unstable            # jasonboukheir fork (work-in-progress)

      Iterate against a local checkout (no flake edit needed):
        nix build .#vllm-xpu-unstable \
          --override-input vllm-xpu-unstable-src path:../vllm
        nix build .#vllm-xpu-kernels-unstable \
          --override-input vllm-xpu-kernels-unstable-src path:../vllm-xpu-kernels

      Lint local checkouts (pinned ruff matching each project's pins):
        lint                       # defaults to ../vllm and ../vllm-xpu-kernels
        lint /path/to/vllm /path/to/vllm-xpu-kernels

      Quantization workspace:
        quantize --help
        quantize init OWNER/MODEL

      The kernels-dev / vllm-dev shells below are wired to the *unstable*
      (fork) variant — the one actually deployed — so entering them never
      builds the stable closure.

      Develop the kernels (editable install, edit/build/test in tree):
        cd /path/to/vllm-xpu-kernels
        nix develop /path/to/vllm-xpu-nix#kernels-dev
        pip install -e . --no-build-isolation
        pytest tests/                 # e.g. tests/gdn_attn for the GDN work

      Develop vllm (editable install, edit/run/test in tree):
        cd /path/to/vllm
        nix develop /path/to/vllm-xpu-nix#vllm-dev
        pip install -e . --no-build-isolation --no-deps
        vllm serve <model> --enforce-eager
      EOF
    '';
  };

  kernels-dev = pkgs.mkShell {
    name = "vllm-xpu-kernels-dev";
    # inputsFrom a kernel-*lib* derivation, not the python package: the
    # package's buildInputs list the five prebuilt kernel .so closures
    # (attn-kernels-xe-2 etc.), so inputsFrom = [ vllmKernels ] would
    # realize them — a ~600-TU FA2 compile on a cache miss. A lib deriv is
    # built *with* the same toolchain but has no prebuilt-lib deps, so this
    # gives the identical compiler/cutlass/oneDNN/torch-xpu env with the
    # whole closure already cached (a kernel dev rebuilds the libs in-tree).
    inputsFrom = [ kernelLibs.gdn-attn-kernels-xe-2 ];
    packages =
      with pkgs;
      [
        cmake
        ninja
        # Lint/format tooling mirroring .pre-commit-config.yaml so style
        # issues (e.g. ruff F841, clang-format drift) are caught locally
        # before push instead of in CI. NB: the repo's C++ gate is
        # clang-format (--style=file, .clang-format), not the vestigial
        # [tool.cpplint] in pyproject.toml. The `lint` wrapper runs both.
        ruff
        llvmPackages_20.clang-tools # clang-format; matches CI's v20.1.3
        codespell
        pre-commit
        # mem_info.cpp includes level_zero/ze_api.h directly. Keep this as a
        # direct shell input so the header is present in compiler include
        # paths, rather than relying on a transitive runtime-only dependency.
        level-zero
        # `lint` mirrors the .pre-commit-config gate on changed files: ruff
        # over the tree + clang-format (--style=file) over changed C/C++.
        # A real command (not a shellHook function) so it works under
        # `nix develop --command`, child shells and scripts too.
        (writeShellScriptBin "lint" ''
          set -u
          rc=0
          # Files changed vs HEAD plus new untracked ones (what you're about
          # to commit) -- avoids scanning vendored third_party/ that the
          # pre-commit config excludes.
          changed() {
            { git diff --name-only --diff-filter=ACMR HEAD -- "$@"
              git ls-files --others --exclude-standard -- "$@"
            } 2>/dev/null | sort -u
          }
          py=$(changed '*.py')
          cxx=$(changed '*.h' '*.hpp' '*.cc' '*.cpp' '*.cu' '*.cuh')
          if [ -n "$py" ]; then
            echo "== ruff =="; ${ruff}/bin/ruff check $py || rc=1
          fi
          if [ -n "$cxx" ]; then
            echo "== clang-format =="
            ${llvmPackages_20.clang-tools}/bin/clang-format \
              --style=file --dry-run --Werror $cxx || rc=1
          fi
          [ -z "$py$cxx" ] && echo "lint: no changed py/c++ files vs HEAD"
          [ "$rc" -eq 0 ] && echo "lint: clean" || echo "lint: issues found"
          exit $rc
        '')
      ]
      ++ (with pkgs.python312Packages; [
        pip
        setuptools
        setuptools-scm
        wheel
        packaging
        jinja2
        psutil
        pytest
        torch-xpu
        triton-xpu
      ]);
    shellHook = syclToolchainShellHook + ''
      # pre-commit currently propagates a newer Python executable. Native
        # extensions and all Python dependencies in this shell target 3.12,
        # so make the matching interpreter authoritative.
        export PATH="${pkgs.python312}/bin:$PATH"
        # icx/icpx do not consume nixpkgs' compiler-wrapper include flags.
        # Expose the direct Level Zero input through a compiler-native path.
        export CPATH="${pkgs.level-zero}/include''${CPATH:+:$CPATH}"
        export VLLM_XPU_AOT_DEVICES="''${VLLM_XPU_AOT_DEVICES:-bmg}"
      export VLLM_XPU_XE2_AOT_DEVICES="''${VLLM_XPU_XE2_AOT_DEVICES:-bmg}"
      export MAX_JOBS="''${MAX_JOBS:-2}"
      cat <<'EOF'
      vllm-xpu-nix kernels-dev shell.

      Toolchain + the full vllm-xpu-kernels (unstable fork) closure
      (torch-xpu, triton-xpu, oneAPI MKL/SYCL, cutlass src) wired up.
      MAX_JOBS=2 by default — each
      SYCL-TLA template instantiation holds ~40 GiB during icpx, raise with
      care.

      Quick start (against a local kernels checkout):
        cd /path/to/vllm-xpu-kernels
        git submodule update --init --recursive
        pip install -e . --no-build-isolation

      Incremental rebuild after editing a .cpp:
        ninja -C build/temp.*/release install

      Tests:
        pytest tests/

      Lint (matches CI .pre-commit-config):
        lint                        # ruff + clang-format on changed files
        pre-commit run --all-files  # full gate (fetches hook envs, needs net)

      Tune feature flags via cmake -D… or env (BASIC_KERNELS_ENABLED,
      FA2_KERNELS_ENABLED, MOE_KERNELS_ENABLED, GDN_KERNELS_ENABLED, etc.).
      EOF
    '';
  };

  vllm-dev = pkgs.mkShell {
    name = "vllm-xpu-vllm-dev";
    inputsFrom = [ vllmPkg ];
    packages =
      with pkgs.python312Packages;
      [
        pip
        setuptools
        wheel
        packaging
        jinja2
        torch-xpu
        triton-xpu
      ]
      ++ [ vllmKernels ];
    shellHook = syclToolchainShellHook + ''
      export VLLM_TARGET_DEVICE=xpu
      export VLLM_VERSION_OVERRIDE="''${VLLM_VERSION_OVERRIDE:-0.0.0.dev}"

      # BMG single-card oneCCL safe defaults — match what the systemd
      # module bakes in. Override per-session to taste.
      export CCL_PROCESS_LAUNCHER="''${CCL_PROCESS_LAUNCHER:-none}"
      export CCL_ATL_TRANSPORT="''${CCL_ATL_TRANSPORT:-ofi}"
      export CCL_ZE_IPC_EXCHANGE="''${CCL_ZE_IPC_EXCHANGE:-sockets}"
      export CCL_LOG_LEVEL="''${CCL_LOG_LEVEL:-warn}"

      cat <<'EOF'
      vllm-xpu-nix vllm-dev shell.

      Toolchain + full vllm-xpu (unstable fork) closure (torch-xpu,
      triton-xpu, vllm-xpu-kernels, runtime python deps) wired up.
      VLLM_TARGET_DEVICE=xpu and
      BMG single-card oneCCL env are pre-set.

      Editable install:
        cd /path/to/vllm
        pip install -e . --no-build-isolation --no-deps
        python -c 'import vllm; print(vllm.__version__)'

      Run server:
        vllm serve <model> --enforce-eager

      Tests:
        pytest tests/
      EOF
    '';
  };
}
