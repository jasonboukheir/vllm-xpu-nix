# `lint` wrapper for the local vllm / vllm-xpu-kernels checkouts.
#
# Pinned ruff binaries matching each project's .pre-commit-config.yaml, so
# `lint` produces the same diagnostics CI would. Nixpkgs' ruff drifts ahead
# of the projects' pins, and `uv tool run` trips on uv's bundled glibc
# python under NixOS, so we vendor the official musl static binary per
# version.
{ pkgs }: let
  mkRuff = {
    version,
    sha256,
  }:
    pkgs.stdenvNoCC.mkDerivation {
      pname = "ruff";
      inherit version;
      src = pkgs.fetchurl {
        url = "https://github.com/astral-sh/ruff/releases/download/${version}/ruff-x86_64-unknown-linux-musl.tar.gz";
        inherit sha256;
      };
      sourceRoot = "ruff-x86_64-unknown-linux-musl";
      dontConfigure = true;
      dontBuild = true;
      installPhase = ''
        install -Dm755 ruff "$out/bin/ruff"
      '';
    };
  ruffVllm = mkRuff {
    version = "0.14.0";
    sha256 = "sha256-7W0bhAeh0ijcMy+xkFfobgSmzTwr6s2zJK1v8qP5Bxs=";
  };
  ruffKernels = mkRuff {
    version = "0.11.7";
    sha256 = "sha256-DK8yqww5m/ugjRixkNwarsjqRzaIZKS/Oz8RPjbo/Wg=";
  };
in
  # Lint local vllm / vllm-xpu-kernels checkouts with the pinned ruff
  # (or pre-commit if available). Defaults to sibling checkouts
  # (../vllm, ../vllm-xpu-kernels relative to $PWD — the ~/Projects
  # layout); override with positional args: lint [VLLM_SRC] [KERNELS_SRC].
  pkgs.writeShellApplication {
    name = "lint";
    runtimeInputs = [pkgs.git];
    text = ''
      vllm_src="''${1:-../vllm}"
      kernels_src="''${2:-../vllm-xpu-kernels}"
      run_in() {
        local dir="$1" ruff="$2"
        if [ ! -d "$dir" ]; then
          echo "[$dir] not found — skipping"
          return 0
        fi
        echo "=== $dir ==="
        (
          cd "$dir"
          if [ -f .pre-commit-config.yaml ] && command -v pre-commit >/dev/null 2>&1; then
            echo "[$dir] pre-commit run --all-files"
            pre-commit run --all-files
          else
            if [ ! -f .pre-commit-config.yaml ]; then
              echo "[$dir] no .pre-commit-config.yaml — running ruff only"
            else
              echo "[$dir] pre-commit not installed — falling back to ruff only"
            fi
            echo "[$dir] ruff check ."
            "$ruff" check .
            echo "[$dir] ruff format --check ."
            "$ruff" format --check .
          fi
        )
      }
      fail=0
      run_in "$vllm_src"    "${ruffVllm}/bin/ruff"    || fail=1
      run_in "$kernels_src" "${ruffKernels}/bin/ruff" || fail=1
      exit "$fail"
    '';
  }
