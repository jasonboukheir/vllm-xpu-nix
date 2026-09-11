{ pkgs, library }:
pkgs.runCommand "kernel-build-parallelism" { } ''
  check_limit() {
    export NIX_BUILD_CORES="$1"
    ${library.preBuild}
    test "$NIX_BUILD_CORES" -eq "$2"
  }

  check_limit 1 1
  check_limit 4 4
  check_limit 12 12
  check_limit 24 12
  touch "$out"
''
