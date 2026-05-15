{
  lib,
  stdenvNoCC,
  torch-xpu,
  python3Packages,
}:

# Header-only slice of torch-xpu for per-TU compile contexts.
#
# Why this exists: `torch-xpu` is a ~700 MB closure that bumps on every
# torch wheel rev, every intel-oneapi point release, every oneccl-bmg
# bump, every nixpkgs python rev, etc. A per-TU SYCL compile only reads
# headers — the `.so` files and compiled python modules only matter at
# link time / runtime. Pulling the full closure into per-TU `buildInputs`
# means every unrelated torch-xpu input bump invalidates 600+ per-TU
# drv input hashes, even though the .o bytes would be byte-identical.
#
# This drv copies only:
#   - lib/pythonX.Y/site-packages/torch/include/      (headers)
#   - lib/pythonX.Y/site-packages/torch/share/cmake/  (find_package(Torch) data, if present)
#   - lib/pythonX.Y/site-packages/torch/lib/cmake/    (find_package(Torch) data, if present)
#   - lib/pythonX.Y/site-packages/torch/version.py    (header lookups sometimes touch it)
#
# Layout mirrors torch-xpu's site-packages tree so consumers can swap
# `torch-xpu` -> `torch-xpu-headers` in `${pkg}/${sitePackages}/torch/include`
# style paths without other edits.
#
# Content-addressed: torch-xpu store-path bumps that don't actually
# change header bytes leave torch-xpu-headers' CA hash stable, so
# downstream per-TU drv input hashes stay put and the per-TU CA cache
# preserves its realisation entries. The expected common case — most
# torch-xpu rebuilds come from intel-oneapi / pti / oneccl / nixpkgs
# python churn, not upstream torch header edits — turns into a no-op
# at the per-TU layer.

let
  sitePackages = python3Packages.python.sitePackages;
in
stdenvNoCC.mkDerivation {
  pname = "torch-xpu-headers";
  inherit (torch-xpu) version;

  dontUnpack = true;
  dontStrip = true;

  __contentAddressed = true;

  buildPhase = ''
    runHook preBuild

    src=${torch-xpu}/${sitePackages}/torch
    dst=$out/${sitePackages}/torch
    mkdir -p "$dst"

    cp -a "$src/include" "$dst/"
    if [ -d "$src/share/cmake" ]; then
      mkdir -p "$dst/share"
      cp -a "$src/share/cmake" "$dst/share/"
    fi
    if [ -d "$src/lib/cmake" ]; then
      mkdir -p "$dst/lib"
      cp -a "$src/lib/cmake" "$dst/lib/"
    fi
    if [ -f "$src/version.py" ]; then
      cp "$src/version.py" "$dst/"
    fi

    runHook postBuild
  '';

  dontInstall = true;

  meta = torch-xpu.meta // {
    description =
      "Header-only slice of torch-xpu (include/, cmake configs, version.py) "
      + "for compile-time inclusion without the full ~700 MB runtime closure";
  };
}
