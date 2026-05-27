{
  lib,
  stdenv,
  ncurses5,
  bc,
  bubblewrap,
  autoPatchelfHook,
  python3,
  libgcc,
  glibc,
  writableTmpDirAsHomeHook,
}:

# Standalone Intel oneAPI offline-installer derivation factory.
#
# Restores the `mkIntelOneApi` helper that nixpkgs deleted when it merged
# the base + hpc derivations into a single `intel-oneapi-toolkit` package
# (`pkgs/by-name/in/intel-oneapi-toolkit/package.nix`). We still need the
# factory for component packages whose installer URL is *not* the unified
# toolkit — e.g. the standalone oneCCL 2021.15.x release with Battlemage
# support (`oneccl-bmg.nix`).
lib.extendMkDerivation {
  constructDrv = stdenv.mkDerivation;

  excludeDrvArgNames = [
    "depsByComponent"
    "components"
  ];

  extendDrvArgs =
    finalAttrs:
    {
      pname,
      versionYear,
      versionMajor,
      versionMinor,
      versionRel,
      src,
      meta,
      depsByComponent ? { },
      postInstall ? "",
      components ? [ "default" ],
      ...
    }@args:
    let
      shortName = name: builtins.elemAt (lib.splitString "." name) 3;
    in
    {
      version = "${finalAttrs.versionYear}.${finalAttrs.versionMajor}.${finalAttrs.versionMinor}.${finalAttrs.versionRel}";

      nativeBuildInputs = [
        ncurses5
        bc
        bubblewrap
        autoPatchelfHook
        writableTmpDirAsHomeHook
      ];

      buildInputs = [ python3 ]
      ++ lib.concatMap (
        comp:
        if comp == "all" || comp == "default" then
          lib.concatLists (builtins.attrValues depsByComponent)
        else
          depsByComponent.${shortName comp} or [ ]
      ) components;

      dontUnpack = true;

      installPhase = ''
        runHook preInstall
        mkdir -p "$out"

        export LD_LIBRARY_PATH="${lib.makeLibraryPath [ libgcc.lib ]}"

        mkdir -p fhs-root/{lib,lib64}
        ln -s "${glibc}/lib/"* fhs-root/lib/
        ln -s "${glibc}/lib/"* fhs-root/lib64/
        bwrap \
          --bind fhs-root / \
          --bind /nix /nix \
          --ro-bind /bin /bin \
          --dev /dev \
          --proc /proc \
          bash "$src" \
            -a \
            --silent \
            --eula accept \
            --install-dir "$out" \
            --components ${lib.concatStringsSep ":" components}

        rm -rf "$out"/logs
        rm -rf "$out"/.toolkit_linking_tool

        ln -s "$out/$versionYear.$versionMajor"/{lib,etc,bin,share,opt} "$out"

        runHook postInstall
      '';
    };
}
