# Match compute-runtime 26.27.39122.11/manifests/manifest.yml.
# Standalone profiling dependencies; do not modify the host driver package.
{
  pkgs ? (builtins.getFlake (toString ../.)).inputs.nixpkgs.legacyPackages.${builtins.currentSystem},
}:
let
  mkMetrics = { component, version, rev, hash }:
    pkgs.stdenv.mkDerivation {
      pname = "intel-metrics-${component}";
      inherit version;
      src = pkgs.fetchzip {
        url = "https://github.com/intel/metrics-${component}/archive/${rev}.tar.gz";
        sha256 = hash;
      };
      nativeBuildInputs = [ pkgs.cmake ];
      buildInputs = [ pkgs.libdrm ];
      enableParallelBuilding = true;
      cmakeFlags = [ "-DCMAKE_INSTALL_LIBDIR=lib" ];
      meta = {
        description = "Intel GPU metrics ${component} for isolated profiling";
        homepage = "https://github.com/intel/metrics-${component}";
        license = pkgs.lib.licenses.mit;
        platforms = [ "x86_64-linux" ];
      };
    };
in
{
  library = mkMetrics {
    component = "library";
    version = "1.0.234";
    rev = "8d50c43cb3d2e8c3781985655ffe47d1af3f17d7";
    hash = "0qxsc7wsf58ww22amzm7136i9sgcflj5kwnikgvsr4p1j9cq8ib5";
  };
  discovery = mkMetrics {
    component = "discovery";
    version = "1.16.189";
    rev = "1f7d2cb7190491283b8cdaa4f10e9341c3bb0d40";
    hash = "1yi235m50cp5cy1y144qpa4zwjnc63klpxdgkb2hmbwdhaagzzjh";
  };
}
