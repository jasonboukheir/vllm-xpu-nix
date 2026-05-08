{
  lib,
  fetchurl,
  intel-oneapi,
  intel-oneapi-base,
  level-zero,
  zlib,
  ucx,
  rdma-core,
  libpsm2,
  libuuid,
  numactl,
  libffi,
}:

intel-oneapi.mkIntelOneApi (finalAttrs: {
  pname = "intel-oneccl-bmg";

  src = fetchurl {
    url = "https://github.com/uxlfoundation/oneCCL/releases/download/2021.15.9/intel-oneccl-2021.15.9.14_offline.sh";
    hash = "sha256-96uBtu0bEN01+t7DZqeARtivIUiI39YlBHzolT1apO8=";
  };

  versionYear = "2021";
  versionMajor = "15";
  versionMinor = "9";
  versionRel = "14";

  components = [ "intel.oneapi.lin.ccl.devel" ];

  depsByComponent.ccl = [
    intel-oneapi-base
    level-zero
    zlib
    ucx
    rdma-core
    libpsm2
    libuuid
    numactl
    libffi
  ];

  autoPatchelfIgnoreMissingDeps = [
    "libcuda.so.1"
  ];

  postInstall = ''
    rm -f $out/{lib,etc,bin,share,opt}
  '';

  meta = {
    description = "Intel oneCCL 2021.15.9.14 with Battlemage (BMG) support";
    homepage = "https://github.com/uxlfoundation/oneCCL";
    license = with lib.licenses; [
      intel-eula
      asl20
    ];
    platforms = [ "x86_64-linux" ];
  };
})
