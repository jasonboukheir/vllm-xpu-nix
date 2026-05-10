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
  libfabric,
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
    libfabric
  ];

  autoPatchelfIgnoreMissingDeps = [
    "libcuda.so.1"
  ];

  postInstall = ''
    rm -f $out/{lib,etc,bin,share,opt}
    ln -s ccl/2021.15/lib $out/lib

    # libccl dlopens libfabric.so when CCL_ATL_TRANSPORT=ofi: first by name,
    # then as `<dirname libccl.so>/libfabric.so`. With neither resolvable,
    # OFI init silently fails ("OFI transport was not initialized, fallback
    # to MPI transport") and libccl falls back to libmpi.so.12 — which is
    # on libtorch_xpu's RPATH via intel-oneapi-base but never MPI_Init-ed,
    # so the next allreduce segfaults inside libmpi (#39).
    #
    # The libfabric the offline installer bundles is unusable here: it's
    # configured with a hardcoded `/usr/local/lib/libfabric` provider path
    # and ships its providers as separate DSOs, so without a runtime
    # FI_PROVIDER_PATH it loads zero providers ("fi_getinfo error: ret -61,
    # providers 0"). nixpkgs' libfabric has the providers we need (tcp,
    # shm, sockets, rxm) compiled into libfabric.so itself, so it works
    # without env-var coaxing — symlink it as a sibling of libccl so the
    # relative-path dlopen succeeds.
    ln -s ${libfabric}/lib/libfabric.so $out/ccl/2021.15/lib/libfabric.so
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
