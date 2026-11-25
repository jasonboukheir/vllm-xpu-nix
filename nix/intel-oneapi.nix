# The Torch 2.15 Linux build uses Intel Deep Learning Essentials 2026.1.3.
# Its 2026.1.1+325 compiler components contain the same SYCL, UR, OpenMP,
# OpenCL and compiler runtime DSOs as Intel’s 2026.1.2 PyPI wheels.
{
  pkgs,
  torch215 ? false,
}:
(pkgs.intel-oneapi-toolkit.override {
  components = [
    "intel.oneapi.lin.dpcpp-cpp-compiler"
    "intel.oneapi.lin.mkl.devel"
    "intel.oneapi.lin.dpl"
    "intel.oneapi.lin.ccl.devel"
  ];
}).overrideAttrs
  (old: {
    versionYear = "2026";
    versionMajor = "1";
    versionMinor = if torch215 then "3" else "0";
    versionRel = if torch215 then "17" else "192";
    src =
      if torch215 then
        pkgs.fetchurl {
          url = "https://registrationcenter-download.intel.com/akdlm/IRC_NAS/bedab612-cf6b-4ece-872e-e72e9534ccab/intel-deep-learning-essentials-2026.1.3.17_offline.sh";
          hash = "sha256-A4Ae7ToORKK2wkyuoVVz4FG4LP/mT7NtTgdo/Rra0X4=";
        }
      else
        pkgs.fetchurl {
          url = "https://registrationcenter-download.intel.com/akdlm/IRC_NAS/33cb2a22-ddf1-4aa9-8d68-1f5a118acaf2/intel-oneapi-toolkit-2026.1.0.192_offline.sh";
          hash = "sha256-KJEZ4us+D37l0fO4muyyDWLgy3lkHKmSJfbR/6MEgW8=";
        };
    # nixpkgs' intel-oneapi-toolkit has no `depsByComponent.ccl` entry
    # (only dpcpp-cpp-compiler / mpi / pti / vtune / mkl / etc.), so
    # adding the ccl component to the install list pulls the binaries
    # in without their native deps. autoPatchelfHook then fails on
    # libccl.so.1 -> libfabric/librdmacm/libibverbs/libpsm2/libucp/...
    # Mirror the depsByComponent.ccl set our standalone oneccl-bmg
    # derivation used.
    # TODO: upstream a depsByComponent.ccl entry to nixpkgs'
    # intel-oneapi-toolkit and drop this list.
    buildInputs =
      (old.buildInputs or [ ])
      ++ (with pkgs; [
        rdma-core
        libpsm2
        ucx
        numactl
        libffi
        libuuid
        libfabric
      ]);
    postInstall = (old.postInstall or "") + ''
      # libccl dlopens libfabric.so when CCL_ATL_TRANSPORT=ofi: first by
      # name, then as `<dirname libccl.so>/libfabric.so`. With neither
      # resolvable, OFI init silently fails ("OFI transport was not
      # initialized, fallback to MPI transport") and libccl falls back
      # to libmpi.so.12 — which is on libtorch_xpu's RPATH via the
      # toolkit but never MPI_Init-ed, so the next allreduce segfaults
      # inside libmpi (vllm-xpu-nix #39).
      #
      # The libfabric the toolkit bundles is unusable: it's configured
      # with a hardcoded `/usr/local/lib/libfabric` provider path and
      # ships its providers as separate DSOs, so without a runtime
      # FI_PROVIDER_PATH it loads zero providers ("fi_getinfo error:
      # ret -61, providers 0"). nixpkgs' libfabric has the providers we
      # need (tcp, shm, sockets, rxm) compiled into libfabric.so
      # itself, so it works without env-var coaxing — symlink it as a
      # sibling of libccl so the relative-path dlopen succeeds.
      for cclLibDir in "$out"/ccl/*/lib; do
        if [ -d "$cclLibDir" ] && [ ! -e "$cclLibDir/libfabric.so" ]; then
          ln -s ${pkgs.libfabric}/lib/libfabric.so "$cclLibDir/libfabric.so"
        fi
      done
    '';
  })
