# Unified oneAPI 2026.0 base toolkit (libsycl.so.9, libmkl_*.so.3 /
# libmkl_sycl_*.so.6, oneCCL 2022.0.0). Pairs with the torch+xpu nightly we
# track in nix/torch-xpu.nix.
#
# oneCCL 2022.0.0 ships with the unified toolkit and supersedes the
# standalone oneccl-bmg (2021.15.9.14) we shipped against the earlier 2025.3
# base toolkit:
#   - 2021.17.2 added BMG single-process / multi-thread support for
#     allreduce, allgatherv, reduce_scatter
#   - 2022.0.0 added Arc Pro B-Series SPMD allreduce / allgather / alltoall
#     / reduce_scatter / broadcast / pt2pt
# The 2021.15.9.x line has no further patches; it also pinned libsycl.so.8,
# which the 2026.0 toolkit's libsycl.so.9 cannot satisfy — so going back to
# a stable torch (which links 2025.x) is the only way to use the standalone
# oneccl-bmg.
{ pkgs }:
(pkgs.intel-oneapi-toolkit.override {
  components = [
    "intel.oneapi.lin.dpcpp-cpp-compiler"
    "intel.oneapi.lin.mkl.devel"
    "intel.oneapi.lin.dpl"
    "intel.oneapi.lin.ccl.devel"
  ];
}).overrideAttrs (old: {
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
    (old.buildInputs or [])
    ++ (with pkgs; [
      rdma-core
      libpsm2
      ucx
      numactl
      libffi
      libuuid
      libfabric
    ]);
  postInstall =
    (old.postInstall or "")
    + ''
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
