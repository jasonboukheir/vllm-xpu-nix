{
  lib,
  stdenvNoCC,
  python3Packages,
  patchelf,
  torch-xpu,
  baseGlue,
  fa2Binding,
  version,
}:

python3Packages.toPythonModule (
  stdenvNoCC.mkDerivation {
    pname = "vllm-xpu-kernels";
    inherit version;

    dontUnpack = true;
    nativeBuildInputs = [
      patchelf
      python3Packages.python
      python3Packages.packaging
    ];
    propagatedBuildInputs = [ torch-xpu ];

    installPhase = ''
          runHook preInstall

          mkdir -p "$out"
          cp -a ${baseGlue}/. "$out/"
          chmod -R u+w "$out"

          site="$out/${python3Packages.python.sitePackages}"
          package="$site/vllm_xpu_kernels"
          test -d "$package"

          shopt -s nullglob
          inheritedFa2=("$package"/_vllm_fa2_C*.so)
          test "''${#inheritedFa2[@]}" -eq 0
          for extension in _C _moe_C _xpu_C xpumem_allocator; do
            candidates=("$package"/$extension*.so)
            test "''${#candidates[@]}" -eq 1
          done
          fa2=("${fa2Binding}/${python3Packages.python.sitePackages}/vllm_xpu_kernels"/_vllm_fa2_C*.so)
          test "''${#fa2[@]}" -eq 1
          cp -a "''${fa2[0]}" "$package/"

          # The reusable base has a projection-hash version so implementation-only
          # attention commits do not invalidate it. Restamp only the cheap composed
          # package with the user-visible upstream version and rebuild RECORD after
      # adding the FA2 extension.
      distInfos=("$site"/vllm_xpu_kernels-*.dist-info)
      test "''${#distInfos[@]}" -eq 1
      ${python3Packages.python}/bin/python ${./scripts/compose-kernel-package.py} \
        "$site" "''${distInfos[0]}" "${version}"

          # The final extension must resolve exactly the current attention DSO. The
          # basename check prevents an absolute donor path from becoming DT_NEEDED;
          # the RPATH check proves autoPatchelf retained the selected split library.
          composedFa2=("$package"/_vllm_fa2_C*.so)
          test "''${#composedFa2[@]}" -eq 1
          patchelf --print-needed "''${composedFa2[0]}" \
            | grep -Fx 'libattn_kernels_xe_2.so'
          patchelf --print-rpath "''${composedFa2[0]}" \
            | tr ':' '\n' \
            | grep -Fx '${fa2Binding.attentionLibrary}/lib'

          runHook postInstall
    '';

    # Copying components must not make either donor package part of the runtime
    # closure. Their extension DSOs may reference only actual runtime libraries.
    disallowedReferences = [
      baseGlue
      fa2Binding
    ];

    meta = {
      description = "Composed vLLM XPU kernels package with reusable native glue";
      homepage = "https://github.com/vllm-project/vllm-xpu-kernels";
      license = lib.licenses.asl20;
      platforms = [ "x86_64-linux" ];
    };
  }
)
