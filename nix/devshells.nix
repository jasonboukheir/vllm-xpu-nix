# Editor-facing development shells. Build, test, runtime, and accelerator
# dependencies belong to package derivations and flake checks, not these shells.
{
  pkgs,
  lint,
}:

let
  editorTools = with pkgs; [
    # Nix editing.
    nil
    nixfmt

    # Python editing and linting. ruff also provides `ruff server` for LSP.
    python312
    ruff

    # C/C++ editing and formatting; clang-tools provides clangd and
    # clang-format without pulling in the project build toolchain.
    llvmPackages_20.clang-tools

    # Repository lint wrapper and spelling checks. Keep pre-commit out of the
    # shell: its hook environments fetch dependencies outside Nix.
    codespell
    lint
  ];

  mkEditorShell =
    name: project:
    pkgs.mkShellNoCC {
      inherit name;
      packages = editorTools;
      shellHook = ''
        cat <<'EOF'
        ${project} editor shell.

        This shell intentionally contains only code-writing, LSP, formatting,
        and lint tooling. Builds and tests are owned by package derivations and
        `nix flake check`.

        Useful commands:
          nil                        # Nix language server
          ruff server                # Python language server
          clangd                     # C/C++ language server
          nixfmt --check flake.nix nix/devshells.nix
          lint
          nix flake check
        EOF
      '';
    };
in
{
  default = mkEditorShell "vllm-xpu-nix-dev" "vllm-xpu-nix";

  # Retain the named entry points for editor configuration compatibility.
  # They deliberately do not inherit the kernels or vLLM build closures.
  kernels-dev = mkEditorShell "vllm-xpu-kernels-dev" "vllm-xpu-kernels";
  vllm-dev = mkEditorShell "vllm-xpu-vllm-dev" "vLLM XPU";
}
