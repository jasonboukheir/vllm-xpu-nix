# Brutus handoff: xpu-v1.12.0

The host configuration is committed at `59db9d0a729ad388c6f9c035396558c979fde858` and ready for
the operator's rebuild. No system switch or production service restart was run.

`modules/flake/nixos/server/flake.nix` selects the permanent packaging branch
`refs/heads/releases/xpu-v1.12.0`, locked to
`d7f516f09bbcd7de8a45e87c3d28bdedaeebee00`. Both anonymous HTTPS source
overrides select the matching `xpu-v1.12.0` tags. Only those three lock nodes
changed. The root lock, model settings, package override and independent
packaging nixpkgs pin remain unchanged. Host pre-commit formatting passed.

The clean committed host evaluates to the exact tested package:

```text
/nix/store/gkr9g6s5bsi0zkyfnnss6r2z9h481yby-python3.12-vllm-xpu-0.31.0+unstable.2026.10.10.gef92dc5.drv
```

Its realized output is already in the local Nix store:

```text
/nix/store/mcyp3ljp8yl1hwfllnfzf22mw3j1z42y-python3.12-vllm-xpu-0.31.0+unstable.2026.10.10.gef92dc5
```

Brutus system evaluation passed:

```text
/nix/store/shd04qj058azg9lrz51ksg7ja72if9jh-nixos-system-brutus-26.05.20261008.7c8764b.drv
```

Use the repository's rebuild command or:

```bash
sudo nixos-rebuild switch --flake ~/.config/nix#brutus
```

All owned foreground servers were stopped and GPU memory released. The original
production state remains: chat failed with MainPID=0, embedding/STT inactive.
After the operator rebuild, verify the enabled chat unit, `/health`, model
readiness and a representative request through the host's normal endpoint.
The earlier release branches/tags remain available for rollback.

[Publication verification](publication-verification.json) records all three
repositories, their immutable release commits, preserved prior refs and exact
host-package equality. Packaging main's following documentation commit records
this handoff; the release branch/tag remain fixed at the qualified snapshot.
