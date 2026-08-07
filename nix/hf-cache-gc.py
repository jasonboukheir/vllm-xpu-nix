#!/usr/bin/env python3
"""Collect Hugging Face cache entries not rooted by NixOS generations."""

import argparse
import fcntl
import json
import os
from pathlib import Path

from huggingface_hub import scan_cache_dir
from huggingface_hub.utils import HFCacheInfo


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    default_cache = os.environ.get(
        "HF_CACHE_GC_CACHE_DIR", "/var/cache/huggingface/hub"
    )
    parser.add_argument("--cache-dir", type=Path, default=Path(default_cache))
    parser.add_argument(
        "--profiles-dir", type=Path, default=Path("/nix/var/nix/profiles")
    )
    parser.add_argument("--runtime-dir", type=Path, default=Path("/run"))
    parser.add_argument(
        "--lock-file", type=Path, default=Path("/run/lock/hf-cache-gc.lock")
    )
    parser.add_argument("--delete", action="store_true", help="execute the deletion")
    return parser.parse_args()


def manifests(profiles_dir, runtime_dir):
    systems = sorted(profiles_dir.glob("system-*-link"))
    systems.extend(
        path
        for path in (
            profiles_dir / "system",
            runtime_dir / "current-system",
            runtime_dir / "booted-system",
        )
        if path.exists()
    )
    paths = []
    missing = []
    seen = set()
    for system in systems:
        closures = [system]
        specialisations = system / "specialisation"
        if specialisations.exists():
            closures.extend(path for path in specialisations.iterdir() if path.is_dir())
        for closure in closures:
            manifest = closure / "etc/huggingface/cache-roots.json"
            if manifest.exists():
                resolved = manifest.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    paths.append(manifest)
            else:
                missing.append(closure)
    if missing:
        names = ", ".join(path.name for path in missing)
        raise SystemExit(
            "refusing to collect: retained NixOS generations have no Hugging Face "
            f"cache manifest: {names}\n"
            "delete those generations first, or rebuild them with manifest support"
        )
    return paths


def load_roots(paths):
    roots = {}
    generations = []
    for path in paths:
        data = json.loads(path.read_text())
        generations.append(str(path))
        for root in data.get("roots", []):
            roots.setdefault((root["type"], root["repo"]), set()).add(
                revision.lower() if (revision := root.get("revision")) else None
            )
    return roots, generations


def roots_snapshot(args):
    paths = manifests(args.profiles_dir, args.runtime_dir)
    roots, generations = load_roots(paths)
    return roots, generations, {path.resolve() for path in paths}


def plan_strategies(cache, roots):
    strategies = []
    revision_count = 0
    for repo in sorted(cache.repos, key=lambda item: (item.repo_type, item.repo_id)):
        wanted = roots.get((repo.repo_type, repo.repo_id), set())
        if None in wanted:
            print(f"KEEP {repo.repo_type}/{repo.repo_id} (unpinned root)")
            continue
        stale = [rev for rev in repo.revisions if rev.commit_hash not in wanted]
        kept = len(repo.revisions) - len(stale)
        status = "KEEP" if not stale else "GC"
        print(
            f"{status} {repo.repo_type}/{repo.repo_id}: "
            f"{kept} rooted, {len(stale)} stale revision(s)"
        )
        if stale:
            repo_cache = HFCacheInfo(
                size_on_disk=repo.size_on_disk,
                repos=frozenset({repo}),
                warnings=[],
            )
            strategies.append(
                repo_cache.delete_revisions(*(rev.commit_hash for rev in stale))
            )
            revision_count += len(stale)
    return strategies, revision_count


def main():
    args = parse_args()
    roots, generations, manifest_paths = roots_snapshot(args)
    if not generations:
        raise SystemExit(
            f"refusing to collect: no cache-root manifests under {args.profiles_dir}"
        )

    cache = scan_cache_dir(args.cache_dir)
    if cache.warnings:
        details = "\n".join(f"- {warning}" for warning in cache.warnings)
        raise SystemExit(
            f"refusing to collect: cache scan reported corruption:\n{details}"
        )

    print(f"Retained generations: {len(generations)}")
    strategies, revision_count = plan_strategies(cache, roots)

    if not strategies:
        print("Nothing to collect.")
        return

    expected_size = sum(strategy.expected_freed_size for strategy in strategies)
    action = "Deleting" if args.delete else "Would delete"
    print(f"{action} {revision_count} revision(s), freeing {expected_size} bytes.")
    if args.delete:
        args.lock_file.parent.mkdir(parents=True, exist_ok=True)
        with args.lock_file.open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            current_roots, _, current_paths = roots_snapshot(args)
            if current_roots != roots or current_paths != manifest_paths:
                raise SystemExit(
                    "refusing to collect: retained system generations changed "
                    "during the cache scan; retry when NixOS switching is quiescent"
                )
            for strategy in strategies:
                strategy.execute()
    else:
        print("Dry run; pass --delete to execute.")


if __name__ == "__main__":
    main()
