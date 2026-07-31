#!/usr/bin/env python3
"""Collect Hugging Face cache entries not rooted by NixOS generations."""

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import scan_cache_dir


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    default_cache = os.environ.get(
        "HF_CACHE_GC_CACHE_DIR", "/var/cache/huggingface/hub"
    )
    parser.add_argument("--cache-dir", type=Path, default=Path(default_cache))
    parser.add_argument(
        "--profiles-dir", type=Path, default=Path("/nix/var/nix/profiles")
    )
    parser.add_argument("--delete", action="store_true", help="execute the deletion")
    return parser.parse_args()


def manifests(profiles_dir):
    generations = sorted(profiles_dir.glob("system-*-link"))
    paths = []
    missing = []
    for generation in generations:
        manifest = generation / "etc/huggingface/cache-roots.json"
        if manifest.exists():
            paths.append(manifest)
        else:
            missing.append(generation)
    if missing:
        names = ", ".join(path.name for path in missing)
        raise SystemExit(
            "refusing to collect: retained NixOS generations have no Hugging Face "
            f"cache manifest: {names}\n"
            "delete those generations first, or rebuild them with manifest support"
        )
    current = profiles_dir / "system/etc/huggingface/cache-roots.json"
    if current.exists():
        paths.append(current)
    return paths


def load_roots(paths):
    roots = {}
    generations = []
    for path in paths:
        data = json.loads(path.read_text())
        generations.append(str(path))
        for root in data.get("roots", []):
            roots.setdefault((root["type"], root["repo"]), set()).add(
                root.get("revision")
            )
    return roots, generations


def main():
    args = parse_args()
    roots, generations = load_roots(manifests(args.profiles_dir))
    if not generations:
        raise SystemExit(
            f"refusing to collect: no cache-root manifests under {args.profiles_dir}"
        )

    cache = scan_cache_dir(args.cache_dir)
    revisions = []
    print(f"Retained generations: {len(generations)}")
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
        revisions.extend(rev.commit_hash for rev in stale)

    if not revisions:
        print("Nothing to collect.")
        return

    strategy = cache.delete_revisions(*revisions)
    action = "Deleting" if args.delete else "Would delete"
    print(
        f"{action} {len(revisions)} revision(s), freeing {strategy.expected_freed_size_str}."
    )
    if args.delete:
        strategy.execute()
    else:
        print("Dry run; pass --delete to execute.")


if __name__ == "__main__":
    main()
