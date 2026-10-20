import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

from huggingface_hub import scan_cache_dir


MODULE_PATH = Path(__file__).parents[1] / "nix/hf-cache-gc.py"
SPEC = importlib.util.spec_from_file_location("hf_cache_gc", MODULE_PATH)
HF_CACHE_GC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HF_CACHE_GC)


class CacheGcTest(unittest.TestCase):
    def add_repo(self, cache, repo_id, revision, contents):
        repo = cache / f"models--{repo_id.replace('/', '--')}"
        blob = repo / "blobs" / revision
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_text(contents)
        snapshot = repo / "snapshots" / revision
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").symlink_to(f"../../blobs/{revision}")
        return repo

    def test_same_commit_hash_is_deleted_from_only_the_stale_repo(self):
        revision = "a" * 40
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            rooted = self.add_repo(cache, "owner/rooted", revision, "rooted")
            stale = self.add_repo(cache, "owner/stale", revision, "stale")
            info = scan_cache_dir(cache)

            strategies, count = HF_CACHE_GC.plan_strategies(
                info, {("model", "owner/rooted"): {revision}}
            )
            for strategy in strategies:
                strategy.execute()

            self.assertEqual(count, 1)
            self.assertTrue(rooted.exists())
            self.assertFalse(stale.exists())

    @unittest.skipUnless(
        os.environ.get("HF_CACHE_TEST_ROOTS"), "requires the Nix module evaluation"
    )
    def test_evaluated_draft_roots_keep_named_revisions_and_single_component_ids(self):
        """Use actual module roots to collect a cache containing external drafts."""
        manifest = Path(os.environ["HF_CACHE_TEST_ROOTS"])
        data = json.loads(manifest.read_text())
        drafts = {
            root["source"]: (root["repo"], root["revision"])
            for root in data["roots"]
            if root["source"].endswith(".draft")
        }
        self.assertEqual(
            drafts,
            {
                "vllm-xpu.pinned.draft": ("example/pinned", "a" * 40),
                "vllm-xpu.branch.draft": ("example/branch", None),
                "vllm-xpu.tag.draft": ("example/tag", None),
                "vllm-xpu.single.draft": ("gpt2", None),
                # Ambiguous relative paths are conservatively treated as Hub IDs.
                "vllm-xpu.relative.draft": ("models/dflash", None),
            },
        )
        self.assertEqual(data["serving"]["branch"]["revision"], "main")
        self.assertEqual(data["serving"]["tag"]["revision"], "v2")
        roots, _ = HF_CACHE_GC.load_roots([manifest])
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            for repo, _ in drafts.values():
                for revision in ("a" * 40, "b" * 40):
                    self.add_repo(cache, repo, revision, revision)
            stale = self.add_repo(cache, "unused/draft", "a" * 40, "disabled")
            strategies, count = HF_CACHE_GC.plan_strategies(scan_cache_dir(cache), roots)
            for strategy in strategies:
                strategy.execute()
            self.assertEqual(count, 2)
            self.assertFalse(stale.exists())
            for repo, pin in drafts.values():
                snapshots = cache / f"models--{repo.replace('/', '--')}" / "snapshots"
                self.assertTrue((snapshots / ("a" * 40)).exists())
                self.assertEqual((snapshots / ("b" * 40)).exists(), pin is None)


if __name__ == "__main__":
    unittest.main()
