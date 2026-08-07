import importlib.util
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
        blob = repo / "blobs" / "blob"
        blob.parent.mkdir(parents=True)
        blob.write_text(contents)
        snapshot = repo / "snapshots" / revision
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").symlink_to("../../blobs/blob")
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


if __name__ == "__main__":
    unittest.main()
