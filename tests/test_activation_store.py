import importlib.util
from pathlib import Path
import tempfile
import unittest

import torch


SCRIPT = Path(__file__).parents[1] / "scripts" / "activation_store.py"
SPEC = importlib.util.spec_from_file_location("activation_store", SCRIPT)
store_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(store_module)


class ActivationStoreTests(unittest.TestCase):
    def test_nested_round_trip_shared_tensor_and_order(self):
        with tempfile.TemporaryDirectory() as directory:
            store = store_module.ActivationStore(Path(directory))
            writer = store.writer("fp-0000", identity={"model": "a", "block": 0})
            shared = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
            writer.append(((shared,), {"mask": shared, "flag": True, "none": None}))
            writer.append(([torch.tensor([9], dtype=torch.int64)], {"label": "second"}))
            handle = writer.commit()
            corpus = store.open(handle, verify_all=True)
            first = corpus[0]
            self.assertTrue(torch.equal(first[0][0], shared))
            self.assertIs(first[0][0], first[1]["mask"])
            self.assertEqual(first[1]["flag"], True)
            self.assertEqual([item[1].get("label") for item in corpus.iter_indices([1, 0])], ["second", None])

    def test_unpublished_and_corrupt_generations_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = store_module.ActivationStore(root)
            writer = store.writer("q-0001", identity={"block": 1})
            writer.append(torch.ones(4))
            handle = writer.commit()
            extent = next(handle.path.glob("extent-*.bin"))
            extent.write_bytes(extent.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError, "truncated|checksum"):
                store.open(handle, verify_all=True)

    def test_orphan_collection_only_removes_temporary_generations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = store_module.ActivationStore(root)
            orphan = root / ".generation-dead"
            orphan.mkdir()
            committed_writer = store.writer("fp", identity={})
            committed_writer.append(torch.ones(1))
            handle = committed_writer.commit()
            self.assertEqual(store.collect_orphans(), [orphan.name])
            self.assertTrue(handle.path.exists())


if __name__ == "__main__":
    unittest.main()
