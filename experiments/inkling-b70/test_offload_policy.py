"""CPU regression tests. Run with the experimental Nix package's bin/python."""

import unittest
from types import SimpleNamespace

import numpy as np
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHash, make_block_hash_with_group_id
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_states.default import DefaultModelState


class OffloadPolicyTest(unittest.TestCase):
    def test_idle_hashes_and_transfer_fences(self):
        for active_only in (False, True):
            for release_fences in ((False, False), (True, False), (False, True)):
                with self.subTest(
                    active_only=active_only, release_fences=release_fences
                ):
                    self.check_release(active_only, release_fences)

    def check_release(self, active_only, release_fences):
        pool = BlockPool(2, True, 128)
        extra = {"gpu_cache_policy": "active_only"} if active_only else {}
        connector = object.__new__(OffloadingConnector)
        connector._kv_transfer_config = SimpleNamespace(kv_connector_extra_config=extra)
        connector.bind_kv_cache_manager(SimpleNamespace(block_pool=pool))
        (block,) = pool.get_new_blocks(1)
        sealed = make_block_hash_with_group_id(BlockHash(b"sealed"), 2)
        alias = make_block_hash_with_group_id(BlockHash(b"alias"), 2)
        pool._insert_block_hash(sealed, block, num_tokens=128)
        pool._insert_block_hash(alias, block, num_tokens=64)
        pool.touch([block])
        done = []
        for ref, release_fence in enumerate(release_fences):
            if release_fence:
                pool.unpin_blocks([block], done.append)
            else:
                pool.free_blocks([block])
            self.assertEqual(block.ref_cnt, 1 - ref)
            expected = None if active_only and ref == 1 else block
            for key in (sealed, alias):
                self.assertIs(
                    pool.cached_block_hash_to_block.get_one_block(key), expected
                )
        self.assertEqual(done, [])
        self.assertEqual(pool.get_new_blocks(1), [block])
        self.assertEqual(done, [block] if any(release_fences) else [])

    def test_generation_survives_reorder_and_changes_on_readmission(self):
        # Construct real MRv2 model state and CPU input tensors. The backend
        # returns the CommonAttentionMetadata instance handed to its builder.
        model_cfg = SimpleNamespace(
            get_inputs_embeds_size=lambda: 1,
            dtype=torch.float32,
            enable_prompt_embeds=False,
            uses_mrope=False,
            uses_xdrope_dim=0,
            supports_multimodal_inputs=False,
            max_model_len=1024,
        )
        cfg = SimpleNamespace(
            model_config=model_cfg,
            scheduler_config=SimpleNamespace(max_num_seqs=3, max_num_batched_tokens=8),
        )
        state = DefaultModelState(cfg, torch.nn.Identity(), None, torch.device("cpu"))
        builder = SimpleNamespace(supports_update_block_table=False)
        group = SimpleNamespace(
            metadata_builders=[builder],
            get_metadata_builder=lambda _: builder,
            build_metadata=lambda common, _: common,
            layer_names=["test"],
        )
        batch = InputBatch.make_dummy(2, 4, InputBuffers(3, 8, torch.device("cpu")))

        def metadata(slots):
            batch.idx_mapping_np = np.array(slots, dtype=np.intp)
            return state.prepare_attn(
                batch,
                CUDAGraphMode.NONE,
                (torch.zeros((2, 1), dtype=torch.int32),),
                torch.zeros((1, 4), dtype=torch.int64),
                [[group]],
                SimpleNamespace(kv_cache_groups=[None]),
            )["test"]

        self.assertEqual(metadata([0, 1]).req_generation.tolist(), [0, 0])
        req = SimpleNamespace(req_id="same-request-id")
        state.add_request(0, req)
        state.add_request(1, SimpleNamespace(req_id="other"))
        original = metadata([0, 1])
        self.assertEqual(original.req_generation.tolist(), [1, 1])
        self.assertEqual(original.req_generation.dtype, np.dtype("int64"))
        # Preemption/stream re-add can have identical ID, slot and block IDs.
        state.remove_request(req.req_id)
        state.add_request(0, req)
        ordered = metadata([1, 0])
        self.assertEqual(ordered.req_idx.tolist(), [1, 0])
        self.assertEqual(ordered.req_generation.tolist(), [1, 2])
        self.assertEqual(original.req_generation.tolist(), [1, 1])
        self.assertEqual(ordered.unpadded(2, 1).req_generation.tolist(), [1])
        # Verification/rollback updates computed lengths in place, with no add.
        batch.num_computed_tokens_np.fill(127)
        self.assertEqual(metadata([0, 1]).req_generation.tolist(), [2, 1])


if __name__ == "__main__":
    unittest.main()
