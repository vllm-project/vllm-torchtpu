import unittest
from unittest.mock import MagicMock

import torch

from vllm_torchtpu.layers.adapter.quantization.compressed_tensors.compressed_tensors_moe.utils import (  # noqa: E501
    get_cpu_weight_loader_hook,
)


class TestGetTpuCpuWeightLoaderHook(unittest.TestCase):
    def test_hook_creation_and_execution(self):
        # Mock layer
        layer = MagicMock()
        layer._map_global_expert_id_to_local_expert_id = (
            lambda x: x
        )  # Identity map for testing

        orig_loader = MagicMock()
        tp_size = 2
        tp_rank = 0

        hook = get_cpu_weight_loader_hook(layer, orig_loader, tp_size, tp_rank)

        self.assertTrue(callable(hook))

        # Test fallback
        param = MagicMock()
        param.shape = (4, 10, 8)  # Unused for fallback
        loaded_weight = torch.ones(10, 10)
        weight_name = "some_other_weight"

        res = hook(param, loaded_weight, weight_name, shard_id="", expert_id=0)
        self.assertTrue(orig_loader.called)

        # Test main weight loading
        orig_loader.reset_mock()
        param = torch.nn.Parameter(torch.empty(4, 16, 16, dtype=torch.float16))
        # Expected untransposed shape: (4, 16, 16)

        weight_name = "w13_weight_packed"
        # Since tp_size=2, tp_rank=0, we expect the first half of dim 0 to be taken.
        # loaded_weight shape should have dim 0 divisible by tp_size
        # (e.g. 16 -> 8 loaded per rank)
        loaded_weight = torch.ones((16, 16), dtype=torch.float16) * 2.0

        # Call hook for w1
        res = hook(param, loaded_weight, weight_name, shard_id="w1", expert_id=1)
        self.assertTrue(res)
        self.assertFalse(orig_loader.called)
        self.assertTrue(hasattr(param, "_cpu_scratch"))

        # param._cpu_scratch should have shape (4, 16, 16)
        scratch = param._cpu_scratch
        self.assertEqual(scratch.shape, (4, 16, 16))

        # Check that the data is loaded into the correct offset
        # loaded_per_rank is 16 // 2 = 8. weight_shard is 8x16.
        # offset for w1 is 0.
        # It copies to scratch.data[expert_id, offset:offset+8, :]
        self.assertEqual(scratch.data[1, 0:8, :].mean().item(), 2.0)

        # Call hook for w3
        loaded_weight = torch.ones((16, 16), dtype=torch.float16) * 3.0
        res = hook(param, loaded_weight, weight_name, shard_id="w3", expert_id=1)
        self.assertTrue(res)
        self.assertFalse(orig_loader.called)

        # offset for w3 is scratch.shape[1] // 2 = 16 // 2 = 8
        self.assertEqual(scratch.data[1, 8:16, :].mean().item(), 3.0)


if __name__ == "__main__":
    unittest.main()
