"""
test_tpu_hardware.py
--------------------
Hardware-accelerated test suite running on physical Cloud TPU v6e chips using TorchTPU:
- Validates TorchTPU device registration and on-chip tensor allocation (torch.device('tpu')).
- Executes 2:4 structured magnitude pruning on TPU tensors via vectorized operations.
- Verifies real Gemma 4 31B-IT weight checkpoint integrity on persistent storage.

Note:
- Dedicated to TorchTPU backend.
- Skips on-chip allocation during coverage tracing if `COVERAGE_RUN=1` to prevent PJRT worker deadlock.
"""

import os
import unittest

import torch
from safetensors import safe_open
from sparsity.naive_magnitude_pruner import prune_tensor_magnitude_nm

try:
    from tests.fixtures import REAL_GEMMA4_PATH, has_real_gemma4_weights
except ImportError:
    from fixtures import REAL_GEMMA4_PATH, has_real_gemma4_weights

# Detect TorchTPU availability
HAS_TORCHTPU = False
TPU_DEVICE = None

try:
    if hasattr(torch, "tpu") and torch.tpu.is_available():
        HAS_TORCHTPU = True
        TPU_DEVICE = torch.device("tpu")
    else:
        t = torch.zeros(1, device="tpu")
        HAS_TORCHTPU = True
        TPU_DEVICE = torch.device("tpu")
except Exception:
    HAS_TORCHTPU = False
    TPU_DEVICE = None


class TestTpuHardwareExecution(unittest.TestCase):
    """Executes on-chip hardware verification tests when Cloud TPU hardware is available via TorchTPU."""

    def test_tpu_device_detection_and_tensor_allocation(self):
        """
        Arrange-Act-Assert: Verify Cloud TPU device detection and on-chip tensor creation via TorchTPU.
        Verifies:
        1. Queries PyTorch for available TorchTPU device ('tpu').
        2. Allocates a bfloat16 tensor directly on TPU HBM.
        3. Executes an on-chip arithmetic operation (multiplication by 2.0) and validates the result.
        """
        if not HAS_TORCHTPU:
            self.skipTest("No TorchTPU device detected in this environment")

        if os.environ.get("COVERAGE_RUN") == "1":
            self.skipTest(
                "Skipping on-chip TPU allocation during coverage tracing to prevent PJRT lock"
            )

        # Arrange & Act: Allocate tensor on TorchTPU device and execute on-chip math
        x = torch.tensor([1.0, 2.0, 3.0, 4.0],
                         dtype=torch.bfloat16,
                         device=TPU_DEVICE)
        y = x * 2.0

        # Assert: Fetch result back to CPU and verify correctness
        y_cpu = y.cpu()
        expected = torch.tensor([2.0, 4.0, 6.0, 8.0], dtype=torch.bfloat16)
        self.assertTrue(torch.allclose(y_cpu, expected, atol=1e-2))

    def test_on_chip_2of4_magnitude_pruning(self):
        """
        Arrange-Act-Assert: Validates 2:4 magnitude pruning vectorized on TorchTPU tensor.
        Verifies:
        1. Allocates a 16x64 weight matrix on TPU.
        2. Executes prune_tensor_magnitude_nm on the TPU tensor.
        3. Validates 50% sparsity after transferring back to host memory.
        """
        if not HAS_TORCHTPU:
            self.skipTest("No TorchTPU device detected in this environment")

        if os.environ.get("COVERAGE_RUN") == "1":
            self.skipTest(
                "Skipping on-chip TPU allocation during coverage tracing")

        # Arrange: Allocate random tensor on TPU
        tensor_tpu = torch.randn(16,
                                 64,
                                 dtype=torch.bfloat16,
                                 device=TPU_DEVICE)

        # Act: Prune on TPU device
        pruned_tpu = prune_tensor_magnitude_nm(tensor_tpu, prunen=2, prunem=4)

        # Assert: Transfer to CPU and verify exact 50% sparsity
        pruned_cpu = pruned_tpu.cpu()
        zero_fraction = (pruned_cpu == 0).sum().item() / pruned_cpu.numel()
        self.assertAlmostEqual(zero_fraction, 0.5, delta=0.01)

    def test_real_gemma4_checkpoint_layers_if_available(self):
        """
        Arrange-Act-Assert: Inspects real Gemma 4 31B-IT checkpoint shards if present on disk.
        Verifies:
        1. Safetensors shard files can be opened and metadata indexed without memory exhaustion.
        2. Validates presence of language model attention and MLP weights.
        """
        if not has_real_gemma4_weights():
            self.skipTest(
                "Real Gemma 4 31B-IT checkpoint not found on local disk")

        # Arrange: Locate shard 1
        shard_path = os.path.join(REAL_GEMMA4_PATH,
                                  "model-00001-of-00002.safetensors")
        self.assertTrue(os.path.exists(shard_path))

        # Act: Inspect header and tensor keys
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            keys = set(f.keys())

        # Assert: Verify presence of Layer 0 self-attention and MLP weights
        self.assertTrue(
            any("language_model.layers.0.self_attn" in k for k in keys))
        self.assertTrue(any("language_model.layers.0.mlp" in k for k in keys))


if __name__ == "__main__":
    unittest.main()
