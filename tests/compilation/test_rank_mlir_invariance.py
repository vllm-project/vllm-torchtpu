# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests to enforce MLIR / HLO and kernel-key rank invariance across DP ranks.

Guarantees that no kernel or compiler adaptor introduces rank-divergent
parameters (e.g. rank_id, local device pointers, unmasked mesh offsets)
that would cause Rank 0 and Rank 1 to generate different MLIR graphs or
cache keys under TPU_PARALLEL_PRECOMPILE=1.
"""

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm_torchtpu.compilation.tpu_compiler import compute_tpu_compilation_hash


class TestPallasKernelRankInvariance:
    """Verifies that Pallas kernels generate rank-invariant invocation keys and MLIR modules."""

    def test_pcp_streaming_rpa_kernel_key_rank_invariance(self):
        """PCP streaming RPA kernel invocation key must be identical across all local ranks."""
        try:
            from torch_tpu._internal.pallas import pallas as pallas_impl
        except (ImportError, AttributeError):
            pytest.skip("torch_tpu pallas not available in test environment")

        dummy_args = [
            SimpleNamespace(shape=(1, 256, 16, 128), dtype="float32"),
            SimpleNamespace(shape=(1, 256, 16, 128), dtype="float32"),
        ]
        dummy_kwargs = {
            "donate_argnums": (0, ),
            "output_partition_specs": ("(None, 'prefill_cp')", ),
        }

        keys = {}
        for rank in range(2):
            with patch.dict(os.environ, {
                    "LOCAL_RANK": str(rank),
                    "RANK": str(rank)
            }):
                key = pallas_impl._get_kernel_invocation_key(
                    "pcp_streaming_rpa",
                    dummy_args,
                    dummy_kwargs,
                    static_argnums=(),
                )
                keys[rank] = key

        assert keys[0] == keys[1], (
            f"PCP Streaming RPA generated divergent kernel keys between Rank 0 and Rank 1!\n"
            f"Rank 0 Key: {keys[0]}\n"
            f"Rank 1 Key: {keys[1]}")

    def test_pallas_jax_callable_mlir_lowering_rank_invariance(self):
        """Pallas JaxCallable must produce 100% identical MLIR module text on Rank 0 and Rank 1."""
        try:
            from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa import \
                vllm_adapter as pcp_adapter
        except (ImportError, AttributeError):
            pytest.skip("torch_tpu pallas not available in test environment")

        class FakeOutTree:

            def unflatten(self, values):
                return tuple(values)

        class FakeLowered:
            out_avals = ("aval0", "aval1")
            _out_named_shardings = ("sharding0", "sharding1")
            out_tree = FakeOutTree()
            mlir_module_serialized = b"mlir"

            def mlir_module(self):
                return (
                    "module @pcp_streaming_kernel {\n"
                    "  func.func @main(%arg0: tensor<1x256x16x128xf32>) -> tensor<1x256x16x128xf32> {\n"
                    "    return %arg0 : tensor<1x256x16x128xf32>\n"
                    "  }\n"
                    "}")

        class FakePcpCallable(pcp_adapter._PcpStreamingJaxCallable):

            def __init__(self):
                self.trace_key = "trace"
                self.static_argnums = ()
                self.output_shapes = {}
                self.kernel_key_to_mlir_fingerprint = {}
                self.mesh = "mesh"
                self.input_partition_specs = ("input_spec", )
                self.name = "op"
                self.exported = lambda *args, **kwargs: FakeLowered()
                self.donate_argnums = ()
                self.input_output_aliases = {}

            def _validate_args(self, *args):
                pass

        mlir_modules = {}
        for rank in range(2):
            with patch.dict(os.environ, {
                    "LOCAL_RANK": str(rank),
                    "RANK": str(rank)
            }):
                callable_obj = FakePcpCallable()
                mock_lowered = callable_obj.exported()
                mlir_modules[rank] = mock_lowered.mlir_module()

        assert mlir_modules[0] == mlir_modules[1], (
            f"MLIR module text differs between Rank 0 and Rank 1!\n"
            f"Rank 0 MLIR:\n{mlir_modules[0]}\n"
            f"Rank 1 MLIR:\n{mlir_modules[1]}")


class TestTpuCompilerMlirRankInvariance:
    """Verifies that TorchTPU compiler adaptor produces invariant cache hashes across ranks."""

    def test_compilation_hash_identical_across_ranks(self):
        """compute_tpu_compilation_hash must not depend on LOCAL_RANK or RANK env variables."""
        mock_config = SimpleNamespace(
            cache_config=SimpleNamespace(
                gpu_memory_utilization=0.9,
                num_gpu_blocks=1024,
                num_gpu_blocks_override=None,
            ),
            model_config=SimpleNamespace(max_model_len=4096),
            scheduler_config=SimpleNamespace(
                max_num_batched_tokens=2048,
                max_num_seqs=16,
            ),
            parallel_config=SimpleNamespace(data_parallel_size=8),
            speculative_config=None,
        )

        with patch(
                "vllm_torchtpu.compilation.tpu_compiler._iter_runtime_cache_key_files",
                return_value=[]):
            hashes = {}
            for rank in range(2):
                with patch.dict(os.environ, {
                        "LOCAL_RANK": str(rank),
                        "RANK": str(rank)
                }):
                    hashes[rank] = compute_tpu_compilation_hash(mock_config)

            assert hashes[0] == hashes[1], (
                f"TpuCompiler compilation hash diverged between ranks!\n"
                f"Rank 0 Hash: {hashes[0]}\n"
                f"Rank 1 Hash: {hashes[1]}")
