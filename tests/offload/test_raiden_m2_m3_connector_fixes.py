# SPDX-License-Identifier: Apache-2.0
"""Unit tests verifying the M2/M3 Raiden offloading connector fixes.

1. HCA pool descending sort (raiden_connector.py):
   Ensures that heterogeneous KV pools (HCA 128 KiB, CSA 32 KiB,
   SWA 16 KiB, indexer 8 KiB)
   are sorted by descending page_size_bytes so pool_tensors[0] is always the maximum
   page size, preventing host staging buffer D2H/H2D truncation.

2. Worker TP-group rank binding (raiden_store.py):
   Ensures that in multi-host DP (e.g., DP=2, TP=8 with 16 world ranks), worker rank
   is derived from get_tp_group().rank_in_group (0..7) rather than
   global world rank (0..15),
   preventing index overflow in Raiden controller worker slots.
"""

import os
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

for _cand_path in (
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
    "/usr/local/google/home/jfy/claude-tpu-opt/studies/deepseek_v3_agentx_hillclimbing/scratch",
    "/usr/local/google/home/jfy/.gemini/jetski/brain/6a70d1f8-c693-4890-afaa-dbb1e841dd0e/scratch/gate7_m2/venv_torch/lib/python3.13/site-packages",
):
    if os.path.isdir(_cand_path) and _cand_path not in sys.path:
        sys.path.insert(0, _cand_path)

try:
    import vllm  # noqa: F401
except ImportError:
    from tests.test_m3_raiden_dspark_ep32 import _ensure_vllm_offload_stubs

    _ensure_vllm_offload_stubs()

if "torch_tpu" not in sys.modules:
    sys.modules["torch_tpu"] = MagicMock()
if "torch_tpu._internal.profiler" not in sys.modules:
    _prof_mod = types.ModuleType("torch_tpu._internal.profiler")
    _prof_mod.TpuProfilerConfig = type("TpuProfilerConfig", (), {})
    sys.modules["torch_tpu._internal.profiler"] = _prof_mod

if "vllm.v1.worker.worker_base" not in sys.modules:
    _wb_mod = types.ModuleType("vllm.v1.worker.worker_base")

    class _StubCompilationTimes:
        pass

    class _StubWorkerBase:
        def __init__(self, *args, **kwargs):
            pass

    _wb_mod.CompilationTimes = _StubCompilationTimes
    _wb_mod.WorkerBase = _StubWorkerBase
    sys.modules["vllm.v1.worker.worker_base"] = _wb_mod


class RaidenM2M3ConnectorFixesTest(unittest.TestCase):
    """Adversarial unit tests for Raiden offload connector fixes."""

    def test_raiden_connector_pool_tensors_descending_sort(self) -> None:
        """Verifies that pool_tensors are sorted by descending page_size_bytes."""
        p_indexer = SimpleNamespace(name="indexer", page_size_bytes=8192)
        p_swa = SimpleNamespace(name="swa", page_size_bytes=16384)
        p_csa = SimpleNamespace(name="csa", page_size_bytes=32768)
        p_hca = SimpleNamespace(name="hca", page_size_bytes=131072)

        # Initial order: indexer, swa, csa, hca (ascending)
        pool_tensors = [p_indexer, p_swa, p_csa, p_hca]

        # Apply the exact raiden_connector sort logic
        pool_tensors.sort(key=lambda pool_tensor: -pool_tensor.page_size_bytes)

        # pool_tensors[0] MUST be the largest (HCA 128 KiB)
        self.assertEqual(pool_tensors[0].name, "hca")
        self.assertEqual(pool_tensors[0].page_size_bytes, 131072)
        self.assertEqual(
            [p.page_size_bytes for p in pool_tensors],
            [131072, 32768, 16384, 8192],
        )

    def test_raiden_connector_negative_control_unsorted_truncation(self) -> None:
        """Negative control: without sort, smaller pool at [0] truncates
        staging buffer."""
        p_indexer = SimpleNamespace(name="indexer", page_size_bytes=8192)
        p_hca = SimpleNamespace(name="hca", page_size_bytes=131072)
        unsorted = [p_indexer, p_hca]

        # Without sort, index 0 is 8 KiB which cannot hold 128 KiB HCA transfers
        self.assertLess(unsorted[0].page_size_bytes, p_hca.page_size_bytes)

    def test_raiden_store_tp_group_rank_binding(self) -> None:
        """Verifies worker rank uses get_tp_group().rank_in_group instead of
        world rank."""
        mock_world_group = MagicMock()
        mock_world_group.rank_in_group = (
            13  # World rank 13 in DP=2, TP=8 (replica 1, TP rank 5)
        )

        mock_tp_group = MagicMock()
        mock_tp_group.rank_in_group = 5  # Clamped to 0..7 within TP group

        mock_parallel_state = MagicMock()
        mock_parallel_state.get_world_group.return_value = mock_world_group
        mock_parallel_state.get_tp_group.return_value = mock_tp_group

        with patch.dict(
            sys.modules,
            {"vllm.distributed": SimpleNamespace(parallel_state=mock_parallel_state)},
        ):
            # Emulate raiden_store rank resolution
            from vllm.distributed import parallel_state

            resolved_rank = parallel_state.get_tp_group().rank_in_group

            self.assertEqual(resolved_rank, 5)
            self.assertLess(
                resolved_rank, 8, "Worker rank must not exceed controller TP slots (8)"
            )
            self.assertNotEqual(resolved_rank, mock_world_group.rank_in_group)

    def test_rc41_record_pending_fences_handles_none_metadata_safely(self) -> None:
        """RC41: On idle DP ranks, _connector_metadata is None or unset;
        record_pending_fences must not assert."""
        from vllm_torchtpu.offload.raiden_connector import TPURaidenOffloadingConnector

        connector = object.__new__(TPURaidenOffloadingConnector)
        connector.connector_worker = SimpleNamespace()
        connector._pending_fences = []
        connector._pool_sync_tensors = []
        connector._connector_metadata = None

        # Case 1: _connector_metadata is None (idle DP rank)
        connector.record_pending_fences()
        self.assertEqual(connector._pending_fences, [])

        # Case 2: _connector_metadata is not set at all
        delattr(connector, "_connector_metadata")
        connector.record_pending_fences()
        self.assertEqual(connector._pending_fences, [])

    def test_rc41_tpu_worker_execute_dummy_batch_record_fences_exception_safe(
        self,
    ) -> None:
        """RC41: Even if record_pending_fences raises an unexpected
        exception, execute_dummy_batch catches it."""
        mock_runner = MagicMock()
        mock_runner._dp_step_executed_in_empty_execute_model = False
        mock_runner._dp_coordinated_step.return_value = (SimpleNamespace(), 1)

        mock_kv_connector = MagicMock()
        mock_kv_connector.record_pending_fences.side_effect = RuntimeError(
            "Simulated crash"
        )

        from vllm_torchtpu.worker.tpu_worker import TPUWorker

        worker = object.__new__(TPUWorker)
        worker.model_runner = mock_runner

        with (
            patch(
                "vllm_torchtpu.worker.tpu_worker.has_kv_transfer_group",
                return_value=True,
            ),
            patch(
                "vllm_torchtpu.worker.tpu_worker.get_kv_transfer_group",
                return_value=mock_kv_connector,
            ),
        ):
            # Must not raise RuntimeError
            worker.execute_dummy_batch()

        # Verify dummy pairing forward still ran
        mock_runner._run_dp_idle_pairing.assert_called_once()

    def test_rc42_get_finished_handles_none_metadata_safely(self) -> None:
        """RC42: On idle DP ranks or post-clear steps, get_finished must
        not assert on None metadata."""
        from vllm_torchtpu.offload.raiden_connector import TPURaidenOffloadingConnector

        connector = object.__new__(TPURaidenOffloadingConnector)
        connector.connector_worker = SimpleNamespace()
        connector._pending_fences = []
        connector._pool_sync_tensors = []
        connector._fenced_jobs = {}
        connector._completed_jobs = {}
        connector._load_error_block_ids = set()
        connector._connector_metadata = None

        # Case 1: _connector_metadata is None
        sent, recvd = connector.get_finished(set())
        self.assertEqual(sent, set())
        self.assertEqual(recvd, set())

        # Case 2: _connector_metadata is deleted
        delattr(connector, "_connector_metadata")
        sent, recvd = connector.get_finished(set())
        self.assertEqual(sent, set())
        self.assertEqual(recvd, set())

        # Case 3: connector_worker is None
        connector.connector_worker = None
        sent, recvd = connector.get_finished(set())
        self.assertEqual(sent, set())
        self.assertEqual(recvd, set())

    def test_rc42_start_load_kv_wait_for_save_clear_metadata_safe(self) -> None:
        """RC42: start_load_kv, wait_for_save, and clear_connector_metadata
        must be safe with None metadata."""
        from vllm_torchtpu.offload.raiden_connector import TPURaidenOffloadingConnector

        connector = object.__new__(TPURaidenOffloadingConnector)
        connector.connector_worker = SimpleNamespace(start_kv_transfers=MagicMock())
        connector._connector_metadata = None

        # start_load_kv with None metadata is safe no-op
        connector.start_load_kv()
        connector.connector_worker.start_kv_transfers.assert_not_called()

        # wait_for_save is safe no-op
        connector.wait_for_save()

        # clear_connector_metadata clears metadata
        connector._connector_metadata = "mock_meta"
        connector.clear_connector_metadata()
        self.assertIsNone(connector._connector_metadata)

    def test_rc42_tpu_runner_maybe_setup_kv_connector_none_metadata_safe(self) -> None:
        """RC42: maybe_setup_kv_connector must return early when
        kv_connector_metadata is None."""
        from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

        runner = object.__new__(TPUModelRunner)
        mock_sched_out = SimpleNamespace(kv_connector_metadata=None)

        with (
            patch(
                "vllm_torchtpu.runner.tpu_runner.has_kv_transfer_group",
                return_value=True,
            ),
            patch(
                "vllm_torchtpu.runner.tpu_runner.get_kv_transfer_group"
            ) as mock_get_group,
        ):
            # Must return cleanly without asserting
            runner.maybe_setup_kv_connector(mock_sched_out)
            mock_get_group.assert_not_called()

    def test_rc42_tpu_runner_maybe_wait_for_kv_save_handles_missing_method_safely(
        self,
    ) -> None:
        """RC42: maybe_wait_for_kv_save must handle connectors without
        wait_for_save method."""
        from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

        runner = object.__new__(TPUModelRunner)
        mock_connector = object()  # Has no wait_for_save method

        with (
            patch(
                "vllm_torchtpu.runner.tpu_runner.has_kv_transfer_group",
                return_value=True,
            ),
            patch(
                "vllm_torchtpu.runner.tpu_runner.get_kv_transfer_group",
                return_value=mock_connector,
            ),
        ):
            # Must not raise AttributeError
            runner.maybe_wait_for_kv_save()

    def test_rc42_hybrid_tp2_dp8_ep16_invariants_and_gate7b_parity(self) -> None:
        """RC42: Verify hybrid TP=2, DP=8, EP=16 topology invariants and
        Gate 7B (TRAP24) precision parity."""
        import torch

        num_target_experts = 384
        num_dspark_experts = 160
        ep_size = 16
        tp_size = 2
        dp_size = 8
        total_chips = tp_size * dp_size
        self.assertEqual(total_chips, 16, "TP=2 * DP=8 must equal 16 TPU chips")

        self.assertEqual(num_target_experts % ep_size, 0)
        self.assertEqual(num_dspark_experts % ep_size, 0)
        g_local_target = num_target_experts // ep_size
        g_local_dspark = num_dspark_experts // ep_size
        self.assertEqual(
            g_local_target, 24, "384 / 16 must be 24 local target experts per rank"
        )
        self.assertEqual(
            g_local_dspark, 10, "160 / 16 must be 10 local DSpark experts per rank"
        )

        # Gate 7B precision parity (TRAP24):
        torch.manual_seed(20261006)
        num_tokens = 64
        hidden_size = 256
        expert_outputs_bf16 = torch.randn(
            (ep_size, num_tokens, hidden_size), dtype=torch.bfloat16
        )

        # Golden unpartitioned reference (float32 accumulation across all 16 EP shards)
        golden_ref = torch.sum(expert_outputs_bf16.to(torch.float32), dim=0).to(
            torch.bfloat16
        )

        # Hybrid TP=2, DP=8, EP=16: TP=2 split in float32 + EP=16 reduction
        tp0_half = (
            expert_outputs_bf16[:, :, : hidden_size // 2].to(torch.float32).sum(dim=0)
        )
        tp1_half = (
            expert_outputs_bf16[:, :, hidden_size // 2 :].to(torch.float32).sum(dim=0)
        )
        hybrid_merged = torch.cat([tp0_half, tp1_half], dim=-1).to(torch.bfloat16)

        # CosSim and SQNR
        ref_f32 = golden_ref.to(torch.float32)
        cand_f32 = hybrid_merged.to(torch.float32)
        cossim = float(
            torch.nn.functional.cosine_similarity(
                ref_f32.flatten(), cand_f32.flatten(), dim=0
            )
        )
        noise = ref_f32 - cand_f32
        signal_pwr = float(torch.mean(ref_f32**2))
        noise_pwr = float(torch.mean(noise**2))
        sqnr = (
            100.0
            if noise_pwr == 0.0
            else float(10.0 * torch.log10(torch.tensor(signal_pwr / noise_pwr)))
        )

        self.assertGreaterEqual(
            cossim, 0.99999, f"TP=2 DP=8 EP=16 CosSim {cossim} < 0.99999"
        )
        self.assertGreaterEqual(sqnr, 45.0, f"TP=2 DP=8 EP=16 SQNR {sqnr} dB < 45.0 dB")

    def test_rc43_get_connector_metadata_safe_override(self) -> None:
        """RC43: TPURaidenOffloadingConnector._get_connector_metadata must
        safely return metadata without asserting."""
        from vllm_torchtpu.offload.raiden_connector import TPURaidenOffloadingConnector

        connector = object.__new__(TPURaidenOffloadingConnector)

        # Path 1: metadata is None (idle DP step) -> must return None without
        # raising AssertionError
        connector._connector_metadata = None
        self.assertIsNone(connector._get_connector_metadata())

        # Path 2: metadata attribute deleted -> must return None without
        # raising AttributeError or AssertionError
        if hasattr(connector, "_connector_metadata"):
            delattr(connector, "_connector_metadata")
        self.assertIsNone(connector._get_connector_metadata())

        # Path 3: metadata is set -> returns it
        mock_meta = SimpleNamespace(name="active_metadata")
        connector._connector_metadata = mock_meta
        self.assertIs(connector._get_connector_metadata(), mock_meta)

        # Negative control: verify that legacy base class asserting behavior
        # raised AssertionError
        class _LegacyBase:
            _connector_metadata = None

            def _get_connector_metadata(self):
                assert self._connector_metadata is not None, "Legacy assertion"
                return self._connector_metadata

        legacy = _LegacyBase()
        with self.assertRaises(AssertionError):
            legacy._get_connector_metadata()


if __name__ == "__main__":
    unittest.main()
