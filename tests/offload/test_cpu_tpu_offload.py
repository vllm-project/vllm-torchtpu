# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for the TPU↔CPU KV cache offloading stack.

Covers the pieces that don't require a real TPU:
- `_resolve_kv_cache_dtype` strictness (raise on "auto") and
  `get_kv_cache_shape` probe-path placeholder semantics in
  `vllm_torchtpu.layers.vllm.attention`.
- `TPUCPUOffloadingSpec.estimate_hbm_reserve_bytes` size formula scales
  proportionally with `KV_H2D_POOL_MAX_BLOCKS`.
- `TPUCPUOffloadingSpec.prewarm_shapes` / `prewarm_shape` delegate to the
  H2D handler when `_tpu_handlers` is populated; no-op before.
- `Transfer` dataclass defaults and the error-capture wrapping pattern
  used by `_h2d_task` / `_d2h_task`.
- `transfer_async` H2D chunking math: `chunks_total = ceil(len(src_ids)
  / chunk_size)` at the boundary cases.
- `utils.estimate_kv_connector_hbm_reserve` dynamic spec resolution
  (no kv_connector / missing metadata / unimportable module → 0).
- `expand_hybrid_pool_block_ids` grouped row mapping for the unified
  block pool (attention + mamba segments, partial first offloaded block,
  null-prefixed mamba segments, empty groups).

Heavy dependencies (real Pallas kernels, torch_tpu DMA, vllm config
plumbing) are mocked at construction time so the tests run on a host
without a TPU.
"""
import os
import unittest
from unittest.mock import MagicMock, patch

import torch


class TestResolveKvCacheDtype(unittest.TestCase):
    """attention._resolve_kv_cache_dtype contract."""

    def test_passes_through_torch_dtype(self):
        from vllm_torchtpu.layers.vllm.attention import _resolve_kv_cache_dtype
        self.assertEqual(_resolve_kv_cache_dtype(torch.bfloat16),
                         torch.bfloat16)

    def test_resolves_known_strings(self):
        from vllm_torchtpu.layers.vllm.attention import _resolve_kv_cache_dtype
        self.assertEqual(_resolve_kv_cache_dtype("bfloat16"), torch.bfloat16)
        self.assertEqual(_resolve_kv_cache_dtype("fp8_e4m3"),
                         torch.float8_e4m3fn)
        self.assertEqual(_resolve_kv_cache_dtype("half"), torch.half)

    def test_auto_raises(self):
        from vllm_torchtpu.layers.vllm.attention import _resolve_kv_cache_dtype
        with self.assertRaisesRegex(ValueError, "must be resolved"):
            _resolve_kv_cache_dtype("auto")

    def test_unknown_raises(self):
        from vllm_torchtpu.layers.vllm.attention import _resolve_kv_cache_dtype
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            _resolve_kv_cache_dtype("not-a-dtype")


class TestGetKvCacheShapeProbe(unittest.TestCase):
    """OffloadingConnectorWorker probes get_kv_cache_shape without a dtype
    to discover the num_blocks dimension. The placeholder return must
    preserve num_blocks at position 0 — see
    vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py
    register_kv_caches."""

    def test_auto_returns_placeholder_with_num_blocks_at_dim_0(self):
        from vllm_torchtpu.layers.vllm.attention import PallasAttentionBackend
        shape = PallasAttentionBackend.get_kv_cache_shape(num_blocks=1234,
                                                          block_size=16,
                                                          num_kv_heads=1,
                                                          head_size=256)
        self.assertEqual(shape[0], 1234)
        self.assertEqual(shape.index(1234), 0)

    def test_concrete_dtype_returns_real_shape(self):
        from vllm_torchtpu.layers.vllm.attention import PallasAttentionBackend
        shape = PallasAttentionBackend.get_kv_cache_shape(
            num_blocks=2048,
            block_size=16,
            num_kv_heads=8,
            head_size=128,
            cache_dtype_str="bfloat16")
        self.assertEqual(shape[0], 2048)
        self.assertEqual(shape[1], 16)  # block_size


class TestEstimateHbmReserveBytes(unittest.TestCase):
    """TPUCPUOffloadingSpec.estimate_hbm_reserve_bytes scales linearly
    with both num_layers and KV_H2D_POOL_MAX_BLOCKS."""

    @staticmethod
    def _make_vllm_config(num_layers=32,
                          num_kv_heads=8,
                          head_size=128,
                          block_size=16,
                          is_hybrid=False,
                          num_attn_layers=None):
        cfg = MagicMock()
        cfg.model_config.get_num_layers.return_value = num_layers
        cfg.model_config.get_num_kv_heads.return_value = num_kv_heads
        cfg.model_config.get_head_size.return_value = head_size
        cfg.model_config.dtype = "bfloat16"
        cfg.model_config.is_hybrid = is_hybrid
        cfg.model_config.get_num_layers_by_block_type.return_value = (
            num_attn_layers if num_attn_layers is not None else num_layers)
        cfg.cache_config.block_size = block_size
        cfg.cache_config.cache_dtype = "auto"  # resolved via model_dtype
        return cfg

    def test_doubling_pool_doubles_reserve(self):
        from vllm_torchtpu.offload.cpu_tpu import TPUCPUOffloadingSpec
        cfg = self._make_vllm_config()
        with patch.dict(os.environ, {"KV_H2D_POOL_MAX_BLOCKS": "1024"}):
            half = TPUCPUOffloadingSpec.estimate_hbm_reserve_bytes(cfg)
        with patch.dict(os.environ, {"KV_H2D_POOL_MAX_BLOCKS": "2048"}):
            full = TPUCPUOffloadingSpec.estimate_hbm_reserve_bytes(cfg)
        self.assertEqual(full, 2 * half)

    def test_doubling_layers_doubles_reserve(self):
        from vllm_torchtpu.offload.cpu_tpu import TPUCPUOffloadingSpec
        with patch.dict(os.environ, {"KV_H2D_POOL_MAX_BLOCKS": "2048"}):
            small = TPUCPUOffloadingSpec.estimate_hbm_reserve_bytes(
                self._make_vllm_config(num_layers=32))
            large = TPUCPUOffloadingSpec.estimate_hbm_reserve_bytes(
                self._make_vllm_config(num_layers=64))
        self.assertEqual(large, 2 * small)

    def test_pool_blocks_rounds_to_next_power_of_2(self):
        from vllm_torchtpu.offload.cpu_tpu import TPUCPUOffloadingSpec
        cfg = self._make_vllm_config()
        with patch.dict(os.environ, {"KV_H2D_POOL_MAX_BLOCKS": "1500"}):
            reserve_1500 = TPUCPUOffloadingSpec.estimate_hbm_reserve_bytes(cfg)
        with patch.dict(os.environ, {"KV_H2D_POOL_MAX_BLOCKS": "2048"}):
            reserve_2048 = TPUCPUOffloadingSpec.estimate_hbm_reserve_bytes(cfg)
        # 1500 rounds up to 2048
        self.assertEqual(reserve_1500, reserve_2048)

    def test_hybrid_uses_attention_layer_count_and_hybrid_pool_default(self):
        """Hybrid (unified block pool) staging is one buffer per attention
        layer position at the pool page, with the smaller hybrid
        KV_H2D_POOL_MAX_BLOCKS default (512 vs 2048)."""
        from vllm_torchtpu.offload.cpu_tpu import TPUCPUOffloadingSpec
        num_layers, num_attn = 60, 15
        with patch.dict(os.environ, {"KV_H2D_POOL_MAX_BLOCKS": "2048"}):
            dense = TPUCPUOffloadingSpec.estimate_hbm_reserve_bytes(
                self._make_vllm_config(num_layers=num_layers))
            hybrid = TPUCPUOffloadingSpec.estimate_hbm_reserve_bytes(
                self._make_vllm_config(num_layers=num_layers,
                                       is_hybrid=True,
                                       num_attn_layers=num_attn))
        # Same pool cap: only the layer count differs (60 vs 15 buffers).
        self.assertEqual(dense * num_attn, hybrid * num_layers)

        orig = os.environ.pop("KV_H2D_POOL_MAX_BLOCKS", None)
        try:
            hybrid_default = TPUCPUOffloadingSpec.estimate_hbm_reserve_bytes(
                self._make_vllm_config(num_layers=num_layers,
                                       is_hybrid=True,
                                       num_attn_layers=num_attn))
            # Default hybrid pool cap is 512 (vs 2048 explicit above).
            self.assertEqual(hybrid, 4 * hybrid_default)
        finally:
            if orig is not None:
                os.environ["KV_H2D_POOL_MAX_BLOCKS"] = orig

    def test_raiden_defaults_to_zero_reserve(self):
        from vllm_torchtpu.offload.cpu_tpu import TPUCPUOffloadingSpec
        cfg = self._make_vllm_config()
        # Temporarily pop global env variable so we can test the fallback path
        # when it is absent. We use try/finally to ensure it is restored even
        # if the test assertions fail (avoiding environment leakage to other tests).
        orig_max_blocks = os.environ.pop("KV_H2D_POOL_MAX_BLOCKS", None)
        try:
            with patch("vllm_torchtpu.offload.cpu_tpu._USE_RAIDEN_OFFLOAD",
                       True):
                raiden_reserve = TPUCPUOffloadingSpec.estimate_hbm_reserve_bytes(
                    cfg)
            self.assertEqual(raiden_reserve, 0)
        finally:
            if orig_max_blocks is not None:
                os.environ["KV_H2D_POOL_MAX_BLOCKS"] = orig_max_blocks


class TestPrewarmDelegation(unittest.TestCase):
    """TPUCPUOffloadingSpec.prewarm_shapes / prewarm_shape forward to the
    H2D handler. Before get_handlers() runs (_tpu_handlers is None) they
    are no-ops so the runner's collective_rpc doesn't crash."""

    def test_prewarm_shapes_none_when_handlers_not_built(self):
        from vllm_torchtpu.offload.cpu_tpu import TPUCPUOffloadingSpec
        spec = TPUCPUOffloadingSpec.__new__(TPUCPUOffloadingSpec)
        spec._tpu_handlers = None
        self.assertEqual(spec.prewarm_shapes, [])
        spec.prewarm_shape(2048)  # must not raise

    def test_prewarm_shapes_delegates_to_h2d_handler(self):
        from vllm_torchtpu.offload.cpu_tpu import TPUCPUOffloadingSpec
        spec = TPUCPUOffloadingSpec.__new__(TPUCPUOffloadingSpec)
        spec._tpu_handlers = MagicMock()
        spec._tpu_handlers.cpu_to_gpu_handler.prewarm_shapes = [1, 2, 4, 8]
        self.assertEqual(spec.prewarm_shapes, [1, 2, 4, 8])

    def test_prewarm_shape_forwards_argument(self):
        from vllm_torchtpu.offload.cpu_tpu import TPUCPUOffloadingSpec
        spec = TPUCPUOffloadingSpec.__new__(TPUCPUOffloadingSpec)
        spec._tpu_handlers = MagicMock()
        spec.prewarm_shape(512)
        spec._tpu_handlers.cpu_to_gpu_handler.prewarm_shape.\
            assert_called_once_with(512)


class TestTransferDataclass(unittest.TestCase):
    """Transfer's chunk-state fields and error-capture pattern."""

    def test_defaults(self):
        from vllm_torchtpu.offload.cpu_tpu import Transfer
        t = Transfer(job_id=1, num_bytes=100, n=10, dma_done=None)
        self.assertIsNone(t.host_buffer)
        self.assertIsNone(t.device_buffer)
        self.assertIsNone(t.dst_ids)
        self.assertIsNone(t.src_ids)
        self.assertEqual(t.chunks_total, 0)
        self.assertEqual(t.chunks_done, 0)
        self.assertIsNone(t.dst_ids_i32)
        self.assertIsNone(t.error)

    def test_error_capture_pattern(self):
        """The wrapper pattern in _h2d_task / _d2h_task sets t.error
        before re-raising so get_finished can report success=False."""
        from vllm_torchtpu.offload.cpu_tpu import Transfer
        t = Transfer(job_id=1, num_bytes=100, n=10, dma_done=None)
        try:
            try:
                raise RuntimeError("simulated DMA failure")
            except BaseException as e:
                t.error = e
                raise
        except RuntimeError:
            pass
        self.assertIsNotNone(t.error)
        self.assertIsInstance(t.error, RuntimeError)
        self.assertEqual(str(t.error), "simulated DMA failure")


class TestChunkingMath(unittest.TestCase):
    """transfer_async H2D branch computes
    `chunks_total = max(1, ceil(len(src_ids) / chunk_size))`. Exercise
    boundary cases at the arithmetic level without spinning up a real
    handler (which would need a TPU device for the staging buffer)."""

    @staticmethod
    def _chunks(n_ids: int, chunk_size: int) -> int:
        return max(1, (n_ids + chunk_size - 1) // chunk_size)

    def test_single_chunk_when_fits(self):
        for n in (0, 1, 100, 2047, 2048):
            self.assertEqual(self._chunks(n, 2048), 1, f"n={n}")

    def test_multi_chunk_when_oversized(self):
        self.assertEqual(self._chunks(2049, 2048), 2)
        self.assertEqual(self._chunks(4096, 2048), 2)
        self.assertEqual(self._chunks(4097, 2048), 3)
        self.assertEqual(self._chunks(5120, 2048), 3)

    def test_smaller_pool_more_chunks(self):
        self.assertEqual(self._chunks(2048, 512), 4)
        self.assertEqual(self._chunks(2049, 512), 5)


class TestMultiChunkH2D(unittest.TestCase):
    """Chunked H2D loads must progress chunk-by-chunk through the
    `get_finished` / `flush_pending_scatters` cycle even before the
    owning request enters `scatter_now_req_ids`.

    The owning request is in WAITING_FOR_REMOTE_KVS until the LAST
    chunk's `finished_recving` is reported. That means the req_id is
    *not* in `scheduler_output.num_scheduled_tokens` (the source of
    `scatter_now_req_ids`) for any intermediate chunk. So
    `flush_pending_scatters` MUST treat intermediate chunks as eligible
    for dispatch independently of the scheduler-gate set — otherwise no
    chunked transfer can ever complete.

    The test drives the handler's chunk FSM directly without TPU
    dependencies: `__init__` is bypassed (it would allocate the
    device-staging buffer via `torch.zeros`) and `_start_h2d_dma` is
    stubbed to immediately signal `dma_done`.
    """

    def _make_handler(self):
        """Construct a barely-initialized H2D handler — just enough state
        for `get_finished` and `flush_pending_scatters` to run."""
        import threading
        from collections import deque

        from vllm_torchtpu.offload.cpu_tpu import \
            SingleDirectionOffloadingHandler

        h = SingleDirectionOffloadingHandler.__new__(
            SingleDirectionOffloadingHandler)
        h.tpu_to_cpu = False
        h.src_tensors = []
        h.dst_tensors = []
        h._transfer_map = {}
        h._transfers = deque()
        h._req_id_by_job_id = {}
        h._h2d_buffer_free = threading.Event()
        h._h2d_buffer_free.set()
        h._pending_scatters = []
        h._dma_worker = MagicMock()
        h.transfer_type = ("CPU", "GPU")
        h.total_block_size_in_bytes = 1
        # Record chunks the handler dispatches.
        h._dispatched_chunks: list[int] = []

        def fake_start(t):
            h._dispatched_chunks.append(t.chunks_done)
            ev = threading.Event()
            ev.set()
            t.dma_done = ev
            t.device_buffer = [MagicMock()]
            t.dst_ids_i32 = MagicMock()
            t.dst_ids_i32.to = lambda dtype: MagicMock()

        h._start_h2d_dma = fake_start
        return h

    def _build_chunked_transfer(self, n_ids: int, req_id: str = "req-A"):
        import numpy as np

        from vllm_torchtpu.offload.cpu_tpu import Transfer
        return Transfer(
            job_id=1,
            num_bytes=0,
            n=n_ids,
            dma_done=None,
            src_ids=np.arange(n_ids, dtype=np.int64),
            dst_ids=np.arange(n_ids, dtype=np.int64),
            chunks_total=4,  # 4 chunks of 2 ids each
            chunks_done=0,
            req_id=req_id,
        )

    def test_intermediate_chunks_dispatch_without_scheduler_gate(self):
        """Each intermediate chunk must dispatch before `finished_recving`
        is reported (otherwise the owning request never gets promoted and
        the LAST chunk never lands). Only the LAST chunk is subject to
        `scatter_now_req_ids`.

        Contract: starting from chunk 0 in flight, looping
        `get_finished` + `flush_pending_scatters(empty_set)` must
        eventually dispatch every chunk and report `finished_recving` on
        the last one."""
        h = self._make_handler()
        t = self._build_chunked_transfer(n_ids=8)
        h._transfer_map[t.job_id] = t
        h._transfers.append(t)
        h._start_h2d_dma(t)  # chunk 0 in flight

        empty_scatter_now: set[str] = set()
        for _ in range(t.chunks_total + 2):
            results = h.get_finished()
            h.flush_pending_scatters(empty_scatter_now)
            if results:
                self.assertEqual(results[0].job_id, t.job_id)
                self.assertEqual(t.chunks_done, t.chunks_total)
                self.assertEqual(h._dispatched_chunks,
                                 list(range(t.chunks_total)))
                return

        self.fail(
            "Chunked transfer stalled: dispatched only chunks "
            f"{h._dispatched_chunks} of {t.chunks_total} total; "
            f"{len(h._pending_scatters)} chunks left in `_pending_scatters`. "
            "Intermediate chunks were not eligible for flush despite the "
            "owning req_id correctly being absent from `scatter_now_req_ids` "
            "(the request is still in WAITING_FOR_REMOTE_KVS).")


class TestExpandHybridPoolBlockIds(unittest.TestCase):
    """Grouped row mapping for hybrid unified-block-pool transfers.

    GPU block IDs pass through unchanged (pool rows ARE scheduler blocks);
    CPU rows are derived per group from sequentially-consumed CPU blocks,
    honoring the partial-first-block skip (block_indices[g] % factor)."""

    @staticmethod
    def _expand(gpu, group_sizes, block_indices, cpu, factor):
        import numpy as np

        from vllm_torchtpu.offload.cpu_tpu import expand_hybrid_pool_block_ids
        gpu_rows, cpu_rows = expand_hybrid_pool_block_ids(
            np.array(gpu, dtype=np.int64),
            group_sizes,
            block_indices,
            np.array(cpu, dtype=np.int64),
            factor,
        )
        return list(gpu_rows), list(cpu_rows)

    def test_factor_1_attention_plus_mamba_tail(self):
        # Qwen3.5-like load: 3 attention blocks + 1 mamba boundary-state
        # block (mamba block table null-prefixed, so its segment starts at
        # logical position 3).
        gpu_rows, cpu_rows = self._expand(
            gpu=[5, 9, 2, 40],
            group_sizes=[3, 1],
            block_indices=[0, 3],
            cpu=[10, 11, 12, 20],
            factor=1,
        )
        self.assertEqual(gpu_rows, [5, 9, 2, 40])
        self.assertEqual(cpu_rows, [10, 11, 12, 20])

    def test_factor_2_partial_first_offloaded_block(self):
        # Resume mid-offloaded-block: skip=1 row of the first CPU block.
        _, cpu_rows = self._expand(
            gpu=[7, 8, 9],
            group_sizes=[3],
            block_indices=[1],
            cpu=[5, 6],
            factor=2,
        )
        self.assertEqual(cpu_rows, [5 * 2 + 1, 6 * 2 + 0, 6 * 2 + 1])

    def test_factor_2_null_prefixed_mamba_segment(self):
        _, cpu_rows = self._expand(
            gpu=[1, 2, 3, 4, 77],
            group_sizes=[4, 1],
            block_indices=[0, 3],
            cpu=[7, 8, 9],
            factor=2,
        )
        self.assertEqual(cpu_rows, [14, 15, 16, 17, 9 * 2 + 1])

    def test_stored_key_gap_stays_factor_aligned(self):
        # Store with an already-stored middle key: the gpu segment covers
        # keys K0 (2 rows) and K2 (2 rows) with K1 skipped; K1 consumes
        # neither gpu rows nor a CPU block, so alignment is preserved.
        _, cpu_rows = self._expand(
            gpu=[11, 12, 15, 16],
            group_sizes=[4],
            block_indices=[0],
            cpu=[3, 9],
            factor=2,
        )
        self.assertEqual(cpu_rows, [6, 7, 18, 19])

    def test_empty_group_consumes_nothing(self):
        _, cpu_rows = self._expand(
            gpu=[1, 2],
            group_sizes=[2, 0],
            block_indices=[0, 0],
            cpu=[3, 4],
            factor=1,
        )
        self.assertEqual(cpu_rows, [3, 4])

    def test_unused_cpu_blocks_assert(self):
        with self.assertRaises(AssertionError):
            self._expand(gpu=[1, 2],
                         group_sizes=[2],
                         block_indices=[0],
                         cpu=[3, 4, 5],
                         factor=1)


class TestHybridTransferAsyncMapping(unittest.TestCase):
    """`transfer_async` in hybrid pool mode must route grouped GPU specs
    through expand_hybrid_pool_block_ids. Uses the same __new__-based
    handler construction as TestMultiChunkH2D to avoid TPU deps."""

    def _make_h2d_handler(self, cpu_factor=1, num_groups=2):
        import threading
        from collections import deque

        from vllm_torchtpu.offload.cpu_tpu import \
            SingleDirectionOffloadingHandler

        h = SingleDirectionOffloadingHandler.__new__(
            SingleDirectionOffloadingHandler)
        h.tpu_to_cpu = False
        h.src_tensors = []
        h.dst_tensors = []
        h.src_block_size_factor = cpu_factor
        h.dst_block_size_factor = 1
        h._hybrid_num_groups = num_groups
        h._transfer_map = {}
        h._transfers = deque()
        h._req_id_by_job_id = {}
        h._h2d_buffer_free = threading.Event()
        h._h2d_buffer_free.set()
        h._pending_scatters = []
        h._dma_worker = MagicMock()
        h.transfer_type = ("CPU", "GPU")
        h.total_block_size_in_bytes = 1
        h._h2d_max_padded = 1024
        h._start_h2d_dma = lambda t: None
        return h

    def test_h2d_load_maps_grouped_gpu_spec(self):
        from vllm.v1.kv_offload.base import GPULoadStoreSpec
        from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec

        h = self._make_h2d_handler(cpu_factor=1, num_groups=2)
        src = CPULoadStoreSpec([10, 11, 12, 20])
        dst = GPULoadStoreSpec([5, 9, 2, 40],
                               group_sizes=[3, 1],
                               block_indices=[0, 3])
        self.assertTrue(h.transfer_async(1, (src, dst)))
        t = h._transfer_map[1]
        self.assertEqual(list(t.src_ids), [10, 11, 12, 20])
        self.assertEqual(list(t.dst_ids), [5, 9, 2, 40])

    def test_h2d_load_with_factor_2_partial_block(self):
        from vllm.v1.kv_offload.base import GPULoadStoreSpec
        from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec

        h = self._make_h2d_handler(cpu_factor=2, num_groups=1)
        src = CPULoadStoreSpec([5, 6])
        dst = GPULoadStoreSpec([7, 8, 9], group_sizes=[3], block_indices=[1])
        self.assertTrue(h.transfer_async(2, (src, dst)))
        t = h._transfer_map[2]
        self.assertEqual(list(t.src_ids), [11, 12, 13])
        self.assertEqual(list(t.dst_ids), [7, 8, 9])

    def test_ungrouped_spec_asserts_in_hybrid_mode(self):
        from vllm.v1.kv_offload.base import BlockIDsLoadStoreSpec
        from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec

        class _FlatGpuSpec(BlockIDsLoadStoreSpec):

            @staticmethod
            def medium() -> str:
                return "GPU"

        h = self._make_h2d_handler(cpu_factor=1, num_groups=2)
        src = CPULoadStoreSpec([0])
        dst = _FlatGpuSpec([1])
        with self.assertRaises(AssertionError):
            h.transfer_async(3, (src, dst))


class TestEstimateKvConnectorHbmReserve(unittest.TestCase):
    """utils.estimate_kv_connector_hbm_reserve dynamically resolves
    the configured spec via kv_connector_extra_config without hardcoding
    connector or spec class names. Shared by tpu_worker budgeting and the
    runner's num_gpu_blocks_override paths."""

    @staticmethod
    def _reserve(kv_transfer_config):
        from vllm_torchtpu import utils
        vllm_config = MagicMock()
        vllm_config.kv_transfer_config = kv_transfer_config
        return utils.estimate_kv_connector_hbm_reserve(vllm_config)

    def test_no_kv_connector_returns_zero(self):
        self.assertEqual(self._reserve(kv_transfer_config=None), 0)

    def test_missing_spec_metadata_returns_zero(self):
        """A kv_connector without spec_name / spec_module_path keys
        (e.g., a P/D disagg connector that doesn't use OffloadingConnector's
        factory mechanism) gets a zero reserve — its budget must not be
        shrunk."""
        kv_tc = MagicMock()
        kv_tc.kv_connector_extra_config = {}
        self.assertEqual(self._reserve(kv_transfer_config=kv_tc), 0)

    def test_unimportable_spec_module_returns_zero(self):
        kv_tc = MagicMock()
        kv_tc.kv_connector_extra_config = {
            "spec_name": "Nonexistent",
            "spec_module_path": "does.not.exist.module",
        }
        self.assertEqual(self._reserve(kv_transfer_config=kv_tc), 0)

    def test_spec_without_estimate_method_returns_zero(self):
        """Even if the spec class resolves, returning 0 when it doesn't
        advertise an HBM reserve preserves the duck-typing contract."""
        kv_tc = MagicMock()
        # Point at this very test module — has no estimate_hbm_reserve_bytes
        # classmethod, so duck-typing falls through to 0.
        kv_tc.kv_connector_extra_config = {
            "spec_name": "TestEstimateKvConnectorHbmReserve",
            "spec_module_path": "tests.offload.test_cpu_tpu_offload",
        }
        self.assertEqual(self._reserve(kv_transfer_config=kv_tc), 0)


class TestRaidenOffloadingHandlerTransferAsync(unittest.TestCase):
    """`_RaidenOffloadingHandler.transfer_async` must call the raiden
    wheel's public KVCacheManager wrapper methods, which are lowercase
    `d2h`/`h2d` — the capitalized `D2h`/`H2d` spellings only exist on the
    private `_impl` object and die with AttributeError on the wrapper."""

    def _make_handler(self, tpu_to_cpu: bool):
        from vllm_torchtpu.offload.cpu_tpu import _RaidenOffloadingHandler

        # spec-limited: any call other than d2h/h2d (e.g. the legacy D2h)
        # raises AttributeError, pinning the public wrapper API.
        mgr = MagicMock(spec=["d2h", "h2d"])
        handler = _RaidenOffloadingHandler(mgr,
                                           tpu_to_cpu=tpu_to_cpu,
                                           src_block_size_factor=1,
                                           dst_block_size_factor=1,
                                           bytes_per_kernel_block=64)
        return handler, mgr

    @staticmethod
    def _spec(block_ids):
        from vllm.v1.kv_offload.base import BlockIDsLoadStoreSpec

        class _TestSpec(BlockIDsLoadStoreSpec):

            @staticmethod
            def medium() -> str:
                return "TEST"

        return _TestSpec(block_ids)

    def test_d2h_store_uses_lowercase_wrapper_method(self):
        handler, mgr = self._make_handler(tpu_to_cpu=True)
        fut = MagicMock()
        mgr.d2h.return_value = fut

        with patch("vllm_torchtpu.offload.cpu_tpu._tpu_sync") as sync:
            self.assertTrue(
                handler.transfer_async(
                    7, (self._spec([3, 5]), self._spec([1, 2]))))

        # The store must barrier on the forward step's in-place KV write
        # before raiden reads the device blocks.
        sync.assert_called_once_with(wait=True)
        mgr.d2h.assert_called_once_with([3, 5], [1, 2], [1, 1])
        mgr.h2d.assert_not_called()
        self.assertEqual(handler._pending[7], (fut, 2 * 64))

    def test_h2d_load_uses_lowercase_wrapper_method(self):
        handler, mgr = self._make_handler(tpu_to_cpu=False)
        fut = MagicMock()
        mgr.h2d.return_value = fut

        self.assertTrue(
            handler.transfer_async(8,
                                   (self._spec([1, 2]), self._spec([3, 5]))))

        mgr.h2d.assert_called_once_with([1, 2], [3, 5], [1, 1])
        mgr.d2h.assert_not_called()
        self.assertEqual(handler._pending[8], (fut, 2 * 64))


if __name__ == "__main__":
    unittest.main()
