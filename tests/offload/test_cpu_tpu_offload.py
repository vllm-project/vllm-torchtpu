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
  TPU offloading worker when it is populated; no-op before.
- `Transfer` dataclass defaults and the error-capture wrapping pattern
  used by `_h2d_task` / `_d2h_task`.
- `transfer_async` H2D chunking math: `chunks_total = ceil(len(src_ids)
  / chunk_size)` at the boundary cases.
- `wait()` flush semantics: D2H blocks on dma_done; H2D drains the chunk
  pipeline (force-dispatching parked scatters past the scatter-now gate,
  re-buffering TransferResults for exactly-once reporting via
  `get_finished`), unparks orphaned last-chunk scatters of aborted
  requests, and times out instead of hanging on a stalled DMA.
- `utils.estimate_kv_connector_hbm_reserve` dynamic spec resolution
  (no kv_connector / missing metadata / unimportable module → 0).
- `expand_hybrid_pool_block_ids` grouped row mapping for the unified
  block pool (attention + mamba segments, partial first offloaded block,
  null-prefixed mamba segments, empty groups).
- `_expand_transfer_ids` non-hybrid flat path: the partial-first-block
  skip must not shift the compactly-written source expansion (regression).
- `TPUCPUOffloadingSpec.__init__` consistency assert: model_config.is_hybrid
  must agree with MambaSpec presence in kv_cache_groups.

Heavy dependencies (real Pallas kernels, torch_tpu DMA, vllm config
plumbing) are mocked at construction time so the tests run on a host
without a TPU.
"""
import os
import time
import unittest
from types import SimpleNamespace
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

    def test_hybrid_uses_attention_layer_count(self):
        """Hybrid (unified block pool) staging is one buffer per attention
        layer position at the pool page. The pool cap is NOT specialized
        for hybrid — launch configs set KV_H2D_POOL_MAX_BLOCKS lower
        (recipes use 512) since hybrid pool pages are MiB-scale."""
        from vllm_torchtpu.offload.cpu_tpu import TPUCPUOffloadingSpec
        num_layers, num_attn = 60, 15
        dense = TPUCPUOffloadingSpec.estimate_hbm_reserve_bytes(
            self._make_vllm_config(num_layers=num_layers))
        hybrid = TPUCPUOffloadingSpec.estimate_hbm_reserve_bytes(
            self._make_vllm_config(num_layers=num_layers,
                                   is_hybrid=True,
                                   num_attn_layers=num_attn))
        # Same pool cap: only the layer count differs (60 vs 15 buffers).
        self.assertEqual(dense * num_attn, hybrid * num_layers)

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
    H2D handler. Before get_worker() runs (_tpu_worker is None) they
    are no-ops so the runner's collective_rpc doesn't crash."""

    def test_prewarm_shapes_none_when_handlers_not_built(self):
        from vllm_torchtpu.offload.cpu_tpu import TPUCPUOffloadingSpec
        spec = TPUCPUOffloadingSpec.__new__(TPUCPUOffloadingSpec)
        spec._tpu_worker = None
        self.assertEqual(spec.prewarm_shapes, [])
        spec.prewarm_shape(2048)  # must not raise

    def test_prewarm_shapes_delegates_to_h2d_handler(self):
        from vllm_torchtpu.offload.cpu_tpu import TPUCPUOffloadingSpec
        spec = TPUCPUOffloadingSpec.__new__(TPUCPUOffloadingSpec)
        spec._tpu_worker = MagicMock()
        spec._tpu_worker.prewarm_shapes = [1, 2, 4, 8]
        self.assertEqual(spec.prewarm_shapes, [1, 2, 4, 8])

    def test_prewarm_shape_forwards_argument(self):
        from vllm_torchtpu.offload.cpu_tpu import TPUCPUOffloadingSpec
        spec = TPUCPUOffloadingSpec.__new__(TPUCPUOffloadingSpec)
        spec._tpu_worker = MagicMock()
        spec.prewarm_shape(512)
        spec._tpu_worker.prewarm_shape.assert_called_once_with(512)


class TestTPUCPUOffloadingWorker(unittest.TestCase):

    def setUp(self):
        from vllm_torchtpu.offload.cpu_tpu import TPUCPUOffloadingWorker
        self.handlers = MagicMock()
        self.worker = TPUCPUOffloadingWorker(self.handlers)

    def test_submit_routes_by_direction(self):
        src_gpu = MagicMock()
        dst_cpu = MagicMock()
        src_cpu = MagicMock()
        dst_gpu = MagicMock()
        self.handlers.gpu_to_cpu_handler.transfer_async.return_value = True
        self.handlers.cpu_to_gpu_handler.transfer_async.return_value = True

        self.assertTrue(self.worker.submit_store(1, src_gpu, dst_cpu))
        self.assertTrue(self.worker.submit_load(2, src_cpu, dst_gpu))

        self.handlers.gpu_to_cpu_handler.transfer_async.assert_called_once_with(
            1, (src_gpu, dst_cpu))
        self.handlers.cpu_to_gpu_handler.transfer_async.assert_called_once_with(
            2, (src_cpu, dst_gpu))

    def test_completion_wait_and_shutdown_cover_both_directions(self):
        store_result = MagicMock()
        load_result = MagicMock()
        self.handlers.gpu_to_cpu_handler.get_finished.return_value = [
            store_result
        ]
        self.handlers.cpu_to_gpu_handler.get_finished.return_value = [
            load_result
        ]

        self.assertEqual(self.worker.get_finished(),
                         [store_result, load_result])
        self.worker.wait({1, 2})
        self.worker.shutdown()

        self.handlers.gpu_to_cpu_handler.wait.assert_called_once_with({1, 2})
        self.handlers.cpu_to_gpu_handler.wait.assert_called_once_with({1, 2})
        self.handlers.gpu_to_cpu_handler.shutdown.assert_called_once_with()
        self.handlers.cpu_to_gpu_handler.shutdown.assert_called_once_with()

    def test_spec_returns_cached_worker(self):
        from vllm.v1.kv_offload.base import OffloadingWorker

        from vllm_torchtpu.offload.cpu_tpu import TPUCPUOffloadingSpec
        spec = TPUCPUOffloadingSpec.__new__(TPUCPUOffloadingSpec)
        spec._tpu_worker = self.worker

        self.assertIsInstance(spec.get_worker(MagicMock()), OffloadingWorker)
        self.assertIs(spec.get_worker(MagicMock()), self.worker)


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
        h._flushed_results = []
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


class TestWaitFlushDrain(unittest.TestCase):
    """`wait()` is the connector's flush primitive, and the scheduler
    legitimately names H2D *load* jobs in it (it flushes ALL pending jobs
    once every tracked request has finished, and on reset_cache). There
    may be no future engine step to run `get_finished` /
    `flush_pending_scatters` afterwards, so the H2D handler must drain
    its chunk pipeline inside `wait()` itself (regression: this used to
    be an `AssertionError` crash on the H2D path).

    Same TPU-free construction pattern as TestMultiChunkH2D: `__init__`
    is bypassed and `_start_h2d_dma` is stubbed to signal `dma_done`
    immediately. The stub also clears `_h2d_buffer_free`, mirroring the
    real worker task, so the tests verify the drain releases the buffer.
    """

    def _make_h2d_handler(self):
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
        h._flushed_results = []
        h._dma_worker = MagicMock()
        h.transfer_type = ("CPU", "GPU")
        h.total_block_size_in_bytes = 1
        h._dispatched_chunks: list[int] = []

        def fake_start(t):
            h._dispatched_chunks.append(t.chunks_done)
            ev = threading.Event()
            ev.set()
            t.dma_done = ev
            # Mirror the real _h2d_task: the DMA claims the buffer; only
            # a scatter dispatch (flush_pending_scatters) releases it.
            h._h2d_buffer_free.clear()
            t.device_buffer = [MagicMock()]
            t.dst_ids_i32 = MagicMock()
            t.dst_ids_i32.to = lambda dtype: MagicMock()

        h._start_h2d_dma = fake_start
        return h

    @staticmethod
    def _make_transfer(job_id: int, chunks_total: int, req_id: str = "req-A"):
        import numpy as np

        from vllm_torchtpu.offload.cpu_tpu import Transfer
        n_ids = 2 * chunks_total
        return Transfer(
            job_id=job_id,
            num_bytes=0,
            n=n_ids,
            dma_done=None,
            src_ids=np.arange(n_ids, dtype=np.int64),
            dst_ids=np.arange(n_ids, dtype=np.int64),
            chunks_total=chunks_total,
            chunks_done=0,
            req_id=req_id,
        )

    def _submit(self, h, t):
        h._transfer_map[t.job_id] = t
        h._transfers.append(t)
        h._start_h2d_dma(t)

    def test_h2d_wait_drains_chunked_transfer(self):
        """wait() must complete a multi-chunk load without any engine
        step's get_finished/flush cycle, and the TransferResult must
        still be reported by the NEXT get_finished — exactly once."""
        h = self._make_h2d_handler()
        t = self._make_transfer(job_id=1, chunks_total=4)
        self._submit(h, t)

        h.wait({1})

        self.assertNotIn(1, h._transfer_map)
        self.assertEqual(h._dispatched_chunks, list(range(t.chunks_total)))
        self.assertEqual(h._pending_scatters, [])
        self.assertTrue(h._h2d_buffer_free.is_set())

        results = h.get_finished()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].job_id, 1)
        self.assertTrue(results[0].success)
        # Exactly once: the re-buffered result must not be re-emitted.
        self.assertEqual(h.get_finished(), [])

    def test_h2d_wait_unparks_orphaned_last_chunk_scatter(self):
        """A last-chunk scatter parks in the same get_finished() that
        pops its transfer from `_transfer_map`. If the owning request is
        then aborted (never promoted, so never in `scatter_now_req_ids`),
        the parked scatter holds `_h2d_buffer_free` with no map entry
        left — wait() must still force-dispatch it even though none of
        the named jobs are in `_transfer_map` anymore."""
        h = self._make_h2d_handler()
        t = self._make_transfer(job_id=1, chunks_total=1)
        self._submit(h, t)

        # Engine step: result reported, last-chunk scatter parked behind
        # the scatter-now gate (req never promoted → empty set).
        results = h.get_finished()
        self.assertEqual([r.job_id for r in results], [1])
        h.flush_pending_scatters(set())
        self.assertEqual(len(h._pending_scatters), 1)
        self.assertFalse(h._h2d_buffer_free.is_set())

        h.wait({1})

        self.assertEqual(h._pending_scatters, [])
        self.assertTrue(h._h2d_buffer_free.is_set())
        # The result was already consumed above; wait() must not
        # resurrect it.
        self.assertEqual(h.get_finished(), [])

    def test_h2d_wait_raises_on_stalled_dma(self):
        """A DMA that has not completed by the timeout raises instead of
        letting the caller overwrite the source blocks."""
        import threading

        h = self._make_h2d_handler()
        t = self._make_transfer(job_id=1, chunks_total=1)
        h._transfer_map[1] = t
        h._transfers.append(t)
        t.dma_done = threading.Event()  # never set — stalled DMA

        with patch(
                "vllm_torchtpu.offload.cpu_tpu._RAIDEN_OFFLOAD_WAIT_TIMEOUT_S",
                0.05):
            with self.assertRaisesRegex(RuntimeError,
                                        "timed out.*jobs=\\[1\\]"):
                h.wait({1})

    def test_h2d_wait_ignores_unknown_jobs(self):
        h = self._make_h2d_handler()
        h.wait({999})  # must not raise or hang
        self.assertTrue(h._h2d_buffer_free.is_set())

    def test_d2h_wait_blocks_until_dma_done(self):
        """D2H wait blocks on each named transfer's dma_done (set only
        after the host-pool scatter, so set ⇒ fully landed) and skips
        job ids it doesn't know about."""
        import threading

        from vllm_torchtpu.offload.cpu_tpu import (
            SingleDirectionOffloadingHandler, Transfer)

        h = SingleDirectionOffloadingHandler.__new__(
            SingleDirectionOffloadingHandler)
        h.tpu_to_cpu = True
        h._transfer_map = {}

        ev = threading.Event()
        t = Transfer(job_id=1, num_bytes=0, n=1, dma_done=ev)
        h._transfer_map[1] = t

        # Signal completion from a worker-like thread shortly after wait
        # starts blocking.
        threading.Timer(0.02, ev.set).start()
        start = time.monotonic()
        h.wait({1, 999})  # 999 unknown → skipped
        self.assertGreaterEqual(time.monotonic() - start, 0.015)
        self.assertTrue(ev.is_set())

    def test_force_flush_dispatches_real_scatter_data(self):
        """force=True must dispatch a gated last-chunk scatter (the
        non-force flush with an empty scatter-now set keeps it parked),
        actually writing the staged data into the KV cache rows."""
        from vllm_torchtpu.offload.cpu_tpu import _PendingScatter

        h = self._make_h2d_handler()
        kv_cache = torch.zeros(8, 4)
        h.dst_tensors = [kv_cache]

        t = self._make_transfer(job_id=1, chunks_total=1, req_id="req-A")
        staged = torch.full((2, 4), 5.0)
        h._pending_scatters.append(
            _PendingScatter(
                transfer=t,
                device_buffer=[staged],
                dst_ids_i32=torch.tensor([2, 3], dtype=torch.int32),
                is_last_chunk=True,
            ))
        h._h2d_buffer_free.clear()

        h.flush_pending_scatters(set())  # gated: req-A not promoted
        self.assertEqual(len(h._pending_scatters), 1)
        self.assertFalse(h._h2d_buffer_free.is_set())
        self.assertTrue(torch.all(kv_cache == 0))

        h.flush_pending_scatters(set(), force=True)
        self.assertEqual(h._pending_scatters, [])
        self.assertTrue(h._h2d_buffer_free.is_set())
        self.assertTrue(torch.equal(kv_cache[2:4], staged))
        self.assertTrue(torch.all(kv_cache[:2] == 0))
        self.assertTrue(torch.all(kv_cache[4:] == 0))


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

    def test_too_few_cpu_blocks_assert(self):
        with self.assertRaises(AssertionError):
            self._expand(gpu=[1, 2],
                         group_sizes=[2],
                         block_indices=[0],
                         cpu=[3],
                         factor=1)


class TestExpandTransferIdsFlat(unittest.TestCase):
    """Non-hybrid `_expand_transfer_ids` flat path.

    `expand_block_ids` writes compactly with the skip already applied, so
    the source expansion must be consumed as-is — slicing off the leading
    `src_skip` entries again would drop valid rows and read uninitialized
    tail memory (regression test)."""

    @staticmethod
    def _expand(src, dst, tpu_to_cpu, src_factor, dst_factor):
        from vllm_torchtpu.offload.cpu_tpu import _expand_transfer_ids
        src_ids, dst_ids = _expand_transfer_ids((src, dst), tpu_to_cpu,
                                                src_factor, dst_factor, None)
        return list(src_ids), list(dst_ids)

    def test_h2d_partial_first_cpu_block(self):
        from vllm.v1.kv_offload.base import GPULoadStoreSpec
        from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec

        # 3 GPU blocks from 2 CPU blocks at factor 2: skip row 0 of CPU
        # block 5, then rows 11, 12, 13 map onto GPU blocks 7, 8, 9.
        src_ids, dst_ids = self._expand(
            CPULoadStoreSpec([5, 6]),
            GPULoadStoreSpec([7, 8, 9], group_sizes=[3], block_indices=[1]),
            tpu_to_cpu=False,
            src_factor=2,
            dst_factor=1,
        )
        self.assertEqual(src_ids, [11, 12, 13])
        self.assertEqual(dst_ids, [7, 8, 9])

    def test_d2h_full_blocks(self):
        from vllm.v1.kv_offload.base import GPULoadStoreSpec
        from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec

        src_ids, dst_ids = self._expand(
            GPULoadStoreSpec([1, 2, 3, 4], group_sizes=[4], block_indices=[0]),
            CPULoadStoreSpec([3, 7]),
            tpu_to_cpu=True,
            src_factor=1,
            dst_factor=2,
        )
        self.assertEqual(src_ids, [1, 2, 3, 4])
        self.assertEqual(dst_ids, [6, 7, 14, 15])

    def test_kernel_granular_factors_scale_manager_mapping(self):
        from vllm.v1.kv_offload.base import GPULoadStoreSpec
        from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec

        # Raiden addresses physical kernel rows, so both factors carry the
        # same kernel-block multiplier (4 here). The mapping must be the
        # manager-block mapping of test_h2d_partial_first_cpu_block with
        # every row expanded into its 4 kernel rows — not a different
        # skip/row count.
        src_ids, dst_ids = self._expand(
            CPULoadStoreSpec([5, 6]),
            GPULoadStoreSpec([7, 8, 9], group_sizes=[3], block_indices=[1]),
            tpu_to_cpu=False,
            src_factor=8,
            dst_factor=4,
        )
        self.assertEqual(src_ids,
                         [r * 4 + i for r in (11, 12, 13) for i in range(4)])
        self.assertEqual(dst_ids,
                         [r * 4 + i for r in (7, 8, 9) for i in range(4)])


class TestExpandTransferIdsHybridPhysical(unittest.TestCase):
    """Raiden indexes physical kernel blocks while vLLM's grouped hybrid
    specs contain scheduler-block IDs. Each mapped scheduler pair must expand
    to matching contiguous physical rows on both sides."""

    def test_d2h_expands_scheduler_pairs_to_kernel_rows(self):
        from vllm.v1.kv_offload.base import GPULoadStoreSpec
        from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec

        from vllm_torchtpu.offload.cpu_tpu import _expand_transfer_ids

        src_ids, dst_ids = _expand_transfer_ids(
            (
                GPULoadStoreSpec([5, 9], group_sizes=[2], block_indices=[0]),
                CPULoadStoreSpec([10, 11]),
            ),
            tpu_to_cpu=True,
            src_block_size_factor=4,
            dst_block_size_factor=4,
            hybrid_num_groups=1,
        )
        self.assertEqual(list(src_ids), [20, 21, 22, 23, 36, 37, 38, 39])
        self.assertEqual(list(dst_ids), [40, 41, 42, 43, 44, 45, 46, 47])


class TestTorchPathManagerBlockView(unittest.TestCase):
    """A manager block may span several physical kernel blocks (e.g. a
    4096-token pool page over 256-token CUSTOM kernel blocks). The torch
    handlers index rows with scheduler-block IDs, so their views must keep
    the manager-block leading dimension and fold the kernel blocks into the
    trailing dims — and both sides must be passed matching block factors."""

    def test_keeps_manager_rows_over_physical_storage(self):
        from vllm.v1.kv_offload.base import (CanonicalKVCaches,
                                             CanonicalKVCacheTensor)

        from vllm_torchtpu.offload.cpu_tpu import CpuTpuOffloadingHandlers

        # 8 kernel blocks of 2 int8 elements = 2 manager blocks of 4 kernel
        # blocks; vllm canonicalizes that storage to (2, 8) int8.
        base = torch.empty((8, 2), dtype=torch.int8)
        canonical = torch.empty(0, dtype=torch.int8).set_(
            base.untyped_storage()).view(2, 8)
        caches = CanonicalKVCaches(
            tensors=[
                CanonicalKVCacheTensor(tensor=canonical, page_size_bytes=8)
            ],
            group_data_refs=[[]],
        )

        with patch("vllm_torchtpu.offload.cpu_tpu._USE_RAIDEN_OFFLOAD",
                   False), patch(
                       "vllm_torchtpu.offload.cpu_tpu."
                       "SingleDirectionOffloadingHandler") as handler_cls:
            CpuTpuOffloadingHandlers(
                gpu_block_size=4,
                cpu_block_size=8,
                num_cpu_blocks=2,
                kv_caches=caches,
                kernel_block_size=1,
                kv_dtype=torch.int8,
                per_block_shape=(2, ),
            )

        d2h_kwargs = handler_cls.call_args_list[0].kwargs
        rebuilt = d2h_kwargs["src_tensors"][0]
        self.assertEqual(tuple(rebuilt.shape), (2, 4, 2))
        self.assertEqual(rebuilt.untyped_storage().data_ptr(),
                         base.untyped_storage().data_ptr())
        # CPU pool: 2 offloaded blocks of 2 manager blocks each.
        self.assertEqual(tuple(d2h_kwargs["dst_tensors"][0].shape), (4, 4, 2))
        # Both rows are manager blocks, so the device factor is 1 and the CPU
        # factor is the offloaded/scheduler block ratio.
        self.assertEqual(d2h_kwargs["src_block_size_factor"], 1)
        self.assertEqual(d2h_kwargs["dst_block_size_factor"], 2)


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
        h._flushed_results = []
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


class TestHybridSpecConsistencyAssert(unittest.TestCase):
    """TPUCPUOffloadingSpec.__init__ cross-checks model_config.is_hybrid
    against MambaSpec presence in kv_cache_groups: the hybrid handler
    mapping and the HBM reserve estimate key on the former, the transfer
    specs on the latter, so a disagreement must fail at construction."""

    @staticmethod
    def _make_spec(is_hybrid: bool, with_mamba_group: bool):
        from vllm.v1.kv_cache_interface import MambaSpec
        from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec

        from vllm_torchtpu.offload.cpu_tpu import TPUCPUOffloadingSpec

        vllm_config = MagicMock()
        vllm_config.model_config.is_hybrid = is_hybrid
        kv_cache_config = MagicMock()
        attn_group = MagicMock()
        attn_group.kv_cache_spec = object()
        groups = [attn_group]
        if with_mamba_group:
            mamba_group = MagicMock()
            mamba_group.kv_cache_spec = MagicMock(spec=MambaSpec)
            groups.append(mamba_group)
        kv_cache_config.kv_cache_groups = groups

        # Bypass the base spec's block-size / eviction-policy plumbing;
        # only the hybrid consistency check is under test.
        with patch.object(CPUOffloadingSpec, "__init__",
                          lambda self, *a, **k: None):
            offloading_config = MagicMock()
            offloading_config.groups = ()
            offloading_config.cache.blocks_per_chunk = 1
            return TPUCPUOffloadingSpec(offloading_config, vllm_config,
                                        kv_cache_config)

    def test_hybrid_with_mamba_group_ok(self):
        spec = self._make_spec(is_hybrid=True, with_mamba_group=True)
        self.assertTrue(spec.is_hybrid_model)

    def test_dense_without_mamba_group_ok(self):
        spec = self._make_spec(is_hybrid=False, with_mamba_group=False)
        self.assertFalse(spec.is_hybrid_model)

    def test_hybrid_without_mamba_group_asserts(self):
        with self.assertRaises(AssertionError):
            self._make_spec(is_hybrid=True, with_mamba_group=False)

    def test_dense_with_mamba_group_asserts(self):
        with self.assertRaises(AssertionError):
            self._make_spec(is_hybrid=False, with_mamba_group=True)


class TestOffloadingConnectorSpecPatch(unittest.TestCase):
    """The OffloadingConnector ctor patch keeps raw cache metadata across
    vLLM 0.26's normalized OffloadingConfig boundary for the TPU spec."""

    def test_worker_spec_receives_normalized_and_raw_configs(self):
        from vllm.distributed.kv_transfer.kv_connector.v1 import \
            KVConnectorRole
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading import \
            config as offloading_config_mod
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading import \
            worker as offloading_worker_mod
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import \
            OffloadingConnector

        from vllm_torchtpu import _patch_vllm_offloading_connector_spec
        from vllm_torchtpu.offload import cpu_tpu

        _patch_vllm_offloading_connector_spec()

        vllm_config = MagicMock()
        vllm_config.kv_transfer_config.kv_connector_extra_config = {
            "spec_name": "TPUCPUOffloadingSpec",
            "spec_module_path": "vllm_torchtpu.offload.cpu_tpu",
        }
        kv_cache_config = MagicMock()
        offloading_config = MagicMock()
        spec = MagicMock()
        worker = MagicMock()

        with (
                patch.object(offloading_config_mod,
                             "build_offloading_config",
                             return_value=offloading_config) as build,
                patch.object(cpu_tpu,
                             "TPUCPUOffloadingSpec",
                             return_value=spec) as spec_cls,
                patch.object(offloading_worker_mod,
                             "OffloadingConnectorWorker",
                             return_value=worker) as worker_cls,
        ):
            connector = OffloadingConnector(vllm_config,
                                            KVConnectorRole.WORKER,
                                            kv_cache_config)

        build.assert_called_once_with(vllm_config, kv_cache_config)
        spec_cls.assert_called_once_with(offloading_config, vllm_config,
                                         kv_cache_config)
        worker_cls.assert_called_once_with(spec, vllm_config, kv_cache_config)
        self.assertIs(connector.connector_worker, worker)
        self.assertIsNone(connector.connector_scheduler)

    def test_non_tpu_spec_uses_stock_construction(self):
        from vllm.distributed.kv_transfer.kv_connector.v1 import \
            KVConnectorRole
        from vllm.distributed.kv_transfer.kv_connector.v1 import \
            offloading_connector as connector_mod
        from vllm.v1.kv_offload.factory import OffloadingSpecFactory

        from vllm_torchtpu import _patch_vllm_offloading_connector_spec

        _patch_vllm_offloading_connector_spec()

        vllm_config = MagicMock()
        vllm_config.kv_transfer_config.kv_connector_extra_config = {}
        kv_cache_config = MagicMock()

        # The stock construction path resolves these in the
        # offloading_connector module namespace.
        with (
                patch.object(connector_mod,
                             "build_offloading_config",
                             return_value=MagicMock()),
                patch.object(OffloadingSpecFactory,
                             "create_spec",
                             return_value=MagicMock()) as create_spec,
                patch.object(connector_mod,
                             "OffloadingConnectorWorker",
                             return_value=MagicMock()),
        ):
            connector_mod.OffloadingConnector(vllm_config,
                                              KVConnectorRole.WORKER,
                                              kv_cache_config)

        # The stock path routes through the factory; the TPU path never
        # touches it.
        create_spec.assert_called_once()


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


class _FakeTpuEvent:
    """Stand-in for torch.tpu.Event: `ready` controls what query() reports."""

    ready = True

    def __init__(self):
        self.recorded = False
        self.synchronized = False

    def record(self):
        self.recorded = True
        return self

    def query(self):
        return self.ready

    def synchronize(self):
        self.synchronized = True
        self.ready = True


def _patch_tpu_event():
    # CPU-only torch has no `tpu` namespace at all; give it one for the test.
    if not hasattr(torch, "tpu"):
        return patch.object(torch,
                            "tpu",
                            SimpleNamespace(Event=_FakeTpuEvent),
                            create=True)
    return patch.object(torch.tpu, "Event", _FakeTpuEvent, create=True)


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

        _FakeTpuEvent.ready = True
        with patch("vllm_torchtpu.offload.cpu_tpu.synchronize_tensors"
                   ) as sync, _patch_tpu_event():
            self.assertTrue(
                handler.transfer_async(
                    7, (self._spec([3, 5]), self._spec([1, 2]))))

        # No device wait on the worker thread: the store is gated on a
        # device-event snapshot, and this one already reports complete, so
        # the DMA is issued at once.
        sync.assert_not_called()
        mgr.d2h.assert_called_once_with([3, 5], [1, 2], [1, 1])
        mgr.h2d.assert_not_called()
        self.assertEqual(handler._pending[7], (fut, 2 * 64))
        self.assertEqual(handler._deferred_d2h, {})

    def test_d2h_store_waits_for_device_snapshot(self):
        handler, mgr = self._make_handler(tpu_to_cpu=True)
        fut = MagicMock()
        mgr.d2h.return_value = fut

        _FakeTpuEvent.ready = False
        with patch("vllm_torchtpu.offload.cpu_tpu.synchronize_tensors"
                   ), _patch_tpu_event():
            self.assertTrue(
                handler.transfer_async(
                    7, (self._spec([3, 5]), self._spec([1, 2]))))
            # The forward that wrote the blocks is still running: no DMA,
            # nothing reported finished.
            mgr.d2h.assert_not_called()
            self.assertEqual(handler.get_finished(), [])
            self.assertIn(7, handler._deferred_d2h)

            # Once the snapshot drains, the next poll submits the DMA and
            # its completion is reported like any other job.
            _FakeTpuEvent.ready = True
            fut.is_ready.return_value = True
            fut.ok.return_value = True
            results = handler.get_finished()
        mgr.d2h.assert_called_once_with([3, 5], [1, 2], [1, 1])
        self.assertEqual([r.job_id for r in results], [7])
        self.assertEqual(handler._deferred_d2h, {})
        self.assertEqual(handler._pending, {})

    def test_d2h_stores_submit_in_order(self):
        handler, mgr = self._make_handler(tpu_to_cpu=True)
        mgr.d2h.return_value = MagicMock()
        _FakeTpuEvent.ready = False
        with patch("vllm_torchtpu.offload.cpu_tpu.synchronize_tensors"
                   ), _patch_tpu_event():
            handler.transfer_async(1, (self._spec([3]), self._spec([1])))
            handler.transfer_async(2, (self._spec([5]), self._spec([2])))
            # Only the later snapshot drained: the earlier store still gates
            # submission, so neither is issued.
            handler._deferred_d2h[2][0].ready = True
            handler.submit_deferred_d2h()
            mgr.d2h.assert_not_called()
            handler._deferred_d2h[1][0].ready = True
            handler.submit_deferred_d2h()
        self.assertEqual([c.args[0] for c in mgr.d2h.call_args_list],
                         [[3], [5]])

    def test_flush_wait_synchronizes_deferred_store(self):
        handler, mgr = self._make_handler(tpu_to_cpu=True)
        fut = MagicMock()
        fut.is_ready.return_value = True
        mgr.d2h.return_value = fut
        _FakeTpuEvent.ready = False
        with patch("vllm_torchtpu.offload.cpu_tpu.synchronize_tensors"
                   ), _patch_tpu_event():
            handler.transfer_async(7, (self._spec([3]), self._spec([1])))
            event = handler._deferred_d2h[7][0]
            # The scheduler is about to reuse the source blocks: the flush
            # waits for the snapshot, submits, then polls the DMA.
            handler.wait({7})
        self.assertTrue(event.synchronized)
        mgr.d2h.assert_called_once_with([3], [1], [1])
        self.assertIn(7, handler._pending)

    def test_flush_wait_raises_on_stalled_raiden_dma(self):
        handler, mgr = self._make_handler(tpu_to_cpu=True)
        fut = MagicMock()
        fut.is_ready.return_value = False  # never completes
        mgr.d2h.return_value = fut
        _FakeTpuEvent.ready = True
        with patch("vllm_torchtpu.offload.cpu_tpu.synchronize_tensors"
                   ), _patch_tpu_event(), patch(
                       "vllm_torchtpu.offload.cpu_tpu."
                       "_RAIDEN_OFFLOAD_WAIT_TIMEOUT_S", 0.05):
            handler.transfer_async(7, (self._spec([3]), self._spec([1])))
            with self.assertRaisesRegex(RuntimeError,
                                        r"timed out.*jobs=\[7\]"):
                handler.wait({7})
        self.assertIn(7, handler._pending)

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


class TestRaidenHybridPoolMapping(unittest.TestCase):
    """`_RaidenOffloadingHandler` in hybrid pool mode routes grouped GPU
    specs through expand_hybrid_pool_block_ids, matching the torch
    handler's row mapping exactly."""

    def _make_handler(self, tpu_to_cpu: bool, cpu_factor=1, num_groups=2):
        from vllm_torchtpu.offload.cpu_tpu import _RaidenOffloadingHandler

        mgr = MagicMock(spec=["d2h", "h2d"])
        handler = _RaidenOffloadingHandler(
            mgr,
            tpu_to_cpu=tpu_to_cpu,
            src_block_size_factor=1 if tpu_to_cpu else cpu_factor,
            dst_block_size_factor=cpu_factor if tpu_to_cpu else 1,
            bytes_per_kernel_block=64,
            hybrid_num_groups=num_groups,
        )
        return handler, mgr

    def test_d2h_store_grouped_mapping(self):
        from vllm.v1.kv_offload.base import GPULoadStoreSpec
        from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec

        handler, mgr = self._make_handler(tpu_to_cpu=True)
        mgr.d2h.return_value = MagicMock()
        src = GPULoadStoreSpec([5, 9, 2, 40],
                               group_sizes=[3, 1],
                               block_indices=[0, 3])
        dst = CPULoadStoreSpec([10, 11, 12, 20])
        _FakeTpuEvent.ready = True
        with patch("vllm_torchtpu.offload.cpu_tpu.synchronize_tensors"
                   ), _patch_tpu_event():
            self.assertTrue(handler.transfer_async(1, (src, dst)))
        mgr.d2h.assert_called_once_with([5, 9, 2, 40], [10, 11, 12, 20],
                                        [1, 1, 1, 1])

    def test_h2d_load_grouped_mapping_factor_2(self):
        from vllm.v1.kv_offload.base import GPULoadStoreSpec
        from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec

        handler, mgr = self._make_handler(tpu_to_cpu=False,
                                          cpu_factor=2,
                                          num_groups=1)
        mgr.h2d.return_value = MagicMock()
        src = CPULoadStoreSpec([5, 6])
        dst = GPULoadStoreSpec([7, 8, 9], group_sizes=[3], block_indices=[1])
        self.assertTrue(handler.transfer_async(2, (src, dst)))
        mgr.h2d.assert_called_once_with([11, 12, 13], [7, 8, 9], [1, 1, 1])


if __name__ == "__main__":
    unittest.main()
