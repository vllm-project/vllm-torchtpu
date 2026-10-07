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

"""100% full-module line and branch coverage tests for

streamindex_topk kernel modules:
- bref_override.py
- config.py
- dcp_sc_compact.py
- metadata.py
- streamindex_topk.py
on CPU under JAX_PLATFORMS=cpu TPU_SKIP_MDS_QUERY=true.
"""

from __future__ import annotations

import dataclasses
import importlib
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TPU_SKIP_MDS_QUERY", "true")

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl

from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
    bref_override,
    config,
    dcp_sc_compact,
    metadata,
)

st_module = importlib.import_module(
    "vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.streamindex_topk"
)


# ==============================================================================
# Helpers & Mocks for CPU Testing
# ==============================================================================


def _norm_idx(idx):
    if isinstance(idx, tuple):
        return tuple(_norm_idx(i) for i in idx)
    if hasattr(idx, "start") and hasattr(idx, "size"):
        stride = getattr(idx, "stride", 1)
        stride = 1 if stride is None else int(stride)
        start = int(np.asarray(idx.start))
        size = int(np.asarray(idx.size))
        return slice(start, start + size, stride)
    return idx


class MockArrayRef:
    """Mock HBM / VMEM buffer reference for simulating Pallas pipelines on CPU."""

    def __init__(self, arr: np.ndarray | jax.Array):
        self.arr = np.asarray(arr).copy()
        self.shape = self.arr.shape
        self.dtype = self.arr.dtype

    def __getitem__(self, idx):
        if idx is Ellipsis:
            return jnp.asarray(self.arr)
        res = self.arr[_norm_idx(idx)]
        return jnp.asarray(res)

    def __setitem__(self, idx, val):
        if isinstance(val, MockArrayRef):
            val = val.arr
        if idx is Ellipsis:
            self.arr[...] = np.asarray(val)
        else:
            self.arr[_norm_idx(idx)] = np.asarray(val)

    @property
    def at(self):
        class _AtIndexer:
            def __init__(self, ref):
                self.ref = ref

            def __getitem__(self, idx):
                if idx is Ellipsis:
                    return jnp.asarray(self.ref.arr)
                res = self.ref.arr[_norm_idx(idx)]
                return MockArrayRef(res)

            def __setitem__(self, idx, val):
                if isinstance(val, MockArrayRef):
                    val = val.arr
                if idx is Ellipsis:
                    self.ref.arr[...] = np.asarray(val)
                else:
                    self.ref.arr[_norm_idx(idx)] = np.asarray(val)

        return _AtIndexer(self)

    def bitcast(self, dtype):
        return MockArrayRef(self.arr.view(dtype))

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return MockArrayRef(self.arr.reshape(*shape))


class MockAsyncCopy:
    """Mock async DMA copy handle."""

    def __init__(self):
        self.started = False
        self.waited = False

    def start(self):
        self.started = True
        return self

    def wait(self):
        self.waited = True
        return self


def fake_when(cond):
    """Executes when body unconditionally for branch/statement coverage."""

    def decorator(fn):
        fn()
        return fn

    return decorator


def fake_bitcast(x, ty):
    """CPU fallback for pltpu.bitcast."""
    try:
        return x.view(ty)
    except Exception:
        return jnp.zeros(x.shape, dtype=ty)


class FakeMesh:
    """Hashable fake mesh for testing jax.jit static arguments."""

    def __init__(self, axis_name: str = "dcp", size: int = 2):
        self.axis_name = axis_name
        self.size = size
        self.axis_names = (axis_name,)
        self.shape = {axis_name: size}

    def __hash__(self):
        return hash((self.axis_name, self.size))

    def __eq__(self, other):
        return (
            isinstance(other, FakeMesh)
            and self.axis_name == other.axis_name
            and self.size == other.size
        )


# ==============================================================================
# 1. config.py Tests
# ==============================================================================


class TestConfig:
    """100% full-module line and branch coverage on config.py."""

    def test_mla_case(self):
        assert config.MlaCase.DECODE.symbol == "d"
        assert config.MlaCase.PREFILL.symbol == "p"
        assert config.MlaCase.MIXED.symbol == "m"

    def test_kv_layout_parse(self):
        # None defaults to HEAD_ALONG_SUBLANE
        assert config.KVLayout.parse(None) == config.KVLayout.HEAD_ALONG_SUBLANE

        # KVLayout instance passes through
        assert (
            config.KVLayout.parse(config.KVLayout.SEQ_ALONG_LANE)
            == config.KVLayout.SEQ_ALONG_LANE
        )
        assert (
            config.KVLayout.parse(config.KVLayout.HEAD_ALONG_SUBLANE)
            == config.KVLayout.HEAD_ALONG_SUBLANE
        )

        # String aliases and case-insensitivity
        assert (
            config.KVLayout.parse("head_along_sublane")
            == config.KVLayout.HEAD_ALONG_SUBLANE
        )
        assert (
            config.KVLayout.parse("  HEAD_ALONG_SUBLANE  ")
            == config.KVLayout.HEAD_ALONG_SUBLANE
        )
        assert config.KVLayout.parse("seq_along_lane") == config.KVLayout.SEQ_ALONG_LANE
        assert config.KVLayout.parse("SEQ_ALONG_LANE") == config.KVLayout.SEQ_ALONG_LANE
        # Typo tolerance
        assert config.KVLayout.parse("seq_alone_lane") == config.KVLayout.SEQ_ALONG_LANE

        # Invalid string raises ValueError
        with pytest.raises(ValueError, match="Unknown kv_layout 'invalid'"):
            config.KVLayout.parse("invalid")

    def test_scheduling_group_ids(self):
        config.reset_scheduling_group_ids()

        with pytest.raises(ValueError, match="must be positive"):
            config.reserve_scheduling_group_ids(0)

        with pytest.raises(ValueError, match="must be positive"):
            config.reserve_scheduling_group_ids(-5)

        # Single id reservation
        first1 = config.reserve_scheduling_group_ids(1)
        assert first1 == 1

        # Multiple ids reservation
        first2 = config.reserve_scheduling_group_ids(3)
        assert first2 == 2

        # Reset restarts from 1
        config.reset_scheduling_group_ids()
        first_reset = config.reserve_scheduling_group_ids(1)
        assert first_reset == 1


# ==============================================================================
# 2. dcp_sc_compact.py Tests
# ==============================================================================


class TestDcpScCompact:
    """100% full-module line and branch coverage on dcp_sc_compact.py."""

    def test_pack_nonnegative_validations(self):
        # Non-int32 dtype raises ValueError
        with pytest.raises(ValueError, match="vals must be int32"):
            dcp_sc_compact.pack_nonnegative(jnp.zeros((4, 16), dtype=jnp.float32))

        # Missing SparseCore info raises NotImplementedError
        with patch("jax.experimental.pallas.tpu.get_tpu_info") as mock_info:
            mock_info.return_value = SimpleNamespace(sparse_core=None)
            with pytest.raises(
                NotImplementedError, match="SparseCore is not available"
            ):
                dcp_sc_compact.pack_nonnegative(jnp.zeros((4, 16), dtype=jnp.int32))

        # width not multiple of lanes raises ValueError
        with patch("jax.experimental.pallas.tpu.get_tpu_info") as mock_info:
            mock_info.return_value = SimpleNamespace(
                sparse_core=SimpleNamespace(num_lanes=16, num_cores=4, num_subcores=4)
            )
            with pytest.raises(ValueError, match="width .* must be a multiple of 16"):
                dcp_sc_compact.pack_nonnegative(jnp.zeros((4, 10), dtype=jnp.int32))

    def test_pack_nonnegative_execution(self):
        sc_info = SimpleNamespace(num_lanes=16, num_cores=2, num_subcores=2)
        fake_tpu_info = SimpleNamespace(sparse_core=sc_info)

        def fake_kernel(body_fn, out_type=None, *args, **kwargs):
            def run(vals_flat):
                return jnp.zeros(out_type.shape, dtype=out_type.dtype)

            return run

        with (
            patch(
                "jax._src.pallas.mosaic.tpu_info.get_tpu_info",
                return_value=fake_tpu_info,
            ),
            patch(
                "jax.experimental.pallas.tpu.get_tpu_info",
                return_value=fake_tpu_info,
            ),
            patch("jax.experimental.pallas.kernel", side_effect=fake_kernel),
        ):
            # Test padded_b > b branch (e.g. b=3, rows_per_wave=4, padded_b=4)
            vals_pad = jnp.zeros((3, 32), dtype=jnp.int32)
            out_pad = dcp_sc_compact.pack_nonnegative(vals_pad)
            assert out_pad.shape == (3, 32)

            # Test padded_b == b branch (e.g. b=4, rows_per_wave=4, padded_b=4)
            vals_nopad = jnp.zeros((4, 32), dtype=jnp.int32)
            out_nopad = dcp_sc_compact.pack_nonnegative(vals_nopad)
            assert out_nopad.shape == (4, 32)

    def test_pack_body_and_closures(self):
        captured = {}

        def fake_emit_pipeline(step_fn, grid=None, in_specs=None, out_specs=None):
            captured["step_fn"] = step_fn
            captured["in_specs"] = in_specs
            captured["out_specs"] = out_specs

            def call(*args):
                pass

            return call

        def fake_parallel_loop(start, end, unroll=1, carry=None):
            def decorator(fn):
                c = carry
                iters = int(np.asarray(end)) - int(np.asarray(start))
                for i in range(max(0, iters)):
                    if c is not None:
                        c = fn(i, c)
                    else:
                        fn(i)
                return c

            return decorator

        def fake_store_compressed(ref, val, mask=None):
            pass

        def fake_all_reduce_popcount(mask):
            return jnp.array([jnp.sum(mask.astype(jnp.int32))], dtype=jnp.int32)

        vals_hbm = jnp.zeros(64, dtype=jnp.int32)
        out_hbm = jnp.zeros(64, dtype=jnp.int32)
        rowbuf_vmem = MockArrayRef(np.zeros(64, dtype=np.int32))

        # Test num_waves == 1 (buf_count = 1)
        with (
            patch("jax.lax.axis_index", return_value=0),
            patch(
                "jax.experimental.pallas.tpu.emit_pipeline",
                side_effect=fake_emit_pipeline,
            ),
        ):
            dcp_sc_compact._pack_body(
                vals_hbm,
                out_hbm,
                rowbuf_vmem,
                b=2,
                width=32,
                num_waves=1,
                lanes=16,
                num_cores=2,
                num_subcores=2,
            )
            # Evaluate BlockSpec index map lambdas
            assert captured["in_specs"][0].index_map(0) == (0,)
            assert captured["out_specs"].index_map(0) == (0,)

        # Test num_waves == 2 (buf_count = 2)
        with (
            patch("jax.lax.axis_index", return_value=0),
            patch(
                "jax.experimental.pallas.tpu.emit_pipeline",
                side_effect=fake_emit_pipeline,
            ),
        ):
            dcp_sc_compact._pack_body(
                vals_hbm,
                out_hbm,
                rowbuf_vmem,
                b=8,
                width=32,
                num_waves=2,
                lanes=16,
                num_cores=2,
                num_subcores=2,
            )

        step_fn = captured["step_fn"]
        vals_vmem = MockArrayRef(
            np.array([1, -1, 3, -1] * 8, dtype=np.int32)
        )  # mixed >=0 and <0
        out_vmem = MockArrayRef(np.zeros(32, dtype=np.int32))

        # Test step_fn with row < b
        with (
            patch("jax.experimental.pallas.program_id", return_value=0),
            patch(
                "jax.experimental.pallas.tpu_sc.parallel_loop",
                side_effect=fake_parallel_loop,
            ),
            patch(
                "jax.experimental.pallas.tpu_sc.store_compressed",
                side_effect=fake_store_compressed,
            ),
            patch(
                "jax.experimental.pallas.tpu_sc.all_reduce_population_count",
                side_effect=fake_all_reduce_popcount,
            ),
        ):
            step_fn(vals_vmem, out_vmem)

        # Test step_fn with row >= b
        with (
            patch("jax.experimental.pallas.program_id", return_value=10),
            patch(
                "jax.experimental.pallas.tpu_sc.parallel_loop",
                side_effect=fake_parallel_loop,
            ),
            patch(
                "jax.experimental.pallas.tpu_sc.store_compressed",
                side_effect=fake_store_compressed,
            ),
            patch(
                "jax.experimental.pallas.tpu_sc.all_reduce_population_count",
                side_effect=fake_all_reduce_popcount,
            ),
        ):
            step_fn(vals_vmem, out_vmem)

    def test_resolve_owned_winners(self):
        local_idxs = jnp.array([[10, 20, 30, 40], [50, 60, 70, 80]])
        # slots: slot // k == dcp_rank determines ownership
        slots = jnp.array([[0, 5, -1, 2], [1, 4, 7, -1]])
        k = 4
        dcp_rank = 0

        with patch.object(
            dcp_sc_compact, "pack_nonnegative", side_effect=lambda x: x
        ) as mock_pack:
            res = dcp_sc_compact.resolve_owned_winners(local_idxs, slots, k, dcp_rank)
            assert mock_pack.called
            # slots for row 0:
            # slot 0: 0//4 == 0 (ours) -> local_idxs[0, 0] = 10
            # slot 5: 5//4 == 1 != 0 (not ours) -> -1
            # slot -1: <0 (not ours) -> -1
            # slot 2: 2//4 == 0 (ours) -> local_idxs[0, 2] = 30
            np.testing.assert_array_equal(res[0], np.array([10, -1, -1, 30]))


# ==============================================================================
# 3. metadata.py Tests
# ==============================================================================


class TestMetadata:
    """100% full-module line and branch coverage on metadata.py."""

    def test_cp_helpers(self):
        # cp_size == 1 branch
        assert metadata.cp_local_to_global(5, 0, 1, 4) == 5
        assert metadata.cp_local_length(10, 0, 1, 4) == 10
        assert metadata.cp_owner_rank(jnp.array([5]), 1, 4) == 0
        assert metadata.cp_global_to_local(7, 1, 4) == 7

        # cp_size > 1 branch
        # cp_size = 2, interleave_c = 4 -> cycle_c = 8
        # rank 0 gets [0..3], rank 1 gets [4..7],
        # rank 0 gets [8..11], rank 1 gets [12..15]
        local_idx = 5  # local 5 = 1*4 + 1 -> global 1*8 + 1*4 + 1 = 13 for rank 1
        global_pos = metadata.cp_local_to_global(
            local_idx, cp_rank=1, cp_size=2, interleave_c=4
        )
        assert global_pos == 13

        # cp_local_length
        len_rank0 = metadata.cp_local_length(10, cp_rank=0, cp_size=2, interleave_c=4)
        assert len_rank0 == 6  # [0..3] (4) + [8..9] (2) = 6
        len_rank1 = metadata.cp_local_length(10, cp_rank=1, cp_size=2, interleave_c=4)
        assert len_rank1 == 4  # [4..7] (4) = 4

        # cp_owner_rank
        assert metadata.cp_owner_rank(jnp.array(2), 2, 4) == 0
        assert metadata.cp_owner_rank(jnp.array(5), 2, 4) == 1

        # cp_global_to_local
        assert (
            metadata.cp_global_to_local(13, 2, 4) == 5
        )  # (13//8)*4 + 13%4 = 4 + 1 = 5

    def test_cp_rank_as_data(self):
        with patch("jax.lax.all_to_all", side_effect=lambda x, *a, **kw: x):
            rank = metadata.cp_rank_as_data("dcp", 2)
            assert int(np.asarray(rank)) == 0

    def test_metadata_ref(self):
        ref = metadata.MetadataRef.create(
            num_steps=jnp.array([10]),
            batch_tile_idx=jnp.array([0, 1]),
            bq_idx=jnp.array([0, 0]),
            bkv_idx=jnp.array([0, 1]),
        )
        assert ref.start_seq_idx is ref.batch_tile_idx

        # Test pytree flatten/unflatten
        leaves, treedef = jax.tree_util.tree_flatten(ref)
        reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)
        assert reconstructed == ref

    def test_next_power_of_2(self):
        assert metadata.next_power_of_2(1) == 1
        assert metadata.next_power_of_2(2) == 2
        assert metadata.next_power_of_2(3) == 4
        assert metadata.next_power_of_2(4) == 4
        assert metadata.next_power_of_2(15) == 16
        assert metadata.next_power_of_2(16) == 16
        assert metadata.next_power_of_2(17) == 32

    def test_generate_max_steps(self):
        fake_info = SimpleNamespace(
            smem_capacity_bytes=1024 * 1024,
            num_lanes=128,
        )
        with patch("jax.experimental.pallas.tpu.get_tpu_info", return_value=fake_info):
            steps = metadata.generate_max_steps(num_seqs=4, max_pages_per_seq=16)
            assert steps >= 128
            assert steps % 128 == 0

    def test_compute_batched_seq_metadata_empty_pass(self):
        # start_seq_idx >= end_seq_idx triggers _empty_schedule
        seq_lens = jnp.array([100, 200], dtype=jnp.int32)
        cu_q_lens = jnp.array([0, 1, 2], dtype=jnp.int32)
        ref = metadata.compute_metadata(
            seq_lens=seq_lens,
            cu_q_lens=cu_q_lens,
            start_seq_idx=2,
            end_seq_idx=1,
            bq_sz=16,
            bkv_sz=128,
            pages_per_seq=4,
            page_size=128,
            max_num_tokens=2,
        )
        assert int(ref.num_steps[0]) == 0
        assert int(ref.batch_tile_idx[0]) == 0

    def test_compute_batched_seq_metadata_schedules(self):
        seq_lens = jnp.array([100, 200, 150], dtype=jnp.int32)
        cu_q_lens = jnp.array([0, 16, 32, 48], dtype=jnp.int32)

        # 1. static_q_len == 1 (decode path)
        ref_decode = metadata.compute_metadata(
            seq_lens=seq_lens,
            cu_q_lens=cu_q_lens,
            start_seq_idx=0,
            end_seq_idx=3,
            bq_sz=16,
            bkv_sz=128,
            pages_per_seq=4,
            page_size=128,
            max_num_tokens=48,
            static_q_len=1,
            seq_batch_size=2,  # test seq_batch_size > 1
        )
        assert ref_decode.num_steps[0] > 0

        # 2. static_q_len > 1 (prefill path)
        ref_prefill = metadata.compute_metadata(
            seq_lens=seq_lens,
            cu_q_lens=cu_q_lens,
            start_seq_idx=0,
            end_seq_idx=3,
            bq_sz=16,
            bkv_sz=128,
            pages_per_seq=4,
            page_size=128,
            max_num_tokens=48,
            static_q_len=16,
            seq_batch_size=1,
        )
        assert ref_prefill.num_steps[0] > 0

        # 3. static_q_len is None (mixed path)
        ref_mixed = metadata.compute_metadata(
            seq_lens=seq_lens,
            cu_q_lens=cu_q_lens,
            start_seq_idx=0,
            end_seq_idx=3,
            bq_sz=16,
            bkv_sz=128,
            pages_per_seq=4,
            page_size=128,
            max_num_tokens=48,
            static_q_len=None,
            seq_batch_size=1,
        )
        assert ref_mixed.num_steps[0] > 0

        # 4. Chunked with tuple chunk_starts & cp_size > 1
        ref_chunked_cp = metadata.compute_metadata(
            seq_lens=seq_lens,
            cu_q_lens=cu_q_lens,
            start_seq_idx=0,
            end_seq_idx=3,
            bq_sz=16,
            bkv_sz=128,
            pages_per_seq=4,
            page_size=128,
            max_num_tokens=48,
            static_q_len=16,
            chunk_token_start=(0, 24),
            chunk_tokens=24,
            cp_size=2,
            cp_rank=0,
            interleave_c=2,
        )
        assert len(ref_chunked_cp.num_steps) == 2

        # 5. Chunked with non-tuple chunk_token_start
        ref_chunked_int = metadata.compute_metadata(
            seq_lens=seq_lens,
            cu_q_lens=cu_q_lens,
            start_seq_idx=0,
            end_seq_idx=3,
            bq_sz=16,
            bkv_sz=64,  # test bkv_sz < page_size branch
            pages_per_seq=4,
            page_size=128,
            max_num_tokens=48,
            static_q_len=16,
            chunk_token_start=0,
            chunk_tokens=24,
        )
        assert len(ref_chunked_int.num_steps) == 1


# ==============================================================================
# 4. bref_override.py Tests
# ==============================================================================


class TestBrefOverride:
    """100% full-module line and branch coverage on bref_override.py."""

    def test_step_offset(self):
        meta = metadata.MetadataRef(
            num_steps=jnp.array([10, 20, 30]),
            batch_tile_idx=jnp.array([0]),
            bq_idx=jnp.array([0]),
            bkv_idx=jnp.array([0]),
        )
        assert bref_override._step_offset(meta, 0) == 0
        assert bref_override._step_offset(meta, 2) == 30

    def test_token_window(self):
        cu_q_lens = jnp.array([0, 16, 32], dtype=jnp.int32)
        meta = metadata.MetadataRef(
            num_steps=jnp.array([10]),
            batch_tile_idx=jnp.array([0, 1]),
            bq_idx=jnp.array([0, 0]),
            bkv_idx=jnp.array([0, 0]),
        )
        start_end = jnp.array([0, 2, 0], dtype=jnp.int32)
        refs = (None, cu_q_lens, meta, start_end)

        # 1. seq_batch_size > 1 branch
        lo, sz = bref_override._token_window(refs, 0, 16, bq_sz=16, seq_batch_size=4)
        assert lo == 0
        assert sz == 4

        # 2. bq_sz == 1 branch
        lo, sz = bref_override._token_window(refs, 0, 16, bq_sz=1, seq_batch_size=1)
        assert lo == 0
        assert sz == 1

        # 3. bq_sz > 1 branch
        lo, sz = bref_override._token_window(refs, 0, 16, bq_sz=16, seq_batch_size=1)
        assert lo == 0
        assert sz == 16

    def test_streamindex_kv_buffered_ref(self):
        spec = pl.BlockSpec((16, 128), lambda *args: (0, 0))
        ref = bref_override.StreamIndexKVBufferedRef.input(
            spec=spec,
            dtype_or_type=jnp.uint8,
            buffer_count=2,
            use_lookahead=True,
            bkv_p=2,
            page_size_per_kv_packing=32,
            kv_packing=4,
            pages_per_seq=2,
            seq_batch_size=2,
        )

        # Populate slots and semaphores
        sem_mock = MagicMock()
        window_mock = MagicMock()
        ref_pop = dataclasses.replace(
            ref,
            sem_recvs=sem_mock,
            window_ref=window_mock,
            copy_in_slot=jnp.uint32(0),
            wait_in_slot=jnp.uint32(0),
        )

        cache_kv_hbm = MagicMock(spec=jax.Ref)
        cache_kv_hbm.shape = (4, 32, 4, 256)
        cache_kv_hbm.reshape = MagicMock(return_value=cache_kv_hbm)
        page_indices_ref = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
        meta_ref = metadata.MetadataRef(
            num_steps=jnp.array([1]),
            batch_tile_idx=jnp.array([0]),
            bq_idx=jnp.array([0]),
            bkv_idx=jnp.array([0]),
        )

        with patch(
            "jax.experimental.pallas.tpu.make_async_copy",
            return_value=MockAsyncCopy(),
        ):
            ref_pop.copy_in((cache_kv_hbm, page_indices_ref, meta_ref), (0,))

            # wait_in when buffered
            ref_pop.wait_in((cache_kv_hbm, page_indices_ref, meta_ref), (0,))

            # wait_in when unbuffered (early return branch)
            ref_unbuf = dataclasses.replace(ref_pop, _buffer_count=0)
            ref_unbuf.wait_in((cache_kv_hbm, page_indices_ref, meta_ref), (0,))

    def test_streamindex_kv_seq_along_lane_buffered_ref(self):
        spec = pl.BlockSpec((16, 128), lambda *args: (0, 0))
        ref = bref_override.StreamIndexKVSeqAlongLaneBufferedRef.input(
            spec=spec,
            dtype_or_type=jnp.uint8,
            buffer_count=2,
            use_lookahead=True,
            bkv_p=2,
            page_size=128,
            pages_per_seq=2,
            seq_batch_size=2,
        )

        sem_mock = MagicMock()
        window_mock = MagicMock()
        ref_pop = dataclasses.replace(
            ref,
            sem_recvs=sem_mock,
            window_ref=window_mock,
            copy_in_slot=jnp.uint32(0),
            wait_in_slot=jnp.uint32(0),
        )

        cache_kv_hbm = MagicMock(spec=jax.Ref)
        cache_kv_hbm.shape = (4, 33, 4, 128)
        page_indices_ref = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
        meta_ref = metadata.MetadataRef(
            num_steps=jnp.array([1]),
            batch_tile_idx=jnp.array([0]),
            bq_idx=jnp.array([0]),
            bkv_idx=jnp.array([0]),
        )

        with patch(
            "jax.experimental.pallas.tpu.make_async_copy",
            return_value=MockAsyncCopy(),
        ):
            ref_pop.copy_in((cache_kv_hbm, page_indices_ref, meta_ref), (0,))
            ref_pop.wait_in((cache_kv_hbm, page_indices_ref, meta_ref), (0,))

            # wait_in when unbuffered
            ref_unbuf = dataclasses.replace(ref_pop, _buffer_count=0)
            ref_unbuf.wait_in((cache_kv_hbm, page_indices_ref, meta_ref), (0,))

    def test_streamindex_q_buffered_ref(self):
        spec = pl.BlockSpec((16, 128), lambda *args: (0, 0))
        ref = bref_override.StreamIndexQBufferedRef.input(
            spec=spec,
            dtype_or_type=jnp.bfloat16,
            buffer_count=2,
            use_lookahead=True,
            bq_sz=16,
            chunk_tokens=16,
            seq_batch_size=1,
        )

        sem_mock = MagicMock()
        window_mock = MagicMock()
        ref_pop = dataclasses.replace(
            ref,
            sem_recvs=sem_mock,
            window_ref=window_mock,
            copy_in_slot=jnp.uint32(0),
            wait_in_slot=jnp.uint32(0),
        )

        q_hbm = MagicMock()
        cu_q_lens = jnp.array([0, 16], dtype=jnp.int32)
        meta_ref = metadata.MetadataRef(
            num_steps=jnp.array([1]),
            batch_tile_idx=jnp.array([0]),
            bq_idx=jnp.array([0]),
            bkv_idx=jnp.array([0]),
        )
        start_end = jnp.array([0, 1, 0], dtype=jnp.int32)
        src_ref = (q_hbm, cu_q_lens, meta_ref, start_end)

        with patch(
            "jax.experimental.pallas.tpu.make_async_copy",
            return_value=MockAsyncCopy(),
        ):
            ref_pop.copy_in(src_ref, (0,))
            ref_pop.wait_in(src_ref, (0,))

            # wait_in when unbuffered
            ref_unbuf = dataclasses.replace(ref_pop, _buffer_count=0)
            ref_unbuf.wait_in(src_ref, (0,))

    def test_streamindex_o_buffered_ref(self):
        spec = pl.BlockSpec((16, 128), lambda *args: (0, 0))
        ref = bref_override.StreamIndexOBufferedRef.output(
            spec=spec,
            dtype_or_type=jnp.int32,
            buffer_count=2,
            use_lookahead=False,
            bq_sz=16,
            num_sublanes=1,
            seq_batch_size=1,
        )

        sem_mock = MagicMock()
        window_mock = MagicMock()
        ref_pop = dataclasses.replace(
            ref,
            sem_sends=sem_mock,
            window_ref=window_mock,
            copy_out_slot=jnp.uint32(0),
            wait_out_slot=jnp.uint32(0),
        )

        scores_hbm = MagicMock()
        scores_hbm.shape = (16, 1, 128)
        cu_q_lens = jnp.array([0, 16], dtype=jnp.int32)
        meta_ref = metadata.MetadataRef(
            num_steps=jnp.array([1]),
            batch_tile_idx=jnp.array([0]),
            bq_idx=jnp.array([0]),
            bkv_idx=jnp.array([0]),
        )
        start_end = jnp.array([0, 1, 0], dtype=jnp.int32)
        dst_ref = (scores_hbm, cu_q_lens, meta_ref, start_end)

        with patch(
            "jax.experimental.pallas.tpu.make_async_copy",
            return_value=MockAsyncCopy(),
        ):
            ref_pop.copy_out(dst_ref, (0,))
            ref_pop.wait_out(dst_ref, (0,))

            # wait_out when unbuffered
            ref_unbuf = dataclasses.replace(ref_pop, _buffer_count=0)
            ref_unbuf.wait_out(dst_ref, (0,))


# ==============================================================================
# 5. streamindex_topk.py Helpers & Pure Functions Tests
# ==============================================================================


class TestStreamIndexTopKHelpers:
    """100% line and branch coverage on helper functions in streamindex_topk.py."""

    def test_cdiv_align_to(self):
        assert st_module.cdiv(10, 3) == 4
        assert st_module.cdiv(9, 3) == 3
        assert st_module.align_to(10, 4) == 12
        assert st_module.align_to(12, 4) == 12

    def test_dtype_helpers(self):
        assert st_module.get_dtype_bitwidth(jnp.int32) == 32
        assert st_module.get_dtype_bitwidth(jnp.uint8) == 8
        assert st_module.get_dtype_packing(jnp.int32) == 1
        assert st_module.get_dtype_packing(jnp.uint8) == 4

    def test_prepare_q_inputs_and_index_weights(self):
        q = jnp.ones((4, 3, 100), dtype=jnp.bfloat16)  # actual_num_q_heads=3
        padded_q = st_module.prepare_q_inputs(q)
        # packing for bfloat16 (16 bits) is 32//16 = 2, so num_q_heads aligned to 2 -> 4
        # head_dim aligned to 128 -> 128
        assert padded_q.shape == (4, 4, 128)

        w = jnp.ones((4, 3), dtype=jnp.float32)
        padded_w = st_module.prepare_index_weights(w, jnp.bfloat16)
        assert padded_w.shape == (4, 4)
        assert padded_w.dtype == jnp.float32

        # prepare_outputs 3D vs 2D
        out3d = jnp.zeros((4, 2, 8))
        assert st_module.prepare_outputs(out3d).shape == (4, 16)
        out2d = jnp.zeros((4, 16))
        assert st_module.prepare_outputs(out2d).shape == (4, 16)

    def test_get_kv_cache_shape(self):
        # SEQ_ALONG_LANE invalid page_size
        with pytest.raises(
            ValueError, match="SEQ_ALONG_LANE requires page_size to be a multiple"
        ):
            st_module.get_kv_cache_shape(
                total_num_pages=4,
                page_size=64,
                actual_head_dim=128,
                kv_layout=config.KVLayout.SEQ_ALONG_LANE,
            )

        # SEQ_ALONG_LANE valid shape: [total_num_pages, head_dim // 4 + 1, 4, page_size]
        shape_sal = st_module.get_kv_cache_shape(
            total_num_pages=4,
            page_size=128,
            actual_head_dim=128,
            kv_layout=config.KVLayout.SEQ_ALONG_LANE,
        )
        assert shape_sal == (4, 33, 4, 128)

        # HEAD_ALONG_SUBLANE invalid page_size
        with pytest.raises(
            ValueError,
            match="HEAD_ALONG_SUBLANE requires page_size to be a multiple",
        ):
            st_module.get_kv_cache_shape(
                total_num_pages=4,
                page_size=5,
                actual_head_dim=128,
                kv_layout=config.KVLayout.HEAD_ALONG_SUBLANE,
            )

        # HEAD_ALONG_SUBLANE valid shape:
        # head_dim=128, record_width = 128 + 1 = 129 -> align_to(129, 128) = 256
        shape_has = st_module.get_kv_cache_shape(
            total_num_pages=4,
            page_size=128,
            actual_head_dim=128,
            kv_layout=config.KVLayout.HEAD_ALONG_SUBLANE,
        )
        assert shape_has == (4, 32, 4, 256)

    def test_convert_cache_to_seq_along_lane(self):
        # HEAD_ALONG_SUBLANE cache of shape [pages, page_size // 4, 4, width]
        cache_has = jnp.zeros((2, 32, 4, 256), dtype=jnp.uint8)
        cache_sal = st_module.convert_cache_to_seq_along_lane(
            cache_has, actual_head_dim=128
        )
        # Result shape: [2, 33, 4, 128]
        assert cache_sal.shape == (2, 33, 4, 128)

    def test_effective_row_lengths(self):
        seq_lens = jnp.array([128, 256], dtype=jnp.int32)
        cu_q_lens = jnp.array([0, 16, 32], dtype=jnp.int32)
        distribution = jnp.array([0, 0, 2], dtype=jnp.int32)

        # cp_size = 1
        lengths1 = st_module._effective_row_lengths(
            seq_lens=seq_lens,
            cu_q_lens=cu_q_lens,
            distribution=distribution,
            num_tokens=32,
            num_positions=128,
            compression_ratio=1,
            cp_size=1,
        )
        assert lengths1.shape == (32,)
        assert int(lengths1[0]) > 0

        # cp_size = 2
        lengths2 = st_module._effective_row_lengths(
            seq_lens=seq_lens,
            cu_q_lens=cu_q_lens,
            distribution=distribution,
            num_tokens=32,
            num_positions=128,
            compression_ratio=1,
            cp_rank=0,
            cp_size=2,
            interleave_c=2,
        )
        assert lengths2.shape == (32,)


# ==============================================================================
# 6. streamindex_topk.py Kernel Direct Execution Tests
# ==============================================================================


class TestStreamIndexTopKKernelDirect:
    """100% full-module line and branch coverage on kernel() in streamindex_topk.py."""

    def test_kernel_construction_and_step_body_execution(self):
        captured = {}

        def fake_with_scoped(final_allocs=None):
            def decorator(fn):
                def wrapper(*args, **kwargs):
                    return fn(final_allocs)

                return wrapper

            return decorator

        def fake_emit_pipeline(step_fn, grid=None, in_specs=None, out_specs=None):
            captured["step_fn"] = step_fn
            captured["in_specs"] = in_specs
            captured["out_specs"] = out_specs

            def pipeline_call(*args, **kwargs):
                pass

            return pipeline_call

        def fake_run_scoped(fn, *allocs):
            return fn(*allocs)

        # Case 1: HEAD_ALONG_SUBLANE, seq_batch_size == 1 and bq_sz > 1
        # (scoped q_hm path)
        with (
            patch("jax.experimental.pallas.run_scoped", side_effect=fake_run_scoped),
            patch(
                "jax.experimental.pallas.with_scoped",
                side_effect=fake_with_scoped,
            ),
            patch(
                "jax.experimental.pallas.tpu.emit_pipeline",
                side_effect=fake_emit_pipeline,
            ),
        ):
            meta = metadata.compute_metadata(
                seq_lens=jnp.array([128]),
                cu_q_lens=jnp.array([0, 16]),
                start_seq_idx=0,
                end_seq_idx=1,
                bq_sz=16,
                bkv_sz=128,
                pages_per_seq=1,
                page_size=128,
                max_num_tokens=16,
                compression_ratio=1,
                static_q_len=16,
            )
            q_ref = jnp.zeros((16, 4, 128), dtype=jnp.bfloat16)
            w_ref = jnp.zeros((16, 4), dtype=jnp.float32)
            cache_ref = jnp.zeros((1, 32, 4, 256), dtype=jnp.uint8)
            scores_ref = jnp.zeros((16, 1, 128), dtype=jnp.int32)
            st_module.kernel(
                jnp.array([128]),
                jnp.array([0]),
                jnp.array([0, 16]),
                jnp.array([0, 1, 0]),
                meta,
                jnp.array([0]),
                q_ref,
                w_ref,
                cache_ref,
                scores_ref,
                compression_ratio=1,
                static_q_len=16,
                bkv_p=1,
                bq_sz=16,
                seq_batch_size=1,
                buffer_count=3,
                case=config.MlaCase.PREFILL,
                kv_layout=config.KVLayout.HEAD_ALONG_SUBLANE,
            )

            # Evaluate index_map lambdas
            for spec in captured["in_specs"]:
                spec.index_map(0)
            captured["out_specs"].index_map(0)

            # Execute step_fn with q_hm_ref
            step_fn = captured["step_fn"]
            q_vmem = MockArrayRef(np.zeros((16, 4, 128), dtype=np.float32))
            w_vmem = MockArrayRef(np.ones((16, 4), dtype=np.float32))
            bkv_vmem = MockArrayRef(np.zeros((1, 32, 4, 256), dtype=np.uint8))
            scores_vmem = MockArrayRef(np.zeros((16, 1, 128), dtype=np.int32))
            q_hm = MockArrayRef(np.zeros((64, 128), dtype=np.float32))

            with (
                patch(
                    "jax.experimental.pallas.program_id",
                    return_value=jnp.int32(0),
                ),
                patch("jax.experimental.pallas.when", side_effect=fake_when),
                patch(
                    "jax.experimental.pallas.tpu.bitcast",
                    side_effect=fake_bitcast,
                ),
            ):
                step_fn(
                    q_vmem,
                    w_vmem,
                    bkv_vmem,
                    scores_vmem,
                    q_hm_ref=q_hm,
                )

        # Case 2: SEQ_ALONG_LANE, seq_batch_size == 2
        # (non-scoped, head_along_lane path, n_tok == 1 decode)
        with (
            patch("jax.experimental.pallas.run_scoped", side_effect=fake_run_scoped),
            patch(
                "jax.experimental.pallas.with_scoped",
                side_effect=fake_with_scoped,
            ),
            patch(
                "jax.experimental.pallas.tpu.emit_pipeline",
                side_effect=fake_emit_pipeline,
            ),
        ):
            meta_sal = metadata.compute_metadata(
                seq_lens=jnp.array([128, 128]),
                cu_q_lens=jnp.array([0, 1, 2]),
                start_seq_idx=0,
                end_seq_idx=2,
                bq_sz=1,
                bkv_sz=128,
                pages_per_seq=1,
                page_size=128,
                max_num_tokens=2,
                compression_ratio=1,
                static_q_len=1,
                seq_batch_size=2,
            )
            q_ref2 = jnp.zeros((2, 4, 128), dtype=jnp.bfloat16)
            w_ref2 = jnp.zeros((2, 4), dtype=jnp.float32)
            cache_ref2 = jnp.zeros((1, 33, 4, 128), dtype=jnp.uint8)
            scores_ref2 = jnp.zeros((2, 1, 128), dtype=jnp.int32)
            st_module.kernel(
                jnp.array([128, 128]),
                jnp.array([0, 1]),
                jnp.array([0, 1, 2]),
                jnp.array([0, 2, 0]),
                meta_sal,
                jnp.array([0]),
                q_ref2,
                w_ref2,
                cache_ref2,
                scores_ref2,
                compression_ratio=1,
                static_q_len=1,
                bkv_p=1,
                bq_sz=1,
                seq_batch_size=2,
                buffer_count=3,
                case=config.MlaCase.DECODE,
                kv_layout=config.KVLayout.SEQ_ALONG_LANE,
            )

            # Evaluate index_map lambdas
            for spec in captured["in_specs"]:
                spec.index_map(0)
            captured["out_specs"].index_map(0)

            # Execute step_fn with seq_along_lane and n_tok=1 decode path
            step_fn_sal = captured["step_fn"]
            q_vmem_sal = MockArrayRef(np.zeros((2, 1, 4, 128), dtype=np.float32))
            w_vmem_sal = MockArrayRef(np.ones((2, 1, 4), dtype=np.float32))
            bkv_vmem_sal = MockArrayRef(np.zeros((2, 33, 4, 128), dtype=np.uint8))
            scores_vmem_sal = MockArrayRef(np.zeros((2, 1, 128), dtype=np.int32))

            with (
                patch(
                    "jax.experimental.pallas.program_id",
                    return_value=jnp.int32(0),
                ),
                patch("jax.experimental.pallas.when", side_effect=fake_when),
                patch(
                    "jax.experimental.pallas.tpu.bitcast",
                    side_effect=fake_bitcast,
                ),
            ):
                step_fn_sal(
                    q_vmem_sal,
                    w_vmem_sal,
                    bkv_vmem_sal,
                    scores_vmem_sal,
                    q_hm_ref=None,
                )


# ==============================================================================
# 7. streamindex_topk.py Outer Dispatch Tests
# ==============================================================================


class TestStreamIndexTopKOuter:
    """100% full-module line and branch coverage on streamindex_topk() dispatch."""

    def test_validation_errors(self):
        q = jnp.zeros((16, 4, 128), dtype=jnp.bfloat16)
        w = jnp.zeros((16, 4), dtype=jnp.float32)
        cache = jnp.zeros((2, 32, 4, 256), dtype=jnp.uint8)
        seq_lens = jnp.array([128, 128], dtype=jnp.int32)
        page_indices = jnp.array([0, 1], dtype=jnp.int32)
        cu_q_lens = jnp.array([0, 8, 16], dtype=jnp.int32)
        dist = jnp.array([0, 0, 2], dtype=jnp.int32)

        # 1. cp_size < 1
        with pytest.raises(ValueError, match="cp_size must be >= 1"):
            st_module.streamindex_topk(
                q,
                w,
                cache,
                seq_lens,
                page_indices,
                cu_q_lens,
                dist,
                k=16,
                compression_ratio=1,
                cp_size=0,
            )

        # 2. cp_size > 1 and interleave_size % compression_ratio != 0
        with pytest.raises(ValueError, match="interleave_size .* must be a multiple"):
            st_module.streamindex_topk(
                q,
                w,
                cache,
                seq_lens,
                page_indices,
                cu_q_lens,
                dist,
                k=16,
                compression_ratio=4,
                cp_size=2,
                interleave_size=6,
            )

        # 3. cp_size > 1 and enable_early_exit
        with pytest.raises(
            NotImplementedError,
            match="enable_early_exit is not supported with cp_size > 1",
        ):
            st_module.streamindex_topk(
                q,
                w,
                cache,
                seq_lens,
                page_indices,
                cu_q_lens,
                dist,
                k=16,
                compression_ratio=1,
                cp_size=2,
                interleave_size=1,
                enable_early_exit=True,
            )

        # 4. enable_early_exit and return_scores
        with pytest.raises(
            NotImplementedError,
            match="return_scores is not supported with enable_early_exit",
        ):
            st_module.streamindex_topk(
                q,
                w,
                cache,
                seq_lens,
                page_indices,
                cu_q_lens,
                dist,
                k=16,
                compression_ratio=1,
                enable_early_exit=True,
                return_scores=True,
            )

        # 5. num_kv_pages_per_block or num_queries_per_block is None
        with pytest.raises(
            ValueError, match="num_kv_pages_per_block and num_queries_per_block"
        ):
            st_module.streamindex_topk(
                q,
                w,
                cache,
                seq_lens,
                page_indices,
                cu_q_lens,
                dist,
                k=16,
                compression_ratio=1,
                num_kv_pages_per_block=None,
                num_queries_per_block=(16, 16, 16),
            )

        # 6. compression_ratio < 1 or not power of 2
        with pytest.raises(ValueError, match="compression_ratio must be a power of 2"):
            st_module.streamindex_topk(
                q,
                w,
                cache,
                seq_lens,
                page_indices,
                cu_q_lens,
                dist,
                k=16,
                compression_ratio=3,
                num_kv_pages_per_block=(1, 1, 1),
                num_queries_per_block=(16, 16, 16),
            )

        # 7. buffer_count invalid length
        with pytest.raises(ValueError, match="buffer_count must be a 3-tuple"):
            st_module.streamindex_topk(
                q,
                w,
                cache,
                seq_lens,
                page_indices,
                cu_q_lens,
                dist,
                k=16,
                compression_ratio=1,
                num_kv_pages_per_block=(1, 1, 1),
                num_queries_per_block=(16, 16, 16),
                buffer_count=(2, 2),
            )

        # 8. chunk_tokens % decode_req_batch_size != 0
        with pytest.raises(ValueError, match="chunk_tokens .* must be a multiple"):
            st_module.streamindex_topk(
                q,
                w,
                cache,
                seq_lens,
                page_indices,
                cu_q_lens,
                dist,
                k=16,
                compression_ratio=1,
                num_kv_pages_per_block=(1, 1, 1),
                num_queries_per_block=(16, 16, 16),
                decode_req_batch_size=4,
                chunk_tokens=6,
            )

        # 9. SEQ_ALONG_LANE cache shape mismatch
        with pytest.raises(
            ValueError, match="SEQ_ALONG_LANE expects cache_kv of shape"
        ):
            st_module.streamindex_topk(
                q,
                w,
                cache,  # wrong shape for SEQ_ALONG_LANE
                seq_lens,
                page_indices,
                cu_q_lens,
                dist,
                k=16,
                compression_ratio=1,
                num_kv_pages_per_block=(1, 1, 1),
                num_queries_per_block=(16, 16, 16),
                kv_layout=config.KVLayout.SEQ_ALONG_LANE,
            )

        # 10. SEQ_ALONG_LANE sublane groups mismatch
        bad_sal_cache = jnp.zeros((2, 30, 4, 128), dtype=jnp.uint8)
        with (
            patch.object(st_module, "get_kv_cache_shape", return_value=(2, 30, 4, 128)),
            pytest.raises(ValueError, match="Inconsistent SEQ_ALONG_LANE cache"),
        ):
            st_module.streamindex_topk(
                q,
                w,
                bad_sal_cache,
                seq_lens,
                page_indices,
                cu_q_lens,
                dist,
                k=16,
                compression_ratio=1,
                num_kv_pages_per_block=(1, 1, 1),
                num_queries_per_block=(16, 16, 16),
                kv_layout=config.KVLayout.SEQ_ALONG_LANE,
            )

        # 11. bkv_sz % 128 != 0
        with pytest.raises(ValueError, match="must be a multiple of 128"):
            st_module.streamindex_topk(
                q,
                w,
                jnp.zeros((2, 8, 4, 256), dtype=jnp.uint8),  # page_size = 8*4 = 32
                seq_lens,
                page_indices,
                cu_q_lens,
                dist,
                k=16,
                compression_ratio=1,
                num_kv_pages_per_block=(1, 1, 1),  # bkv_sz = 32 * 1 = 32 != 0 % 128
                num_queries_per_block=(16, 16, 16),
            )

    def test_early_exit_fast_path(self):
        def fake_pallas_call(kernel_fn, *args, out_shape=None, **kwargs):
            def run(*call_args):
                return jnp.zeros(out_shape.shape, dtype=out_shape.dtype)

            return run

        def fake_sparsecore_topk(scores, k, **kwargs):
            idxs = jnp.zeros((scores.shape[0], k), dtype=jnp.int32)
            scs = jnp.zeros((scores.shape[0], k), dtype=jnp.int32)
            if kwargs.get("return_scores", False):
                return idxs, scs
            return idxs

        q = jnp.zeros((16, 4, 128), dtype=jnp.bfloat16)
        w = jnp.zeros((16, 4), dtype=jnp.float32)
        cache = jnp.zeros((2, 32, 4, 256), dtype=jnp.uint8)
        # max(seq_lens) = 8 <= k (16) triggers fast path
        seq_lens = jnp.array([8, 8], dtype=jnp.int32)
        page_indices = jnp.array([0, 1], dtype=jnp.int32)
        cu_q_lens = jnp.array([0, 8, 16], dtype=jnp.int32)
        dist = jnp.array([0, 0, 2], dtype=jnp.int32)

        with (
            patch(
                "jax.experimental.pallas.pallas_call",
                side_effect=fake_pallas_call,
            ),
            patch.object(
                st_module, "sparsecore_topk", side_effect=fake_sparsecore_topk
            ),
        ):
            res = st_module.streamindex_topk(
                q,
                w,
                cache,
                seq_lens,
                page_indices,
                cu_q_lens,
                dist,
                k=16,
                compression_ratio=1,
                num_kv_pages_per_block=(1, 1, 1),
                num_queries_per_block=(16, 16, 16),
                enable_early_exit=True,
            )
        assert res.shape == (16, 16)

    def test_common_path_mocked_execution(self):
        def fake_pallas_call(kernel_fn, *args, out_shape=None, **kwargs):
            def run(*call_args):
                # Simulated score matrix
                return jnp.zeros(out_shape.shape, dtype=out_shape.dtype)

            return run

        def fake_sparsecore_topk(scores, k, **kwargs):
            num_rows = scores.shape[0]
            idxs = jnp.zeros((num_rows, k), dtype=jnp.int32)
            scs = jnp.zeros((num_rows, k), dtype=jnp.int32)
            if kwargs.get("return_scores", False):
                return idxs, scs
            return idxs

        # 1. Single chunk, return_scores=False, buffer_count as int,
        # num_kv_pages as int, num_queries as int
        with (
            patch(
                "jax.experimental.pallas.pallas_call",
                side_effect=fake_pallas_call,
            ),
            patch.object(
                st_module, "sparsecore_topk", side_effect=fake_sparsecore_topk
            ),
        ):
            q = jnp.zeros((16, 4, 128), dtype=jnp.bfloat16)
            w = jnp.zeros((16, 4), dtype=jnp.float32)
            cache = jnp.zeros((2, 32, 4, 256), dtype=jnp.uint8)
            seq_lens = jnp.array([128, 128], dtype=jnp.int32)
            page_indices = jnp.array([0, 1], dtype=jnp.int32)
            cu_q_lens = jnp.array([0, 8, 16], dtype=jnp.int32)
            dist = jnp.array([1, 1, 2], dtype=jnp.int32)  # decode, prefill, mixed

            out = st_module.streamindex_topk(
                q,
                w,
                cache,
                seq_lens,
                page_indices,
                cu_q_lens,
                dist,
                k=16,
                compression_ratio=1,
                num_kv_pages_per_block=1,  # int
                num_queries_per_block=16,  # int
                buffer_count=3,  # int
                chunk_tokens=None,
                return_scores=False,
            )
            assert out.shape == (16, 16)

        # 2. Multi-chunk (3 chunks) with buffer_count=None, return_scores=True,
        # scores.shape[1] < k padding
        with (
            patch(
                "jax.experimental.pallas.pallas_call",
                side_effect=fake_pallas_call,
            ),
            patch.object(
                st_module, "sparsecore_topk", side_effect=fake_sparsecore_topk
            ),
        ):
            q32 = jnp.zeros((48, 4, 128), dtype=jnp.bfloat16)
            w32 = jnp.zeros((48, 4), dtype=jnp.float32)
            cache_sal = jnp.zeros((2, 33, 4, 128), dtype=jnp.uint8)
            seq_lens32 = jnp.array([128, 128], dtype=jnp.int32)
            page_indices32 = jnp.array([0, 1], dtype=jnp.int32)
            cu_q_lens32 = jnp.array([0, 24, 48], dtype=jnp.int32)
            dist32 = jnp.array([0, 0, 2], dtype=jnp.int32)

            out_idxs, out_scores = st_module.streamindex_topk(
                q32,
                w32,
                cache_sal,
                seq_lens32,
                page_indices32,
                cu_q_lens32,
                dist32,
                # k > num_sublanes_total * 128 (128) -> triggers BITS_NEG_INF padding
                k=256,
                compression_ratio=1,
                num_kv_pages_per_block=(1, 1, 1),
                num_queries_per_block=(16, 16, 16),
                buffer_count=None,  # None
                kv_layout=config.KVLayout.SEQ_ALONG_LANE,
                chunk_tokens=16,  # 48 // 16 = 3 chunks
                cp_size=2,
                cp_rank=0,
                interleave_size=1,
                return_scores=True,
            )
            assert out_idxs.shape == (48, 256)
            assert out_scores.shape == (48, 256)

        # 3. enable_early_exit with condition False (falls through to common path)
        with (
            patch(
                "jax.experimental.pallas.pallas_call",
                side_effect=fake_pallas_call,
            ),
            patch.object(
                st_module, "sparsecore_topk", side_effect=fake_sparsecore_topk
            ),
        ):
            out_fall = st_module.streamindex_topk(
                q,
                w,
                cache,
                seq_lens,  # max(seq_lens) = 128 > k=16 -> False condition in cond
                page_indices,
                cu_q_lens,
                dist,
                k=16,
                compression_ratio=1,
                num_kv_pages_per_block=(1, 1, 1),
                num_queries_per_block=(16, 16, 16),
                enable_early_exit=True,
            )
            assert out_fall.shape == (16, 16)

        # 4. cp_size > 1 with return_scores=False (hits line 1150: return out)
        with (
            patch(
                "jax.experimental.pallas.pallas_call",
                side_effect=fake_pallas_call,
            ),
            patch.object(
                st_module, "sparsecore_topk", side_effect=fake_sparsecore_topk
            ),
        ):
            out_cp_noscores = st_module.streamindex_topk(
                q,
                w,
                cache,
                seq_lens,
                page_indices,
                cu_q_lens,
                dist,
                k=16,
                compression_ratio=1,
                num_kv_pages_per_block=(1, 1, 1),
                num_queries_per_block=(16, 16, 16),
                cp_size=2,
                cp_rank=0,
                interleave_size=1,
                return_scores=False,
            )
            assert out_cp_noscores.shape == (16, 16)


# ==============================================================================
# 8. streamindex_topk.py DCP Wrapper Tests
# ==============================================================================


class TestStreamIndexTopKDcp:
    """100% full-module line and branch coverage on streamindex_topk_dcp()."""

    def test_dcp_validations(self):
        q = jnp.zeros((16, 4, 128), dtype=jnp.bfloat16)
        w = jnp.zeros((16, 4), dtype=jnp.float32)
        cache = jnp.zeros((2, 32, 4, 256), dtype=jnp.uint8)
        seq_lens = jnp.array([128, 128], dtype=jnp.int32)
        page_indices = jnp.array([0, 1], dtype=jnp.int32)
        cu_q_lens = jnp.array([0, 8, 16], dtype=jnp.int32)
        dist = jnp.array([0, 0, 2], dtype=jnp.int32)

        mesh = FakeMesh("dcp", 2)

        # 1. dcp_size <= 1
        with pytest.raises(
            ValueError, match="streamindex_topk_dcp requires dcp_size > 1"
        ):
            st_module.streamindex_topk_dcp(
                q,
                w,
                cache,
                seq_lens,
                page_indices,
                cu_q_lens,
                dist,
                mesh=mesh,
                k=16,
                compression_ratio=1,
                dcp_size=1,
                interleave_size=1,
            )

        # 2. dcp_axis_name not in mesh.axis_names
        bad_axis_mesh = FakeMesh("other_axis", 2)
        with pytest.raises(ValueError, match="has no 'dcp' axis"):
            st_module.streamindex_topk_dcp(
                q,
                w,
                cache,
                seq_lens,
                page_indices,
                cu_q_lens,
                dist,
                mesh=bad_axis_mesh,
                k=16,
                compression_ratio=1,
                dcp_size=2,
                interleave_size=1,
            )

        # 3. mesh.shape[dcp_axis_name] != dcp_size
        bad_size_mesh = FakeMesh("dcp", 4)
        with pytest.raises(ValueError, match="does not match mesh axis"):
            st_module.streamindex_topk_dcp(
                q,
                w,
                cache,
                seq_lens,
                page_indices,
                cu_q_lens,
                dist,
                mesh=bad_size_mesh,
                k=16,
                compression_ratio=1,
                dcp_size=2,
                interleave_size=1,
            )

        # 4. q.shape[0] % dcp_size != 0
        q_odd = jnp.zeros((15, 4, 128), dtype=jnp.bfloat16)
        w_odd = jnp.zeros((15, 4), dtype=jnp.float32)
        with pytest.raises(ValueError, match="must be divisible by dcp_size"):
            st_module.streamindex_topk_dcp(
                q_odd,
                w_odd,
                cache,
                seq_lens,
                page_indices,
                cu_q_lens,
                dist,
                mesh=mesh,
                k=16,
                compression_ratio=1,
                dcp_size=2,
                interleave_size=1,
            )

    def test_dcp_local_execution(self):
        mesh = FakeMesh("dcp", 2)

        def fake_shard_map(
            fn, mesh=None, in_specs=None, out_specs=None, check_vma=False
        ):
            def runner(*args):
                return fn(*args)

            return runner

        def fake_streamindex_topk(*args, **kwargs):
            q_arg = args[0]
            k_arg = kwargs["k"]
            idxs = jnp.zeros((q_arg.shape[0], k_arg), dtype=jnp.int32)
            scores = jnp.zeros((q_arg.shape[0], k_arg), dtype=jnp.int32)
            return idxs, scores

        def fake_sparsecore_topk(scores, k, **kwargs):
            return jnp.zeros((scores.shape[0], k), dtype=jnp.int32)

        def fake_resolve_owned_winners(local_idxs, slots, k, dcp_rank):
            return local_idxs

        with (
            patch("jax.shard_map", side_effect=fake_shard_map),
            patch.object(
                st_module, "streamindex_topk", side_effect=fake_streamindex_topk
            ),
            patch(
                "jax.lax.all_gather",
                side_effect=lambda x, *args, **kwargs: jnp.concatenate([x, x], axis=1),
            ),
            patch.object(
                st_module, "sparsecore_topk", side_effect=fake_sparsecore_topk
            ),
            patch.object(
                st_module.dcp_sc_compact,
                "resolve_owned_winners",
                side_effect=fake_resolve_owned_winners,
            ),
            patch.object(st_module, "cp_rank_as_data", return_value=jnp.int32(0)),
        ):
            q = jnp.zeros((16, 4, 128), dtype=jnp.bfloat16)
            w = jnp.zeros((16, 4), dtype=jnp.float32)
            cache = jnp.zeros((2, 32, 4, 256), dtype=jnp.uint8)
            seq_lens = jnp.array([128, 128], dtype=jnp.int32)
            page_indices = jnp.array([0, 1], dtype=jnp.int32)
            cu_q_lens = jnp.array([0, 8, 16], dtype=jnp.int32)
            dist = jnp.array([0, 0, 2], dtype=jnp.int32)

            res = st_module.streamindex_topk_dcp(
                q,
                w,
                cache,
                seq_lens,
                page_indices,
                cu_q_lens,
                dist,
                mesh=mesh,
                k=16,
                compression_ratio=1,
                dcp_size=2,
                interleave_size=1,
                num_kv_pages_per_block=(1, 1, 1),
                num_queries_per_block=(16, 16, 16),
            )
            assert res.shape == (16, 16)
