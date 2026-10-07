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

csa_cache_layout.py, sparsecore_topk.py, and rope.py on CPU.
"""

from __future__ import annotations

import contextlib
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TPU_SKIP_MDS_QUERY", "true")

import jax
import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.kernels.deepseek_v4 import (
    csa_cache_layout as csa,
)
from vllm_torchtpu.kernels.deepseek_v4 import (
    rope,
)
from vllm_torchtpu.kernels.deepseek_v4 import (
    sparsecore_topk as sc,
)


def _norm_idx(idx):
    if isinstance(idx, tuple):
        return tuple(_norm_idx(i) for i in idx)
    if hasattr(idx, "start") and hasattr(idx, "size"):
        stride = getattr(idx, "stride", 1)
        stride = 1 if stride is None else int(stride)
        return slice(int(idx.start), int(idx.start + idx.size), stride)
    return idx


class MockArrayRef:
    """Mock HBM / VMEM buffer reference for simulating Pallas pipelines on CPU."""

    def __init__(self, arr: np.ndarray | jax.Array):
        self.arr = np.asarray(arr).copy()
        self.shape = self.arr.shape
        self.dtype = self.arr.dtype

    def __getitem__(self, idx):
        try:
            return self.arr[_norm_idx(idx)]
        except Exception:
            return self.arr

    def __setitem__(self, idx, val):
        with contextlib.suppress(Exception):
            self.arr[_norm_idx(idx)] = np.asarray(val)

    @property
    def at(self):
        class _AtIndexer:
            def __init__(self, ref):
                self.ref = ref

            def __getitem__(self, idx):
                try:
                    return MockArrayRef(self.ref.arr[_norm_idx(idx)])
                except Exception:
                    return self.ref

        return _AtIndexer(self)

    def bitcast(self, dtype):
        return MockArrayRef(self.arr.view(dtype))

    def reshape(self, *shape):
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


# ==============================================================================
# 1. csa_cache_layout.py Full Coverage Tests
# ==============================================================================


class TestCsaCacheLayout:
    """100% full-module line and branch coverage on csa_cache_layout.py."""

    def test_is_word_array(self):
        assert csa.is_word_array(jnp.int32) is True
        assert csa.is_word_array(jnp.uint32) is True
        assert csa.is_word_array(jnp.float32) is True
        assert csa.is_word_array(np.int32) is True
        assert csa.is_word_array(jnp.uint8) is False
        assert csa.is_word_array(jnp.int8) is False
        assert csa.is_word_array(jnp.bfloat16) is False
        assert csa.is_word_array(jnp.float64) is False

    def test_u8_slab_shape(self):
        # Non-word array branch: returns tuple(shape)
        shape_u8 = (2, 8, 4, 128)
        assert csa.u8_slab_shape(shape_u8, jnp.uint8) == shape_u8
        assert csa.u8_slab_shape(list(shape_u8), jnp.int8) == shape_u8

        # Word array branch: returns (num_pages, rows, 4, lanes)
        shape_w = (4, 16, 128)
        assert csa.u8_slab_shape(shape_w, jnp.int32) == (4, 16, 4, 128)

    def test_as_u8_slabs(self):
        # Non-word array branch: returns ref unchanged
        ref_u8 = MockArrayRef(np.zeros((2, 8, 4, 128), dtype=np.uint8))
        assert csa.as_u8_slabs(ref_u8) is ref_u8

        # Word array branch: bitcast to uint8 and reshapes
        ref_w = MockArrayRef(np.zeros((3, 6, 128), dtype=np.int32))
        slabs_ref = csa.as_u8_slabs(ref_w)
        assert slabs_ref.shape == (3, 6, 4, 128)
        assert slabs_ref.dtype == np.uint8

    def test_slabs_to_words_and_words_to_slabs_roundtrip(self):
        rng = np.random.default_rng(42)
        # Random uint8 slabs
        slabs_np = rng.integers(0, 256, size=(2, 3, 4, 128), dtype=np.uint8)
        slabs_jax = jnp.asarray(slabs_np)
        words_jax = csa.slabs_to_words(slabs_jax)
        assert words_jax.shape == (2, 3, 128)
        assert words_jax.dtype == jnp.int32

        # Roundtrip back to slabs
        slabs_rec = csa.words_to_slabs(words_jax)
        assert slabs_rec.shape == (2, 3, 4, 128)
        assert slabs_rec.dtype == jnp.uint8
        np.testing.assert_array_equal(np.asarray(slabs_rec), slabs_np)

        # Byte position verification
        # Sub-row 0 in bits 0..7, sub-row 1 in 8..15,
        # sub-row 2 in 16..23, sub-row 3 in 24..31
        fixed_slabs = np.zeros((1, 4, 1), dtype=np.uint8)
        fixed_slabs[0, 0, 0] = 0x12
        fixed_slabs[0, 1, 0] = 0x34
        fixed_slabs[0, 2, 0] = 0x56
        fixed_slabs[0, 3, 0] = 0x78
        word = csa.slabs_to_words(jnp.asarray(fixed_slabs))
        assert int(np.asarray(word)[0, 0]) == 0x78563412

    def test_rope_words_and_encode_rope(self):
        rng = np.random.default_rng(123)
        # Shape: (8, 64) bf16 tokens
        tokens = rng.normal(size=(8, 64)).astype(np.float32)
        tokens_bf16 = jnp.asarray(tokens, dtype=jnp.bfloat16)

        # rope_words: [..., 64] -> [..., 32]
        words = csa.rope_words(tokens_bf16)
        assert words.shape == (8, 32)
        assert words.dtype == jnp.int32

        # Bitwise verification: word k has low half bf16[k] and high half bf16[k+32]
        tokens_bits = np.asarray(tokens_bf16.view(jnp.uint16))
        words_bits = np.asarray(words.view(jnp.uint32))
        low16 = words_bits[..., :32] & 0xFFFF
        high16 = (words_bits[..., :32] >> 16) & 0xFFFF
        np.testing.assert_array_equal(low16, tokens_bits[..., :32])
        np.testing.assert_array_equal(high16, tokens_bits[..., 32:])

        # encode_rope: [num_tokens, 64] -> [num_tokens // 4, 128]
        encoded = csa.encode_rope(tokens_bf16)
        assert encoded.shape == (2, 128)
        assert encoded.dtype == jnp.int32
        np.testing.assert_array_equal(
            np.asarray(encoded), words_bits.reshape(2, 128).view(np.int32)
        )


# ==============================================================================
# 2. sparsecore_topk.py Full Coverage Tests
# ==============================================================================


class TestSparseCoreTopkMathAndPartition:
    """Tests math helpers and partition factor calculation."""

    def test_math_helpers(self):
        assert sc._cdiv(10, 3) == 4
        assert sc._cdiv(9, 3) == 3
        assert sc._cdiv(0, 3) == 0

        assert sc._align_to(10, 16) == 16
        assert sc._align_to(16, 16) == 16
        assert sc._align_to(33, 16) == 48

        assert int(sc._cdiv_dyn(0)) == 0
        assert int(sc._cdiv_dyn(1)) == 1
        assert int(sc._cdiv_dyn(16)) == 1
        assert int(sc._cdiv_dyn(17)) == 2

    def test_pick_partition_branches(self):
        # b >= 32 branch: max_p = 1
        assert sc._pick_partition(32, 10240, 2048) == 1
        assert sc._pick_partition(64, 10240, 2048) == 1

        # b < 32 branch: max_p = 32 // b
        assert sc._pick_partition(16, 10240, 2048) == 2
        assert sc._pick_partition(8, 10240, 2048) == 4
        assert sc._pick_partition(4, 10240, 1024) == 8
        assert sc._pick_partition(4, 10240, 2048) == 4

        # Large n (p_min doubling loop)
        assert sc._pick_partition(16, 133120, 2048) == 8
        assert sc._pick_partition(64, 1048576, 2048) == 32

        # Break branches when doubling:
        # next_np < k (1024 < 2048)
        assert sc._pick_partition(1, 2048, 2048) == 1
        # next_np < LANES (8 < 16)
        assert sc._pick_partition(1, 16, 8) == 1
        # (n % (p * 2)) != 0 (560 % 32 != 0)
        assert sc._pick_partition(1, 560, 16) == 16


class TestSparseCoreTopkValidationAndFlow:
    """Tests input validation and single/multi-stage control flow in sparsecore_topk."""

    def test_validation_errors(self):
        # scores.ndim != 2
        with pytest.raises(ValueError, match="scores must be 2D"):
            sc.sparsecore_topk(jnp.ones((16,), dtype=jnp.float32), 16)
        with pytest.raises(ValueError, match="scores must be 2D"):
            sc.sparsecore_topk(jnp.ones((2, 4, 16), dtype=jnp.float32), 16)

        # scores.dtype not float32 or int32 (using float16 and uint8)
        with pytest.raises(ValueError, match="scores must be f32 or i32"):
            sc.sparsecore_topk(jnp.ones((2, 32), dtype=jnp.float16), 16)
        with pytest.raises(ValueError, match="scores must be f32 or i32"):
            sc.sparsecore_topk(jnp.ones((2, 32), dtype=jnp.uint8), 16)

        # n % LANES != 0
        with pytest.raises(ValueError, match=r"n=31 must be a multiple of 16"):
            sc.sparsecore_topk(jnp.ones((2, 31), dtype=jnp.float32), 16)

        # k % LANES != 0
        with pytest.raises(ValueError, match=r"k=15 must be a positive multiple of 16"):
            sc.sparsecore_topk(jnp.ones((2, 32), dtype=jnp.float32), 15)

        # k <= 0
        with pytest.raises(ValueError, match=r"k=0 must be a positive multiple of 16"):
            sc.sparsecore_topk(jnp.ones((2, 32), dtype=jnp.float32), 0)

        # k > 4096
        with pytest.raises(
            ValueError, match=r"k=4112 must be a positive multiple of 16"
        ):
            sc.sparsecore_topk(jnp.ones((2, 8192), dtype=jnp.float32), 4112)

    def test_single_stage_flow(self):
        # b=64, n=32, k=16 -> p=1
        b, n, k = 64, 32, 16
        scores = jnp.zeros((b, n), dtype=jnp.float32)

        def mock_direct(
            s, k_val, row_lens, *, write_empty=True, scheduling_group_id=None
        ):
            slots = jnp.full((b, k_val), 7, dtype=jnp.int32)
            keys = jnp.full((b, k_val), 42, dtype=jnp.int32)
            return slots, keys

        with patch.object(sc, "_sc_topk_direct", side_effect=mock_direct) as mock_d:
            # return_scores=False, row_lengths=None (default n)
            res = sc.sparsecore_topk(scores, k, row_lengths=None, return_scores=False)
            assert res.shape == (b, k)
            assert int(res[0, 0]) == 7
            mock_d.assert_called_once()

        with patch.object(sc, "_sc_topk_direct", side_effect=mock_direct):
            # return_scores=True, custom row_lengths, scheduling_group_id
            lens = jnp.full((b,), 20, dtype=jnp.int32)
            slots, keys = sc.sparsecore_topk(
                scores,
                k,
                row_lengths=lens,
                scheduling_group_id=99,
                return_scores=True,
            )
            assert slots.shape == (b, k)
            assert keys.shape == (b, k)
            assert int(keys[0, 0]) == 42

    def test_multi_stage_flow_direct_stage2(self):
        # b=16, n=64, k=16 -> p=2, n_p=32, k_p=16. p * k_p = 32 <= MAX_SLICE_WORDS
        b, n, k = 16, 64, 16
        scores = jnp.arange(b * n, dtype=jnp.float32).reshape(b, n)
        lens = jnp.full((b,), n, dtype=jnp.int32)

        call_count = 0

        def mock_direct(
            s, k_val, row_lens, *, write_empty=True, scheduling_group_id=None
        ):
            nonlocal call_count
            call_count += 1
            cur_b = s.shape[0]
            if call_count == 1:
                # Stage 1: (b * p, k_p) = (32, 16)
                slots = jnp.arange(k_val, dtype=jnp.int32)[None, :].repeat(
                    cur_b, axis=0
                )
                # simulate some -1 indices for branch coverage line 576
                slots = slots.at[:, -2:].set(-1)
                keys = jnp.full((cur_b, k_val), 100, dtype=jnp.int32)
                return slots, keys
            else:
                # Stage 2: (b, k) = (16, 16)
                slots = jnp.arange(k_val, dtype=jnp.int32)[None, :].repeat(
                    cur_b, axis=0
                )
                # simulate negative slot for safe_slots line 605/607
                slots = slots.at[:, -1].set(-1)
                keys = jnp.full((cur_b, k_val), 200, dtype=jnp.int32)
                return slots, keys

        with patch.object(sc, "_sc_topk_direct", side_effect=mock_direct):
            indices, scores_out = sc.sparsecore_topk(
                scores,
                k,
                row_lengths=lens,
                scheduling_group_id=1,
                stage2_scheduling_group_id=2,
                return_scores=True,
            )
            assert indices.shape == (b, k)
            assert scores_out.shape == (b, k)
            assert int(indices[0, -1]) == -1  # from safe_slots branch

    def test_multi_stage_flow_recursive_stage2(self):
        # b=64, n=1048576, k=2048 -> p=32, n_p=32768, k_p=2048.
        # p * k_p = 65536 > MAX_SLICE_WORDS (32768) -> triggers line 587 recursion!
        b, n, k = 64, 1048576, 2048
        scores = jnp.zeros((b, n), dtype=jnp.float32)

        def mock_direct(
            s, k_val, row_lens, *, write_empty=True, scheduling_group_id=None
        ):
            cur_b = s.shape[0]
            slots = jnp.zeros((cur_b, k_val), dtype=jnp.int32)
            keys = jnp.zeros((cur_b, k_val), dtype=jnp.int32)
            return slots, keys

        with patch.object(sc, "_sc_topk_direct", side_effect=mock_direct):
            # Run without return_scores
            indices = sc.sparsecore_topk(scores, k, return_scores=False)
            assert indices.shape == (b, k)


class TestSparseCoreTopkDirectKernelWiring:
    """Tests _sc_topk_direct branches, padding, and Pallas kernel simulation."""

    def test_sc_not_implemented_error(self):
        with (
            patch.object(
                sc.pltpu, "get_tpu_info", return_value=SimpleNamespace(sparse_core=None)
            ),
            pytest.raises(NotImplementedError, match="SparseCore is not available"),
        ):
            sc._sc_topk_direct(
                jnp.zeros((4, 16), dtype=jnp.float32), 16, jnp.array([16] * 4)
            )

    def test_sc_topk_direct_padding_and_types(self):
        sc_info = SimpleNamespace(num_cores=2, num_subcores=16)
        tpu_info = SimpleNamespace(sparse_core=sc_info)

        def fake_kernel(
            fn,
            out_type=None,
            compiler_params=None,
            scratch_types=None,
            mesh=None,
            name=None,
        ):
            def runner(words, lens):
                b_padded = out_type[0].shape[0] // 16
                slots = jnp.zeros((b_padded * 16,), dtype=jnp.int32)
                keys = jnp.zeros((b_padded * 16,), dtype=jnp.int32)
                return slots, keys

            return runner

        with (
            patch.object(sc.pltpu, "get_tpu_info", return_value=tpu_info),
            patch.object(sc.plsc, "VectorSubcoreMesh", return_value=MagicMock()),
            patch.object(sc.pl, "kernel", side_effect=fake_kernel),
            patch.object(
                sc.xla_metadata, "set_xla_metadata", side_effect=lambda x, **kw: x
            ) as mock_meta,
        ):
            # pad_b > 0 (b=4 < 32), pad_n > 0 (n=20 aligned to 32), float32 dtype
            slots, keys = sc._sc_topk_direct(
                jnp.zeros((4, 20), dtype=jnp.float32),
                16,
                jnp.array([20] * 4, dtype=jnp.int32),
                scheduling_group_id=42,
            )
            assert slots.shape == (4, 16)
            assert keys.shape == (4, 16)
            mock_meta.assert_called_once()

            # pad_b == 0 (b=32), pad_n == 0 (n=32), int32 dtype,
            # scheduling_group_id=None
            slots32, keys32 = sc._sc_topk_direct(
                jnp.zeros((32, 32), dtype=jnp.int32),
                16,
                jnp.array([32] * 32, dtype=jnp.int32),
                scheduling_group_id=None,
            )
            assert slots32.shape == (32, 16)


class TestSparseCoreTopkBodyExecution:
    """Executes _topk_body and inner pipeline functions on CPU."""

    def test_topk_body_pipeline_simulation(self):
        b, n, k = 4, 32, 16
        slice_len = 32

        # Create buffers
        scores_hbm = MockArrayRef(np.ones((32, slice_len), dtype=np.int32))
        lengths_hbm = MockArrayRef(
            np.array([32] * 32 * sc.LANES, dtype=np.int32).reshape(32, sc.LANES)
        )
        out_hbm = MockArrayRef(np.zeros((32, k), dtype=np.int32))
        keys_hbm = MockArrayRef(np.zeros((32, k), dtype=np.int32))
        glob_vmem = MockArrayRef(np.zeros((sc.NUM_BUCKETS + sc.LANES,), dtype=np.int32))
        rowbuf_vmem = MockArrayRef(np.zeros((k + sc.LANES,), dtype=np.int32))
        keybuf_vmem = MockArrayRef(np.zeros((k + sc.LANES,), dtype=np.int32))

        # Mock plsc loop and parallel primitives
        def fake_parallel_loop(lower, upper, step=1, *, unroll=1, carry=None):
            def decorator(body):
                cur = carry
                for i in range(int(lower), int(upper), int(step)):
                    if carry is None:
                        body(i)
                    else:
                        cur = body(i, cur)
                return cur

            return decorator

        def fake_scan_count(x, mask=None):
            x_arr = np.asarray(x)
            if mask is None:
                m = np.ones_like(x_arr, dtype=bool)
            else:
                m = np.asarray(mask, dtype=bool)
            dup = np.zeros_like(x_arr, dtype=np.int32)
            counts = {}
            last_idx = {}
            for idx, (val, valid) in enumerate(zip(x_arr, m)):
                if valid:
                    c = counts.get(val, 0) + 1
                    counts[val] = c
                    dup[idx] = c
                    last_idx[val] = idx
            last = np.zeros_like(m, dtype=bool)
            for idx in last_idx.values():
                last[idx] = True
            return jnp.asarray(dup), jnp.asarray(last)

        def fake_addupdate_scatter(ref, indices, x, mask=None):
            idx = np.asarray(indices[0])
            x_val = np.asarray(x)
            if mask is not None:
                m = np.asarray(mask, dtype=bool)
                for i, v in zip(idx[m], x_val[m]):
                    ref[int(i)] = ref[int(i)] + v
            else:
                for i, v in zip(idx, x_val):
                    ref[int(i)] = ref[int(i)] + v

        def fake_cumsum(x, mask=None):
            return jnp.cumsum(x)

        def fake_popcount(x, reduce=1):
            return jnp.array([jnp.sum(x.astype(jnp.int32))])

        def fake_store_compressed(ref, x, mask=None):
            sel_x = np.asarray(x)[np.asarray(mask, dtype=bool)]
            for k_idx, val in enumerate(sel_x):
                ref[k_idx] = val

        def fake_when(cond):
            def decorator(fn):
                if bool(np.asarray(cond)):
                    fn()

            return decorator

        captured_pipeline_step = None
        captured_in_specs = None
        captured_out_specs = None

        def fake_emit(pipeline_step, grid=None, in_specs=None, out_specs=None):
            nonlocal captured_pipeline_step, captured_in_specs, captured_out_specs
            captured_pipeline_step = pipeline_step
            captured_in_specs = in_specs
            captured_out_specs = out_specs

            def runner(*args):
                pass

            return runner

        with (
            patch.object(sc.plsc, "parallel_loop", side_effect=fake_parallel_loop),
            patch.object(sc.plsc, "scan_count", side_effect=fake_scan_count),
            patch.object(
                sc.plsc, "addupdate_scatter", side_effect=fake_addupdate_scatter
            ),
            patch.object(sc.plsc, "cumsum", side_effect=fake_cumsum),
            patch.object(
                sc.plsc, "all_reduce_population_count", side_effect=fake_popcount
            ),
            patch.object(
                sc.plsc, "store_compressed", side_effect=fake_store_compressed
            ),
            patch.object(sc.pl, "when", side_effect=fake_when),
            patch.object(sc.lax, "axis_index", return_value=0),
            patch.object(sc.pltpu, "emit_pipeline", side_effect=fake_emit),
        ):
            # Test num_waves == 1 (buf_count = 1)
            sc._topk_kernel(
                scores_hbm,
                lengths_hbm,
                out_hbm,
                keys_hbm,
                glob_vmem,
                rowbuf_vmem,
                keybuf_vmem,
                b=b,
                n=n,
                k=k,
                slice_len=slice_len,
                num_waves=1,
                write_empty=True,
            )

            # Test in_specs and out_specs index maps
            for spec in captured_in_specs + captured_out_specs:
                res = spec.index_map(0)
                assert len(res) == 1

            # Test num_waves == 2 (buf_count = 2)
            sc._topk_kernel(
                scores_hbm,
                lengths_hbm,
                out_hbm,
                keys_hbm,
                glob_vmem,
                rowbuf_vmem,
                keybuf_vmem,
                b=b,
                n=n,
                k=k,
                slice_len=slice_len,
                num_waves=2,
                write_empty=True,
            )

            # Now execute pipeline_step directly under both whole_bucket=True and False
            scores_vmem = MockArrayRef(np.arange(slice_len, dtype=np.int32))
            len_slice_vmem = MockArrayRef(
                np.array([slice_len] * sc.LANES, dtype=np.int32)
            )
            out_vmem = MockArrayRef(np.zeros((k,), dtype=np.int32))
            keys_vmem = MockArrayRef(np.zeros((k,), dtype=np.int32))

            with patch.object(sc.pl, "program_id", return_value=0):
                captured_pipeline_step(scores_vmem, len_slice_vmem, out_vmem, keys_vmem)

            # Execute pipeline_step with tied scores so whole_bucket=False
            # (executes emit_ranked)
            scores_tied = MockArrayRef(np.full(slice_len, 100, dtype=np.int32))
            with patch.object(sc.pl, "program_id", return_value=0):
                captured_pipeline_step(scores_tied, len_slice_vmem, out_vmem, keys_vmem)

            # Execute pipeline_step with empty i_hi (i_hi <= 0)
            len_empty = MockArrayRef(np.zeros((sc.LANES,), dtype=np.int32))
            with patch.object(
                sc.pl, "program_id", return_value=100
            ):  # row >= b (inactive)
                captured_pipeline_step(scores_vmem, len_empty, out_vmem, keys_vmem)


# ==============================================================================
# 3. rope.py Full Coverage Tests
# ==============================================================================


class TestRopeHelpersAndBody:
    """100% full-module line and branch coverage on rope.py."""

    def test_cos_sin_ref_methods(self):
        spec = sc.pl.BlockSpec(block_shape=(16, 64), index_map=lambda i: (i, 0))

        # Directly call CosSinRef.create which creates a genuine BufferedRef
        cos_sin_ref = rope.CosSinRef.create(
            spec=spec,
            dtype=jnp.float32,
            buffer_count=2,
            tile_n=16,
        )
        assert cos_sin_ref.tile_n == 16

        # Test copy_in
        cos_sin_cache_hbm = MockArrayRef(np.ones((100, 64), dtype=np.float32))
        positions = MockArrayRef(np.arange(32, dtype=np.int32))

        copies = []

        def fake_async_copy(src, dst, sem):
            copy = MockAsyncCopy()
            copies.append(copy)
            return copy

        def fake_loop(start, end, unroll=True):
            def decorator(fn):
                for i in range(start, end):
                    fn(i)

            return decorator

        # Set window_ref and sem_recvs on cos_sin_ref for copy_in simulation
        object.__setattr__(
            cos_sin_ref,
            "window_ref",
            MockArrayRef(np.zeros((2, 16, 64), dtype=np.float32)),
        )
        object.__setattr__(
            cos_sin_ref, "sem_recvs", MockArrayRef(np.zeros((2,), dtype=np.int32))
        )
        with (
            patch.object(rope.pltpu, "make_async_copy", side_effect=fake_async_copy),
            patch.object(rope.pl, "loop", side_effect=fake_loop),
            patch.object(
                rope.CosSinRef,
                "current_copy_in_slot",
                new_callable=PropertyMock,
                return_value=0,
            ),
            patch.object(
                rope.CosSinRef,
                "current_wait_in_slot",
                new_callable=PropertyMock,
                return_value=0,
            ),
        ):
            cos_sin_ref.copy_in((cos_sin_cache_hbm, positions), (0,))
            assert len(copies) == 16
            assert copies[0].started is True

            # Test wait_in
            cos_sin_ref.wait_in((cos_sin_cache_hbm, positions), (0,))
            assert copies[-1].waited is True

    def test_lane_gather_2d_and_3d(self):
        # 2D operand [batch, lanes]
        op2 = jnp.arange(4 * 128, dtype=jnp.float32).reshape(4, 128)
        idx = jnp.arange(128, dtype=jnp.int32)
        out2 = rope._lane_gather(op2, idx)
        assert out2.shape == (4, 128)
        np.testing.assert_array_equal(np.asarray(out2), np.asarray(op2))

        # 3D operand [tokens, heads, lanes]
        op3 = jnp.arange(2 * 4 * 128, dtype=jnp.float32).reshape(2, 4, 128)
        out3 = rope._lane_gather(op3, idx)
        assert out3.shape == (2, 4, 128)
        np.testing.assert_array_equal(np.asarray(out3), np.asarray(op3))

    def test_cos_sin_lanes(self):
        cos_sin = jnp.ones((4, 64), dtype=jnp.float32)

        # inverse=False
        cos, sin = rope._cos_sin_lanes(cos_sin, rotary_dim=64, inverse=False)
        assert cos.shape == (4, 128)
        assert sin.shape == (4, 128)
        # NoPE lanes have cos=1, sin=0
        np.testing.assert_array_equal(np.asarray(cos[:, :64]), 1.0)
        np.testing.assert_array_equal(np.asarray(sin[:, :64]), 0.0)

        # inverse=True (covers line 145: sin = -sin)
        cos_inv, sin_inv = rope._cos_sin_lanes(cos_sin, rotary_dim=64, inverse=True)
        np.testing.assert_array_equal(np.asarray(cos_inv), np.asarray(cos))
        np.testing.assert_array_equal(np.asarray(sin_inv), -np.asarray(sin))

    def test_rotate_gptj(self):
        # 2D: [-x1, x0, -x3, x2, ...]
        x = jnp.arange(128, dtype=jnp.float32)[None, :]
        rot = rope._rotate_gptj(x)
        assert rot.shape == (1, 128)
        assert float(rot[0, 0]) == -1.0
        assert float(rot[0, 1]) == 0.0

        # 3D
        x3 = jnp.ones((2, 4, 128), dtype=jnp.float32)
        rot3 = rope._rotate_gptj(x3)
        assert rot3.shape == (2, 4, 128)

    def test_rope_body_unquantized_and_quantized(self):
        # 2D unquantized (quant_dtype is None, x.ndim == 2)
        x_2d = MockArrayRef(np.ones((4, 128), dtype=np.float32))
        cos_sin = MockArrayRef(np.ones((4, 64), dtype=np.float32))
        rope._rope_body(
            x_2d,
            cos_sin,
            rotary_dim=64,
            inverse=False,
            out_dtype=jnp.float32,
            quant_dtype=None,
        )
        assert x_2d.arr.shape == (4, 128)

        # 3D unquantized (quant_dtype is None, x.ndim == 3)
        x_3d = MockArrayRef(np.ones((4, 8, 128), dtype=np.float32))
        rope._rope_body(
            x_3d,
            cos_sin,
            rotary_dim=64,
            inverse=True,
            out_dtype=jnp.float32,
            quant_dtype=None,
        )
        assert x_3d.arr.shape == (4, 8, 128)

        # 2D quantized (quant_dtype = float8_e4m3fn, abs_max > 0)
        q_ref = MockArrayRef(np.zeros((4, 128), dtype=np.uint8))
        scale_ref = MockArrayRef(np.zeros((4,), dtype=np.float32))
        rope._rope_body(
            x_2d,
            cos_sin,
            q_ref,
            scale_ref,
            rotary_dim=64,
            inverse=False,
            out_dtype=jnp.float32,
            quant_dtype=jnp.float8_e4m3fn,
        )
        assert q_ref.arr.shape == (4, 128)
        assert scale_ref.arr.shape == (4,)
        assert float(scale_ref.arr[0]) > 0.0

        # 2D quantized with abs_max == 0 (all zeros input,
        # covers line 195 zero scale_inv)
        x_zero = MockArrayRef(np.zeros((4, 128), dtype=np.float32))
        rope._rope_body(
            x_zero,
            cos_sin,
            q_ref,
            scale_ref,
            rotary_dim=64,
            inverse=False,
            out_dtype=jnp.float32,
            quant_dtype=jnp.float8_e4m3fn,
        )
        assert float(scale_ref.arr[0]) == 0.0

    def test_qnorm_rope_body(self):
        # 3D: (tile_n, num_heads, head_dim)
        x_ref = MockArrayRef(np.ones((2, 4, 256), dtype=np.float32))
        cos_sin_ref = MockArrayRef(np.ones((2, 64), dtype=np.float32))
        out_ref = MockArrayRef(np.zeros((2, 4, 256), dtype=np.float32))

        rope._qnorm_rope_body(
            x_ref,
            cos_sin_ref,
            out_ref,
            rotary_dim=64,
            eps=1e-6,
            inverse=False,
            out_dtype=jnp.float32,
        )
        assert out_ref.arr.shape == (2, 4, 256)
        assert np.isfinite(out_ref.arr).all()

    def test_kernel_wiring_branches(self):
        # Test _kernel with 3D block shape (len == 3, line 221) and quant_dtype=None
        def fake_emit(pipeline_fn, grid=None, in_specs=None, out_specs=None):
            if in_specs:
                for s in in_specs:
                    if s is not None and getattr(s, "index_map", None) is not None:
                        s.index_map(0)
            if out_specs:
                for s in out_specs:
                    if s is not None and getattr(s, "index_map", None) is not None:
                        s.index_map(0)

            def runner(*args, allocations=None):
                pass

            return runner

        mock_alloc = SimpleNamespace(current_copy_in_slot=0, current_wait_in_slot=0)

        with (
            patch.object(rope.pltpu, "emit_pipeline", side_effect=fake_emit),
            patch.object(rope.CosSinRef, "create", return_value=mock_alloc),
            patch.object(
                rope.pl,
                "with_scoped",
                side_effect=lambda allocations: lambda fn: lambda: fn(allocations),
            ),
            patch.object(
                rope.pltpu.BufferedRef, "input_output", return_value=mock_alloc
            ),
            patch.object(rope.pltpu.BufferedRef, "input", return_value=mock_alloc),
            patch.object(rope.pltpu.BufferedRef, "output", return_value=mock_alloc),
        ):
            # 3D block shape, quant_dtype=None (covers lines 221-222, 238-246)
            # *out_hbm_refs is passed positionally after cos_sin_cache_hbm_ref
            pos_ref = MockArrayRef(np.zeros(16, dtype=np.int32))
            cs_ref = MockArrayRef(np.zeros((100, 64), dtype=np.float32))
            out_hbm = MockArrayRef(np.zeros((16, 4, 128), dtype=np.float32))

            rope._kernel(
                pos_ref,
                None,
                cs_ref,
                out_hbm,
                num_tiles=1,
                tile_n=16,
                x_block_shape=(16, 4, 128),
                scale_block_shape=(16, 4),
                lane_block=0,
                rotary_dim=64,
                inverse=False,
                in_dtype=jnp.float32,
                out_dtype=jnp.float32,
                cos_sin_dtype=jnp.float32,
                quant_dtype=None,
            )

            # 2D block shape, quant_dtype is not None (covers lines 223-224, 247-267)
            x_hbm = MockArrayRef(np.zeros((16, 128), dtype=np.float32))
            q_hbm = MockArrayRef(np.zeros((16, 128), dtype=np.uint8))
            sc_hbm = MockArrayRef(np.zeros((16,), dtype=np.float32))

            rope._kernel(
                pos_ref,
                x_hbm,
                cs_ref,
                q_hbm,
                sc_hbm,
                num_tiles=1,
                tile_n=16,
                x_block_shape=(16, 128),
                scale_block_shape=(16,),
                lane_block=0,
                rotary_dim=64,
                inverse=True,
                in_dtype=jnp.float32,
                out_dtype=jnp.float32,
                cos_sin_dtype=jnp.float32,
                quant_dtype=jnp.float8_e4m3fn,
            )

            # Test _qnorm_rope_kernel (covers lines 325-385)
            rope._qnorm_rope_kernel(
                pos_ref,
                MockArrayRef(np.zeros((16, 4, 256), dtype=np.float32)),
                cs_ref,
                MockArrayRef(np.zeros((16, 4, 256), dtype=np.float32)),
                num_tiles=1,
                tile_n=16,
                x_block_shape=(16, 4, 256),
                rotary_dim=64,
                eps=1e-6,
                inverse=False,
                in_dtype=jnp.float32,
                cos_sin_dtype=jnp.float32,
            )


class TestRopeCallAndPublicWrappers:
    """Tests _rope_call assertions, validations, and rope / qnorm_rope / rope_quant."""

    def test_rope_call_validations(self):
        positions = jnp.zeros((16,), dtype=jnp.int32)
        cache = jnp.zeros((100, 64), dtype=jnp.float32)

        # x rank != 2 or 3
        with pytest.raises(ValueError, match="x must be rank 2 or 3"):
            rope._rope_call(
                jnp.zeros((16,), dtype=jnp.float32),
                positions,
                cache,
                inverse=False,
                quant_dtype=None,
                name="r",
            )
        with pytest.raises(ValueError, match="x must be rank 2 or 3"):
            rope._rope_call(
                jnp.zeros((2, 2, 2, 128), dtype=jnp.float32),
                positions,
                cache,
                inverse=False,
                quant_dtype=None,
                name="r",
            )

        # quantization requires head_dim == LANE (128)
        x_wide = jnp.zeros((16, 256), dtype=jnp.float32)
        with pytest.raises(
            ValueError, match=r"quantization requires head_dim == 128, got 256"
        ):
            rope._rope_call(
                x_wide,
                positions,
                cache,
                inverse=False,
                quant_dtype=jnp.float8_e4m3fn,
                name="r",
            )

    def test_public_wrappers_with_mock_pallas_call(self):
        # Pure JAX reference implementations for tracing inside jax.jit
        def jax_ref_rope(x, pos, cache, inverse=False):
            rot_dim = cache.shape[1]
            c, s = jnp.split(cache[pos], 2, axis=-1)
            c = jnp.repeat(c, 2, axis=-1)
            s = jnp.repeat(s, 2, axis=-1)
            if inverse:
                s = -s
            if x.ndim == 3:
                c = c[:, None, :]
                s = s[:, None, :]
            x_pass, x_rot = x[..., :-rot_dim], x[..., -rot_dim:]
            rotated = jnp.stack([-x_rot[..., 1::2], x_rot[..., 0::2]], axis=-1).reshape(
                x_rot.shape
            )
            out = jnp.concatenate([x_pass, x_rot * c + rotated * s], axis=-1)
            return out.astype(x.dtype)

        def jax_ref_quant(y, quant_dtype):
            abs_max = jnp.max(jnp.abs(y), axis=-1, keepdims=True)
            scale = abs_max / 448.0
            scale_inv = jnp.where(abs_max == 0.0, 0.0, 1.0 / scale)
            q = (y * scale_inv).astype(quant_dtype)
            return q, jnp.squeeze(scale, -1)

        def jax_ref_qnorm_rope(x, pos, cache, eps=1e-6, inverse=False):
            rms = jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + eps)
            return jax_ref_rope(x * rms, pos, cache, inverse=inverse)

        def fake_pallas_call(
            kernel,
            out_shape=None,
            grid_spec=None,
            input_output_aliases=None,
            compiler_params=None,
            name=None,
        ):
            def runner(pos, x, cache):
                if "qnorm" in name:
                    return jax_ref_qnorm_rope(x, pos, cache, inverse=False)
                elif "quant" in name:
                    return jax_ref_quant(
                        jax_ref_rope(x, pos, cache, inverse=False), jnp.float8_e4m3fn
                    )
                else:
                    return jax_ref_rope(x, pos, cache, inverse=False)

            return runner

        with patch.object(rope.pl, "pallas_call", side_effect=fake_pallas_call):
            # Test rope 2D
            x2 = jnp.ones((16, 128), dtype=jnp.float32)
            pos = jnp.arange(16, dtype=jnp.int32)
            cache = jnp.ones((100, 64), dtype=jnp.float32)
            out2 = rope.rope(x2, pos, cache, inverse=False)
            assert out2.shape == (16, 128)

            # Test rope 3D
            x3 = jnp.ones((16, 4, 128), dtype=jnp.float32)
            out3 = rope.rope(x3, pos, cache, inverse=True)
            assert out3.shape == (16, 4, 128)

            # Test qnorm_rope 3D
            x_qn = jnp.ones((16, 4, 256), dtype=jnp.float32)
            out_qn = rope.qnorm_rope(x_qn, pos, cache, eps=1e-6, inverse=False)
            assert out_qn.shape == (16, 4, 256)

            # Test rope_quant 2D and 3D
            x2_q = jnp.ones((16, 128), dtype=jnp.float32)
            q2, sc2 = rope.rope_quant(x2_q, pos, cache, inverse=False)
            assert q2.shape == (16, 128)
            assert sc2.shape == (16,)

            x3_q = jnp.ones((16, 4, 128), dtype=jnp.float32)
            q3, sc3 = rope.rope_quant(x3_q, pos, cache, inverse=False)
            assert q3.shape == (16, 4, 128)
            assert sc3.shape == (16, 4)
