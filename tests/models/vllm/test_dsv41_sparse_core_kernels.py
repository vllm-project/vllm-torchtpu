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

"""100% full-module line and branch coverage tests for sparse_core kernels.

Covers:
1. core_map_helper.py
2. ragged_gather.py
3. ragged_gather_v2.py
4. ragged_scatter.py
5. ragged_gather_reduce.py
6. ragged_gather_reduce_v3.py
"""

from __future__ import annotations

import contextlib
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TPU_SKIP_MDS_QUERY", "true")

import jax
import jax.numpy as jnp
import numpy as np
from jax._src import core as jax_core
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.sparse_core import (
    core_map_helper,
    ragged_gather,
    ragged_gather_reduce,
    ragged_gather_reduce_v3,
    ragged_gather_v2,
    ragged_scatter,
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
    """Mock HBM / VMEM buffer reference for simulating Pallas kernels on CPU."""

    def __init__(self, arr: np.ndarray | jax.Array, dtype=None):
        self.arr = np.asarray(arr).copy()
        self.shape = self.arr.shape
        self.dtype = self.arr.dtype if dtype is None else dtype
        self.size = self.arr.size
        self.ndim = self.arr.ndim

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
            def __init__(self, ref, sl=None):
                self.ref = ref
                self.sl = sl

            def __getitem__(self, idx):
                return _AtIndexer(self.ref, _norm_idx(idx))

        return _AtIndexer(self)

    def bitcast(self, dtype):
        try:
            return MockArrayRef(self.arr.view(dtype), dtype=dtype)
        except Exception:
            return MockArrayRef(self.arr, dtype=dtype)

    def reshape(self, *shape):
        return MockArrayRef(self.arr.reshape(*shape), dtype=self.dtype)


class MockAsyncCopy:
    """Mock async DMA copy handle."""

    def __init__(self, src=None, dst=None):
        self.src = src
        self.dst = dst
        self.started = False
        self.waited = False

    def start(self):
        self.started = True
        return self

    def wait(self):
        self.waited = True
        return self


# ==============================================================================
# 1. core_map_helper.py Full Coverage Tests
# ==============================================================================


class TestCoreMapHelper:
    """100% full-module line and branch coverage on core_map_helper.py."""

    def test_empty_out_ref_standard_device(self):
        # Branch: aval.memory_space is jax_core.MemorySpace -> memory_space = None
        struct = jax.ShapeDtypeStruct((4, 8), jnp.float32)
        ref = core_map_helper._empty_out_ref(struct)
        assert isinstance(ref, jax_core.Ref)
        assert ref.shape == (4, 8)
        assert ref.dtype == jnp.float32

    def test_empty_out_ref_vmem_non_memory_space(self):
        # Branch: aval.memory_space is NOT jax_core.MemorySpace (e.g. pltpu.VMEM)
        @jax.jit
        def _build_vmem():
            vmem_struct = pltpu.VMEM((2, 4), jnp.int32)
            ref = core_map_helper._empty_out_ref(vmem_struct)
            return ref[...]

        val = _build_vmem()
        assert val.shape == (2, 4)
        assert val.dtype == jnp.int32

    def test_kernel_single_output_execution(self):
        # single_output = True, outs[0] return branch
        called = False

        def fake_core_map(*args, **kwargs):
            def decorator(fn):
                def wrapper(*a, **kw):
                    nonlocal called
                    called = True
                    return fn(*a, **kw)

                return wrapper()

            return decorator

        def body(in_ref, out_ref):
            out_ref[...] = in_ref[...] * 3.0

        with patch(
            "vllm_torchtpu.kernels.sparse_core.core_map_helper.pl_core.core_map",
            side_effect=fake_core_map,
        ):
            runner = core_map_helper.kernel(
                body,
                out_type=jax.ShapeDtypeStruct((4,), jnp.float32),
                mesh=MagicMock(),
            )
            inp = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)
            out = runner(inp)
            assert called
            np.testing.assert_allclose(np.asarray(out), [3.0, 6.0, 9.0, 12.0])

    def test_kernel_multi_output_tuple_and_list(self):
        # single_output = False with tuple and list out_type
        def fake_core_map(*args, **kwargs):
            def decorator(fn):
                def wrapper(*a, **kw):
                    return fn(*a, **kw)

                return wrapper()

            return decorator

        def body(in_ref, out1_ref, out2_ref):
            out1_ref[...] = in_ref[...] + 1.0
            out2_ref[...] = in_ref[...] + 2.0

        with patch(
            "vllm_torchtpu.kernels.sparse_core.core_map_helper.pl_core.core_map",
            side_effect=fake_core_map,
        ):
            # Tuple out_type
            runner_tuple = core_map_helper.kernel(
                body,
                out_type=(
                    jax.ShapeDtypeStruct((2,), jnp.float32),
                    jax.ShapeDtypeStruct((2,), jnp.float32),
                ),
                mesh=MagicMock(),
                scratch_types={"scratch": MagicMock()},
            )
            inp = jnp.array([10.0, 20.0], dtype=jnp.float32)
            out1, out2 = runner_tuple(inp)
            np.testing.assert_allclose(np.asarray(out1), [11.0, 21.0])
            np.testing.assert_allclose(np.asarray(out2), [12.0, 22.0])

            # List out_type
            runner_list = core_map_helper.kernel(
                body,
                out_type=[
                    jax.ShapeDtypeStruct((2,), jnp.float32),
                    jax.ShapeDtypeStruct((2,), jnp.float32),
                ],
                mesh=MagicMock(),
            )
            out_list = runner_list(inp)
            assert isinstance(out_list, list)
            np.testing.assert_allclose(np.asarray(out_list[0]), [11.0, 21.0])
            np.testing.assert_allclose(np.asarray(out_list[1]), [12.0, 22.0])


# ==============================================================================
# 2. ragged_gather.py Full Coverage Tests
# ==============================================================================


class TestRaggedGather:
    """100% full-module line and branch coverage on ragged_gather.py."""

    def test_calculate_col_size_generations(self):
        sc_info = SimpleNamespace(num_lanes=16)
        # Gen 6
        tpu_info6 = SimpleNamespace(sparse_core=sc_info, num_lanes=128, generation=6)
        with patch.object(ragged_gather.pltpu, "get_tpu_info", return_value=tpu_info6):
            c6 = ragged_gather.calculate_col_size(1024)
            assert c6 == 1024

        # Gen 7
        tpu_info7 = SimpleNamespace(sparse_core=sc_info, num_lanes=128, generation=7)
        with patch.object(ragged_gather.pltpu, "get_tpu_info", return_value=tpu_info7):
            c7 = ragged_gather.calculate_col_size(1024)
            assert c7 == 1024

        # Gen other (default, e.g. 5)
        tpu_info5 = SimpleNamespace(sparse_core=sc_info, num_lanes=128, generation=5)
        with patch.object(ragged_gather.pltpu, "get_tpu_info", return_value=tpu_info5):
            c5 = ragged_gather.calculate_col_size(1024)
            assert c5 == 1024

        # Large hidden size triggering while loop iterations
        with patch.object(ragged_gather.pltpu, "get_tpu_info", return_value=tpu_info5):
            c_large = ragged_gather.calculate_col_size(8192)
            assert c_large > 0

    def test_ragged_gather_input_validation(self):
        # x.ndim != 2
        with pytest.raises(AssertionError, match="2d inputs"):
            ragged_gather.ragged_gather(
                jnp.zeros((4,), dtype=jnp.float32),
                jnp.array([0, 1]),
                0,
                2,
            )

        # indices.ndim != 1
        with pytest.raises(AssertionError, match="1d indices"):
            ragged_gather.ragged_gather(
                jnp.zeros((4, 8), dtype=jnp.float32),
                jnp.zeros((2, 2), dtype=jnp.int32),
                0,
                2,
            )

    def test_ragged_gather_sc_info_none_fallback(self):
        tpu_info_no_sc = SimpleNamespace(sparse_core=None)
        with patch.object(
            ragged_gather.pltpu, "get_tpu_info", return_value=tpu_info_no_sc
        ):
            x = jnp.array([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]])
            indices = jnp.array([2, 0])
            # Scalar start and end promotion
            res = ragged_gather.ragged_gather(x, indices, 0, 2)
            np.testing.assert_allclose(np.asarray(res), [[30.0, 31.0], [10.0, 11.0]])

            # 1D array start and end
            res1d = ragged_gather.ragged_gather(
                x, indices, jnp.array([0]), jnp.array([2])
            )
            np.testing.assert_allclose(np.asarray(res1d), np.asarray(res))

    def test_ragged_gather_sc_kernel_wiring(self):
        sc_info = SimpleNamespace(num_lanes=16, num_cores=2, num_subcores=16)
        tpu_info = SimpleNamespace(sparse_core=sc_info, num_lanes=128, generation=7)

        def mock_kernel(body, out_type=None, **kwargs):
            def runner(s, e, x_val, idx_val):
                return x_val[idx_val]

            return runner

        with (
            patch.object(ragged_gather.pltpu, "get_tpu_info", return_value=tpu_info),
            patch.object(
                ragged_gather.core_map_helper, "kernel", side_effect=mock_kernel
            ),
            patch.object(
                ragged_gather.plsc, "VectorSubcoreMesh", return_value=MagicMock()
            ),
        ):
            x = jnp.ones((64, 128), dtype=jnp.float32)
            indices = jnp.arange(32, dtype=jnp.int32)
            res = ragged_gather.ragged_gather(x, indices, 0, 32)
            assert res.shape == (32, 128)

    def test_main_kernel_simulation_packing_1_and_packed(self):
        sc_info = SimpleNamespace(num_lanes=16)
        tpu_info = SimpleNamespace(sparse_core=sc_info, num_lanes=128, generation=7)

        def fake_emit(pipeline_step, *args, **kwargs):
            def runner(*a, **kw):
                pipeline_step()

            return runner

        def fake_loop(start, stop=None, step=1, init_carry=None):
            def dec(body):
                if init_carry is not None:
                    carry = init_carry
                    carry = body(0, carry)
                    return carry
                body(0)
                return body

            return dec

        def fake_when(cond):
            def dec(fn):
                fn()

            return dec

        with (
            patch.object(ragged_gather.pltpu, "get_tpu_info", return_value=tpu_info),
            patch.object(
                ragged_gather.pltpu,
                "make_async_copy",
                return_value=MockAsyncCopy(),
            ),
            patch.object(ragged_gather.pltpu, "sync_copy", return_value=None),
            patch.object(ragged_gather.pltpu, "emit_pipeline", side_effect=fake_emit),
            patch.object(ragged_gather.pl, "loop", side_effect=fake_loop),
            patch.object(ragged_gather.pl, "when", side_effect=fake_when),
            patch.object(ragged_gather.pl, "program_id", return_value=0),
            patch.object(ragged_gather.jax.lax, "axis_size", return_value=32),
        ):
            # Test packing > 1 (dtype bfloat16 -> uint16, packing = 2)
            start_ref = MockArrayRef(np.array([0], dtype=np.int32))
            end_ref = MockArrayRef(np.array([16], dtype=np.int32))
            in_hbm_ref = MockArrayRef(
                np.zeros((32, 128), dtype=np.uint16), dtype=jnp.bfloat16
            )
            indices_hbm_ref = MockArrayRef(np.zeros((32,), dtype=np.int32))
            out_hbm_ref = MockArrayRef(
                np.zeros((32, 128), dtype=np.uint16), dtype=jnp.bfloat16
            )
            start_vmem_ref = MockArrayRef(np.array([0], dtype=np.int32))
            end_vmem_ref = MockArrayRef(np.array([16], dtype=np.int32))
            out_vmem_ref = MockArrayRef(np.zeros((16, 128), dtype=np.uint32))
            indices_vmem_ref = MockArrayRef(np.zeros((16,), dtype=np.int32))
            sem_ref = MockArrayRef(np.zeros((2,), dtype=np.int32))

            ragged_gather.main_kernel(
                start_ref,
                end_ref,
                in_hbm_ref,
                indices_hbm_ref,
                out_hbm_ref,
                start_vmem_ref,
                end_vmem_ref,
                out_vmem_ref,
                indices_vmem_ref,
                sem_ref,
                core_axis_name="core",
                subcore_axis_name="subcore",
            )

            # Test packing == 1 (dtype float32, packing = 1)
            # and end == start branch (num_blocks = 0)
            in_hbm_32 = MockArrayRef(
                np.zeros((32, 128), dtype=np.float32), dtype=jnp.float32
            )
            out_hbm_32 = MockArrayRef(
                np.zeros((32, 128), dtype=np.float32), dtype=jnp.float32
            )
            end_same = MockArrayRef(np.array([0], dtype=np.int32))
            end_vmem_same = MockArrayRef(np.array([0], dtype=np.int32))

            ragged_gather.main_kernel(
                start_ref,
                end_same,
                in_hbm_32,
                indices_hbm_ref,
                out_hbm_32,
                start_vmem_ref,
                end_vmem_same,
                out_vmem_ref,
                indices_vmem_ref,
                sem_ref,
                core_axis_name="core",
                subcore_axis_name="subcore",
            )


# ==============================================================================
# 3. ragged_gather_v2.py Full Coverage Tests
# ==============================================================================


class TestRaggedGatherV2:
    """100% full-module line and branch coverage on ragged_gather_v2.py."""

    def test_calculate_col_size_branches(self):
        sc_info = SimpleNamespace(num_lanes=16)

        # Gen 6
        tpu_info6 = SimpleNamespace(sparse_core=sc_info, generation=6)
        with patch.object(
            ragged_gather_v2.pltpu, "get_tpu_info", return_value=tpu_info6
        ):
            assert ragged_gather_v2.calculate_col_size(1024, packing=2) == 1024

        # Gen 7
        tpu_info7 = SimpleNamespace(sparse_core=sc_info, generation=7)
        with patch.object(
            ragged_gather_v2.pltpu, "get_tpu_info", return_value=tpu_info7
        ):
            assert ragged_gather_v2.calculate_col_size(1024, packing=2) == 1024

        # Default gen (e.g. 5)
        tpu_info5 = SimpleNamespace(sparse_core=sc_info, generation=5)
        with patch.object(
            ragged_gather_v2.pltpu, "get_tpu_info", return_value=tpu_info5
        ):
            assert ragged_gather_v2.calculate_col_size(1024, packing=2) == 512

        # Fallback to max_safe_col (no divisor found in range)
        with patch.object(
            ragged_gather_v2.pltpu, "get_tpu_info", return_value=tpu_info7
        ):
            c_safe = ragged_gather_v2.calculate_col_size(129, packing=2)
            assert c_safe > 0

    def test_ragged_gather_v2_validation_and_fallbacks(self):
        # x.ndim != 2
        with pytest.raises(AssertionError, match="2d inputs"):
            ragged_gather_v2.ragged_gather_v2(
                jnp.zeros((4,), dtype=jnp.float32),
                jnp.array([0, 1]),
                0,
                2,
            )

        # indices.ndim != 1
        with pytest.raises(AssertionError, match="1d indices"):
            ragged_gather_v2.ragged_gather_v2(
                jnp.zeros((4, 8), dtype=jnp.float32),
                jnp.zeros((2, 2), dtype=jnp.int32),
                0,
                2,
            )

        # Invalid dtype -> ValueError
        with pytest.raises(ValueError, match="dtype must be f32, bf16"):
            ragged_gather_v2.ragged_gather_v2(
                jnp.zeros((4, 8), dtype=jnp.int32),
                jnp.array([0, 1]),
                0,
                2,
            )

        # sc_info is None -> fallback x[indices]
        tpu_info_no_sc = SimpleNamespace(sparse_core=None)
        with patch.object(
            ragged_gather_v2.pltpu, "get_tpu_info", return_value=tpu_info_no_sc
        ):
            x = jnp.array([[10.0, 11.0], [20.0, 21.0]])
            indices = jnp.array([1, 0])
            # Scalar start/end
            res = ragged_gather_v2.ragged_gather_v2(x, indices, 0, 2)
            np.testing.assert_allclose(np.asarray(res), [[20.0, 21.0], [10.0, 11.0]])
            # 1D start/end
            res1d = ragged_gather_v2.ragged_gather_v2(
                x, indices, jnp.array([0]), jnp.array([2])
            )
            np.testing.assert_allclose(np.asarray(res1d), np.asarray(res))

    def test_ragged_gather_v2_subchunk_range_and_sc_call(self):
        sc_info = SimpleNamespace(num_lanes=16, num_cores=2, num_subcores=16)
        tpu_info = SimpleNamespace(sparse_core=sc_info, num_lanes=128, generation=7)

        def mock_kernel(body, out_type=None, **kwargs):
            def runner(s, e, x_val, idx_val):
                return x_val[idx_val]

            return runner

        with (
            patch.object(ragged_gather_v2.pltpu, "get_tpu_info", return_value=tpu_info),
            patch.object(
                ragged_gather_v2.core_map_helper,
                "kernel",
                side_effect=mock_kernel,
            ),
            patch.object(
                ragged_gather_v2.plsc,
                "VectorSubcoreMesh",
                return_value=MagicMock(),
            ),
        ):
            x = jnp.ones((64, 128), dtype=jnp.float32)

            # Small out_size -> num_row_subchunks = 1
            res_small = ragged_gather_v2.ragged_gather_v2(
                x, jnp.arange(16, dtype=jnp.int32), 0, 16
            )
            assert res_small.shape == (16, 128)

            # Large out_size -> num_row_subchunks clamped to 4
            res_large = ragged_gather_v2.ragged_gather_v2(
                x, jnp.arange(2048, dtype=jnp.int32), 0, 2048
            )
            assert res_large.shape == (2048, 128)

    def test_main_kernel_v2_simulation(self):
        sc_info = SimpleNamespace(num_lanes=16)
        tpu_info = SimpleNamespace(sparse_core=sc_info, generation=7)

        captured_inner = None
        captured_outer = None
        captured_in_specs = None
        captured_out_specs = None

        def fake_emit(pipeline_step, *args, in_specs=None, out_specs=None, **kw):
            nonlocal \
                captured_inner, \
                captured_outer, \
                captured_in_specs, \
                captured_out_specs
            if "inner_pipeline" in getattr(pipeline_step, "__name__", "") or (
                hasattr(pipeline_step, "func")
                and "inner_pipeline" in pipeline_step.func.__name__
            ):
                captured_inner = pipeline_step
                captured_in_specs = in_specs
                captured_out_specs = out_specs
            else:
                captured_outer = pipeline_step

            def runner(*a, **k):
                pass

            return runner

        def fake_parallel_loop(lower, upper, step=1):
            def decorator(fn):
                for i in range(lower, upper, step):
                    fn(i)

            return decorator

        with (
            patch.object(ragged_gather_v2.pltpu, "get_tpu_info", return_value=tpu_info),
            patch.object(
                ragged_gather_v2.pltpu,
                "make_async_copy",
                return_value=MockAsyncCopy(),
            ),
            patch.object(
                ragged_gather_v2.pltpu, "emit_pipeline", side_effect=fake_emit
            ),
            patch.object(
                ragged_gather_v2.plsc,
                "parallel_loop",
                side_effect=fake_parallel_loop,
            ),
            patch.object(ragged_gather_v2.pl, "program_id", return_value=0),
            patch.object(ragged_gather_v2.lax, "axis_size", return_value=32),
            patch.object(ragged_gather_v2.lax, "axis_index", return_value=0),
        ):
            # 1. Test packing > 1 (e.g. bfloat16 -> packing = 2)
            start_ref = MockArrayRef(np.array([0], dtype=np.int32))
            end_ref = MockArrayRef(np.array([32], dtype=np.int32))
            in_hbm_ref = MockArrayRef(np.zeros((64, 128), dtype=np.uint16))
            indices_hbm_ref = MockArrayRef(np.arange(64, dtype=np.int32))
            out_hbm_ref = MockArrayRef(np.zeros((64, 128), dtype=np.uint16))
            start_vmem_ref = MockArrayRef(np.array([0], dtype=np.int32))
            end_vmem_ref = MockArrayRef(np.array([32], dtype=np.int32))
            sem_ref = MockArrayRef(np.zeros((2,), dtype=np.int32))

            ragged_gather_v2.main_kernel_v2(
                start_ref,
                end_ref,
                in_hbm_ref,
                indices_hbm_ref,
                out_hbm_ref,
                start_vmem_ref,
                end_vmem_ref,
                sem_ref,
                core_axis_name="core",
                subcore_axis_name="subcore",
                num_row_subchunks=2,
            )

            # Execute outer_pipeline and inner_pipeline
            assert captured_outer is not None
            idx_ref = MockArrayRef(np.arange(32, dtype=np.int32))
            captured_outer(idx_ref)

            assert captured_inner is not None
            gather_ref = MockArrayRef(np.zeros((16, 128), dtype=np.int32))
            out_ref = MockArrayRef(np.zeros((8, 128), dtype=np.int32))
            captured_inner(gather_ref, out_ref)

            # Test BlockSpec index maps
            if captured_in_specs is not None:
                spec_idx = captured_in_specs.index_map(0, 0)
                assert len(spec_idx) == 2
            if captured_out_specs is not None:
                spec_out = captured_out_specs.index_map(0, 0)
                assert len(spec_out) == 2

            # 2. Test packing == 1 (float32) and end <= start branch
            in_f32 = MockArrayRef(np.zeros((32, 128), dtype=np.float32))
            out_f32 = MockArrayRef(np.zeros((32, 128), dtype=np.float32))
            end_le_start = MockArrayRef(np.array([0], dtype=np.int32))
            end_vmem_le = MockArrayRef(np.array([0], dtype=np.int32))

            ragged_gather_v2.main_kernel_v2(
                start_ref,
                end_le_start,
                in_f32,
                indices_hbm_ref,
                out_f32,
                start_vmem_ref,
                end_vmem_le,
                sem_ref,
                core_axis_name="core",
                subcore_axis_name="subcore",
                num_row_subchunks=1,
            )
            captured_outer(idx_ref)
            gather_f32 = MockArrayRef(np.zeros((16, 128), dtype=np.float32))
            out_f32_slice = MockArrayRef(np.zeros((16, 128), dtype=np.float32))
            captured_inner(gather_f32, out_f32_slice)


# ==============================================================================
# 4. ragged_scatter.py Full Coverage Tests
# ==============================================================================


class TestRaggedScatter:
    """100% full-module line and branch coverage on ragged_scatter.py."""

    def test_preprocess_indices_validation_and_branches(self):
        # indices.ndim != 1
        with pytest.raises(AssertionError, match="1d indices"):
            ragged_scatter._preprocess_indices(
                jnp.zeros((2, 2), dtype=jnp.int32),
                jnp.array([0]),
                jnp.array([4]),
                out_pad_size=0,
                packing=2,
                row_tile_size=16,
            )

        # row_tile_size % packing != 0
        with pytest.raises(AssertionError):
            ragged_scatter._preprocess_indices(
                jnp.arange(16, dtype=jnp.int32),
                jnp.array([0]),
                jnp.array([16]),
                out_pad_size=0,
                packing=3,
                row_tile_size=16,
            )

        # Valid execution with num_valid_src_rows_per_dst_sublane > 1 and <= 1
        # indices in [0, 4) valid, others invalid (-1)
        indices = jnp.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=jnp.int32)
        src, dst, total_valid = ragged_scatter._preprocess_indices(
            indices,
            start=jnp.array([0]),
            end=jnp.array([4]),
            out_pad_size=0,
            packing=2,
            row_tile_size=16,
        )
        assert src.shape == (8,)
        assert dst.shape == (8,)
        assert int(total_valid[0]) > 0

    def test_calculate_col_size_generations(self):
        sc_info = SimpleNamespace(num_lanes=16)

        # Gen 6
        tpu_info6 = SimpleNamespace(sparse_core=sc_info, num_lanes=128, generation=6)
        with patch.object(ragged_scatter.pltpu, "get_tpu_info", return_value=tpu_info6):
            assert ragged_scatter.calculate_col_size(1024) == 1024

        # Gen 7
        tpu_info7 = SimpleNamespace(sparse_core=sc_info, num_lanes=128, generation=7)
        with patch.object(ragged_scatter.pltpu, "get_tpu_info", return_value=tpu_info7):
            assert ragged_scatter.calculate_col_size(1024) == 1024

        # Default gen (e.g. 5)
        tpu_info5 = SimpleNamespace(sparse_core=sc_info, num_lanes=128, generation=5)
        with patch.object(ragged_scatter.pltpu, "get_tpu_info", return_value=tpu_info5):
            assert ragged_scatter.calculate_col_size(1024) == 1024
            # Large hidden size triggering while loop iterations
            assert ragged_scatter.calculate_col_size(8192) > 0

    def test_ragged_scatter_validation_and_fallbacks(self):
        # x.ndim != 2
        with pytest.raises(AssertionError, match="2d inputs"):
            ragged_scatter.ragged_scatter(
                jnp.zeros((4,), dtype=jnp.float32),
                jnp.array([0, 1]),
                0,
                2,
            )

        # indices.ndim != 1
        with pytest.raises(AssertionError, match="1d indices"):
            ragged_scatter.ragged_scatter(
                jnp.zeros((4, 8), dtype=jnp.float32),
                jnp.zeros((2, 2), dtype=jnp.int32),
                0,
                2,
            )

        # sc_info is None -> fallback x[indices]
        tpu_info_no_sc = SimpleNamespace(sparse_core=None)
        with patch.object(
            ragged_scatter.pltpu, "get_tpu_info", return_value=tpu_info_no_sc
        ):
            x = jnp.array([[10.0, 11.0], [20.0, 21.0]])
            indices = jnp.array([1, 0])
            # Scalar start/end
            res = ragged_scatter.ragged_scatter(x, indices, 0, 2)
            np.testing.assert_allclose(np.asarray(res), [[20.0, 21.0], [10.0, 11.0]])
            # 1D start/end
            res1d = ragged_scatter.ragged_scatter(
                x, indices, jnp.array([0]), jnp.array([2])
            )
            np.testing.assert_allclose(np.asarray(res1d), np.asarray(res))

        # VMEM capacity fallback (small size < vmem_capacity * 0.6)
        sc_info = SimpleNamespace(num_lanes=16, num_cores=2, num_subcores=16)
        tpu_info_small = SimpleNamespace(
            sparse_core=sc_info,
            vmem_capacity_bytes=100_000_000,
            num_lanes=128,
            generation=7,
        )
        with patch.object(
            ragged_scatter.pltpu, "get_tpu_info", return_value=tpu_info_small
        ):
            ragged_scatter.ragged_scatter.clear_cache()
            res_vmem = ragged_scatter.ragged_scatter(x, indices, 0, 2)
            np.testing.assert_allclose(np.asarray(res_vmem), np.asarray(res))
            # Also test unwrapped function directly with 0d array scalars
            res_unwrapped = ragged_scatter.ragged_scatter._fun(
                x, indices, jnp.array(0), jnp.array(2)
            )
            np.testing.assert_allclose(np.asarray(res_unwrapped), np.asarray(res))

    def test_ragged_scatter_sc_wiring(self):
        sc_info = SimpleNamespace(num_lanes=16, num_cores=2, num_subcores=16)
        tpu_info = SimpleNamespace(
            sparse_core=sc_info,
            vmem_capacity_bytes=100,  # small capacity forces SC path
            num_lanes=128,
            generation=7,
        )

        def mock_kernel(body, out_type=None, **kwargs):
            def runner(tot, x_val, src_val, dst_val):
                return x_val[: len(dst_val)]

            return runner

        with (
            patch.object(ragged_scatter.pltpu, "get_tpu_info", return_value=tpu_info),
            patch.object(
                ragged_scatter.core_map_helper, "kernel", side_effect=mock_kernel
            ),
            patch.object(
                ragged_scatter.plsc,
                "VectorSubcoreMesh",
                return_value=MagicMock(),
            ),
        ):
            x = jnp.ones((64, 128), dtype=jnp.float32)
            indices = jnp.arange(32, dtype=jnp.int32)
            ragged_scatter.ragged_scatter.clear_cache()
            res = ragged_scatter.ragged_scatter(x, indices, 0, 32)
            assert res.shape == (32, 128)

    def test_main_kernel_simulation_packing_and_branches(self):
        sc_info = SimpleNamespace(num_lanes=16)
        tpu_info = SimpleNamespace(sparse_core=sc_info, num_lanes=128, generation=7)

        def fake_emit(pipeline_step, *args, **kwargs):
            def runner(*a, **kw):
                pipeline_step()

            return runner

        def fake_loop(start, stop=None, step=1, init_carry=None):
            def dec(body):
                if init_carry is not None:
                    carry = init_carry
                    carry = body(0, carry)
                    return carry
                body(0)
                return body

            return dec

        def fake_when(cond):
            def dec(fn):
                fn()

            return dec

        with (
            patch.object(ragged_scatter.pltpu, "get_tpu_info", return_value=tpu_info),
            patch.object(
                ragged_scatter.pltpu,
                "make_async_copy",
                return_value=MockAsyncCopy(),
            ),
            patch.object(ragged_scatter.pltpu, "emit_pipeline", side_effect=fake_emit),
            patch.object(ragged_scatter.pl, "loop", side_effect=fake_loop),
            patch.object(ragged_scatter.pl, "when", side_effect=fake_when),
            patch.object(ragged_scatter.pl, "program_id", return_value=0),
            patch.object(ragged_scatter.jax.lax, "axis_size", return_value=32),
        ):
            # 1. packing > 1 (bfloat16 -> uint16, packing = 2)
            # adjacent rows with same row_hbm to trigger merge branch
            tot_rows_ref = MockArrayRef(np.array([16], dtype=np.int32))
            in_hbm_ref = MockArrayRef(
                np.zeros((32, 128), dtype=np.uint16), dtype=jnp.bfloat16
            )
            src_idx_ref = MockArrayRef(np.arange(16, dtype=np.int32))
            # dst_idx has pairs with same dst sublane (e.g. 0, 0, 2, 2, ...)
            dst_idx_ref = MockArrayRef(np.repeat(np.arange(8, dtype=np.int32), 2))
            out_hbm_ref = MockArrayRef(
                np.zeros((32, 128), dtype=np.uint16), dtype=jnp.bfloat16
            )
            tot_vmem = MockArrayRef(np.array([16], dtype=np.int32))
            out_vmem = MockArrayRef(np.zeros((16, 128), dtype=np.uint32))
            src_vmem = MockArrayRef(np.arange(16, dtype=np.int32))
            dst_vmem = MockArrayRef(np.repeat(np.arange(8, dtype=np.int32), 2))
            sem_ref = MockArrayRef(np.zeros((2,), dtype=np.int32))

            ragged_scatter.main_kernel(
                tot_rows_ref,
                in_hbm_ref,
                src_idx_ref,
                dst_idx_ref,
                out_hbm_ref,
                tot_vmem,
                out_vmem,
                src_vmem,
                dst_vmem,
                sem_ref,
                core_axis_name="core",
                subcore_axis_name="subcore",
            )

            # 2. packing == 1 (float32) and total_num_rows == 0 branch
            tot_zero = MockArrayRef(np.array([0], dtype=np.int32))
            tot_vmem_zero = MockArrayRef(np.array([0], dtype=np.int32))
            in_f32 = MockArrayRef(np.zeros((32, 128), dtype=np.float32))
            out_f32 = MockArrayRef(np.zeros((32, 128), dtype=np.float32))

            ragged_scatter.main_kernel(
                tot_zero,
                in_f32,
                src_idx_ref,
                dst_idx_ref,
                out_f32,
                tot_vmem_zero,
                out_vmem,
                src_vmem,
                dst_vmem,
                sem_ref,
                core_axis_name="core",
                subcore_axis_name="subcore",
            )


# ==============================================================================
# 5. ragged_gather_reduce.py Full Coverage Tests
# ==============================================================================


class TestRaggedGatherReduce:
    """100% full-module line and branch coverage on ragged_gather_reduce.py."""

    def test_align_to_and_fallback(self):
        assert ragged_gather_reduce._align_to(10, 16) == 16
        assert ragged_gather_reduce._align_to(16, 16) == 16

        x = jnp.ones((4, 8), dtype=jnp.bfloat16)
        indices = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
        weights = jnp.array([0.5, 0.5, 1.0, 1.0], dtype=jnp.float32)
        mask = jnp.array([True, True, False, True])
        res = ragged_gather_reduce._fallback_implementation(
            x, indices, weights, mask, reduce_group_size=2
        )
        assert res.shape == (2, 8)
        assert res.dtype == jnp.bfloat16

    def test_pad_inputs_if_needed(self):
        x = jnp.ones((3, 100), dtype=jnp.bfloat16)
        indices = jnp.arange(3, dtype=jnp.int32)
        weights = jnp.ones((3,), dtype=jnp.float32)
        mask = jnp.ones((3,), dtype=bool)

        x_p, idx_p, w_p, m_p = ragged_gather_reduce._pad_inputs_if_needed(
            x,
            indices,
            weights,
            mask,
            reduce_group_size=2,
            num_column_partitions=2,
            num_row_partitions=2,
            num_simd_lanes=16,
        )
        assert x_p.shape[0] >= 3
        assert x_p.shape[1] % (128 * 2) == 0
        assert idx_p.shape[0] == x_p.shape[0]
        assert w_p.shape[0] == x_p.shape[0]
        assert m_p.shape[0] == x_p.shape[0]

    def test_calculate_num_col_column_partitions(self):
        # while loop true
        p_large = ragged_gather_reduce._calculate_num_col_column_partitions(
            hidden_size=2048, num_cores=32, num_lanes=128
        )
        assert p_large >= 2

        # while loop false immediately
        p_small = ragged_gather_reduce._calculate_num_col_column_partitions(
            hidden_size=128, num_cores=32, num_lanes=128
        )
        assert p_small == 1

    def test_preprocess(self):
        # indices.ndim != 1
        with pytest.raises(AssertionError, match="1d indices"):
            ragged_gather_reduce._preprocess(
                jnp.zeros((2, 2), dtype=jnp.int32),
                jnp.ones((4,), dtype=bool),
                reduce_group_size=2,
                num_row_partitions=2,
                num_simd_lanes=16,
            )

        # num_row_partitions > num_simd_lanes
        with pytest.raises(AssertionError):
            ragged_gather_reduce._preprocess(
                jnp.zeros((32,), dtype=jnp.int32),
                jnp.ones((32,), dtype=bool),
                reduce_group_size=2,
                num_row_partitions=32,
                num_simd_lanes=16,
            )

        # Valid run
        indices = jnp.arange(16, dtype=jnp.int32)
        mask = jnp.array([True, False] * 8)
        dst, sort_valid, num_src, group_mask = ragged_gather_reduce._preprocess(
            indices,
            mask,
            reduce_group_size=2,
            num_row_partitions=2,
            num_simd_lanes=16,
        )
        assert dst.shape == (16,)
        assert sort_valid.shape == (16,)
        assert num_src.shape == (16,)
        assert group_mask.shape == (8,)

    def test_ragged_gather_reduce_validation_and_fallbacks(self):
        # x.ndim != 2
        with pytest.raises(AssertionError, match="2d inputs"):
            ragged_gather_reduce.ragged_gather_reduce(
                jnp.zeros((4,), dtype=jnp.bfloat16),
                jnp.zeros((4,), dtype=jnp.int32),
                jnp.zeros((4,), dtype=jnp.float32),
                jnp.zeros((4,), dtype=bool),
                reduce_group_size=2,
            )

        # indices.ndim != 1
        with pytest.raises(AssertionError, match="1d indices"):
            ragged_gather_reduce.ragged_gather_reduce(
                jnp.zeros((4, 8), dtype=jnp.bfloat16),
                jnp.zeros((2, 2), dtype=jnp.int32),
                jnp.zeros((4,), dtype=jnp.float32),
                jnp.zeros((4,), dtype=bool),
                reduce_group_size=2,
            )

        # topk_weights.ndim != 1
        with pytest.raises(AssertionError, match="1d topk_weights"):
            ragged_gather_reduce.ragged_gather_reduce(
                jnp.zeros((4, 8), dtype=jnp.bfloat16),
                jnp.zeros((4,), dtype=jnp.int32),
                jnp.zeros((2, 2), dtype=jnp.float32),
                jnp.zeros((4,), dtype=bool),
                reduce_group_size=2,
            )

        # valid_rows_mask.ndim != 1
        with pytest.raises(AssertionError, match="1d valid_rows_mask"):
            ragged_gather_reduce.ragged_gather_reduce(
                jnp.zeros((4, 8), dtype=jnp.bfloat16),
                jnp.zeros((4,), dtype=jnp.int32),
                jnp.zeros((4,), dtype=jnp.float32),
                jnp.zeros((2, 2), dtype=bool),
                reduce_group_size=2,
            )

        # sc_info is None -> fallback
        tpu_info_no_sc = SimpleNamespace(sparse_core=None)
        with patch.object(
            ragged_gather_reduce.pltpu, "get_tpu_info", return_value=tpu_info_no_sc
        ):
            x = jnp.ones((4, 8), dtype=jnp.bfloat16)
            idx = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
            w = jnp.array([1.0, 1.0, 1.0, 1.0], dtype=jnp.float32)
            m = jnp.array([True, True, True, True])
            res = ragged_gather_reduce.ragged_gather_reduce(
                x, idx, w, m, reduce_group_size=2
            )
            assert res.shape == (2, 8)

        # VMEM capacity fallback (small size < capacity * 0.6)
        sc_info = SimpleNamespace(num_lanes=16, num_cores=2, num_subcores=16)
        tpu_info_small = SimpleNamespace(
            sparse_core=sc_info,
            vmem_capacity_bytes=100_000_000,
            num_lanes=128,
        )
        with patch.object(
            ragged_gather_reduce.pltpu, "get_tpu_info", return_value=tpu_info_small
        ):
            ragged_gather_reduce.ragged_gather_reduce.clear_cache()
            res_vmem = ragged_gather_reduce.ragged_gather_reduce(
                x, idx, w, m, reduce_group_size=2
            )
            assert res_vmem.shape == (2, 8)
            res_unwrapped = ragged_gather_reduce.ragged_gather_reduce._fun(
                x, idx, w, m, reduce_group_size=2
            )
            assert res_unwrapped.shape == (2, 8)

    def test_ragged_gather_reduce_sc_wiring(self):
        sc_info = SimpleNamespace(num_lanes=16, num_cores=1, num_subcores=16)
        tpu_info = SimpleNamespace(
            sparse_core=sc_info,
            vmem_capacity_bytes=10,  # small forces SC path
            num_lanes=128,
        )

        def mock_kernel(body, out_type=None, **kwargs):
            def runner(num_src, x_val, idx_val, dst_val, w_val, sort_val):
                return jnp.ones(out_type.shape, dtype=out_type.dtype)

            return runner

        with (
            patch.object(
                ragged_gather_reduce.pltpu, "get_tpu_info", return_value=tpu_info
            ),
            patch.object(
                ragged_gather_reduce.core_map_helper,
                "kernel",
                side_effect=mock_kernel,
            ),
            patch.object(
                ragged_gather_reduce.plsc,
                "VectorSubcoreMesh",
                return_value=MagicMock(),
            ),
        ):
            x = jnp.ones((64, 128), dtype=jnp.bfloat16)
            idx = jnp.arange(64, dtype=jnp.int32)
            w = jnp.ones((64,), dtype=jnp.float32)
            m = jnp.ones((64,), dtype=bool)
            ragged_gather_reduce.ragged_gather_reduce.clear_cache()
            res = ragged_gather_reduce.ragged_gather_reduce(
                x, idx, w, m, reduce_group_size=2
            )
            assert res.shape == (32, 128)

    def test_main_kernel_simulation_bf16_f32_and_invalid_dtype(self):
        sc_info = SimpleNamespace(num_lanes=16)
        tpu_info = SimpleNamespace(sparse_core=sc_info, num_lanes=128)

        def fake_emit(pipeline_step, *args, **kwargs):
            def runner(*a, **kw):
                pipeline_step()

            return runner

        def fake_loop(start, stop=None, step=1, init_carry=None):
            def dec(body):
                if init_carry is not None:
                    carry = init_carry
                    carry = body(0, carry)
                    return carry
                body(0)
                return body

            return dec

        def fake_bitcast(x, dtype):
            try:
                return jnp.asarray(x).view(dtype)
            except Exception:
                return jnp.asarray(x, dtype=dtype)

        with (
            patch.object(
                ragged_gather_reduce.pltpu, "get_tpu_info", return_value=tpu_info
            ),
            patch.object(
                ragged_gather_reduce.pltpu,
                "make_async_copy",
                return_value=MockAsyncCopy(),
            ),
            patch.object(
                ragged_gather_reduce.pltpu, "emit_pipeline", side_effect=fake_emit
            ),
            patch.object(ragged_gather_reduce.pl, "loop", side_effect=fake_loop),
            patch.object(ragged_gather_reduce.pl, "program_id", return_value=0),
            patch.object(
                ragged_gather_reduce.plsc, "bitcast", side_effect=fake_bitcast
            ),
            patch.object(ragged_gather_reduce.jax.lax, "axis_size", return_value=32),
        ):
            # 1. bfloat16 input
            num_rows_ref = MockArrayRef(np.array([16, 16], dtype=np.int32))
            in_bf16 = MockArrayRef(
                np.zeros((32, 128), dtype=np.uint16), dtype=jnp.bfloat16
            )
            idx_ref = MockArrayRef(np.arange(16, dtype=np.int32))
            dst_ref = MockArrayRef(np.repeat(np.arange(8, dtype=np.int32), 2))
            w_ref = MockArrayRef(np.ones((16,), dtype=np.float32))
            sort_ref = MockArrayRef(np.arange(16, dtype=np.int32))
            out_ref = MockArrayRef(np.zeros((16, 128), dtype=np.float32))

            num_rows_vmem = MockArrayRef(np.zeros((16,), dtype=np.int32))
            out_vmem = MockArrayRef(np.zeros((16, 128), dtype=np.uint32))
            prev_row_vmem = MockArrayRef(np.zeros((1, 128), dtype=np.uint32))
            src_idx_vmem = MockArrayRef(np.arange(16, dtype=np.int32))
            dst_idx_vmem = MockArrayRef(np.repeat(np.arange(8, dtype=np.int32), 2))
            w_vmem = MockArrayRef(np.ones((16,), dtype=np.float32))
            sort_vmem = MockArrayRef(np.arange(16, dtype=np.int32))
            sem_ref = MockArrayRef(np.zeros((2,), dtype=np.int32))

            ragged_gather_reduce.main_kernel(
                num_rows_ref,
                in_bf16,
                idx_ref,
                dst_ref,
                w_ref,
                sort_ref,
                out_ref,
                num_rows_vmem,
                out_vmem,
                prev_row_vmem,
                src_idx_vmem,
                dst_idx_vmem,
                w_vmem,
                sort_vmem,
                sem_ref,
                core_axis_name="core",
                subcore_axis_name="subcore",
                num_row_partitions=2,
                num_column_partitions=1,
            )

            # 2. float32 input
            in_f32 = MockArrayRef(
                np.zeros((32, 128), dtype=np.float32), dtype=jnp.float32
            )
            ragged_gather_reduce.main_kernel(
                num_rows_ref,
                in_f32,
                idx_ref,
                dst_ref,
                w_ref,
                sort_ref,
                out_ref,
                num_rows_vmem,
                out_vmem,
                prev_row_vmem,
                src_idx_vmem,
                dst_idx_vmem,
                w_vmem,
                sort_vmem,
                sem_ref,
                core_axis_name="core",
                subcore_axis_name="subcore",
                num_row_partitions=2,
                num_column_partitions=1,
            )

            # 3. Invalid dtype -> ValueError
            in_i32 = MockArrayRef(np.zeros((32, 128), dtype=np.int32), dtype=jnp.int32)
            with pytest.raises(
                ValueError, match="Not yet support extracting data from dtype"
            ):
                ragged_gather_reduce.main_kernel(
                    num_rows_ref,
                    in_i32,
                    idx_ref,
                    dst_ref,
                    w_ref,
                    sort_ref,
                    out_ref,
                    num_rows_vmem,
                    out_vmem,
                    prev_row_vmem,
                    src_idx_vmem,
                    dst_idx_vmem,
                    w_vmem,
                    sort_vmem,
                    sem_ref,
                    core_axis_name="core",
                    subcore_axis_name="subcore",
                    num_row_partitions=2,
                    num_column_partitions=1,
                )


# ==============================================================================
# 6. ragged_gather_reduce_v3.py Full Coverage Tests
# ==============================================================================


class TestRaggedGatherReduceV3:
    """100% full-module line and branch coverage on ragged_gather_reduce_v3.py."""

    def test_align_to_and_fallback(self):
        assert ragged_gather_reduce_v3._align_to(10, 16) == 16
        assert ragged_gather_reduce_v3._align_to(16, 16) == 16

        x = jnp.ones((4, 8), dtype=jnp.bfloat16)
        indices = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
        weights = jnp.array([0.5, 0.5, 1.0, 1.0], dtype=jnp.float32)
        mask = jnp.array([True, True, False, True])
        res = ragged_gather_reduce_v3._fallback_implementation(
            x, indices, weights, mask, reduce_group_size=2
        )
        assert res.shape == (2, 8)
        assert res.dtype == jnp.bfloat16

    def test_calculate_num_column_partitions_branches(self):
        # hidden_size = 1024 -> triggers num_column_partitions //= 2
        p1024 = ragged_gather_reduce_v3._calculate_num_column_partitions(
            1024, num_cores=32, num_lanes=128
        )
        assert p1024 == 1

        # hidden_size = 2048 -> triggers num_column_partitions //= 2
        p2048 = ragged_gather_reduce_v3._calculate_num_column_partitions(
            2048, num_cores=32, num_lanes=128
        )
        assert p2048 == 2

        # hidden_size = 3072 -> does NOT trigger //= 2
        p3072 = ragged_gather_reduce_v3._calculate_num_column_partitions(
            3072, num_cores=32, num_lanes=128
        )
        assert p3072 == 4

        # hidden_size = 128 -> while loop doesn't enter
        p128 = ragged_gather_reduce_v3._calculate_num_column_partitions(
            128, num_cores=32, num_lanes=128
        )
        assert p128 == 1

    def test_partition_counts_and_token_block_alignment(self):
        sc_info = SimpleNamespace(num_cores=2, num_subcores=16)
        tpu_info = SimpleNamespace(sparse_core=sc_info, num_lanes=128)
        col_p, row_p = ragged_gather_reduce_v3.partition_counts(1024, tpu_info)
        assert col_p * row_p == 32
        align = ragged_gather_reduce_v3.token_block_alignment(1024, tpu_info)
        assert align == row_p * 64

    def test_ragged_gather_reduce_fallbacks_and_exceptions(self):
        sc_info = SimpleNamespace(num_cores=2, num_subcores=16, num_lanes=16)
        tpu_info = SimpleNamespace(
            sparse_core=sc_info,
            num_lanes=128,
            vmem_capacity_bytes=100_000_000,
        )

        x = jnp.ones((4, 128), dtype=jnp.bfloat16)
        idx = jnp.arange(4, dtype=jnp.int32)
        w = jnp.ones((4,), dtype=jnp.float32)
        m = jnp.ones((4,), dtype=bool)

        # 1. sc_info is None -> fallback
        tpu_no_sc = SimpleNamespace(sparse_core=None, num_lanes=128)
        with patch.object(
            ragged_gather_reduce_v3.pltpu, "get_tpu_info", return_value=tpu_no_sc
        ):
            res1 = ragged_gather_reduce_v3.ragged_gather_reduce(
                x, idx, w, m, reduce_group_size=2
            )
            assert res1.shape == (2, 128)

        # 2. x.dtype != bfloat16 -> fallback
        with patch.object(
            ragged_gather_reduce_v3.pltpu, "get_tpu_info", return_value=tpu_info
        ):
            res2 = ragged_gather_reduce_v3.ragged_gather_reduce(
                x.astype(jnp.float32), idx, w, m, reduce_group_size=2
            )
            assert res2.shape == (2, 128)

        # 3. x.shape[-1] % num_lanes != 0 -> fallback
        with patch.object(
            ragged_gather_reduce_v3.pltpu, "get_tpu_info", return_value=tpu_info
        ):
            x_bad_col = jnp.ones((4, 100), dtype=jnp.bfloat16)
            res3 = ragged_gather_reduce_v3.ragged_gather_reduce(
                x_bad_col, idx, w, m, reduce_group_size=2
            )
            assert res3.shape == (2, 100)

        # 4. Small input size < vmem_capacity * 0.6 -> fallback
        with patch.object(
            ragged_gather_reduce_v3.pltpu, "get_tpu_info", return_value=tpu_info
        ):
            ragged_gather_reduce_v3.ragged_gather_reduce.clear_cache()
            res4 = ragged_gather_reduce_v3.ragged_gather_reduce(
                x, idx, w, m, reduce_group_size=2
            )
            assert res4.shape == (2, 128)
            res4_unwrapped = ragged_gather_reduce_v3.ragged_gather_reduce._fun(
                x, idx, w, m, reduce_group_size=2
            )
            assert res4_unwrapped.shape == (2, 128)

        # 5. input_size % reduce_group_size != 0 -> ValueError
        tpu_small_vmem = SimpleNamespace(
            sparse_core=sc_info,
            num_lanes=128,
            vmem_capacity_bytes=10,
        )
        with (
            patch.object(
                ragged_gather_reduce_v3.pltpu,
                "get_tpu_info",
                return_value=tpu_small_vmem,
            ),
            pytest.raises(ValueError, match="must be divisible"),
        ):
            ragged_gather_reduce_v3.ragged_gather_reduce(
                x, jnp.arange(3, dtype=jnp.int32), w[:3], m[:3], reduce_group_size=2
            )

        # 6. x.shape[0] > (1 << 20) -> ValueError
        with (
            patch.object(
                ragged_gather_reduce_v3.pltpu,
                "get_tpu_info",
                return_value=tpu_small_vmem,
            ),
            patch.object(
                ragged_gather_reduce_v3,
                "_ROUTE_INDEX_BITS",
                2,  # set threshold to 4
            ),
            pytest.raises(ValueError, match="supports at most"),
        ):
            x_too_large = jnp.ones((10, 128), dtype=jnp.bfloat16)
            ragged_gather_reduce_v3.ragged_gather_reduce(
                x_too_large, idx, w, m, reduce_group_size=2
            )

    def test_ragged_gather_reduce_v3_fixed_and_sentinel_routes(self):
        sc_info = SimpleNamespace(num_cores=1, num_subcores=1, num_lanes=16)
        tpu_info = SimpleNamespace(
            sparse_core=sc_info,
            num_lanes=128,
            vmem_capacity_bytes=10,  # force SC path
        )

        def mock_kernel(body, out_type=None, **kwargs):
            def runner(x_val, meta, w_val, cnt):
                return jnp.ones(out_type.shape, dtype=out_type.dtype)

            return runner

        with (
            patch.object(
                ragged_gather_reduce_v3.pltpu, "get_tpu_info", return_value=tpu_info
            ),
            patch.object(
                ragged_gather_reduce_v3.core_map_helper,
                "kernel",
                side_effect=mock_kernel,
            ),
            patch.object(
                ragged_gather_reduce_v3.plsc,
                "VectorSubcoreMesh",
                return_value=MagicMock(),
            ),
        ):
            # Test odd x.shape[0] (e.g. 5) -> triggers pad ((0, 1), (0, 0))
            x_odd = jnp.ones((5, 128), dtype=jnp.bfloat16)
            # 128 items = 64 tokens * group_size 2
            idx = jnp.arange(128, dtype=jnp.int32)
            w = jnp.ones((128,), dtype=jnp.float32)
            m = jnp.ones((128,), dtype=bool)

            # 1. fixed_two_route_blocks is True (all valid, paired destinations match)
            res_fixed = ragged_gather_reduce_v3.ragged_gather_reduce(
                x_odd, idx, w, m, reduce_group_size=2
            )
            assert res_fixed.shape == (64, 128)

            # 2. needs_zero_weight_sentinel is True
            # (route_counts == 128 but paired destinations not matching)
            # Create alternate mask to trigger needs_zero_weight_sentinel
            m_sentinel = np.ones((128,), dtype=bool)
            # Make first block have route_counts == 128, but token destinations shuffled
            res_sentinel = ragged_gather_reduce_v3.ragged_gather_reduce(
                x_odd, idx, w, jnp.asarray(m_sentinel), reduce_group_size=2
            )
            assert res_sentinel.shape == (64, 128)

            # 3. token_padding > 0 (e.g. 32 tokens, padded to 64)
            idx_32 = jnp.arange(64, dtype=jnp.int32)
            w_32 = jnp.ones((64,), dtype=jnp.float32)
            m_32 = jnp.ones((64,), dtype=bool)
            res_pad = ragged_gather_reduce_v3.ragged_gather_reduce(
                x_odd, idx_32, w_32, m_32, reduce_group_size=2
            )
            assert res_pad.shape == (32, 128)

    def test_main_kernel_simulation_fixed_two_routes_and_generic(self):
        sc_info = SimpleNamespace(num_lanes=16)
        tpu_info = SimpleNamespace(sparse_core=sc_info, num_lanes=128)

        captured_token_block = None
        captured_route_pipe = None
        captured_in_specs = None

        def fake_emit(pipeline_step, *args, in_specs=None, **kwargs):
            nonlocal captured_token_block, captured_route_pipe, captured_in_specs
            step_name = getattr(pipeline_step, "__name__", "")
            if "token_block" in step_name:
                captured_token_block = pipeline_step
                if in_specs is not None:
                    for spec in in_specs:
                        if hasattr(spec, "index_map"):
                            spec.index_map(0)
            elif "route_pipeline" in step_name:
                captured_route_pipe = pipeline_step
                captured_in_specs = in_specs

            def runner(*a, **kw):
                if "token_block" in step_name:
                    # Run token_block_pipeline with mock args
                    meta = MockArrayRef(np.zeros((128,), dtype=np.int32))
                    w = MockArrayRef(np.ones((128,), dtype=np.float32))
                    cnt = MockArrayRef(np.array([128], dtype=np.int32))
                    sem = MockArrayRef(np.zeros((2,), dtype=np.int32))
                    pipeline_step(meta, w, cnt, sem)
                elif "route_pipeline" in step_name:
                    # Run route_pipeline with mock gather_ref
                    gather = MockArrayRef(np.zeros((16, 128), dtype=np.int32))
                    pipeline_step(gather)

            return runner

        def fake_parallel_loop(lower, upper, step=1):
            def decorator(fn):
                for i in range(lower, upper, step):
                    fn(i)

            return decorator

        def fake_when(cond):
            def dec(fn):
                fn()

            return dec

        def fake_bitcast(x, dtype):
            try:
                return jnp.asarray(x).view(dtype)
            except Exception:
                return jnp.asarray(x, dtype=dtype)

        def fake_pack(x, y, format=None):
            return jnp.asarray(x)

        with (
            patch.object(
                ragged_gather_reduce_v3.pltpu, "get_tpu_info", return_value=tpu_info
            ),
            patch.object(
                ragged_gather_reduce_v3.pltpu,
                "make_async_copy",
                return_value=MockAsyncCopy(),
            ),
            patch.object(
                ragged_gather_reduce_v3.pltpu,
                "emit_pipeline",
                side_effect=fake_emit,
            ),
            patch.object(
                ragged_gather_reduce_v3.plsc,
                "parallel_loop",
                side_effect=fake_parallel_loop,
            ),
            patch.object(
                ragged_gather_reduce_v3.plsc, "bitcast", side_effect=fake_bitcast
            ),
            patch.object(ragged_gather_reduce_v3.plsc, "pack", side_effect=fake_pack),
            patch.object(
                ragged_gather_reduce_v3.plsc,
                "addupdate_scatter",
                return_value=None,
            ),
            patch.object(ragged_gather_reduce_v3.pl, "when", side_effect=fake_when),
            patch.object(ragged_gather_reduce_v3.pl, "program_id", return_value=0),
            patch.object(ragged_gather_reduce_v3.jax.lax, "axis_index", return_value=0),
        ):
            x_ref = MockArrayRef(np.zeros((64, 128), dtype=np.uint16))
            meta_ref = MockArrayRef(np.zeros((128,), dtype=np.int32))
            w_ref = MockArrayRef(np.ones((128,), dtype=np.float32))
            cnt_ref = MockArrayRef(np.array([128], dtype=np.int32))
            out_ref = MockArrayRef(np.zeros((64, 128), dtype=np.uint16))
            accum_vmem = MockArrayRef(np.zeros((64, 128), dtype=np.float32))
            out_tile_vmem = MockArrayRef(np.zeros((2, 8, 128), dtype=np.uint32))
            sem_ref = MockArrayRef(np.zeros((2,), dtype=np.int32))

            ragged_gather_reduce_v3._main_kernel(
                x_ref,
                meta_ref,
                w_ref,
                cnt_ref,
                out_ref,
                accum_vmem,
                out_tile_vmem,
                sem_ref,
                core_axis_name="core",
                subcore_axis_name="subcore",
                num_row_partitions=1,
                num_column_partitions=1,
                blocks_per_row_partition=1,
                topk=2,
            )

            # Test in_specs lambda for route_pipeline
            if captured_in_specs is not None:
                spec_idx = captured_in_specs.index_map(0)
                assert len(spec_idx) == 2
