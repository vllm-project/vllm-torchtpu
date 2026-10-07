# SPDX-License-Identifier: Apache-2.0
"""Full-module line and branch coverage tests for megablox and quantized_matmul.

Covers megablox (common, tuned_block_sizes, moe_onehot_unpermute, gmm_v2) and
quantized_matmul (util, tuned_block_sizes, kernel, blockwise_kernel).
"""

import contextlib
import dataclasses
from typing import Any
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.megablox import common as mb_common
from vllm_torchtpu.kernels.megablox import gmm_v2 as mb_gmm
from vllm_torchtpu.kernels.megablox import moe_onehot_unpermute as mb_unpermute
from vllm_torchtpu.kernels.megablox import tuned_block_sizes as mb_tuned
from vllm_torchtpu.kernels.quantized_matmul import blockwise_kernel as qmm_blockwise
from vllm_torchtpu.kernels.quantized_matmul import kernel as qmm_kernel
from vllm_torchtpu.kernels.quantized_matmul import tuned_block_sizes as qmm_tuned
from vllm_torchtpu.kernels.quantized_matmul import util as qmm_util

# ---------------------------------------------------------------------------
# Helpers and Mocks
# ---------------------------------------------------------------------------


def _normalize_idx(idx: Any) -> Any:
    if idx is Ellipsis:
        return slice(None)
    if hasattr(idx, "start") and hasattr(idx, "size"):
        stride = getattr(idx, "stride", 1)
        start = getattr(idx, "start", 0)
        size = getattr(idx, "size", 0)
        if isinstance(start, (jax.core.Tracer, jax.Array)) or isinstance(
            size, (jax.core.Tracer, jax.Array)
        ):
            return slice(None)
        if stride == 0:
            return np.zeros(int(size), dtype=np.intp)
        return slice(int(start), int(start + size), stride)
    if isinstance(idx, tuple):
        return tuple(_normalize_idx(x) for x in idx)
    return idx


class MockBuffer:
    """Mock reference supporting in-place item assignment, .at indexing, and pl.ds."""

    def __init__(self, arr: Any) -> None:
        self.arr = np.asarray(arr)
        try:
            self._jnp = jnp.asarray(arr)
        except Exception:
            self._jnp = None

    @property
    def shape(self) -> tuple[int, ...]:
        return self.arr.shape

    @property
    def dtype(self) -> np.dtype:
        return self.arr.dtype

    @property
    def ndim(self) -> int:
        return self.arr.ndim

    def __len__(self) -> int:
        return len(self.arr)

    def __array__(self, dtype: Any = None) -> np.ndarray:
        return np.asarray(self.arr, dtype=dtype)

    def __getitem__(self, idx: Any) -> Any:
        idx = _normalize_idx(idx)
        if isinstance(idx, tuple) and any(isinstance(x, np.ndarray) for x in idx):
            return jnp.asarray(self.arr[idx])
        if self._jnp is not None:
            try:
                return self._jnp[idx]
            except Exception:
                pass
        res = self.arr[idx]
        return jnp.asarray(res) if isinstance(res, (np.ndarray, np.generic)) else res

    def __setitem__(self, idx: Any, val: Any) -> None:
        if isinstance(idx, (jax.core.Tracer, jax.Array)) and self._jnp is not None:
            self._jnp = self._jnp.at[idx].set(val)
            return
        idx = _normalize_idx(idx)
        if isinstance(val, MockBuffer):
            val = val.arr
        self.arr[idx] = np.asarray(val)
        if self._jnp is not None:
            with contextlib.suppress(Exception):
                self._jnp = self._jnp.at[idx].set(val)

    def __add__(self, other: Any) -> "MockBuffer":
        val = other.arr if isinstance(other, MockBuffer) else np.asarray(other)
        return MockBuffer(self.arr + val)

    def __radd__(self, other: Any) -> "MockBuffer":
        val = other.arr if isinstance(other, MockBuffer) else np.asarray(other)
        return MockBuffer(val + self.arr)

    def __sub__(self, other: Any) -> "MockBuffer":
        val = other.arr if isinstance(other, MockBuffer) else np.asarray(other)
        return MockBuffer(self.arr - val)

    def __rsub__(self, other: Any) -> "MockBuffer":
        val = other.arr if isinstance(other, MockBuffer) else np.asarray(other)
        return MockBuffer(val - self.arr)

    def __mul__(self, other: Any) -> "MockBuffer":
        val = other.arr if isinstance(other, MockBuffer) else np.asarray(other)
        return MockBuffer(self.arr * val)

    def __rmul__(self, other: Any) -> "MockBuffer":
        val = other.arr if isinstance(other, MockBuffer) else np.asarray(other)
        return MockBuffer(val * self.arr)

    def __truediv__(self, other: Any) -> "MockBuffer":
        val = other.arr if isinstance(other, MockBuffer) else np.asarray(other)
        return MockBuffer(self.arr / val)

    def __rtruediv__(self, other: Any) -> "MockBuffer":
        val = other.arr if isinstance(other, MockBuffer) else np.asarray(other)
        return MockBuffer(val / self.arr)

    def reshape(self, *args: Any) -> "MockBuffer":
        shape = (
            args[0] if len(args) == 1 and isinstance(args[0], (tuple, list)) else args
        )
        return MockBuffer(self.arr.reshape(shape))

    def astype(self, dtype: Any) -> "MockBuffer":
        return MockBuffer(self.arr.astype(dtype))

    @property
    def at(self) -> Any:
        class _At:
            def __init__(s, parent: "MockBuffer") -> None:
                s.parent = parent

            def __getitem__(s, idx: Any) -> "MockBuffer":
                idx = _normalize_idx(idx)
                try:
                    return MockBuffer(s.parent.arr[idx])
                except Exception:
                    return s.parent

            def __setitem__(s, idx: Any, val: Any) -> None:
                s.parent[idx] = val

        return _At(self)


class MockTpuInfo:
    def __init__(
        self,
        generation: int = 5,
        mxu_column_size: int = 128,
        vmem_capacity_bytes: int = 128 * 1024 * 1024,
        num_lanes: int = 128,
        fp8_ops_per_second: float = 1e12,
        int8_ops_per_second: float = 1e12,
    ) -> None:
        self.generation = generation
        self.mxu_column_size = mxu_column_size
        self.vmem_capacity_bytes = vmem_capacity_bytes
        self.num_lanes = num_lanes
        self.fp8_ops_per_second = fp8_ops_per_second
        self.int8_ops_per_second = int8_ops_per_second

    def get_sublane_tiling(self, dtype: Any) -> int:
        del dtype
        return 8

    def is_matmul_supported(self, lhs_dtype: Any, rhs_dtype: Any) -> bool:
        return lhs_dtype == rhs_dtype


class MockAsyncCopy:
    def __init__(
        self, src_ref: Any = None, dst_ref: Any = None, sem: Any = None
    ) -> None:
        self.src_ref = src_ref
        self.dst_ref = dst_ref
        self.sem = sem

    def start(self, priority: int = 0) -> "MockAsyncCopy":
        del priority
        return self

    def wait(self) -> "MockAsyncCopy":
        return self


# ---------------------------------------------------------------------------
# Test Suite 1: megablox/common.py
# ---------------------------------------------------------------------------


class TestMegabloxCommon:
    def test_is_tpu_and_tpu_kind(self) -> None:
        fake_tpu_dev = mock.MagicMock()
        fake_tpu_dev.device_kind = "TPU v5p"
        with mock.patch("jax.devices", return_value=[fake_tpu_dev]):
            assert mb_common.is_tpu() is True
            assert mb_common.tpu_kind() == "TPU v5p"

        fake_cpu_dev = mock.MagicMock()
        fake_cpu_dev.device_kind = "cpu"
        with mock.patch("jax.devices", return_value=[fake_cpu_dev]):
            assert mb_common.is_tpu() is False
            assert mb_common.tpu_kind() == "cpu"

    def test_tpu_generation(self) -> None:
        with mock.patch("vllm_torchtpu.kernels.megablox.common.tpu_kind") as m_kind:
            m_kind.return_value = "TPU v5p"
            assert mb_common.tpu_generation() == 5

            m_kind.return_value = "TPU7x"
            assert mb_common.tpu_generation() == 7

            m_kind.return_value = "cpu"
            with pytest.raises(
                NotImplementedError, match="only TPU devices are supported"
            ):
                mb_common.tpu_generation()

    def test_assert_is_supported_dtype(self) -> None:
        supported = [
            jnp.bfloat16,
            jnp.float32,
            jnp.float8_e4m3fn,
            jnp.float8_e5m2,
            jnp.int8,
            jnp.int4,
            jnp.float4_e2m1fn,
            jnp.uint4,
        ]
        for dt in supported:
            mb_common.assert_is_supported_dtype(dt)

        with pytest.raises(ValueError, match="No support for"):
            mb_common.assert_is_supported_dtype(jnp.float64)


# ---------------------------------------------------------------------------
# Test Suite 2: megablox/tuned_block_sizes.py
# ---------------------------------------------------------------------------


class TestMegabloxTunedBlockSizes:
    def test_round_up_to_multiple_of_128_within_limit(self) -> None:
        with pytest.raises(AssertionError):
            mb_tuned.round_up_to_multiple_of_128_within_limit(100, 64)
        with pytest.raises(AssertionError):
            mb_tuned.round_up_to_multiple_of_128_within_limit(100, 200)

        assert mb_tuned.round_up_to_multiple_of_128_within_limit(50, 512) == 128
        assert mb_tuned.round_up_to_multiple_of_128_within_limit(128, 512) == 128
        assert mb_tuned.round_up_to_multiple_of_128_within_limit(150, 512) == 256
        assert mb_tuned.round_up_to_multiple_of_128_within_limit(1024, 1024) == 1024
        assert mb_tuned.round_up_to_multiple_of_128_within_limit(513, 512) == 512

    def test_get_default_gmm_block_sizes(self) -> None:
        tm, tk, tn = mb_tuned.get_default_gmm_block_sizes(128, 256, 256, 4)
        assert tm > 0 and tk > 0 and tn > 0

        # Cover candidate loop fallthrough when tm <= 0 (branch 350->356)
        with mock.patch(
            "vllm_torchtpu.kernels.megablox.tuned_block_sizes.round_up_to_multiple_of_128_within_limit",
            return_value=0,
        ):
            tm0, tk0, tn0 = mb_tuned.get_default_gmm_block_sizes(0, 256, 256, 4)
            assert tm0 == 0

    def test_get_tuned_block_sizes(self) -> None:
        known_key = (128, 320, 6144, 160, 160, "bfloat16", "float8_e4m3fn", 320)
        res = mb_tuned.get_tuned_block_sizes(*known_key)
        assert res == (128, 640, 256 * 5)

        unknown_key = (999, 320, 6144, 160, 160, "bfloat16", "float8_e4m3fn", 320)
        res_default = mb_tuned.get_tuned_block_sizes(*unknown_key)
        assert len(res_default) == 3


# ---------------------------------------------------------------------------
# Test Suite 3: megablox/moe_onehot_unpermute.py
# ---------------------------------------------------------------------------


class TestMegabloxMoeOnehotUnpermute:
    def test_explicit_vmem_bytes(self) -> None:
        bytes_val = mb_unpermute._explicit_vmem_bytes(128, 256, 16)
        assert bytes_val > 0

    def test_can_use_blockwise_onehot_unpermute_predicates(self) -> None:
        route_out = jnp.zeros((128, 128), dtype=jnp.bfloat16)
        tok_idx = jnp.zeros((128,), dtype=jnp.int32)
        topk_w = jnp.zeros((128,), dtype=jnp.bfloat16)
        valid_cnt = jnp.array([128], dtype=jnp.int32)

        mock_info = MockTpuInfo(
            generation=5,
            mxu_column_size=128,
            vmem_capacity_bytes=128 * 1024 * 1024,
        )

        with mock.patch(
            "jax.experimental.pallas.tpu.get_tpu_info", return_value=mock_info
        ):
            # 1. ndim != 2 or dtype != bfloat16
            assert (
                mb_unpermute.can_use_blockwise_onehot_unpermute(
                    jnp.zeros((128,), dtype=jnp.bfloat16),
                    tok_idx,
                    topk_w,
                    valid_cnt,
                    num_tokens=16,
                )
                is False
            )
            assert (
                mb_unpermute.can_use_blockwise_onehot_unpermute(
                    jnp.zeros((128, 128), dtype=jnp.float32),
                    tok_idx,
                    topk_w,
                    valid_cnt,
                    num_tokens=16,
                )
                is False
            )

            # 2. token_indices_sorted mismatch
            assert (
                mb_unpermute.can_use_blockwise_onehot_unpermute(
                    route_out,
                    jnp.zeros((64,), dtype=jnp.int32),
                    topk_w,
                    valid_cnt,
                    num_tokens=16,
                )
                is False
            )
            assert (
                mb_unpermute.can_use_blockwise_onehot_unpermute(
                    route_out,
                    jnp.zeros((128,), dtype=jnp.int16),
                    topk_w,
                    valid_cnt,
                    num_tokens=16,
                )
                is False
            )

            # 3. topk_weights_sorted mismatch
            assert (
                mb_unpermute.can_use_blockwise_onehot_unpermute(
                    route_out,
                    tok_idx,
                    jnp.zeros((64,), dtype=jnp.bfloat16),
                    valid_cnt,
                    num_tokens=16,
                )
                is False
            )
            assert (
                mb_unpermute.can_use_blockwise_onehot_unpermute(
                    route_out,
                    tok_idx,
                    jnp.zeros((128,), dtype=jnp.float32),
                    valid_cnt,
                    num_tokens=16,
                )
                is False
            )

            # 4. valid_count mismatch
            assert (
                mb_unpermute.can_use_blockwise_onehot_unpermute(
                    route_out,
                    tok_idx,
                    topk_w,
                    jnp.zeros((2,), dtype=jnp.int32),
                    num_tokens=16,
                )
                is False
            )
            assert (
                mb_unpermute.can_use_blockwise_onehot_unpermute(
                    route_out,
                    tok_idx,
                    topk_w,
                    jnp.zeros((1,), dtype=jnp.int16),
                    num_tokens=16,
                )
                is False
            )

            # 5. route_capacity % 128 != 0
            bad_rc = jnp.zeros((100, 128), dtype=jnp.bfloat16)
            assert (
                mb_unpermute.can_use_blockwise_onehot_unpermute(
                    bad_rc,
                    jnp.zeros((100,), dtype=jnp.int32),
                    jnp.zeros((100,), dtype=jnp.bfloat16),
                    valid_cnt,
                    num_tokens=16,
                )
                is False
            )

            # 6. hidden_size % mxu_column_size != 0
            bad_hs = jnp.zeros((128, 100), dtype=jnp.bfloat16)
            assert (
                mb_unpermute.can_use_blockwise_onehot_unpermute(
                    bad_hs, tok_idx, topk_w, valid_cnt, num_tokens=16
                )
                is False
            )

            # 7. num_tokens % sublane != 0
            assert (
                mb_unpermute.can_use_blockwise_onehot_unpermute(
                    route_out, tok_idx, topk_w, valid_cnt, num_tokens=3
                )
                is False
            )

            # 8. vmem overflow
            mock_tiny_info = MockTpuInfo(
                generation=5,
                mxu_column_size=128,
                vmem_capacity_bytes=1024,
            )
            with mock.patch(
                "jax.experimental.pallas.tpu.get_tpu_info", return_value=mock_tiny_info
            ):
                assert (
                    mb_unpermute.can_use_blockwise_onehot_unpermute(
                        route_out, tok_idx, topk_w, valid_cnt, num_tokens=16
                    )
                    is False
                )

            # 9. All valid
            assert (
                mb_unpermute.can_use_blockwise_onehot_unpermute(
                    route_out, tok_idx, topk_w, valid_cnt, num_tokens=16
                )
                is True
            )

    def test_accumulate_and_make_route_block_selector(self) -> None:
        mock_info = MockTpuInfo(mxu_column_size=128)
        with mock.patch(
            "jax.experimental.pallas.tpu.get_tpu_info", return_value=mock_info
        ):
            token_ids = jnp.arange(128, dtype=jnp.int32) % 16
            route_weights = jnp.ones((128,), dtype=jnp.bfloat16)
            route_valid = jnp.ones((128,), dtype=bool)

            selector = mb_unpermute._make_route_block_selector(
                token_ids, route_weights, route_valid, num_tokens=16
            )
            assert selector.shape == (16, 128)

            route_rows = jnp.ones((128, 128), dtype=jnp.bfloat16)
            owner_ref = MockBuffer(np.zeros((16, 128), dtype=np.float32))
            route_row_valid = jnp.ones((128, 128), dtype=bool)

            mb_unpermute._accumulate_route_block(
                route_rows,
                owner_ref,
                token_ids,
                route_weights,
                route_valid,
                route_row_valid,
                num_tokens=16,
            )
            assert owner_ref.arr.sum() > 0

    def test_pipeline_body_and_kernel(self) -> None:
        mock_info = MockTpuInfo(mxu_column_size=128)
        with (
            mock.patch(
                "jax.experimental.pallas.tpu.get_tpu_info", return_value=mock_info
            ),
            mock.patch("jax.experimental.pallas.program_id", return_value=0),
            mock.patch(
                "jax.experimental.pallas.multiple_of", side_effect=lambda x, y: x
            ),
            mock.patch(
                "jax.experimental.pallas.ds", side_effect=lambda x, y: slice(x, x + y)
            ),
        ):
            route_block_ref = jnp.ones((128, 128), dtype=jnp.bfloat16)
            owner_ref = MockBuffer(np.zeros((16, 128), dtype=np.float32))
            tok_ref = jnp.arange(128, dtype=jnp.int32) % 16
            topk_w_ref = jnp.ones((128,), dtype=jnp.float32)
            val_cnt_ref = jnp.array([128], dtype=jnp.int32)

            mb_unpermute._pipeline_body(
                route_block_ref,
                owner_ref,
                token_indices_sorted_ref=tok_ref,
                topk_weights_sorted_ref=topk_w_ref,
                valid_count_ref=val_cnt_ref,
                route_capacity=128,
                num_tokens=16,
            )
            assert owner_ref.arr.sum() > 0

            # Test _kernel with mocked emit_pipeline and make_async_copy
            def mock_emit(fn: Any, *args: Any, **kwargs: Any) -> Any:
                def run(r_hbm: Any, scratches: Any) -> None:
                    del r_hbm, scratches

                return run

            store_ref = MockBuffer(np.zeros((16, 128), dtype=np.float32))
            owner_out_ref = MockBuffer(np.zeros((16, 128), dtype=np.float32))
            owner_store_sem_ref = MockBuffer(np.zeros((1,), dtype=np.int32))

            with (
                mock.patch(
                    "jax.experimental.pallas.tpu.emit_pipeline", side_effect=mock_emit
                ),
                mock.patch(
                    "jax.experimental.pallas.tpu.make_async_copy",
                    return_value=MockAsyncCopy(),
                ),
            ):
                mb_unpermute._kernel(
                    val_cnt_ref,
                    tok_ref,
                    topk_w_ref,
                    route_block_ref,
                    owner_out_ref,
                    owner_ref,
                    store_ref,
                    owner_store_sem_ref,
                    route_capacity=128,
                    hidden_size=128,
                    num_tokens=16,
                )

    def test_validate_inputs_and_blockwise_onehot_unpermute(self) -> None:
        route_out = jnp.zeros((128, 128), dtype=jnp.bfloat16)
        tok_idx = jnp.zeros((128,), dtype=jnp.int32)
        topk_w = jnp.zeros((128,), dtype=jnp.bfloat16)
        valid_cnt = jnp.array([128], dtype=jnp.int32)

        with pytest.raises(
            TypeError, match="num_tokens must be a static Python integer"
        ):
            mb_unpermute._validate_inputs(
                route_out,
                tok_idx,
                topk_w,
                valid_cnt,
                num_tokens="not_an_int",  # type: ignore
            )

        with (
            mock.patch(
                "vllm_torchtpu.kernels.megablox.moe_onehot_unpermute.can_use_blockwise_onehot_unpermute",
                return_value=False,
            ),
            mock.patch(
                "jax.experimental.pallas.tpu.get_tpu_info",
                return_value=MockTpuInfo(generation=5),
            ),
            pytest.raises(
                ValueError, match="blockwise one-hot unpermute does not support"
            ),
        ):
            mb_unpermute._validate_inputs(
                route_out, tok_idx, topk_w, valid_cnt, num_tokens=16
            )

        with (
            mock.patch(
                "vllm_torchtpu.kernels.megablox.moe_onehot_unpermute.can_use_blockwise_onehot_unpermute",
                return_value=True,
            ),
            mock.patch(
                "jax.experimental.pallas.tpu.get_tpu_info",
                return_value=MockTpuInfo(generation=5),
            ),
        ):
            rc, hs, nt = mb_unpermute._validate_inputs(
                route_out, tok_idx, topk_w, valid_cnt, num_tokens=16
            )
            assert (rc, hs, nt) == (128, 128, 16)

            with mock.patch("jax.experimental.pallas.pallas_call") as m_pcall:
                m_pcall.return_value = lambda *args: jnp.zeros(
                    (16, 128), dtype=jnp.bfloat16
                )
                out = mb_unpermute.blockwise_onehot_unpermute(
                    route_out, tok_idx, topk_w, valid_cnt, num_tokens=16
                )
                assert out.shape == (16, 128)


# ---------------------------------------------------------------------------
# Test Suite 4: megablox/gmm_v2.py
# ---------------------------------------------------------------------------


class TestMegabloxGmmV2:
    def test_activations(self) -> None:
        x = jnp.ones((2, 4), dtype=jnp.float32)
        y = jnp.ones((2, 4), dtype=jnp.float32) * 2.0

        assert mb_gmm.swigluoai(x, y).shape == (2, 4)
        assert mb_gmm.silu_and_mul_with_clamp(x, y).shape == (2, 4)
        assert mb_gmm.situ_and_mul(x, y, 1.0, None).shape == (2, 4)
        assert mb_gmm.situ_and_mul(x, y, 1.0, 2.0).shape == (2, 4)

        with mock.patch(
            "jax.experimental.pallas.tpu.get_tpu_info",
            return_value=MockTpuInfo(num_lanes=8),
        ):
            x_inter = jnp.arange(16, dtype=jnp.float32).reshape(2, 8)
            inter = mb_gmm.interleave_lane(x_inter, x_inter)
            de_l, de_r = mb_gmm.deinterleave_lane(inter)
            np.testing.assert_allclose(de_l, x_inter)
            np.testing.assert_allclose(de_r, x_inter)

    def test_apply_act_fn(self) -> None:
        acc_inter = jnp.ones((2, 16), dtype=jnp.float32)
        assert mb_gmm.apply_act_fn(acc_inter, None) is acc_inter

        with mock.patch(
            "jax.experimental.pallas.tpu.get_tpu_info",
            return_value=MockTpuInfo(num_lanes=8),
        ):
            for act in [
                "silu",
                "gelu",
                "gelu_tanh",
                "swigluoai",
                "silu_and_mul_with_clamp",
            ]:
                res = mb_gmm.apply_act_fn(acc_inter, act)
                assert res.shape == (2, 8)

            res_situ_none = mb_gmm.apply_act_fn(acc_inter, "situ:1.0:none")
            assert res_situ_none.shape == (2, 8)

            res_situ_float = mb_gmm.apply_act_fn(acc_inter, "situ:1.0:2.0")
            assert res_situ_float.shape == (2, 8)

            with pytest.raises(
                NotImplementedError, match="Unsupported activation function"
            ):
                mb_gmm.apply_act_fn(acc_inter, "invalid_act")

    def test_align_to_and_sublane_size(self) -> None:
        assert mb_gmm.align_to(10, 8) == 16
        with mock.patch(
            "jax.experimental.pallas.tpu.get_tpu_info",
            return_value=MockTpuInfo(generation=5),
        ):
            assert mb_gmm._get_lhs_sublane_size(jnp.bfloat16, 128) == 8
            assert mb_gmm._get_lhs_sublane_size(jnp.bfloat16, 4) == 4

    def test_get_packing_factor(self) -> None:
        assert mb_gmm.get_packing_factor(jnp.int32, None) == 1
        assert mb_gmm.get_packing_factor(jnp.int32, jnp.int32) == 1
        assert mb_gmm.get_packing_factor(jnp.int32, jnp.int4) == 8

        fake_storage = mock.MagicMock()
        fake_quant = mock.MagicMock()
        with (
            mock.patch("jax.dtypes.itemsize_bits", side_effect=[32, 24]),
            pytest.raises(ValueError, match="is not divisible by"),
        ):
            mb_gmm.get_packing_factor(fake_storage, fake_quant)

    def test_weights_ref_and_fused_weights_ref(self) -> None:
        w = MockBuffer(np.ones((4, 4), dtype=np.float32))
        s = MockBuffer(np.ones((2, 1, 4), dtype=np.float32))
        b = MockBuffer(np.ones((1, 4), dtype=np.float32))

        w_ref = mb_gmm.WeightsRef(weight=w, scale=s, bias=b)
        assert w_ref.get_weight().shape == (4, 4)
        assert w_ref.get_scale().shape == (2, 1, 4)
        assert w_ref.get_scale(replicate_size=2).shape == (2, 2, 4)
        assert w_ref.get_bias().shape == (1, 4)

        with mock.patch(
            "jax.experimental.pallas.tpu.get_tpu_info",
            return_value=MockTpuInfo(num_lanes=4),
        ):
            f_ref = mb_gmm.FusedWeightsRef(gate=w_ref, up=w_ref)
            assert f_ref.get_weight().shape == (4, 8)
            assert f_ref.get_scale().shape == (2, 1, 8)
            assert f_ref.get_scale(replicate_size=2).shape == (2, 2, 8)
            assert f_ref.get_bias().shape == (1, 8)

    def test_input_configs_and_gmm_configs_properties(self) -> None:
        mock_info = MockTpuInfo(mxu_column_size=128)
        with mock.patch(
            "jax.experimental.pallas.tpu.get_tpu_info", return_value=mock_info
        ):
            cfg_packed = mb_gmm.InputConfigs(
                quant_dtype=jnp.int4,
                quant_block_size=64,
                dtype=jnp.int32,
                has_scale=True,
            )
            assert cfg_packed.should_unpack is True
            assert cfg_packed.should_dequantize_before_matmul is True
            assert cfg_packed.should_dequantize_after_matmul is False
            assert cfg_packed.should_quantize is True

            cfg_unpacked = mb_gmm.InputConfigs(
                quant_dtype=jnp.bfloat16,
                quant_block_size=256,
                dtype=jnp.bfloat16,
                has_scale=True,
            )
            assert cfg_unpacked.should_unpack is False
            assert cfg_unpacked.should_dequantize_before_matmul is False
            assert cfg_unpacked.should_dequantize_after_matmul is True
            assert cfg_unpacked.should_quantize is False

            cfg_no_scale = mb_gmm.InputConfigs(
                quant_dtype=None,
                quant_block_size=None,
                dtype=jnp.bfloat16,
                has_scale=False,
            )
            assert cfg_no_scale.should_dequantize_before_matmul is False
            assert cfg_no_scale.should_quantize is False

            gmm_cfg = mb_gmm.GmmConfigs(
                tiles=mb_gmm.TileSizes(
                    tile_m=128, tile_k=256, tile_n=128, bucket_base=128
                ),
                dims=mb_gmm.Dimensions(
                    size_m=128,
                    size_k=256,
                    size_n=256,
                    size_group=2,
                    size_lhs_group=2,
                    size_lhs_sublane=8,
                ),
                lhs_cfgs=cfg_unpacked,
                rhs_cfgs=cfg_unpacked,
                out_dtype=jnp.bfloat16,
                acc_dtype=jnp.float32,
                zero_init=False,
                fuse_act=None,
            )
            assert gmm_cfg.num_quant_blocks_per_tile_k == 1
            assert gmm_cfg.out_size_n == 256

            gmm_cfg_fused = dataclasses.replace(gmm_cfg, fuse_act="silu")
            assert gmm_cfg_fused.out_size_n == 128

    def test_index_maps_and_generate_block_specs(self) -> None:
        meta_ref = mb_gmm.MetadataRef(
            gm_id_to_group_id=MockBuffer(np.zeros(10, dtype=np.int32)),
            gm_id_to_m_offset=MockBuffer(np.arange(11, dtype=np.int32) * 16),
        )
        tiles = mb_gmm.TileSizes(tile_m=128, tile_k=256, tile_n=128, bucket_base=128)
        dims = mb_gmm.Dimensions(
            size_m=128,
            size_k=256,
            size_n=256,
            size_group=2,
            size_lhs_group=2,
            size_lhs_sublane=8,
        )
        rhs_cfg = mb_gmm.InputConfigs(
            quant_dtype=jnp.float8_e4m3fn,
            quant_block_size=128,
            dtype=jnp.bfloat16,
            has_bias=True,
            has_scale=True,
        )
        cfgs = mb_gmm.GmmConfigs(
            tiles=tiles,
            dims=dims,
            lhs_cfgs=rhs_cfg,
            rhs_cfgs=rhs_cfg,
            out_dtype=jnp.bfloat16,
            acc_dtype=jnp.float32,
            zero_init=False,
            fuse_act=None,
        )

        im = mb_gmm.IndexMaps(meta_ref, cfgs)
        lhs_map = im.lhs_index_map(0, 0, 0)
        assert len(lhs_map) == 3
        w_map = im.rhs_weight_index_map(0, 0, 0)
        assert len(w_map) == 3
        b_map = im.rhs_bias_index_map(0, 0, 0)
        assert len(b_map) == 3
        s_map = im.rhs_scale_index_map(0, 0, 0)
        assert len(s_map) == 4

        with mock.patch("jax.experimental.pallas.num_programs", return_value=5):
            out_map_mid = im.out_index_map(0, 1, 0)
            assert len(out_map_mid) == 3
            out_map_last = im.out_index_map(0, 4, 0)
            assert len(out_map_last) == 3

        (lhs_s, rhs_s), out_s = mb_gmm.generate_block_specs(meta_ref, cfgs)
        assert lhs_s is not None and rhs_s is not None and out_s is not None

        cfgs_plain = dataclasses.replace(
            cfgs,
            rhs_cfgs=dataclasses.replace(rhs_cfg, has_bias=False, has_scale=False),
        )
        (lhs_s2, rhs_s2), out_s2 = mb_gmm.generate_block_specs(meta_ref, cfgs_plain)
        assert rhs_s2.bias is None and rhs_s2.scale is None

    def test_matmul_and_inner_kernel_branches(self) -> None:
        mock_info = MockTpuInfo(mxu_column_size=128, num_lanes=128)

        def mock_cond(pred: Any, t_fn: Any, f_fn: Any) -> Any:
            return t_fn() if bool(pred) else f_fn()

        def mock_switch(idx: Any, branches: Any) -> Any:
            return branches[min(int(idx), len(branches) - 1)]()

        with (
            mock.patch(
                "jax.experimental.pallas.tpu.get_tpu_info", return_value=mock_info
            ),
            mock.patch("jax.lax.cond", side_effect=mock_cond),
            mock.patch("jax.lax.switch", side_effect=mock_switch),
        ):
            tiles = mb_gmm.TileSizes(
                tile_m=256, tile_k=128, tile_n=128, bucket_base=128
            )
            dims = mb_gmm.Dimensions(
                size_m=256,
                size_k=128,
                size_n=256,
                size_group=2,
                size_lhs_group=2,
                size_lhs_sublane=8,
            )

            # Scenario A: Unquantized matmul, first & last k step, fused silu
            rhs_cfg_after = mb_gmm.InputConfigs(
                quant_dtype=jnp.dtype(jnp.float8_e4m3fn),
                quant_block_size=128,
                dtype=jnp.dtype(jnp.bfloat16),
                has_bias=True,
                has_scale=True,
            )
            lhs_cfg_unquant = mb_gmm.InputConfigs(
                quant_dtype=None, quant_block_size=None, dtype=jnp.dtype(jnp.bfloat16)
            )
            cfgs_a = mb_gmm.GmmConfigs(
                tiles=tiles,
                dims=dims,
                lhs_cfgs=lhs_cfg_unquant,
                rhs_cfgs=rhs_cfg_after,
                out_dtype=jnp.dtype(jnp.bfloat16),
                acc_dtype=jnp.dtype(jnp.float32),
                zero_init=False,
                fuse_act="silu",
            )

            tiled_lhs_ref = jnp.ones((32, 8, 128), dtype=jnp.float32)
            w_gate = mb_gmm.WeightsRef(
                weight=jnp.ones((128, 128), dtype=jnp.float32),
                scale=jnp.ones((1, 1, 128), dtype=jnp.float32),
                bias=jnp.ones((1, 128), dtype=jnp.float32),
            )
            w_up = mb_gmm.WeightsRef(
                weight=jnp.ones((128, 128), dtype=jnp.float32),
                scale=jnp.ones((1, 1, 128), dtype=jnp.float32),
                bias=jnp.ones((1, 128), dtype=jnp.float32),
            )
            tiled_rhs_fused = mb_gmm.FusedWeightsRef(gate=w_gate, up=w_up)

            tiled_out_ref = MockBuffer(np.zeros((32, 8, 128), dtype=np.float32))
            partial_out_ref = MockBuffer(np.zeros((8, 128), dtype=np.float32))
            acc_ref = MockBuffer(np.zeros((256, 128), dtype=np.float32))
            metadata_ref = mb_gmm.MetadataRef(
                gm_id_to_group_id=jnp.zeros(10, dtype=jnp.int32),
                gm_id_to_m_offset=jnp.array([0, 128, 256], dtype=jnp.int32),
            )

            with (
                mock.patch("jax.experimental.pallas.program_id", return_value=0),
                mock.patch("jax.experimental.pallas.num_programs", return_value=1),
            ):
                mb_gmm.inner_kernel(
                    tiled_lhs_ref,
                    tiled_rhs_fused,
                    tiled_out_ref,
                    partial_out_ref,
                    acc_ref,
                    metadata_ref,
                    cfgs=cfgs_a,
                )

            # Scenario B: Quantized matmul, dequantize before matmul, multi-k steps
            rhs_cfg_before = mb_gmm.InputConfigs(
                quant_dtype=jnp.dtype(jnp.float8_e4m3fn),
                quant_block_size=64,
                dtype=jnp.dtype(jnp.bfloat16),
                has_bias=False,
                has_scale=True,
            )
            lhs_cfg_quant = mb_gmm.InputConfigs(
                quant_dtype=jnp.dtype(jnp.float8_e4m3fn),
                quant_block_size=64,
                dtype=jnp.dtype(jnp.bfloat16),
            )
            cfgs_b = mb_gmm.GmmConfigs(
                tiles=tiles,
                dims=dataclasses.replace(dims, size_k=256),
                lhs_cfgs=lhs_cfg_quant,
                rhs_cfgs=rhs_cfg_before,
                out_dtype=jnp.dtype(jnp.bfloat16),
                acc_dtype=jnp.dtype(jnp.float32),
                zero_init=False,
                fuse_act=None,
            )
            w_plain = mb_gmm.WeightsRef(
                weight=jnp.ones((128, 128), dtype=jnp.float32),
                scale=jnp.ones((2, 64, 128), dtype=jnp.float32),
                bias=None,
            )

            # First k step
            with (
                mock.patch("jax.experimental.pallas.program_id", return_value=0),
                mock.patch("jax.experimental.pallas.num_programs", return_value=2),
            ):
                mb_gmm.inner_kernel(
                    tiled_lhs_ref,
                    w_plain,
                    tiled_out_ref,
                    partial_out_ref,
                    acc_ref,
                    metadata_ref,
                    cfgs=cfgs_b,
                )

            # Last k step with unaligned m_end_local
            meta_unaligned = mb_gmm.MetadataRef(
                gm_id_to_group_id=jnp.zeros(10, dtype=jnp.int32),
                gm_id_to_m_offset=jnp.array([0, 125, 256], dtype=jnp.int32),
            )
            with (
                mock.patch("jax.experimental.pallas.program_id", return_value=1),
                mock.patch("jax.experimental.pallas.num_programs", return_value=2),
            ):
                mb_gmm.inner_kernel(
                    tiled_lhs_ref,
                    w_plain,
                    tiled_out_ref,
                    partial_out_ref,
                    acc_ref,
                    meta_unaligned,
                    cfgs=cfgs_b,
                )

            # Scenario C: Integer quantized matmul with zero inputs (block_scale == 0)
            lhs_cfg_int = mb_gmm.InputConfigs(
                quant_dtype=jnp.dtype(jnp.int8),
                quant_block_size=64,
                dtype=jnp.dtype(jnp.bfloat16),
            )
            cfgs_c = dataclasses.replace(cfgs_b, lhs_cfgs=lhs_cfg_int)
            tiled_lhs_zero = jnp.zeros((32, 8, 128), dtype=jnp.float32)
            with (
                mock.patch("jax.experimental.pallas.program_id", return_value=0),
                mock.patch("jax.experimental.pallas.num_programs", return_value=1),
            ):
                mb_gmm.inner_kernel(
                    tiled_lhs_zero,
                    w_plain,
                    tiled_out_ref,
                    partial_out_ref,
                    acc_ref,
                    metadata_ref,
                    cfgs=cfgs_c,
                )

            # Scenario D: Quantized matmul with
            # should_dequantize_after_matmul=True, supported matmul
            # (lhs_q_dtype == block_rhs.dtype), valid_k != 0 on last step,
            # and mid-step matmul (lines 466-467, 541->544, 554-558, 628).
            lhs_cfg_fp8 = mb_gmm.InputConfigs(
                quant_dtype=jnp.dtype(jnp.float8_e4m3fn),
                quant_block_size=128,
                dtype=jnp.dtype(jnp.bfloat16),
            )
            rhs_cfg_fp8_after = mb_gmm.InputConfigs(
                quant_dtype=jnp.dtype(jnp.float8_e4m3fn),
                quant_block_size=128,
                dtype=jnp.dtype(jnp.float8_e4m3fn),
                has_scale=True,
                has_bias=False,
            )
            cfgs_d = mb_gmm.GmmConfigs(
                tiles=tiles,
                dims=dataclasses.replace(dims, size_k=200),  # 200 % 128 = 72 != 0
                lhs_cfgs=lhs_cfg_fp8,
                rhs_cfgs=rhs_cfg_fp8_after,
                out_dtype=jnp.dtype(jnp.bfloat16),
                acc_dtype=jnp.dtype(jnp.float32),
                zero_init=False,
                fuse_act=None,
            )
            w_fp8 = mb_gmm.WeightsRef(
                weight=jnp.ones((128, 128), dtype=jnp.float8_e4m3fn),
                scale=MockBuffer(np.ones((1, 1, 128), dtype=np.float32)),
                bias=None,
            )
            # Last k-step with valid_k != 0 (covers lines 466-467, 541->544, 554-558)
            with (
                mock.patch("jax.experimental.pallas.program_id", return_value=1),
                mock.patch("jax.experimental.pallas.num_programs", return_value=2),
            ):
                mb_gmm.inner_kernel(
                    tiled_lhs_ref,
                    w_fp8,
                    tiled_out_ref,
                    partial_out_ref,
                    acc_ref,
                    metadata_ref,
                    cfgs=cfgs_d,
                )

            # Mid-step matmul (k_id=1, num_programs=3) (covers line 628)
            with (
                mock.patch("jax.experimental.pallas.program_id", return_value=1),
                mock.patch("jax.experimental.pallas.num_programs", return_value=3),
            ):
                mb_gmm.inner_kernel(
                    tiled_lhs_ref,
                    w_plain,
                    tiled_out_ref,
                    partial_out_ref,
                    acc_ref,
                    metadata_ref,
                    cfgs=cfgs_b,
                )

            # Scenario E: Unquantized lhs with
            # should_dequantize_after_matmul=False (branch 487->494)
            # and bucket_m < tile_m for partial m steps (line 647)
            tiles_multi_bucket = mb_gmm.TileSizes(
                tile_m=256, tile_k=128, tile_n=128, bucket_base=128
            )
            rhs_cfg_no_scale = mb_gmm.InputConfigs(
                quant_dtype=None,
                quant_block_size=128,
                dtype=jnp.dtype(jnp.bfloat16),
                has_scale=False,
                has_bias=False,
            )
            lhs_cfg_unquant = mb_gmm.InputConfigs(
                quant_dtype=None,
                quant_block_size=None,
                dtype=jnp.dtype(jnp.bfloat16),
            )
            cfgs_e = mb_gmm.GmmConfigs(
                tiles=tiles_multi_bucket,
                dims=dataclasses.replace(dims, size_m=256, size_k=128),
                lhs_cfgs=lhs_cfg_unquant,
                rhs_cfgs=rhs_cfg_no_scale,
                out_dtype=jnp.dtype(jnp.bfloat16),
                acc_dtype=jnp.dtype(jnp.float32),
                zero_init=False,
                fuse_act=None,
            )
            w_unquant = mb_gmm.WeightsRef(
                weight=jnp.ones((128, 128), dtype=jnp.float32),
                scale=None,
                bias=None,
            )
            # m_end_local = 125 gives bucket_idx = 0 -> bucket_m = 128 < tile_m = 256
            # First-last step (k_id=0, num_k=1)
            with (
                mock.patch("jax.experimental.pallas.program_id", return_value=0),
                mock.patch("jax.experimental.pallas.num_programs", return_value=1),
            ):
                mb_gmm.inner_kernel(
                    tiled_lhs_ref,
                    w_unquant,
                    tiled_out_ref,
                    partial_out_ref,
                    acc_ref,
                    meta_unaligned,
                    cfgs=cfgs_e,
                )
            # Last step (k_id=1, num_k=2)
            with (
                mock.patch("jax.experimental.pallas.program_id", return_value=1),
                mock.patch("jax.experimental.pallas.num_programs", return_value=2),
            ):
                mb_gmm.inner_kernel(
                    tiled_lhs_ref,
                    w_unquant,
                    tiled_out_ref,
                    partial_out_ref,
                    acc_ref,
                    meta_unaligned,
                    cfgs=cfgs_e,
                )

    def test_fill_metadata(self) -> None:
        lhs_group_sizes = jnp.array([32, 64, 0], dtype=jnp.int32)
        group_offset = jnp.array([0], dtype=jnp.int32)
        meta_ref = mb_gmm.MetadataRef(
            gm_id_to_group_id=MockBuffer(np.zeros(10, dtype=np.int32)),
            gm_id_to_m_offset=MockBuffer(np.zeros(11, dtype=np.int32)),
        )
        tiles = mb_gmm.TileSizes(tile_m=128, tile_k=128, tile_n=128, bucket_base=128)
        dims = mb_gmm.Dimensions(
            size_m=128,
            size_k=128,
            size_n=128,
            size_group=3,
            size_lhs_group=3,
            size_lhs_sublane=8,
        )
        cfgs = mb_gmm.GmmConfigs(
            tiles=tiles,
            dims=dims,
            lhs_cfgs=mb_gmm.InputConfigs(
                quant_dtype=None, quant_block_size=None, dtype=jnp.bfloat16
            ),
            rhs_cfgs=mb_gmm.InputConfigs(
                quant_dtype=None, quant_block_size=None, dtype=jnp.bfloat16
            ),
            out_dtype=jnp.bfloat16,
            acc_dtype=jnp.float32,
            zero_init=False,
            fuse_act=None,
        )

        with mock.patch(
            "jax.experimental.pallas.cdiv", side_effect=lambda a, b: (a + b - 1) // b
        ):
            num_gm = mb_gmm.fill_metadata(
                lhs_group_sizes, group_offset, meta_ref, cfgs=cfgs
            )
            assert num_gm > 0

    def test_zero_out_start_and_end(self) -> None:
        mock_info = MockTpuInfo(num_lanes=128)
        with (
            mock.patch(
                "jax.experimental.pallas.tpu.get_tpu_info", return_value=mock_info
            ),
            mock.patch(
                "jax.experimental.pallas.tpu.make_async_copy",
                return_value=MockAsyncCopy(),
            ),
            mock.patch(
                "jax.experimental.pallas.cdiv",
                side_effect=lambda a, b: (a + b - 1) // b,
            ),
        ):
            out_ref = MockBuffer(np.ones((128, 128), dtype=np.float32))
            zero_ref = MockBuffer(np.zeros((128, 128), dtype=np.float32))
            sem_ref = MockBuffer(np.zeros(1, dtype=np.int32))
            meta_ref = mb_gmm.MetadataRef(
                gm_id_to_group_id=MockBuffer(np.zeros(10, dtype=np.int32)),
                gm_id_to_m_offset=MockBuffer(np.array([16, 64, 128], dtype=np.int32)),
            )
            dims = mb_gmm.Dimensions(
                size_m=128,
                size_k=128,
                size_n=128,
                size_group=2,
                size_lhs_group=2,
                size_lhs_sublane=8,
            )

            z_size = mb_gmm.zero_out_start(
                out_ref, zero_ref, sem_ref, meta_ref, num_gm=1, dims=dims
            )
            mb_gmm.zero_out_end(out_ref, sem_ref, z_size, dims=dims)

    def test_kernel_main(self) -> None:
        mock_info = MockTpuInfo(num_lanes=128)
        with (
            mock.patch(
                "jax.experimental.pallas.tpu.get_tpu_info", return_value=mock_info
            ),
            mock.patch(
                "vllm_torchtpu.kernels.megablox.gmm_v2.fill_metadata", return_value=1
            ),
            mock.patch(
                "vllm_torchtpu.kernels.megablox.gmm_v2.zero_out_start", return_value=10
            ),
            mock.patch("vllm_torchtpu.kernels.megablox.gmm_v2.zero_out_end"),
            mock.patch(
                "jax.experimental.pallas.cdiv",
                side_effect=lambda a, b: (a + b - 1) // b,
            ),
        ):
            pipeline_called = False

            def mock_emit(fn: Any, *args: Any, **kwargs: Any) -> Any:
                nonlocal pipeline_called

                def run(lhs_in: Any, rhs_in: Any, out_in: Any, scratches: Any) -> None:
                    nonlocal pipeline_called
                    pipeline_called = True

                return run

            with mock.patch(
                "jax.experimental.pallas.tpu.emit_pipeline", side_effect=mock_emit
            ):
                tiles = mb_gmm.TileSizes(
                    tile_m=128, tile_k=128, tile_n=128, bucket_base=128
                )
                dims = mb_gmm.Dimensions(
                    size_m=128,
                    size_k=128,
                    size_n=256,
                    size_group=1,
                    size_lhs_group=1,
                    size_lhs_sublane=8,
                )
                cfgs = mb_gmm.GmmConfigs(
                    tiles=tiles,
                    dims=dims,
                    lhs_cfgs=mb_gmm.InputConfigs(
                        quant_dtype=None, quant_block_size=None, dtype=jnp.bfloat16
                    ),
                    rhs_cfgs=mb_gmm.InputConfigs(
                        quant_dtype=jnp.int4,
                        quant_block_size=64,
                        dtype=jnp.int32,
                    ),
                    out_dtype=jnp.bfloat16,
                    acc_dtype=jnp.float32,
                    zero_init=True,
                    fuse_act="silu",
                )

                lhs_grp = MockBuffer(np.array([128], dtype=np.int32))
                grp_off = MockBuffer(np.array([0], dtype=np.int32))
                lhs_r = MockBuffer(np.zeros((128, 128), dtype=np.float32))
                w_arr = mock.MagicMock()
                w_arr.bitcast.return_value = MockBuffer(
                    np.zeros((1, 128, 256), dtype=np.int8)
                )
                rhs_r = mb_gmm.WeightsRef(weight=w_arr, scale=None, bias=None)
                out_r = MockBuffer(np.zeros((128, 256), dtype=np.float32))
                part_r = MockBuffer(np.zeros((8, 128), dtype=np.float32))
                acc_r = MockBuffer(np.zeros((128, 256), dtype=np.float32))
                meta_r = mb_gmm.MetadataRef(
                    gm_id_to_group_id=MockBuffer(np.zeros(10, dtype=np.int32)),
                    gm_id_to_m_offset=MockBuffer(np.array([0, 128], dtype=np.int32)),
                )
                zero_r = MockBuffer(np.zeros((128, 128), dtype=np.float32))
                sem_r = MockBuffer(np.zeros(1, dtype=np.int32))

                mb_gmm.kernel_main(
                    lhs_grp,
                    grp_off,
                    lhs_r,
                    rhs_r,
                    out_r,
                    part_r,
                    acc_r,
                    meta_r,
                    zero_r,
                    sem_r,
                    cfgs=cfgs,
                )
                assert pipeline_called is True

                # Second call with zero_init=False, fuse_act=None, should_unpack=False
                # covers lines 870->875, 882->892, 894->904, 918->exit
                cfgs_no_zero_no_fuse = dataclasses.replace(
                    cfgs,
                    rhs_cfgs=dataclasses.replace(cfgs.rhs_cfgs, quant_dtype=None),
                    zero_init=False,
                    fuse_act=None,
                )
                rhs_r_plain = mb_gmm.WeightsRef(
                    weight=MockBuffer(np.zeros((1, 128, 256), dtype=np.float32)),
                    scale=None,
                    bias=None,
                )
                mb_gmm.kernel_main(
                    lhs_grp,
                    grp_off,
                    lhs_r,
                    rhs_r_plain,
                    out_r,
                    part_r,
                    acc_r,
                    meta_r,
                    zero_r,
                    sem_r,
                    cfgs=cfgs_no_zero_no_fuse,
                )

    def test_calculate_tiling_branches(self) -> None:
        mock_info = MockTpuInfo(mxu_column_size=128, num_lanes=128)
        with (
            mock.patch(
                "jax.experimental.pallas.tpu.get_tpu_info", return_value=mock_info
            ),
            mock.patch(
                "jax.experimental.pallas.cdiv",
                side_effect=lambda a, b: (a + b - 1) // b,
            ),
        ):
            dims = mb_gmm.Dimensions(
                size_m=256,
                size_k=512,
                size_n=512,
                size_group=2,
                size_lhs_group=2,
                size_lhs_sublane=8,
            )
            lhs_cfgs = mb_gmm.InputConfigs(
                quant_dtype=jnp.dtype(jnp.float8_e4m3fn),
                quant_block_size=128,
                dtype=jnp.dtype(jnp.bfloat16),
            )
            rhs_cfgs = mb_gmm.InputConfigs(
                quant_dtype=jnp.dtype(jnp.float8_e4m3fn),
                quant_block_size=128,
                dtype=jnp.dtype(jnp.bfloat16),
                has_scale=True,
                has_bias=True,
            )

            # Normal fit
            tiles = mb_gmm.calculate_tiling(
                dims,
                lhs_cfgs,
                rhs_cfgs,
                vmem_limit_bytes=32 * 1024 * 1024,
                fuse_act="silu",
            )
            assert tiles.tile_m > 0 and tiles.tile_k > 0 and tiles.tile_n > 0

            # Constrained vmem with tile_n < tile_n_limit, unquantized lhs,
            # no scale on rhs
            # (covers branch 944->946, 962, 978->985, 987->990, 1021-1030)
            mock_info_narrow = MockTpuInfo(mxu_column_size=128, num_lanes=64)
            with mock.patch(
                "jax.experimental.pallas.tpu.get_tpu_info",
                return_value=mock_info_narrow,
            ):
                lhs_cfgs_plain = mb_gmm.InputConfigs(
                    quant_dtype=None,
                    quant_block_size=None,
                    dtype=jnp.dtype(jnp.bfloat16),
                )
                rhs_cfgs_plain = mb_gmm.InputConfigs(
                    quant_dtype=None,
                    quant_block_size=128,
                    dtype=jnp.dtype(jnp.bfloat16),
                    has_scale=False,
                    has_bias=False,
                )
                dims_narrow = mb_gmm.Dimensions(
                    size_m=256,
                    size_k=1024,
                    size_n=384,
                    size_group=2,
                    size_lhs_group=2,
                    size_lhs_sublane=8,
                )
                tiles_tight = mb_gmm.calculate_tiling(
                    dims_narrow,
                    lhs_cfgs_plain,
                    rhs_cfgs_plain,
                    vmem_limit_bytes=1_500_000,
                    fuse_act=None,
                )
                assert tiles_tight.tile_n == 384 and tiles_tight.tile_k == 384

            # Full completion of bucket loop without break (branch 1043->1051)
            dims_large_m = mb_gmm.Dimensions(
                size_m=1024,
                size_k=128,
                size_n=256,
                size_group=1,
                size_lhs_group=1,
                size_lhs_sublane=8,
            )
            tiles_all_buckets = mb_gmm.calculate_tiling(
                dims_large_m,
                lhs_cfgs_plain,
                rhs_cfgs_plain,
                vmem_limit_bytes=100 * 1024 * 1024,
                fuse_act=None,
            )
            assert tiles_all_buckets.tile_m == 512

            # Impossible size_k = 0 triggering ValueError
            dims_zero = mb_gmm.Dimensions(
                size_m=256,
                size_k=0,
                size_n=512,
                size_group=2,
                size_lhs_group=2,
                size_lhs_sublane=8,
            )
            with pytest.raises(ValueError, match="Could not find valid tile sizes"):
                mb_gmm.calculate_tiling(
                    dims_zero, lhs_cfgs, rhs_cfgs, vmem_limit_bytes=100, fuse_act=None
                )

    def test_validate_inputs(self) -> None:
        lhs = jnp.zeros((128, 256), dtype=jnp.bfloat16)
        rhs = jnp.zeros((2, 256, 128), dtype=jnp.bfloat16)
        rhs_scale = jnp.zeros((2, 2, 1, 128), dtype=jnp.float32)
        rhs_bias = jnp.zeros((2, 1, 128), dtype=jnp.float32)
        grp_sizes = jnp.array([64, 64], dtype=jnp.int32)
        grp_offset = jnp.array([0], dtype=jnp.int32)

        with mock.patch(
            "jax.experimental.pallas.tpu.get_tpu_info",
            return_value=MockTpuInfo(num_lanes=128),
        ):
            dims = mb_gmm.validate_inputs(
                lhs, rhs, rhs_scale, rhs_bias, grp_sizes, grp_offset, packing_factor=1
            )
            assert dims.size_m == 128

            with pytest.raises(ValueError, match="divisible by 2 \\* num_lanes"):
                mb_gmm.validate_inputs(
                    lhs,
                    rhs,
                    rhs_scale,
                    rhs_bias,
                    grp_sizes,
                    grp_offset,
                    fuse_act="silu",
                    packing_factor=1,
                )

            # rhs_scale=None and fuse_act="silu" with size_n divisible
            # by 2 * num_lanes (covers 1078->1083, 1088->1094)
            rhs_wide = jnp.zeros(
                (2, 256, 256), dtype=jnp.bfloat16
            )  # 256 is divisible by 2 * 128
            dims_fused = mb_gmm.validate_inputs(
                lhs,
                rhs_wide,
                rhs_scale=None,
                rhs_bias=None,
                group_sizes=grp_sizes,
                group_offset=grp_offset,
                fuse_act="silu",
                packing_factor=1,
            )
            assert dims_fused.size_n == 256

    def test_cost_estimate_scope_name_and_metadata(self) -> None:
        tiles = mb_gmm.TileSizes(tile_m=128, tile_k=128, tile_n=128, bucket_base=128)
        dims = mb_gmm.Dimensions(
            size_m=128,
            size_k=128,
            size_n=128,
            size_group=2,
            size_lhs_group=2,
            size_lhs_sublane=8,
        )
        rhs_cfgs = mb_gmm.InputConfigs(
            quant_dtype=jnp.dtype(jnp.float8_e4m3fn),
            quant_block_size=128,
            dtype=jnp.dtype(jnp.bfloat16),
            has_scale=True,
            has_bias=True,
        )
        cfgs = mb_gmm.GmmConfigs(
            tiles=tiles,
            dims=dims,
            lhs_cfgs=rhs_cfgs,
            rhs_cfgs=rhs_cfgs,
            out_dtype=jnp.dtype(jnp.bfloat16),
            acc_dtype=jnp.dtype(jnp.float32),
            zero_init=False,
            fuse_act=None,
        )

        with mock.patch(
            "jax.experimental.pallas.cdiv", side_effect=lambda a, b: (a + b - 1) // b
        ):
            cost = mb_gmm.get_cost_estimate(cfgs)
            assert cost.flops > 0 and cost.bytes_accessed > 0

            cfgs_plain = dataclasses.replace(
                cfgs,
                rhs_cfgs=dataclasses.replace(rhs_cfgs, has_scale=False, has_bias=False),
            )
            cost_plain = mb_gmm.get_cost_estimate(cfgs_plain)
            assert cost_plain.bytes_accessed > 0

        scope = mb_gmm.get_scope_name(cfgs)
        assert "gmm_v2" in scope

        meta = mb_gmm.get_metadata(cfgs)
        assert "tiles.tile_m" in meta

    def test_make_gmm_configs_and_gmm_v2(self) -> None:
        lhs = jnp.zeros((120, 512), dtype=jnp.bfloat16)
        rhs = jnp.zeros((2, 256, 256), dtype=jnp.bfloat16)
        grp_sizes = jnp.array([60, 60], dtype=jnp.int32)

        mock_info = MockTpuInfo(
            num_lanes=128,
            fp8_ops_per_second=1e12,
            int8_ops_per_second=1e12,
            vmem_capacity_bytes=128 * 1024 * 1024,
        )

        with (
            mock.patch(
                "jax.experimental.pallas.tpu.get_tpu_info", return_value=mock_info
            ),
            mock.patch(
                "jax.experimental.pallas.cdiv",
                side_effect=lambda a, b: (a + b - 1) // b,
            ),
        ):
            cfg_fp8 = mb_gmm.make_gmm_configs(
                lhs,
                rhs,
                rhs_scale=jnp.zeros((2, 2, 1, 256), dtype=jnp.float32),
                rhs_bias=None,
                group_sizes=grp_sizes,
                group_offset=jnp.array([0], dtype=jnp.int32),
                tile_info=mb_gmm.calculate_tiling,
                vmem_limit_bytes=32 * 1024 * 1024,
                out_dtype=None,
                acc_dtype=None,
                maybe_quantize_lhs=True,
                zero_initialize=False,
                fuse_act=None,
                rhs_quant_dtype=jnp.float8_e4m3fn,
            )
            assert cfg_fp8.lhs_cfgs.quant_dtype == jnp.float8_e4m3fn.dtype

            lhs_plain = jnp.zeros((120, 256), dtype=jnp.bfloat16)
            cfg_int8 = mb_gmm.make_gmm_configs(
                lhs_plain,
                rhs,
                rhs_scale=jnp.zeros((2, 2, 1, 256), dtype=jnp.float32),
                rhs_bias=jnp.zeros((2, 1, 256), dtype=jnp.float32),
                group_sizes=grp_sizes,
                group_offset=jnp.array([0], dtype=jnp.int32),
                tile_info=cfg_fp8.tiles,
                vmem_limit_bytes=32 * 1024 * 1024,
                out_dtype=jnp.bfloat16,
                acc_dtype=jnp.float32,
                maybe_quantize_lhs=False,
                zero_initialize=True,
                fuse_act=None,
                rhs_quant_dtype=jnp.bfloat16,
            )
            assert cfg_int8.lhs_cfgs.quant_dtype is None

            # rhs_scale=None, acc_dtype=None, maybe_quantize_lhs=False
            # (lines 1186-1187, 1228)
            cfg_no_scale = mb_gmm.make_gmm_configs(
                lhs_plain,
                rhs,
                rhs_scale=None,
                rhs_bias=None,
                group_sizes=grp_sizes,
                group_offset=jnp.array([0], dtype=jnp.int32),
                tile_info=cfg_fp8.tiles,
                vmem_limit_bytes=32 * 1024 * 1024,
                out_dtype=None,
                acc_dtype=None,
                maybe_quantize_lhs=False,
                zero_initialize=False,
            )
            assert cfg_no_scale.rhs_cfgs.has_scale is False
            assert cfg_no_scale.acc_dtype == jnp.float32.dtype

            # int8 rhs with fp8_ops_per_second=0 and int8_ops_per_second>0
            # (branches 1203->1210, 1208->1210, line 1211)
            mock_info_no_fp8 = MockTpuInfo(
                num_lanes=128,
                fp8_ops_per_second=0,
                int8_ops_per_second=1e12,
                vmem_capacity_bytes=128 * 1024 * 1024,
            )
            with mock.patch(
                "jax.experimental.pallas.tpu.get_tpu_info",
                return_value=mock_info_no_fp8,
            ):
                cfg_int8_lhs = mb_gmm.make_gmm_configs(
                    lhs,
                    rhs,
                    rhs_scale=jnp.zeros((2, 2, 1, 256), dtype=jnp.float32),
                    rhs_bias=None,
                    group_sizes=grp_sizes,
                    group_offset=jnp.array([0], dtype=jnp.int32),
                    tile_info=cfg_fp8.tiles,
                    vmem_limit_bytes=32 * 1024 * 1024,
                    out_dtype=None,
                    acc_dtype=None,
                    maybe_quantize_lhs=True,
                    zero_initialize=False,
                    rhs_quant_dtype=jnp.int8,
                )
                assert cfg_int8_lhs.lhs_cfgs.quant_dtype == jnp.int8.dtype

            # int8 rhs with fp8_ops_per_second > 0 and int8_ops_per_second > 0
            # (covers branch 1208->1210)
            cfg_int8_fp8_avail = mb_gmm.make_gmm_configs(
                lhs,
                rhs,
                rhs_scale=jnp.zeros((2, 2, 1, 256), dtype=jnp.float32),
                rhs_bias=None,
                group_sizes=grp_sizes,
                group_offset=jnp.array([0], dtype=jnp.int32),
                tile_info=cfg_fp8.tiles,
                vmem_limit_bytes=32 * 1024 * 1024,
                out_dtype=None,
                acc_dtype=None,
                maybe_quantize_lhs=True,
                zero_initialize=False,
                rhs_quant_dtype=jnp.int8,
            )
            assert cfg_int8_fp8_avail.lhs_cfgs.quant_dtype == jnp.int8.dtype

            with mock.patch("jax.experimental.pallas.pallas_call") as m_pcall:
                m_pcall.return_value = lambda *args: jnp.zeros(
                    (128, 256), dtype=jnp.bfloat16
                )
                out = mb_gmm.gmm_v2(
                    lhs_plain,
                    rhs,
                    grp_sizes,
                    rhs_scale=jnp.zeros((2, 2, 1, 256), dtype=jnp.float32),
                    rhs_bias=jnp.zeros((2, 1, 256), dtype=jnp.float32),
                    group_offset=None,
                    zero_initialize=True,
                )
                assert out.shape == (120, 256)

                # Unaligned size_m (125 % 8 != 0), scalar group_offset,
                # explicit vmem_limit_bytes, rhs_scale=None, rhs_bias=None,
                # zero_initialize=False (covers lines 1332, 1337-1338,
                # branch 1340->1343, 1364->1367, 1367->1372, line 1410)
                lhs_unaligned = jnp.zeros((125, 256), dtype=jnp.bfloat16)
                out_unaligned = mb_gmm.gmm_v2(
                    lhs_unaligned,
                    rhs,
                    grp_sizes,
                    rhs_scale=None,
                    rhs_bias=None,
                    group_offset=jnp.int32(0),
                    vmem_limit_bytes=32 * 1024 * 1024,
                    zero_initialize=False,
                )
                assert out_unaligned.shape == (125, 256)

                # Non-scalar array group_offset (covers branch 1337->1340)
                out_arr_offset = mb_gmm.gmm_v2(
                    lhs_plain,
                    rhs,
                    grp_sizes,
                    rhs_scale=None,
                    rhs_bias=None,
                    group_offset=jnp.array([0], dtype=jnp.int32),
                    vmem_limit_bytes=32 * 1024 * 1024,
                    zero_initialize=False,
                )
                assert out_arr_offset.shape == (120, 256)


# ---------------------------------------------------------------------------
# Test Suite 5: quantized_matmul/util.py
# ---------------------------------------------------------------------------


class TestQuantizedMatmulUtil:
    def test_unfold_args(self) -> None:
        rec = []

        def target(a: bool, b: bool) -> None:
            rec.append((a, b))

        # Python bools
        qmm_util.unfold_args((True, False), (), target)
        assert rec == [(True, False)]

        # JAX bools (unfolds all branches during eager tracing)
        rec.clear()
        qmm_util.unfold_args((jnp.array(False), jnp.array(True)), (), target)
        assert len(rec) == 4
        assert set(rec) == {(True, True), (True, False), (False, True), (False, False)}

    def test_quantize_tensor_and_array(self) -> None:
        x = jnp.ones((2, 4), dtype=jnp.float32) * 5.0
        xq_c, scale_c = qmm_util.quantize_tensor(x, jnp.int8, dim=-1)
        assert xq_c.shape == (2, 4) and scale_c.shape == (2, 1)

        xq_b, scale_b = qmm_util.quantize_tensor(x, jnp.int8, dim=-1, block_size=2)
        assert xq_b.shape == (2, 4) and scale_b.ndim == 3

        x_abs_max = jnp.array([[5.0, 5.0]], dtype=jnp.float32)
        arr_f, s_f = qmm_util.quantize_array(x, x_abs_max, jnp.float8_e4m3fn)
        assert arr_f.shape == (2, 4)
        arr_i, s_i = qmm_util.quantize_array(x, x_abs_max, jnp.int8)
        assert arr_i.shape == (2, 4)

    def test_next_multiple_and_kernel_name(self) -> None:
        assert qmm_util.next_multiple(10, 8) == 16
        assert qmm_util.next_multiple(16, 8) == 16
        kname = qmm_util.get_kernel_name(qmm_tuned.TunedValue(16, 32, 64))
        assert kname == "quantized_matmul_kernel_16_32_64"

    def test_xla_quantized_matmul(self) -> None:
        x = jnp.ones((4, 8), dtype=jnp.float32)
        w_int = jnp.ones((8, 16), dtype=jnp.int8)
        w_fp = jnp.ones((8, 16), dtype=jnp.float8_e4m3fn)
        w_scale = jnp.ones((16,), dtype=jnp.float32)

        out_int = qmm_util.xla_quantized_matmul(
            x, w_int, w_scale, quantize_activation=True
        )
        assert out_int.shape == (4, 16)

        out_fp = qmm_util.xla_quantized_matmul(
            x, w_fp, w_scale, quantize_activation=True
        )
        assert out_fp.shape == (4, 16)

        out_unquant = qmm_util.xla_quantized_matmul(
            x, w_int, w_scale, quantize_activation=False
        )
        assert out_unquant.shape == (4, 16)

    def test_xla_quantized_batched_matmul(self) -> None:
        x = jnp.ones((2, 4, 8), dtype=jnp.float32)
        w_int = jnp.ones((2, 8, 16), dtype=jnp.int8)
        w_fp = jnp.ones((2, 8, 16), dtype=jnp.float8_e4m3fn)
        w_scale = jnp.ones((16,), dtype=jnp.float32)
        dims = (((2,), (1,)), ((0,), (0,)))

        out_int = qmm_util.xla_quantized_batched_matmul(
            x, w_int, w_scale, dimension_numbers=dims, quantize_activation=True
        )
        assert out_int.shape == (2, 4, 16)

        out_fp = qmm_util.xla_quantized_batched_matmul(
            x, w_fp, w_scale, dimension_numbers=dims, quantize_activation=True
        )
        assert out_fp.shape == (2, 4, 16)

        out_unquant = qmm_util.xla_quantized_batched_matmul(
            x, w_int, w_scale, dimension_numbers=dims, quantize_activation=False
        )
        assert out_unquant.shape == (2, 4, 16)

        # Non-identity perm triggering x_scale transpose (lines 163-164)
        x_perm = jnp.ones((4, 2, 8), dtype=jnp.float32)
        w_perm = jnp.ones((2, 8, 16), dtype=jnp.int8)
        dims_perm = (((2,), (1,)), ((1,), (0,)))
        out_perm = qmm_util.xla_quantized_batched_matmul(
            x_perm,
            w_perm,
            w_scale,
            dimension_numbers=dims_perm,
            quantize_activation=True,
        )
        assert out_perm.shape == (2, 4, 16)

    def test_get_vmem_limit(self) -> None:
        limit_large = qmm_util.get_vmem_limit(
            n_batch=2,
            n_out=2,
            n_in=2,
            batch_block_size=128,
            out_block_size=128,
            in_block_size=128,
            x_dtype=jnp.bfloat16,
            x_q_dtype=jnp.int8,
            w_q_dtype=jnp.int8,
            scale_dtype=jnp.float32,
            out_dtype=jnp.bfloat16,
            acc_dtype=jnp.int32,
            save_acc=True,
            save_x_q=True,
            upper_limit_bytes=96 * 1024 * 1024,
        )
        assert limit_large > 0

        limit_clamped = qmm_util.get_vmem_limit(
            n_batch=1,
            n_out=1,
            n_in=1,
            batch_block_size=128,
            out_block_size=128,
            in_block_size=128,
            x_dtype=jnp.bfloat16,
            x_q_dtype=jnp.int8,
            w_q_dtype=jnp.int8,
            scale_dtype=jnp.float32,
            out_dtype=jnp.bfloat16,
            acc_dtype=jnp.int32,
            save_acc=False,
            save_x_q=False,
            upper_limit_bytes=100,
        )
        assert limit_clamped == 100

    def test_validate_inputs(self) -> None:
        x = jnp.zeros((16, 32))
        wq = jnp.zeros((64, 32))
        ws2d = jnp.zeros((1, 64))
        ws3d = jnp.zeros((1, 1, 64))
        x_abs_max = jnp.zeros((1, 16))

        # Shape mismatch x vs wq
        with pytest.raises(ValueError, match="must be equal"):
            qmm_util.validate_inputs(
                jnp.zeros((16, 20)), wq, ws2d, x_abs_max, 16, 64, 32
            )

        # Shape mismatch wq vs ws 3D
        with pytest.raises(ValueError, match="must be equal"):
            qmm_util.validate_inputs(
                x, wq, jnp.zeros((1, 10, 30)), x_abs_max, 16, 64, 32
            )

        # Shape mismatch x_abs_max
        with pytest.raises(ValueError, match="must be equal"):
            qmm_util.validate_inputs(x, wq, ws2d, jnp.zeros((1, 10)), 16, 64, 32)

        # Non-multiple batch block size
        with pytest.raises(ValueError, match="must be a multiple"):
            qmm_util.validate_inputs(x, wq, ws2d, x_abs_max, 10, 64, 32)

        # Non-multiple out block size
        with pytest.raises(ValueError, match="must be a multiple"):
            qmm_util.validate_inputs(x, wq, ws2d, x_abs_max, 16, 50, 32)

        # Non-multiple in block size
        with pytest.raises(ValueError, match="must be a multiple"):
            qmm_util.validate_inputs(x, wq, ws2d, x_abs_max, 16, 64, 20)

        # Success with 2D and 3D
        qmm_util.validate_inputs(x, wq, ws2d, x_abs_max, 16, 64, 32)
        qmm_util.validate_inputs(x, wq, ws3d, None, 16, 64, 32)

    def test_get_max_min_and_quantize_block(self) -> None:
        f_max, f_min = qmm_util.get_max_min(jnp.float8_e4m3fn)
        assert f_max > 0
        i_max, i_min = qmm_util.get_max_min(jnp.int8)
        assert i_max > 0

        data_zero = jnp.zeros((2, 4), dtype=jnp.float32)
        q_fz, s_fz = qmm_util.quantize_block(data_zero, -1, jnp.float8_e4m3fn)
        assert q_fz.shape == (2, 4)

        data_nonzero = jnp.ones((2, 4), dtype=jnp.float32) * 3.0
        q_fn, s_fn = qmm_util.quantize_block(data_nonzero, -1, jnp.float8_e4m3fn)
        assert q_fn.shape == (2, 4)

        q_in, s_in = qmm_util.quantize_block(data_nonzero, -1, jnp.int8)
        assert q_in.shape == (2, 4)


# ---------------------------------------------------------------------------
# Test Suite 6: quantized_matmul/tuned_block_sizes.py
# ---------------------------------------------------------------------------


class TestQuantizedMatmulTunedBlockSizes:
    def test_device_vmem_limit_and_tpu_version(self) -> None:
        with mock.patch(
            "vllm_torchtpu.kernels.quantized_matmul.tuned_block_sizes.get_tpu_version"
        ) as m_ver:
            m_ver.return_value = 6
            assert qmm_tuned.get_device_vmem_limit() == 96 * 1024 * 1024

            m_ver.return_value = 7
            assert qmm_tuned.get_device_vmem_limit() == 48 * 1024 * 1024

            m_ver.return_value = -1
            assert qmm_tuned.get_device_vmem_limit() == 96 * 1024 * 1024

    def test_get_tpu_version(self) -> None:
        fake_dev = mock.MagicMock()
        fake_dev.device_kind = "cpu"
        with mock.patch("jax.devices", return_value=[fake_dev]):
            assert qmm_tuned.get_tpu_version() == -1

        fake_dev.device_kind = "TPU v6"
        with mock.patch("jax.devices", return_value=[fake_dev]):
            assert qmm_tuned.get_tpu_version() == 6

        fake_dev.device_kind = "TPU7x"
        with mock.patch("jax.devices", return_value=[fake_dev]):
            assert qmm_tuned.get_tpu_version() == 7

    def test_get_tuned_block_sizes(self) -> None:
        with mock.patch(
            "vllm_torchtpu.kernels.quantized_matmul.tuned_block_sizes.get_tpu_version",
            return_value=6,
        ):
            val = qmm_tuned.get_tuned_block_sizes(1024, 1024, 4096, "int8", "int8")
            assert val == qmm_tuned.TunedValue(1024, 256, 4096)

            val_def = qmm_tuned.get_tuned_block_sizes(999, 999, 999, "int8", "int8")
            assert val_def == qmm_tuned.TunedValue(128, 128, 128)


# ---------------------------------------------------------------------------
# Test Suite 7: quantized_matmul/kernel.py
# ---------------------------------------------------------------------------


class TestQuantizedMatmulKernel:
    def test_quantize_array_and_vmem_limit(self) -> None:
        x = jnp.ones((2, 4), dtype=jnp.float32)
        x_abs_max = jnp.array([[2.0, 2.0]], dtype=jnp.float32)

        arr_f, s_f = qmm_kernel.quantize_array(x, x_abs_max, jnp.float8_e4m3fn)
        assert arr_f.shape == (2, 4)
        arr_i, s_i = qmm_kernel.quantize_array(x, x_abs_max, jnp.int8)
        assert arr_i.shape == (2, 4)

        limit_large = qmm_kernel.get_vmem_limit(
            n_batch=2,
            n_out=2,
            n_in=2,
            batch_block_size=128,
            out_block_size=128,
            in_block_size=128,
            x_dtype=jnp.bfloat16,
            x_q_dtype=jnp.int8,
            w_q_dtype=jnp.int8,
            scale_dtype=jnp.float32,
            out_dtype=jnp.bfloat16,
            acc_dtype=jnp.int32,
            save_acc=True,
            save_x_q=True,
            upper_limit_bytes=96 * 1024 * 1024,
        )
        assert limit_large > 0

        limit_single = qmm_kernel.get_vmem_limit(
            n_batch=1,
            n_out=1,
            n_in=1,
            batch_block_size=128,
            out_block_size=128,
            in_block_size=128,
            x_dtype=jnp.bfloat16,
            x_q_dtype=jnp.int8,
            w_q_dtype=jnp.int8,
            scale_dtype=jnp.float32,
            out_dtype=jnp.bfloat16,
            acc_dtype=jnp.int32,
            save_acc=False,
            save_x_q=False,
            upper_limit_bytes=96 * 1024 * 1024,
        )
        assert limit_single > 0

    def test_validate_inputs(self) -> None:
        x = jnp.zeros((16, 32), dtype=jnp.bfloat16)
        wq_int = jnp.zeros((64, 32), dtype=jnp.int8)
        ws = jnp.zeros((1, 64), dtype=jnp.float32)
        x_abs_max = jnp.zeros((1, 16), dtype=jnp.float32)

        with pytest.raises(ValueError, match="must be the same int or float type"):
            qmm_kernel.validate_inputs(
                x, wq_int, ws, x_abs_max, jnp.float8_e4m3fn, 16, 64, 32
            )

        with pytest.raises(ValueError, match="must be equal"):
            qmm_kernel.validate_inputs(
                jnp.zeros((16, 20), dtype=jnp.bfloat16),
                wq_int,
                ws,
                x_abs_max,
                jnp.int8,
                16,
                64,
                32,
            )

        with pytest.raises(ValueError, match="must be equal"):
            qmm_kernel.validate_inputs(
                x,
                wq_int,
                jnp.zeros((1, 30), dtype=jnp.float32),
                x_abs_max,
                jnp.int8,
                16,
                64,
                32,
            )

        with pytest.raises(ValueError, match="must be equal"):
            qmm_kernel.validate_inputs(
                x,
                wq_int,
                ws,
                jnp.zeros((1, 10), dtype=jnp.float32),
                jnp.int8,
                16,
                64,
                32,
            )

        with pytest.raises(ValueError, match="must be a multiple"):
            qmm_kernel.validate_inputs(x, wq_int, ws, x_abs_max, jnp.int8, 10, 64, 32)

        with pytest.raises(ValueError, match="must be a multiple"):
            qmm_kernel.validate_inputs(x, wq_int, ws, x_abs_max, jnp.int8, 16, 50, 32)

        with pytest.raises(ValueError, match="must be a multiple"):
            qmm_kernel.validate_inputs(x, wq_int, ws, x_abs_max, jnp.int8, 16, 64, 20)

        qmm_kernel.validate_inputs(x, wq_int, ws, x_abs_max, jnp.int8, 16, 64, 32)

    def test_matmul_kernel_body_branches(self) -> None:
        x_ref = jnp.ones((16, 32), dtype=jnp.float32)
        wq_ref_int = jnp.ones((64, 32), dtype=jnp.int8)
        ws_ref = jnp.ones((1, 64), dtype=jnp.float32)
        x_abs_max_ref = jnp.ones((1, 16), dtype=jnp.float32)
        out_ref = MockBuffer(np.zeros((16, 64), dtype=np.float32))

        acc_scratch = MockBuffer(np.zeros((16, 64), dtype=np.int32))
        xq_scratch = MockBuffer(np.zeros((16, 32), dtype=np.int8))
        xscale_scratch = MockBuffer(np.zeros((16, 1), dtype=np.float32))

        # Branch 1: save_x_q=True, save_acc=True, quantize_activation=True, out_idx=0
        with (
            mock.patch("jax.experimental.pallas.program_id", side_effect=[0, 0]),
            mock.patch("jax.experimental.pallas.num_programs", return_value=2),
        ):
            qmm_kernel.matmul_kernel(
                x_ref,
                wq_ref_int,
                ws_ref,
                x_abs_max_ref,
                out_ref,
                acc_scratch,
                xq_scratch,
                xscale_scratch,
                x_q_dtype=jnp.int8,
                save_acc=True,
                save_x_q=True,
            )

        # Branch 2: out_idx > 0, in_idx = 1
        with (
            mock.patch("jax.experimental.pallas.program_id", side_effect=[1, 1]),
            mock.patch("jax.experimental.pallas.num_programs", return_value=2),
        ):
            qmm_kernel.matmul_kernel(
                x_ref,
                wq_ref_int,
                ws_ref,
                x_abs_max_ref,
                out_ref,
                acc_scratch,
                xq_scratch,
                xscale_scratch,
                x_q_dtype=jnp.int8,
                save_acc=True,
                save_x_q=True,
            )

        # Branch 3: unquantized activation, save_x_q=False, save_acc=False
        out_ref_f = MockBuffer(np.zeros((16, 64), dtype=np.float32))
        wq_ref_f = jnp.ones((64, 32), dtype=np.float32)
        with (
            mock.patch("jax.experimental.pallas.program_id", side_effect=[0, 0]),
            mock.patch("jax.experimental.pallas.num_programs", return_value=1),
        ):
            qmm_kernel.matmul_kernel(
                x_ref,
                wq_ref_f,
                ws_ref,
                x_abs_max_ref,
                out_ref_f,
                None,
                None,
                None,
                x_q_dtype=jnp.float32,
                save_acc=False,
                save_x_q=False,
            )
        # Branch 4: quantize_activation=True, save_x_q=False (branch 182->192)
        out_ref_q_nosave = MockBuffer(np.zeros((16, 64), dtype=np.float32))
        with (
            mock.patch("jax.experimental.pallas.program_id", side_effect=[0, 0]),
            mock.patch("jax.experimental.pallas.num_programs", return_value=1),
        ):
            qmm_kernel.matmul_kernel(
                x_ref,
                wq_ref_int,
                ws_ref,
                x_abs_max_ref,
                out_ref_q_nosave,
                None,
                None,
                None,
                x_q_dtype=jnp.int8,
                save_acc=False,
                save_x_q=False,
            )

        # Branch 5: out_idx = 1, in_idx = 0
        # (mid step, not last step, quant=False, branch 189->192)
        with (
            mock.patch("jax.experimental.pallas.program_id", side_effect=[1, 0]),
            mock.patch("jax.experimental.pallas.num_programs", return_value=2),
        ):
            qmm_kernel.matmul_kernel(
                x_ref,
                wq_ref_int,
                ws_ref,
                x_abs_max_ref,
                out_ref,
                acc_scratch,
                xq_scratch,
                xscale_scratch,
                x_q_dtype=jnp.int8,
                save_acc=True,
                save_x_q=True,
            )

    def test_quantized_matmul_kernel_wrapper(self) -> None:
        x = jnp.zeros((10, 30), dtype=jnp.bfloat16)
        wq = jnp.zeros((60, 30), dtype=jnp.int8)
        ws = jnp.zeros((60,), dtype=jnp.float16)

        with pytest.raises(NotImplementedError, match="zero_point is not supported"):
            qmm_kernel.quantized_matmul_kernel(x, wq, ws, w_zp=jnp.zeros(60))

        with pytest.raises(NotImplementedError, match="block_size is not supported"):
            qmm_kernel.quantized_matmul_kernel(x, wq, ws, block_size=16)

        with (
            mock.patch("jax.experimental.pallas.pallas_call") as m_pcall,
            mock.patch(
                "vllm_torchtpu.kernels.quantized_matmul.kernel.get_tuned_block_sizes",
                return_value=qmm_tuned.TunedValue(16, 64, 32),
            ),
            mock.patch("vllm_torchtpu.kernels.quantized_matmul.kernel.validate_inputs"),
        ):
            m_pcall.return_value = lambda *args: jnp.zeros((16, 64), dtype=jnp.bfloat16)

            out = qmm_kernel.quantized_matmul_kernel(
                x, wq, ws, x_q_dtype=jnp.int8, tuned_value=None
            )
            assert out.shape == (10, 60)

            # Cover exact block size matches and x_q_dtype=None
            # (lines 261, 290->293, 294->297, 298->302, 302->304, 317->320)
            tuned_exact = qmm_tuned.TunedValue(16, 64, 32)
            x_exact = jnp.zeros((16, 32), dtype=jnp.bfloat16)
            wq_exact = jnp.zeros((64, 32), dtype=jnp.int8)
            ws_exact = jnp.zeros((64,), dtype=jnp.float32)
            out_exact = qmm_kernel.quantized_matmul_kernel(
                x_exact, wq_exact, ws_exact, x_q_dtype=None, tuned_value=tuned_exact
            )
            assert out_exact.shape == (16, 64)


# ---------------------------------------------------------------------------
# Test Suite 8: quantized_matmul/blockwise_kernel.py
# ---------------------------------------------------------------------------


class TestQuantizedMatmulBlockwiseKernel:
    def test_blockwise_kernel_exceptions(self) -> None:
        x = jnp.zeros((16, 32), dtype=jnp.bfloat16)
        wq = jnp.zeros((64, 32), dtype=jnp.int8)
        ws_valid = jnp.zeros((1, 1, 64), dtype=jnp.float32)

        with pytest.raises(ValueError, match="Block size was not specified"):
            qmm_blockwise.quantized_matmul_kernel(x, wq, ws_valid, block_size=None)

        with pytest.raises(ValueError, match="Block size must be positive"):
            qmm_blockwise.quantized_matmul_kernel(x, wq, ws_valid, block_size=-1)

        with pytest.raises(NotImplementedError, match="zero_point is not supported"):
            qmm_blockwise.quantized_matmul_kernel(
                x, wq, ws_valid, w_zp=jnp.zeros(64), block_size=32
            )

        with pytest.raises(ValueError, match="w_scale must be rank 3"):
            qmm_blockwise.quantized_matmul_kernel(
                x, wq, jnp.zeros((1, 64), dtype=jnp.float32), block_size=32
            )

        with pytest.raises(ValueError, match="w_scale.shape\\[1\\] must be 1"):
            qmm_blockwise.quantized_matmul_kernel(
                x, wq, jnp.zeros((1, 2, 64), dtype=jnp.float32), block_size=32
            )

        with pytest.raises(ValueError, match="w_scale output dim"):
            qmm_blockwise.quantized_matmul_kernel(
                x, wq, jnp.zeros((1, 1, 30), dtype=jnp.float32), block_size=32
            )

        with pytest.raises(ValueError, match="must be divisible by block_size"):
            qmm_blockwise.quantized_matmul_kernel(x, wq, ws_valid, block_size=10)

        with pytest.raises(ValueError, match="w_scale block dim"):
            qmm_blockwise.quantized_matmul_kernel(
                x, wq, jnp.zeros((5, 1, 64), dtype=jnp.float32), block_size=32
            )

        tuned_bad_in = qmm_tuned.TunedValue(16, 256, 128)
        with pytest.raises(ValueError, match="Channelwise weight scales require"):
            qmm_blockwise.quantized_matmul_kernel(
                x, wq, ws_valid, block_size=32, tuned_value=tuned_bad_in
            )

        tuned_bad_mod = qmm_tuned.TunedValue(16, 256, 32)
        with pytest.raises(ValueError, match="must be divisible by block_size"):
            qmm_blockwise.quantized_matmul_kernel(
                x,
                wq,
                jnp.zeros((2, 1, 64), dtype=jnp.float32),
                block_size=16,
                tuned_value=tuned_bad_mod._replace(in_block_size=20),
            )

        tuned_large_in = qmm_tuned.TunedValue(16, 256, 64)
        with pytest.raises(ValueError, match="must be a multiple of in_block_size"):
            qmm_blockwise.quantized_matmul_kernel(
                x,
                wq,
                jnp.zeros((2, 1, 64), dtype=jnp.float32),
                block_size=16,
                tuned_value=tuned_large_in,
            )

        tuned_bad_out = qmm_tuned.TunedValue(16, 300, 32, n_lane_multiplier=1)
        with pytest.raises(ValueError, match="must be divisible by compute_tile_n"):
            qmm_blockwise.quantized_matmul_kernel(
                x, wq, ws_valid, block_size=32, tuned_value=tuned_bad_out
            )

    def test_blockwise_kernel_accum_execution(self) -> None:
        orig_n_batch, orig_n_in = 16, 32
        orig_n_out = 256
        x = jnp.ones((orig_n_batch, orig_n_in), dtype=jnp.bfloat16)
        w_q = jnp.ones((orig_n_out, orig_n_in), dtype=jnp.int8)
        w_scale = jnp.ones((2, 1, orig_n_out), dtype=jnp.float16)

        tuned = qmm_tuned.TunedValue(16, 256, 32, n_lane_multiplier=1)

        captured = []

        def mock_pcall(k_fn: Any, *args: Any, **kwargs: Any) -> Any:
            captured.append(k_fn)
            return lambda *call_args: jnp.zeros((16, 256), dtype=jnp.bfloat16)

        with (
            mock.patch("jax.experimental.pallas.pallas_call", side_effect=mock_pcall),
            mock.patch("vllm_torchtpu.kernels.quantized_matmul.util.validate_inputs"),
        ):
            qmm_blockwise.quantized_matmul_kernel(
                x, w_q, w_scale, block_size=16, x_q_dtype=jnp.int8, tuned_value=tuned
            )

        assert len(captured) > 0
        captured_kernel = captured[0]

        lhs_ref = jnp.ones((16, 32), dtype=jnp.float32) * 256
        rhs_ref = jnp.ones((256, 32), dtype=jnp.int8)
        w_scales_ref = jnp.ones((2, 1, 256), dtype=jnp.float32)
        out_ref = MockBuffer(np.zeros((16, 256), dtype=np.float32))
        acc_scratch = MockBuffer(np.zeros((16, 256), dtype=np.float32))

        # First and last step (n_in = 1 step)
        with mock.patch.object(qmm_blockwise.pl, "program_id", return_value=0):
            captured_kernel(lhs_ref, rhs_ref, w_scales_ref, out_ref, acc_scratch)
        assert out_ref.arr.sum() > 0

        # Mid step & last step
        with (
            mock.patch("jax.experimental.pallas.pallas_call", side_effect=mock_pcall),
            mock.patch("vllm_torchtpu.kernels.quantized_matmul.util.validate_inputs"),
        ):
            qmm_blockwise.quantized_matmul_kernel(
                jnp.ones((16, 64), dtype=jnp.bfloat16),
                jnp.ones((256, 64), dtype=jnp.float8_e4m3fn),
                jnp.ones((4, 1, 256), dtype=jnp.float32),
                block_size=16,
                x_q_dtype=jnp.float8_e4m3fn,
                tuned_value=tuned._replace(in_block_size=32),
            )

        kernel_multi_k = captured[-1]
        out_ref = MockBuffer(np.zeros((16, 256), dtype=np.float32))
        acc_scratch = MockBuffer(np.zeros((16, 256), dtype=np.float32))
        rhs_multi_ref = jnp.ones((256, 32), dtype=jnp.float8_e4m3fn)

        # Step 0 (first step, not last step) -> updates acc_scratch
        with mock.patch.object(qmm_blockwise.pl, "program_id", return_value=0):
            kernel_multi_k(lhs_ref, rhs_multi_ref, w_scales_ref, out_ref, acc_scratch)
        assert acc_scratch.arr.sum() > 0

        # Step 1 (last step, not first step) -> updates out_ref
        with mock.patch.object(qmm_blockwise.pl, "program_id", return_value=1):
            kernel_multi_k(lhs_ref, rhs_multi_ref, w_scales_ref, out_ref, acc_scratch)
        assert out_ref.arr.sum() > 0

    def test_blockwise_kernel_success_wrapper(self) -> None:
        orig_n_batch, orig_n_in = 10, 32
        orig_n_out = 200
        x = jnp.ones((orig_n_batch, orig_n_in), dtype=jnp.bfloat16)
        w_q = jnp.ones((orig_n_out, orig_n_in), dtype=jnp.int8)
        w_scale = jnp.ones((1, 1, orig_n_out), dtype=jnp.float16)

        tuned = qmm_tuned.TunedValue(16, 256, 32, n_lane_multiplier=1)

        with (
            mock.patch("jax.experimental.pallas.pallas_call") as m_pcall,
            mock.patch("vllm_torchtpu.kernels.quantized_matmul.util.validate_inputs"),
        ):
            m_pcall.return_value = lambda *args: jnp.zeros(
                (16, 256), dtype=jnp.bfloat16
            )

            out = qmm_blockwise.quantized_matmul_kernel(
                x, w_q, w_scale, block_size=32, x_q_dtype=None, tuned_value=tuned
            )
            assert out.shape == (orig_n_batch, orig_n_out)

            # Cover tuned_value is None branch
            with mock.patch(
                "vllm_torchtpu.kernels.quantized_matmul.blockwise_kernel.get_tuned_block_sizes",
                return_value=tuned,
            ):
                out_none = qmm_blockwise.quantized_matmul_kernel(
                    x, w_q, w_scale, block_size=32, x_q_dtype=None, tuned_value=None
                )
                assert out_none.shape == (orig_n_batch, orig_n_out)
