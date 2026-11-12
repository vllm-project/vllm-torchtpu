# SPDX-License-Identifier: Apache-2.0
"""EP16/TP2 fused prefill MoE on CPU: the loader's interleaved gate/up tile,
the stepped down reduction and the single-launch 16K geometry."""

import types

import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest

from vllm_torchtpu.megakernel.kimi_k3.kernels import prefill_megakernel as kernels
from vllm_torchtpu.megakernel.kimi_k3.kernels import rank_loader
from vllm_torchtpu.megakernel.kimi_k3.kernels.prefill_moe import fused

pytestmark = pytest.mark.cpu_test

FP4_VALUES = np.array(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], np.float32
)


class _Ref:
    """A numpy array behind the slicing a Pallas kernel's refs offer."""

    def __init__(self, array):
        self.array = array

    @property
    def shape(self):
        return self.array.shape

    @property
    def dtype(self):
        return self.array.dtype

    @property
    def at(self):
        array = self.array

        class _At:
            def __getitem__(self, index):
                index = index if isinstance(index, tuple) else (index,)
                index = tuple(
                    slice(int(i.start), int(i.start) + int(i.size))
                    if hasattr(i, "size") and not isinstance(i, slice)
                    else i
                    for i in index
                )
                return _Ref(array[index])

        return _At()

    def reshape(self, *shape):
        return _Ref(self.array.reshape(*shape))

    def __getitem__(self, index):
        return jnp.asarray(self.array[index])


@pytest.fixture(autouse=True)
def tpu_info(monkeypatch):
    info = types.SimpleNamespace(mxu_column_size=256, num_lanes=128)
    monkeypatch.setattr(fused.pltpu, "get_tpu_info", lambda: info)
    monkeypatch.setattr(kernels.pltpu, "get_tpu_info", lambda: info)


def _mxfp4(rng, rows, cols):
    """Random MXFP4 weights: FP4 codes [rows, cols] and E8M0 exponents per 32
    rows, with their dequantized FP32 value."""
    codes = rng.integers(0, 16, size=(rows, cols))
    exponents = rng.integers(122, 130, size=(rows // 32, cols)).astype(np.uint8)
    values = FP4_VALUES[codes] * np.repeat(
        2.0 ** (exponents.astype(np.float32) - 127), 32, 0
    )
    weights = FP4_VALUES[codes].astype(ml_dtypes.float4_e2m1fn)
    return weights, exponents, values


def _cfgs(*, tile_m, tile_k, fuse_act):
    return types.SimpleNamespace(
        dims=types.SimpleNamespace(size_k=tile_k),
        tiles=types.SimpleNamespace(tile_m=tile_m, tile_k=tile_k),
        rhs_cfgs=types.SimpleNamespace(has_bias=False, quant_block_size=32),
        lhs_cfgs=types.SimpleNamespace(dtype=jnp.bfloat16),
        acc_dtype=jnp.float32,
        fuse_act=fuse_act,
    )


def _interleave_quarters(gate, up):
    """An EP16 rank's resident gate/up row as the loader builds it from the
    checkpoint's TP4 quarters ([gate | up] of 768 columns each)."""
    columns = [slice(768 * q, 768 * q + 768) for q in range(gate.shape[-1] // 768)]
    quarters = [
        np.ascontiguousarray(np.concatenate((gate[..., c], up[..., c]), -1))
        for c in columns
    ]
    return rank_loader._join_quarters("expert_gu", quarters)


def test_loader_interleave_is_the_gmm_lane_interleave():
    rng = np.random.default_rng(0)
    gate = rng.standard_normal((8, 1536)).astype(np.float32)
    up = rng.standard_normal((8, 1536)).astype(np.float32)
    np.testing.assert_array_equal(
        _interleave_quarters(gate, up),
        np.asarray(kernels.interleave_lane(jnp.asarray(gate), jnp.asarray(up))),
    )


def test_interleaved_gate_up_tile_matches_separate_projections():
    rng = np.random.default_rng(1)
    k, mid, rows = 512, 1536, 256
    gate_w, gate_s, gate = _mxfp4(rng, k, mid)
    up_w, up_s, up = _mxfp4(rng, k, mid)
    weights = _interleave_quarters(gate_w, up_w)
    scales = _interleave_quarters(gate_s, up_s)
    lhs = rng.standard_normal((rows, k)).astype(ml_dtypes.bfloat16)
    rhs = kernels.WeightsRef(weight=_Ref(weights), scale=_Ref(scales), bias=None)
    cfgs = _cfgs(tile_m=256, tile_k=k, fuse_act="situ:4:25")
    panels = [
        kernels.apply_act_fn(value, cfgs.fuse_act)
        for _, value in fused._product_panels(
            _Ref(lhs), rhs, cfgs=cfgs, bucket_rows=rows, overlap_work=lambda _: None
        )
    ]
    actual = np.concatenate([np.asarray(p) for p in panels], axis=1)
    x = lhs.astype(np.float32)
    deq = lambda w: w.astype(ml_dtypes.bfloat16).astype(np.float32)  # noqa: E731
    expected = np.asarray(
        kernels.situ_and_mul(
            jnp.asarray(x @ deq(gate)), jnp.asarray(x @ deq(up)), 4.0, 25.0
        )
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("width", [896, 512])
def test_down_tile_reduces_in_768_row_steps(width):
    rng = np.random.default_rng(2)
    k, rows = 1536, 256
    down_w, down_s, down = _mxfp4(rng, k, width)
    lhs = rng.standard_normal((rows, k)).astype(ml_dtypes.bfloat16)
    rhs = kernels.WeightsRef(weight=_Ref(down_w), scale=_Ref(down_s), bias=None)
    combined = []
    actual = fused._expert_product(
        _Ref(lhs),
        rhs,
        cfgs=_cfgs(tile_m=256, tile_k=k, fuse_act=None),
        bucket_rows=rows,
        overlap_work=combined.append,
    )
    # The combine of the previous tile runs once per column panel.
    assert combined == list(range(-(-width // 256)))
    expected = lhs.astype(np.float32) @ down.astype(ml_dtypes.bfloat16).astype(
        np.float32
    )
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-4, atol=1e-2)


def test_route_capacity_and_single_launch_16k():
    assert [fused._route_capacity(t) for t in (512, 2048, 4096, 8192, 16384)] == [
        8192,
        32768,
        65536,
        65536,
        65536,
    ]
    # Token ids and weights of every compact route fit half of SMEM.
    assert fused._route_capacity(16384) * 8 <= 512 * 1024
    assert 16384 not in fused._COMPOUND_PARTS
    assert fused._OVERFLOW_PARTS[16384] == (8192, 8192)
    for window, parts in fused._COMPOUND_PARTS.items():
        assert sum(parts) == window and max(parts) <= 8192


def test_16k_reduce_ring_regions_cover_the_column_block():
    # 512 rows per rank, 512-column blocks: twelve streams tile the block.
    regions = kernels._regions(512, 512)
    covered = np.zeros((512, 512), np.int32)
    for row, rows, col, cols in regions:
        covered[row : row + rows, col : col + cols] += 1
    assert len(regions) == 12
    np.testing.assert_array_equal(covered, 1)
