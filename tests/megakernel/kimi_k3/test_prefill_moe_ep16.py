# SPDX-License-Identifier: Apache-2.0
"""EP16/TP2 fused prefill MoE on CPU: the loader's interleaved gate/up tile,
the stepped down reduction and the single-launch 16K geometry."""

import functools
import json
import types

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

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


def _mxfp4(rng, rows, cols, exponent_range=(122, 130)):
    """Random MXFP4 weights: FP4 codes [rows, cols] and E8M0 exponents per 32
    rows, with their dequantized FP32 value."""
    codes = rng.integers(0, 16, size=(rows, cols))
    exponents = rng.integers(*exponent_range, size=(rows // 32, cols)).astype(np.uint8)
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
            _Ref(lhs),
            rhs,
            cfgs=cfgs,
            bucket_rows=rows,
            overlap_work=lambda _: None,
            fold=rank_loader.GATE_UP_E5M2_FOLD,
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


@pytest.mark.parametrize("fold", [0, rank_loader.GATE_UP_E5M2_FOLD])
def test_e5m2_weights_are_the_mxfp4_values_over_the_window(fold):
    # Every code at both ends of the window and between: the E5M2 cast is the
    # MXFP4 value times 2 ** fold, bit for bit.
    low, high = rank_loader.E5M2_BLOCK_EXPONENTS
    exponents = np.arange(127 + low - fold, 127 + high - fold + 1, dtype=np.uint8)
    codes = np.tile(np.arange(16), (32 * exponents.size, 8))  # [rows, 128]
    columns = np.broadcast_to(exponents[:, None], (exponents.size, 128))
    values = FP4_VALUES[codes] * np.repeat(
        2.0 ** (columns.astype(np.float32) - 127), 32, 0
    )
    rhs = kernels.WeightsRef(
        weight=_Ref(FP4_VALUES[codes].astype(ml_dtypes.float4_e2m1fn)),
        scale=_Ref(np.ascontiguousarray(columns)),
        bias=None,
    )
    rows = codes.shape[0]
    cast = fused._dequantized(
        rhs,
        _cfgs(tile_m=256, tile_k=rows, fuse_act=None),
        pl.ds(0, rows),
        pl.ds(0, 128),
        fold,
        jnp.float8_e5m2,
    )
    assert cast.dtype == jnp.float8_e5m2
    np.testing.assert_array_equal(np.asarray(cast, np.float32), values * 2.0**fold)


def test_folded_gate_up_products_equal_unfolded():
    # The fold scales weights and products by powers of two: FP32 results match
    # bit for bit (CPU dots; the MXU check is the device's).
    rng = np.random.default_rng(4)
    k, mid, rows = 512, 512, 128
    weights, exponents, _ = _mxfp4(rng, k, 2 * mid)
    lhs = rng.standard_normal((rows, k)).astype(ml_dtypes.bfloat16)
    rhs = kernels.WeightsRef(weight=_Ref(weights), scale=_Ref(exponents), bias=None)
    cfgs = _cfgs(tile_m=128, tile_k=k, fuse_act="situ:4:25")
    folded, plain = (
        [
            np.asarray(value)
            for _, value in fused._product_panels(
                _Ref(lhs),
                rhs,
                cfgs=cfgs,
                bucket_rows=rows,
                overlap_work=lambda _: None,
                fold=fold,
            )
        ]
        for fold in (rank_loader.GATE_UP_E5M2_FOLD, 0)
    )
    for a, b in zip(folded, plain, strict=True):
        np.testing.assert_array_equal(a, b)


def _gate_up_window():
    """The gate/up E8M0 exponents the E5M2 cast holds, smallest and largest."""
    low, high = rank_loader.E5M2_BLOCK_EXPONENTS
    fold = rank_loader.GATE_UP_E5M2_FOLD
    return 127 + low - fold, 127 + high - fold


def test_ep16_expert_readers_reject_gate_up_scales_outside_the_e5m2_window():
    first, last = _gate_up_window()
    inside = np.array([[first, last]], np.uint8)
    for key, value in (
        ("expert_gus@0/1", inside),
        ("expert_ds@0/1", np.array([[1, 254]], np.uint8)),
        ("expert_gus@0/2", np.zeros((1, 1), np.float32)),  # not E8M0 bytes
    ):
        _, _, reader = rank_loader._e5m2_checked((key, value.nbytes, lambda v=value: v))
        np.testing.assert_array_equal(reader(), value)
    for value in (inside - 1, inside + 1):
        job = ("expert_gus@1/4", value.nbytes, lambda v=value: v)
        with pytest.raises(ValueError, match="E5M2 window"):
            rank_loader._e5m2_checked(job)[2]()


@pytest.mark.parametrize(("name", "columns"), [("expert_gus", 512), ("expert_ds", 4)])
def test_regrouped_expert_scales_are_checked_as_read(tmp_path, name, columns):
    # A two-quarter pre-shard of uint8 E8M0 scales, one byte below the gate/up
    # window: the gate/up read fails, the down read passes.
    first, _ = _gate_up_window()
    shape = (2, 112, 2, columns)
    dirs = []
    for rank in range(2):
        directory = tmp_path / f"rank{rank:02d}"
        directory.mkdir()
        scales = np.full(shape, first, np.uint8)
        scales[1, 3, 0, 0] = first - 1 if rank == 1 else first
        scales.tofile(directory / f"{name}.bin")
        spec = {"files": [f"{name}.bin"], "layer_ranges": [[0, 2]]}
        spec.update(shape=list(shape), dtype="uint8")
        index = {"rank": rank, "layers": 3, "families": {name: spec}}
        (directory / "index.json").write_text(json.dumps(index))
        dirs.append(rank_loader.RankDir(tmp_path, rank, 3))
    ((_, _, read),) = dirs[0].regrouped_expert_parts(name, dirs, 0, 2)
    if name == "expert_ds":
        assert read().min() == first - 1
        return
    with pytest.raises(ValueError, match="E5M2 window"):
        read()


def _latent_rows(rng, rows):
    """BF16 latent rows over a wide spread of magnitudes, one of them zero."""
    scales = 2.0 ** rng.integers(-12, 8, size=(rows, 1))
    latent = (rng.standard_normal((rows, 3584)) * scales).astype(ml_dtypes.bfloat16)
    latent[3] = 0
    return latent


def _slab_round_trip(latent):
    """``_encode_latent`` then ``_decode_latent`` in interpret mode."""
    rows = latent.shape[0]

    def kernel(latent_ref, slab_ref, rows_ref, inverse_ref):
        fused._encode_latent(latent_ref, slab_ref)
        fused._decode_latent(slab_ref, rows_ref, inverse_ref)

    return pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct((rows, 4, fused._SLAB_LANES), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((rows, 3584), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((rows, 128), jnp.float32),
        ),
        interpret=True,
    )(jnp.asarray(latent))


def test_latent_slabs_round_trip_with_power_of_two_row_scales():
    rng = np.random.default_rng(5)
    latent = _latent_rows(rng, 64)
    slab, rows, inverse = (np.asarray(x) for x in _slab_round_trip(latent))
    values = latent.astype(np.float32)
    largest = np.abs(values).max(axis=1, keepdims=True)
    exponent = np.clip(np.floor(np.log2(np.where(largest > 0, largest, 1))), -100, 120)
    exponent[largest == 0] = -100
    expected = (values * 2.0 ** (7 - exponent)).astype(ml_dtypes.float8_e4m3fn)
    np.testing.assert_array_equal(rows.view(np.uint8), expected.view(np.uint8))
    np.testing.assert_array_equal(
        inverse, np.broadcast_to(2.0 ** (exponent - 7), (64, 128))
    )
    # The slab holds each row's four 896-wide sub-rows, one word per lane.
    np.testing.assert_array_equal(
        slab[:, :, :896].reshape(64, 3584).view(np.uint8), expected.view(np.uint8)
    )
    restored = rows.astype(np.float32) * inverse[:, :1]
    assert np.all(np.abs(restored) < 448 * inverse[:, :1])  # never saturated
    error = np.abs(restored - values)
    # E4M3 rounding: half a mantissa step (2^-4 relative) for normal values,
    # half the subnormal step (2^-10 of the row scale's unit) below.
    bound = np.maximum(np.abs(values) * 2.0**-4, inverse[:, :1] * 2.0**-10)
    assert np.all(error <= bound)


def test_uncapped_consumer_reads_the_slabs_as_their_bf16_values():
    rng = np.random.default_rng(6)
    latent = _latent_rows(rng, 32)
    slab, rows, inverse = (np.asarray(x) for x in _slab_round_trip(latent))
    with pltpu.force_tpu_interpret_mode():
        decoded = np.asarray(fused._latent_bf16(jnp.asarray(slab)))
    expected = (rows.astype(np.float32) * inverse[:, :1]).astype(ml_dtypes.bfloat16)
    assert decoded.shape == (32, 2, 1792)
    np.testing.assert_array_equal(decoded.reshape(32, 3584), expected)


def test_e4m3_gate_up_tile_stays_within_the_bound_of_the_float64_product():
    # The e4m3 latent rows x exact E5M2 weights, rescaled per row: products
    # within 3.5% RMS of float64 BF16-latent x MXFP4-value products, rows within
    # cosine 0.999, all finite.
    rng = np.random.default_rng(7)
    k, mid, rows = 3584, 256, 128
    latent = _latent_rows(rng, rows)
    latent[3] = latent[4]  # keep every row's product nonzero
    weights, exponents, values = _mxfp4(rng, k, 2 * mid)
    _, decoded, inverse = (np.asarray(x) for x in _slab_round_trip(latent))
    fold = rank_loader.GATE_UP_E5M2_FOLD
    rhs = kernels.WeightsRef(weight=_Ref(weights), scale=_Ref(exponents), bias=None)
    cfgs = _cfgs(tile_m=128, tile_k=k, fuse_act="situ:4:25")
    actual = np.concatenate(
        [
            np.asarray(value)
            for _, value in fused._product_panels(
                _Ref(decoded),
                rhs,
                cfgs=cfgs,
                bucket_rows=rows,
                overlap_work=lambda _: None,
                fold=fold,
                rescale=jnp.asarray(inverse[:, :1] * 2.0**-fold),
            )
        ],
        axis=1,
    )
    expected = latent.astype(np.float64) @ values.astype(np.float64)
    assert np.all(np.isfinite(actual))
    diff = actual - expected
    assert np.sqrt(np.mean(diff**2)) <= 0.035 * np.sqrt(np.mean(expected**2))
    cosine = np.sum(actual * expected, 1) / (
        np.linalg.norm(actual, axis=1) * np.linalg.norm(expected, axis=1)
    )
    assert cosine.min() >= 0.999


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


def _combine_call(kernel_body, sums, ids, source):
    """Run ``kernel_body(bounds, combine, source)`` over token sums in Pallas
    interpret mode; returns the updated sums."""
    tokens = sums.shape[0] - fused._RMW_DUMMY_SLOTS

    def kernel(bounds, ids_ref, source_ref, sums_in, sums_out):
        sums_out[...] = sums_in[...]
        combine = fused._TokenCombine(sums_out, fused._FlatMetadata(ids_ref), tokens)
        kernel_body(bounds, combine, source_ref)

    def call(bounds):
        return np.asarray(
            pl.pallas_call(
                kernel,
                out_shape=jax.ShapeDtypeStruct(sums.shape, jnp.float32),
                grid_spec=pltpu.PrefetchScalarGridSpec(
                    num_scalar_prefetch=2,
                    in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)] * 2,
                    out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
                ),
                interpret=True,
            )(
                jnp.asarray(bounds, jnp.int32),
                jnp.asarray(ids.reshape(-1, 128)),
                jnp.asarray(source),
                jnp.asarray(sums),
            )
        )

    return call


def _combine_inputs(rng, *, tokens, start, end):
    """Token sums, an expert tile's packed rows and the routes' distinct
    ascending token ids (one expert holds a token at most once)."""
    sums = rng.standard_normal((tokens + fused._RMW_DUMMY_SLOTS, 2, 256))
    source = rng.standard_normal((256, 2, 256)).astype(ml_dtypes.bfloat16)
    ids = np.zeros(1024, np.int32)
    ids[start:end] = np.sort(rng.choice(tokens, end - start, replace=False))
    return sums.astype(np.float32), source, ids


@pytest.mark.parametrize(
    "start, rows, first, unroll",
    [
        (37, 211, 64, 32),  # later column: chunks, batches and a clamped tail
        (16, 240, 112, 8),  # first column (rows past the GU and first-down overlap)
        (5, 37, 0, 32),  # a short last tile: batches and tail only
        (100, 30, 64, 32),  # every row already combined during the products
    ],
)
def test_token_combine_visit_is_the_scatter_add(start, rows, first, unroll):
    rng = np.random.default_rng(start)
    tokens, end = 1024, start + rows
    sums, source, ids = _combine_inputs(rng, tokens=tokens, start=start, end=end)

    def body(bounds, combine, source_ref):
        combine.visit(
            source_ref, bounds[0], bounds[1], sublane=16, first=first, unroll=unroll
        )

    actual = _combine_call(body, sums, ids, source)([start, end])
    expected = sums.copy()
    begin = start // 16 * 16
    for route in range(start + first, end):
        expected[ids[route]] += source[route - begin].astype(np.float32)
    # Bit-identical: each token takes one FP32 add per tile; padding rows of
    # the clamped tail land in the dummy slots past the token rows.
    np.testing.assert_array_equal(actual[:tokens], expected[:tokens])


def test_token_combine_clamped_rows_past_the_tile_go_to_dummy_slots():
    rng = np.random.default_rng(7)
    tokens, start, end = 1024, 20, 25
    sums, source, ids = _combine_inputs(rng, tokens=tokens, start=start, end=end)

    def body(bounds, combine, source_ref):
        combine.clamped(source_ref, 16, bounds[0], bounds[1], 8, guard=True)
        combine.clamped(source_ref, 16, bounds[0], bounds[1], 2, guard=False)

    actual = _combine_call(body, sums, ids, source)([start, end])
    expected = sums.copy()
    for route in range(start, end):
        expected[ids[route]] += source[route - 16].astype(np.float32)
    np.testing.assert_array_equal(actual[:tokens], expected[:tokens])
    assert (actual[tokens:] != sums[tokens:]).any()


@pytest.mark.parametrize("half, width", [(256, 512), (512, 896)])
def test_token_combine_emit_copies_the_column_and_restarts_the_sums(half, width):
    rng = np.random.default_rng(width)
    tokens = 512
    sums = rng.standard_normal((tokens + fused._RMW_DUMMY_SLOTS, 2, half))
    sums = sums.astype(np.float32)

    def kernel(sums_in, sums_out, subtotals):
        sums_out[...] = sums_in[...]
        combine = fused._TokenCombine(sums_out, None, tokens)
        combine.emit(subtotals, rows=128)

    restarted, subtotals = pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct(sums.shape, jnp.float32),
            jax.ShapeDtypeStruct((tokens, width), jnp.bfloat16),
        ),
        interpret=True,
    )(jnp.asarray(sums))
    expected = sums[:tokens].reshape(tokens, 2 * half)[:, :width]
    np.testing.assert_array_equal(
        np.asarray(subtotals), expected.astype(ml_dtypes.bfloat16)
    )
    restarted = np.asarray(restarted)
    assert not restarted[:tokens].any()
    np.testing.assert_array_equal(restarted[tokens:], sums[tokens:])


def _tiles(counts, tile, sublane):
    """fill_metadata's tiles of expert-major routes: group ids and offsets."""
    groups, offsets, start = [], [0], 0
    for expert, count in enumerate(counts):
        cur, end = start, start + count
        while cur < end:
            cur += min(tile - cur % sublane, end - cur)
            groups.append(expert)
            offsets.append(cur)
        start = end
    return np.asarray(groups, np.int32), np.asarray(offsets, np.int32)


def test_single_tile_columns_feed_every_tile_its_rows_and_expert_block():
    tile, sublane, depth, width, columns = 32, 8, 64, 128, (1, 3)
    counts = [5, 0, 40, 7, 33, 1]
    groups, offsets = _tiles(counts, tile, sublane)
    ng, experts = len(groups), len(counts)
    rng = np.random.default_rng(4)
    rows = rng.standard_normal((sum(counts) + tile, 128)).astype(np.float32)
    weight = rng.standard_normal((experts, depth, 3 * width)).astype(np.float32)
    scale = rng.integers(0, 255, (experts, depth // 32, 3 * width)).astype(np.uint8)
    steps = (columns[1] - columns[0]) * ng

    def kernel(groups_ref, offsets_ref, rows_hbm, weight_hbm, scale_hbm, seen, used):
        metadata = kernels.MetadataRef(
            gm_id_to_group_id=groups_ref, gm_id_to_m_offset=offsets_ref
        )

        def step(x, w, g, n):
            index = (n - columns[0]) * ng + g
            seen[index] = x[...]
            used[index] = w.weight[...] + w.scale[...][:1].astype(jnp.float32)

        pl.run_scoped(
            functools.partial(
                fused._single_tile_columns,
                step,
                lambda n: None,
                metadata,
                ng,
                rows_hbm,
                kernels.WeightsRef(weight=weight_hbm, scale=scale_hbm, bias=None),
                columns=columns,
                sublane=sublane,
            ),
            rows_buf=pltpu.VMEM((2, tile, 128), jnp.float32),
            weight_buf=pltpu.VMEM((2, depth, width), jnp.float32),
            scale_buf=pltpu.VMEM((2, depth // 32, width), jnp.uint8),
            sems=pltpu.SemaphoreType.DMA((3, 2)),
            run=pltpu.SMEM((1,), jnp.int32),
        )

    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    # DMAs land at their waits and races are reported: a tile reading a
    # buffer before its copy was waited for, or while the next copy into it
    # runs, sees the wrong data.
    seen, used = pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct((steps, tile, 128), jnp.float32),
            jax.ShapeDtypeStruct((steps, depth, width), jnp.float32),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2, in_specs=[hbm] * 3, out_specs=(vmem, vmem)
        ),
        interpret=pltpu.InterpretParams(
            dma_execution_mode="on_wait", detect_races=True, uninitialized_memory="nan"
        ),
    )(jnp.asarray(groups), jnp.asarray(offsets), rows, weight, scale)
    seen, used = np.asarray(seen), np.asarray(used)
    for n in range(*columns):
        cols = slice(n * width, (n + 1) * width)
        for g in range(ng):
            index = (n - columns[0]) * ng + g
            first = offsets[g] // sublane * sublane
            count = -(-offsets[g + 1] // sublane) * sublane - first
            np.testing.assert_array_equal(seen[index, :count], rows[first:][:count])
            block = weight[groups[g], :, cols] + scale[groups[g], :1, cols]
            np.testing.assert_array_equal(used[index], block)
