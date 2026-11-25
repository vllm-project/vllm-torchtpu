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
import types
from typing import NamedTuple

import jax.numpy as jnp
import pytest

from vllm_torchtpu.kernels.experimental.push_fused_moe import kernel as push_kernel
from vllm_torchtpu.kernels.experimental.push_fused_moe import layer as push_layer
from vllm_torchtpu.kernels.fused_moe.v2 import host, kernel
from vllm_torchtpu.kernels.fused_moe.v2 import layer as v2_layer
from vllm_torchtpu.kernels.fused_moe.v2.layer import density_build_kwargs

BIG = dict(g_local=8, capacity=128, hidden=7168, inter=2048)
SMALL = dict(g_local=32, capacity=128, hidden=512, inter=256)
SHAPES = [pytest.param(BIG, id="dsv4_pro"), pytest.param(SMALL, id="small")]

FORMATS = [
    pytest.param(host.WeightFormat.FP4, id="fp4"),
    pytest.param(host.WeightFormat.FP8, id="fp8"),
]


class _FakeChip:
    """The fields the VMEM accounting reads off a device record.

    The sublane tiling is the 32-byte rule the real chips report -- f32 tiles
    to 8 sublanes, bf16 to 16, eight-bit to 32 -- so the byte figures here are
    the shape of the real ones, not just self-consistent.
    """

    num_lanes = 128
    generation = host.MIN_GENERATION

    def __init__(self, vmem_capacity_bytes=128 * 2**20):
        self.vmem_capacity_bytes = vmem_capacity_bytes

    def get_sublane_tiling(self, dtype):
        return 32 // jnp.dtype(dtype).itemsize


@pytest.fixture
def chip():
    return _FakeChip()


def _estimate(shape, nbuf, chip, weight_format=host.WeightFormat.FP8, **kw):
    """One candidate's cost, at `fit_weight_slots`'s own defaults."""
    return host.vmem_estimate_bytes(
        **shape, nbuf=nbuf, weight_format=weight_format, info=chip, **kw
    )


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("weight_format", FORMATS)
def test_scale_tables_are_slot_deep_not_expert_deep(shape, weight_format):
    """A scale table is staged with the slab it scales, so it is nbuf deep.

    While they were `g_local` deep they were fixed cost, and giving a weight
    slot back bought proportionally less -- on the four-bit path, where the
    block-scale tables are some of the largest buffers, it bought almost
    nothing.
    """
    nbuf = 2
    assert nbuf != shape["g_local"]
    arrays = dict(
        (name, s)
        for name, s, _ in host.vmem_scratch_arrays(
            **shape, nbuf=nbuf, weight_format=weight_format
        )
    )
    for name in ("w1_vm", "w2_vm", "w1s_vm", "w2s_vm"):
        assert arrays[name][0] == nbuf, f"{name} is not slot deep"


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("weight_format", FORMATS)
def test_a_shallower_schedule_costs_strictly_less(shape, weight_format, chip):
    """What the descent trades on: fewer slots, fewer bytes, monotonically."""
    costs = [
        _estimate(shape, host.nbuf_for(d), chip, weight_format)
        for d in range(host.MAX_WEIGHT_PREFETCH_DISTANCE + 1)
    ]
    assert costs == sorted(costs)
    assert len(set(costs)) == len(costs)


@pytest.mark.parametrize("shape", SHAPES)
def test_room_at_the_ceiling_is_taken(shape, chip):
    """No descent when the full runway fits: the ceiling is the default."""
    ceiling = _estimate(shape, host.MAX_NBUF, chip)
    slots = host.fit_weight_slots(**shape, info=chip, limit=ceiling)
    assert slots.fits
    assert slots.distance == host.MAX_WEIGHT_PREFETCH_DISTANCE
    assert slots.nbuf == host.MAX_NBUF
    assert (slots.vmem_bytes, slots.limit) == (ceiling, ceiling)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("distance", range(host.MAX_WEIGHT_PREFETCH_DISTANCE + 1))
def test_the_descent_stops_at_the_first_depth_that_fits(shape, distance, chip):
    """One slot at a time, and it stops the moment it clears.

    The budget is a byte under what the next depth up costs, so the deepest
    schedule that fits is exactly `distance` -- a descent that skipped a rung,
    or dropped straight to zero, lands somewhere else.
    """
    want = _estimate(shape, host.nbuf_for(distance), chip)
    over = (
        _estimate(shape, host.nbuf_for(distance + 1), chip)
        if distance < host.MAX_WEIGHT_PREFETCH_DISTANCE
        else want + 1
    )

    slots = host.fit_weight_slots(**shape, info=chip, limit=over - 1)

    assert slots.fits
    assert slots.distance == distance
    assert slots.nbuf == host.nbuf_for(distance)
    assert slots.vmem_bytes == want


@pytest.mark.parametrize("shape", SHAPES)
def test_over_budget_at_distance_zero_is_reported_not_raised(shape, chip):
    """The builder owns the refusal, so the walk hands back the bottom rung.

    Its figures are the ones the build quotes by name: the cost with the
    runway already given up, which is what says no further shortening helps.
    """
    slots = host.fit_weight_slots(**shape, info=chip, limit=1)

    assert not slots.fits
    assert (slots.distance, slots.nbuf) == (0, 1)
    assert slots.vmem_bytes == _estimate(shape, 1, chip)
    assert slots.limit == 1


# tile_m, and a rows_alloc small enough that the ladder is buildable at all:
# the ladder is for shards whose routed experts fit inside one tile.
TILE_M = 512
ROWS_ALLOC = 1024
LADDER_ARGS = (TILE_M, ROWS_ALLOC, 8, 8)


@pytest.mark.parametrize(
    "override,want",
    [
        (None, tuple(range(128, 513, 128))),
        ("0", tuple(range(128, 513, 128))),
        ("2", (256, 512)),
        ("8", tuple(range(64, 513, 64))),
        ("1", (TILE_M,)),
        ("3", (TILE_M,)),
        ("512", (TILE_M,)),
        ("-1", (TILE_M,)),
    ],
)
def test_the_height_ladder_override(monkeypatch, override, want):
    """The override is a request the ladder still has to be able to honour.

    Unset and zero leave the built-in count. One is the ladder off, by the same
    path a count that cannot be built takes: a step that is not whole row
    blocks, or that does not divide the tile height into equal rungs, or a
    nonsense count. All of them fall back to the single full height rather than
    emitting a ladder whose rungs do not cover the tile.
    """
    monkeypatch.delenv("MOE_FUSED_EP_V2_HEIGHT_RUNGS_OVERRIDE", raising=False)
    if override is not None:
        monkeypatch.setenv("MOE_FUSED_EP_V2_HEIGHT_RUNGS_OVERRIDE", override)

    rungs = kernel.tile_height_rungs(*LADDER_ARGS)

    assert rungs == want
    # Whatever the count, the top rung is the full tile: a full tile's body is
    # never the short one.
    assert rungs[-1] == TILE_M
    assert all(step % host.ROWBLK == 0 for step in rungs)


INPUT_COMBINATIONS = [(2, None), (3, None), (2, 32), (3, 32)]
TUNED = dict(g_local=51, capacity=128, hidden=5120, inter=2560)


@pytest.mark.parametrize("input_slots,cache_rows", INPUT_COMBINATIONS)
def test_input_mechanisms_have_independent_scratch_costs(input_slots, cache_rows, chip):
    """Input3 owns a raw/scale slot; cache32 owns a separate compute buffer."""
    kw = dict(weight_format=host.WeightFormat.FP4, nbuf=2)
    legacy = host.vmem_scratch_arrays(**TUNED, **kw)
    arrays = host.vmem_scratch_arrays(
        **TUNED, **kw, input_slots=input_slots, cache_layout_rows=cache_rows
    )
    shapes = {name: shape for name, shape, _ in arrays}
    assert shapes["lhs_vm"][0] == shapes["ls_vm"][0] == input_slots
    assert shapes["out_vm"][0] == host.OUT_PARITIES
    assert ("lhs_compute_vm" in shapes) == (cache_rows is not None)
    legacy_bytes = sum(host.array_vmem_bytes(s, d, chip) for _, s, d in legacy)
    expected_extra = sum(
        host.array_vmem_bytes((1,) + s[1:], d, chip)
        for name, s, d in legacy
        if input_slots == 3 and name in ("lhs_vm", "ls_vm")
    )
    if cache_rows is not None:
        expected_extra += host.array_vmem_bytes((128, 5120), host.FP8, chip)
    scratch_bytes = sum(host.array_vmem_bytes(s, d, chip) for _, s, d in arrays)
    assert scratch_bytes == legacy_bytes + expected_extra
    assert (
        _estimate(
            TUNED,
            2,
            chip,
            host.WeightFormat.FP4,
            input_slots=input_slots,
            cache_layout_rows=cache_rows,
        )
        == _estimate(TUNED, 2, chip, host.WeightFormat.FP4) + expected_extra
    )


@pytest.mark.parametrize("input_slots,cache_rows", INPUT_COMBINATIONS)
def test_weight_depth_uses_the_selected_input_combination(
    input_slots, cache_rows, chip
):
    """An exact byte limit must select the depth for the actual scratch set."""
    kw = dict(
        weight_format=host.WeightFormat.FP4,
        input_slots=input_slots,
        cache_layout_rows=cache_rows,
    )
    limit = host.vmem_estimate_bytes(**TUNED, nbuf=2, info=chip, **kw)
    slots = host.fit_weight_slots(**TUNED, info=chip, limit=limit, **kw)
    assert slots.fits and slots.nbuf == 2 and slots.vmem_bytes == limit
    shallower = host.fit_weight_slots(**TUNED, info=chip, limit=limit - 1, **kw)
    assert shallower.fits and shallower.nbuf == 1


def _build_input_case(monkeypatch, input_slots, cache_rows, *, g_local=51, limit=None):
    from vllm_torchtpu.kernels.fused_moe.v2 import tuned_block_sizes

    info = kernel.pltpu.get_tpu_info_for_chip(kernel.pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(kernel.pltpu, "get_tpu_info", lambda: info)
    key = (info.chip_version, 128, 5120, 2560, 512)
    monkeypatch.setattr(tuned_block_sizes, "INPUT_SLOTS", {key: input_slots})
    monkeypatch.setattr(
        tuned_block_sizes,
        "INPUT_LAYOUT_ROWS",
        {} if cache_rows is None else {key: cache_rows},
    )
    if limit is not None:
        monkeypatch.setattr(host, "vmem_limit", lambda info=None: limit)
    selected = []
    original_fit = host.fit_weight_slots

    def fit(*args, **kwargs):
        result = original_fit(*args, **kwargs)
        selected.append(
            (kwargs.get("input_slots", 2), kwargs.get("cache_layout_rows"), result)
        )
        return result

    monkeypatch.setattr(host, "fit_weight_slots", fit)
    fn = kernel._build_fused_ep_moe_kernel(
        **(TUNED | dict(g_local=g_local)),
        ep=1,
        weight_format=host.WeightFormat.FP4,
        rhs_qb=512,
        ragged_rows_alloc=128,
        has_w1_bias=True,
        has_w2_bias=True,
    )
    return fn, selected, info


@pytest.mark.parametrize("input_slots,cache_rows", INPUT_COMBINATIONS)
def test_builder_selects_input_mechanisms_independently(
    monkeypatch, input_slots, cache_rows
):
    _, selected, _ = _build_input_case(monkeypatch, input_slots, cache_rows)
    assert len(selected) == 1
    assert selected[0][:2] == (input_slots, cache_rows)
    assert selected[0][2].fits


def test_builder_shrinks_weight_depth_with_input3_cache32_and_bias(monkeypatch):
    """The former g160 double-bias startup failure now fits with one weight slot."""
    _, selected, info = _build_input_case(monkeypatch, 3, 32, g_local=160)
    legacy = host.fit_weight_slots(
        **(TUNED | dict(g_local=160)),
        info=info,
        weight_format=host.WeightFormat.FP4,
        has_w1_bias=True,
        has_w2_bias=True,
    )
    assert legacy.nbuf == 2
    assert selected[0][0:2] == (3, 32)
    assert selected[0][2].fits and selected[0][2].nbuf == 1


def test_builder_retries_legacy_if_selected_input_minimum_does_not_fit(monkeypatch):
    """Extra input scratch must not remove support for a legacy-fitting shape."""
    info = kernel.pltpu.get_tpu_info_for_chip(kernel.pltpu.ChipVersion.TPU_7X, 1)
    limit = host.vmem_estimate_bytes(
        **TUNED,
        nbuf=1,
        info=info,
        weight_format=host.WeightFormat.FP4,
        has_w1_bias=True,
        has_w2_bias=True,
    )
    _, selected, _ = _build_input_case(monkeypatch, 3, 32, limit=limit)
    assert len(selected) == 2
    assert not selected[0][2].fits
    assert selected[1][:2] == (2, None)
    assert selected[1][2].fits and selected[1][2].nbuf == 1


def test_builder_still_refuses_when_legacy_minimum_does_not_fit(monkeypatch):
    with pytest.raises(ValueError, match="over the .* budget"):
        _build_input_case(monkeypatch, 3, 32, limit=1)


def test_short_height_disables_cache_without_disabling_input3(monkeypatch):
    monkeypatch.setenv("MOE_FUSED_EP_V2_HEIGHT_RUNGS_OVERRIDE", "8")
    _, selected, _ = _build_input_case(monkeypatch, 3, 32)
    assert selected[0][:2] == (3, None)


@pytest.mark.parametrize("input_slots,cache_rows", INPUT_COMBINATIONS)
@pytest.mark.parametrize("height_rungs", [4, 8])
@pytest.mark.parametrize("nbuf", [1, 2])
def test_input_combinations_trace_the_real_pallas_body(
    monkeypatch, input_slots, cache_rows, height_rungs, nbuf
):
    """Trace Ref reads, DMA semaphores and scratch unpacking for all four paths."""
    import jax

    monkeypatch.setenv("MOE_FUSED_EP_V2_HEIGHT_RUNGS_OVERRIDE", str(height_rungs))
    info = kernel.pltpu.get_tpu_info_for_chip(kernel.pltpu.ChipVersion.TPU_7X, 1)
    active_cache = cache_rows if height_rungs == 4 else None
    limit = host.vmem_estimate_bytes(
        **(TUNED | dict(g_local=1)),
        nbuf=nbuf,
        info=info,
        weight_format=host.WeightFormat.FP4,
        input_slots=input_slots,
        cache_layout_rows=active_cache,
        has_w1_bias=True,
        has_w2_bias=True,
    )
    fn, selected, _ = _build_input_case(
        monkeypatch, input_slots, cache_rows, g_local=1, limit=limit
    )
    assert selected[0][2].nbuf == nbuf
    s = jax.ShapeDtypeStruct
    tables = tuple(s((1, 1), jnp.int32) for _ in range(5))
    args = (
        tables,
        s((256,), jnp.int32),
        s((1,), jnp.int32),
        s((1,), jnp.int32),
        s((128, 5120), host.FP8),
        s((2, 128), jnp.float32),
        s((1, 640, 5120), jnp.uint32),
        s((1, 320, 5120), jnp.uint32),
        s((1, 10, 5120), jnp.float32),
        s((1, 5, 5120), jnp.float32),
        s((1, 5120), jnp.float32),
        s((1, 5120), jnp.float32),
        s((1,), jnp.int32),
        s((host.N_COUNTS, 1), jnp.int32),
        s((1, 1), jnp.int32),
    )

    def call(*xs):
        return fn(*xs[:12], recv_rows=128, visit=xs[12], counts=xs[13], rank=xs[14])

    traced = jax.make_jaxpr(call)(*args)
    calls = [eqn for eqn in traced.jaxpr.eqns if eqn.primitive.name == "pallas_call"]
    assert len(calls) == 1
    pallas = calls[0]
    name = str(pallas.params["name"])
    assert f"_nb{nbuf}" in name
    assert ("_i3" in name) == (input_slots == 3)
    assert ("_lhscache32" in name) == (cache_rows is not None and height_rungs == 4)
    # This is frontend validation, not device execution or backend allocation.
    assert pallas.params["jaxpr"].eqns


def test_multi_tile_index_window_disables_input3_without_disabling_cache(monkeypatch):
    monkeypatch.setattr(host, "TOKEN_GATHER_TILES_PER_WINDOW", 2)
    _, selected, _ = _build_input_case(monkeypatch, 3, 32)
    assert selected[0][:2] == (2, 32)


@pytest.mark.parametrize(
    "options,message",
    [
        ({"input_slots": 1}, "input_slots"),
        ({"input_slots": 4}, "input_slots"),
        ({"cache_layout_rows": 0}, "cache layout rows"),
        ({"cache_layout_rows": -32}, "cache layout rows"),
        ({"cache_layout_rows": 48}, "cache layout rows"),
    ],
)
def test_input_scratch_refuses_invalid_geometry(options, message):
    with pytest.raises(ValueError, match=message):
        host.vmem_scratch_arrays(**TUNED, nbuf=2, **options)


# --------------------------------------------------------------------------- #
# Density choices with an expert layout.
#
# A dynamic-EPLB layout gives each shard more physical slots than logical
# experts: at DeepSeek-V4.1 EP8, 48 logical experts + 1 replica + staging slots,
# 56 slots in all. Staging slots never hold rows and a replica splits an
# expert's rows without adding any, so the tile-height ladder and the VMEM-push
# choice are made from the logical experts per shard and from the slab the
# bucket gets without the extra slots.
# --------------------------------------------------------------------------- #

DENSITY_TILE_M, DENSITY_EP, DENSITY_TOPK = 128, 8, 6
# Global token buckets, 16 to 1,024 tokens per shard at EP8.
DENSITY_BUCKETS = [128, 256, 512, 1024, 2048, 4096, 8192]


def _slab(tokens, slots_per_shard):
    return host.ragged_stride_bound(
        tokens, DENSITY_TOPK, DENSITY_EP * slots_per_shard, DENSITY_TILE_M
    )


@pytest.mark.parametrize("g_local", [8, 16, 48, 56])
@pytest.mark.parametrize("rows", [1280, 2048, 7424, 12800, 49664, 65536])
def test_without_occupied_slots_the_density_choices_are_unchanged(g_local, rows):
    rungs, push = kernel.density_choices(DENSITY_TILE_M, rows, g_local, DENSITY_EP)
    assert rungs == kernel.tile_height_rungs(DENSITY_TILE_M, rows, g_local, DENSITY_EP)
    assert push == (rows <= DENSITY_EP * g_local * DENSITY_TILE_M)


@pytest.mark.parametrize("tokens", DENSITY_BUCKETS)
@pytest.mark.parametrize(
    "logical, slots",
    [pytest.param(48, 56, id="dsv41"), pytest.param(16, 24, id="small")],
)
def test_staging_slots_get_the_unplaced_density_choices(logical, slots, tokens):
    plain = kernel.density_choices(
        DENSITY_TILE_M, _slab(tokens, logical), logical, DENSITY_EP
    )
    placed = kernel.density_choices(
        DENSITY_TILE_M,
        _slab(tokens, slots),
        slots,
        DENSITY_EP,
        occupied_slots=logical,
        density_rows_alloc=_slab(tokens, logical),
    )
    assert placed == plain


def test_the_unplaced_slab_decides_at_the_one_tile_bound():
    """At 8 x 48 x 128 = 49,152 rows the unplaced slab sits on the bound and
    the placed one (49,280) just past it: occupied slots alone would flip
    both choices; the unplaced slab keeps the unplaced build's."""
    plain = kernel.density_choices(DENSITY_TILE_M, 49152, 48, DENSITY_EP)
    assert plain == ((32, 64, 96, 128), True)
    occupied_only = kernel.density_choices(
        DENSITY_TILE_M, 49280, 56, DENSITY_EP, occupied_slots=48
    )
    assert occupied_only == ((128,), False)
    both = kernel.density_choices(
        DENSITY_TILE_M,
        49280,
        56,
        DENSITY_EP,
        occupied_slots=48,
        density_rows_alloc=49152,
    )
    assert both == plain


def test_density_inputs_never_exceed_the_build():
    """Each case would flip the push choice if its input were not clamped:
    99 slots would put 49,664 rows under the bound, and 10**6 rows would put
    a build sitting on the bound (49,152) past it."""
    assert kernel.density_choices(
        DENSITY_TILE_M, 49664, 48, DENSITY_EP, occupied_slots=99
    ) == kernel.density_choices(DENSITY_TILE_M, 49664, 48, DENSITY_EP)
    assert kernel.density_choices(
        DENSITY_TILE_M, 49152, 48, DENSITY_EP, density_rows_alloc=10**6
    ) == kernel.density_choices(DENSITY_TILE_M, 49152, 48, DENSITY_EP)


@pytest.mark.parametrize(
    "e_total, e_logical, tokens, occupied",
    [
        pytest.param(448, 384, 8192, 48, id="dsv41"),
        # 380 / 8 = 47.5 logical experts per shard: a shard holds up to 48.
        pytest.param(448, 380, 64, 48, id="rounds_up"),
    ],
)
def test_density_build_kwargs_give_a_layout_the_unplaced_inputs(
    e_total, e_logical, tokens, occupied
):
    assert density_build_kwargs(e_total, e_logical, 8, tokens, 6, 128) == {
        "occupied_slots": occupied,
        "density_rows_alloc": host.ragged_stride_bound(tokens, 6, e_logical, 128),
    }


def test_density_build_kwargs_are_empty_without_extra_slots():
    assert density_build_kwargs(384, 384, 8, 8 * 1024, 6, 128) == {}


class _BuilderReached(Exception):
    pass


class _Call(NamedTuple):
    """One layer call: `tokens` rows of x over `ep` shards, `logical` experts
    in `slots` physical slots (an identity layout when slots > logical), and
    the token count the layer routes after splitting rows over `groups`."""

    ep: int
    logical: int
    slots: int
    tokens: int
    groups: tuple | None
    routed: int


CALL_TOPK = 6
PAIRS = ((0, 1), (2, 3), (4, 5), (6, 7))
# One shard with 48 logical experts in 56 slots; and eight shards with 380 in
# 448, where a shard holds up to 48 logical experts and pairs of replicas
# split each shard's 11 tokens into 6, aligned to 8, so the layer routes 64
# tokens rather than 88. The three slabs then all differ: unplaced 896,
# placed 1,024, and 1,152 if the slab were sized from the 88 tokens given.
PLACED_CALLS = [
    pytest.param(_Call(1, 48, 56, 56, None, 56), id="ep1"),
    pytest.param(_Call(8, 380, 448, 88, PAIRS, 64), id="ep8_pairs"),
]
UNPLACED_CALLS = [
    pytest.param(_Call(1, 48, 48, 56, None, 56), id="ep1"),
    pytest.param(_Call(8, 384, 384, 88, PAIRS, 64), id="ep8_pairs"),
]
LAYER_BUILDERS = [
    pytest.param(v2_layer, "build_fused_ep_moe_kernel", "fused_ep_moe_v2", id="v2"),
    pytest.param(push_layer, "build_push_moe_kernel", "fused_ep_moe_push", id="push"),
]


def _builder_kwargs(monkeypatch, module, builder, entry, call):
    """Call a layer entry point up to its kernel build; return the build's
    kwargs.

    The mesh is a stand-in carrying what the layers read off one before the
    build -- the axis name and its size -- so every case runs on one CPU
    device."""
    seen = {}

    def spy(**kwargs):
        seen.update(kwargs)
        raise _BuilderReached

    monkeypatch.setattr(module, builder, spy)
    hidden, inter = 256, 128
    mesh = types.SimpleNamespace(axis_names=("ep",), shape={"ep": call.ep})
    layout = (
        dict(expert_slot=jnp.arange(call.logical, dtype=jnp.int32)[:, None])
        if call.slots > call.logical
        else {}
    )
    with pytest.raises(_BuilderReached):
        getattr(module, entry)(
            jnp.zeros((call.tokens, hidden), jnp.bfloat16),
            jnp.zeros((call.slots, hidden, 2 * inter), jnp.float8_e4m3fn),
            jnp.zeros((call.slots, inter, hidden), jnp.float8_e4m3fn),
            jnp.zeros((call.slots, 2 * inter), jnp.float32),
            jnp.zeros((call.slots, hidden), jnp.float32),
            jnp.zeros((call.tokens, call.logical), jnp.float32),
            topk=CALL_TOPK,
            renormalize=True,
            mesh=mesh,
            capacity=128,
            token_replica_groups=call.groups,
            **layout,
        )
    return seen


@pytest.mark.parametrize("call", PLACED_CALLS)
@pytest.mark.parametrize("module, builder, entry", LAYER_BUILDERS)
def test_layers_build_a_layout_with_the_unplaced_density_inputs(
    monkeypatch, module, builder, entry, call
):
    seen = _builder_kwargs(monkeypatch, module, builder, entry, call)
    assert seen["g_local"] == call.slots // call.ep
    assert seen["ragged_rows_alloc"] == host.ragged_stride_bound(
        call.routed, CALL_TOPK, call.slots, 128
    )
    assert seen["occupied_slots"] == -(-call.logical // call.ep)
    assert seen["density_rows_alloc"] == host.ragged_stride_bound(
        call.routed, CALL_TOPK, call.logical, 128
    )
    assert seen["density_rows_alloc"] < seen["ragged_rows_alloc"]


@pytest.mark.parametrize("call", UNPLACED_CALLS)
@pytest.mark.parametrize("module, builder, entry", LAYER_BUILDERS)
def test_layers_pass_no_density_inputs_without_a_layout(
    monkeypatch, module, builder, entry, call
):
    seen = _builder_kwargs(monkeypatch, module, builder, entry, call)
    assert seen["g_local"] == call.logical // call.ep
    assert "occupied_slots" not in seen
    assert "density_rows_alloc" not in seen


# The kernel modules and their builders' arguments beyond the shared ones.
KERNEL_BUILDS = [
    pytest.param(kernel, "_build_fused_ep_moe_kernel", {}, id="v2"),
    pytest.param(
        push_kernel,
        "_build_push_moe_kernel",
        dict(push_tokens=1024, sc_prepush=2),
        id="push",
    ),
]


@pytest.mark.parametrize("module, builder, extra", KERNEL_BUILDS)
def test_kernels_make_their_density_choices_from_the_density_inputs(
    monkeypatch, module, builder, extra
):
    """The 1024-token bucket at EP8, 48 logical experts in 56 slots: counting
    the staging slots puts the placed slab (49,792 rows) under the one-tile
    bound, which takes the short-tile ladder and ships from VMEM; the density
    inputs keep the unplaced build's choices, full tiles and the HBM slab."""
    info = module.pltpu.get_tpu_info_for_chip(module.pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(module.pltpu, "get_tpu_info", lambda: info)
    placed, unplaced = _slab(8192, 56), _slab(8192, 48)
    calls = []
    real = kernel.density_choices

    def recording(*args):
        result = real(*args)
        calls.append((args, result))
        return result

    monkeypatch.setattr(module, "density_choices", recording)
    getattr(module, builder)(
        **(TUNED | dict(g_local=56)),
        ep=8,
        weight_format=host.WeightFormat.FP4,
        rhs_qb=512,
        ragged_rows_alloc=placed,
        occupied_slots=48,
        density_rows_alloc=unplaced,
        **extra,
    )
    assert calls == [((128, placed, 56, 8, 48, unplaced), ((128,), False))]
    assert kernel.density_choices(128, placed, 56, 8) == ((32, 64, 96, 128), True)
