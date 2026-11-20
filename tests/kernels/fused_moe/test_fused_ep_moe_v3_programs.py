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
"""Two of fused EP MoE v3's Pallas programs, run directly in TPU interpret
mode on CPU against references written out here (the routing message's
padding on a TPU7x chip too).

``select_kernel.select_top_k`` (the router's top-k and its renormalized
weights) and ``combine.weighted_sum`` (each token's k-ordered weighted sum of
its expert rows) hold the layer's selection and combine arithmetic and run
as single-device programs. The layer as a whole does not run in the
interpreter: its token exchange returns DMA semaphores from one Pallas
program to the next, and the TPU interpreter cannot return a semaphore from
a program. The layer's own TPU test, which comes with the layer, runs it end
to end on an eight-core TPU host.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src.pallas.mosaic import tpu_info
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.fused_moe.v3 import combine, select_kernel

LANES = 128


def _tpu7x():
    return pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)


@pytest.fixture
def interpret(monkeypatch):
    """TPU interpret mode with a TPU7x chip record (the programs size their
    VMEM from it)."""
    monkeypatch.setattr(pltpu, "get_tpu_info", _tpu7x)
    monkeypatch.setattr(tpu_info, "get_tpu_info", _tpu7x)
    with pltpu.force_tpu_interpret_mode(pltpu.InterpretParams()):
        yield


@pytest.fixture(
    params=[
        pytest.param("interpret", marks=pytest.mark.cpu_test),
        pytest.param("tpu", marks=pytest.mark.tpu_test),
    ]
)
def interpret_or_chip(request, monkeypatch):
    """The program in TPU interpret mode on CPU, or on the TPU chip."""
    if request.param == "tpu":
        yield
        return
    monkeypatch.setattr(pltpu, "get_tpu_info", _tpu7x)
    monkeypatch.setattr(tpu_info, "get_tpu_info", _tpu7x)
    with pltpu.force_tpu_interpret_mode(pltpu.InterpretParams()):
        yield


def _logits(rows, experts, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((rows, experts)).astype(np.float32)


# ---- select_top_k ----


@pytest.mark.cpu_test
@pytest.mark.parametrize("renormalize", [False, True], ids=["raw", "renormalized"])
@pytest.mark.parametrize("experts,top_k", [(64, 4), (512, 10)])
def test_select_top_k_matches_lax_top_k(interpret, renormalize, experts, top_k):
    logits = _logits(32, experts)
    weights, indices = select_kernel.select_top_k(
        jnp.asarray(logits), top_k=top_k, block_rows=8, renormalize=renormalize
    )
    values, want_indices = jax.lax.top_k(jnp.asarray(logits), top_k)
    np.testing.assert_array_equal(np.asarray(indices), np.asarray(want_indices))
    if renormalize:
        want = jax.nn.softmax(values, axis=-1)
        np.testing.assert_allclose(np.asarray(weights), want, rtol=1e-6, atol=0)
    else:
        np.testing.assert_array_equal(np.asarray(weights), np.asarray(values))


@pytest.mark.cpu_test
def test_select_top_k_breaks_ties_toward_the_lower_expert(interpret):
    logits = _logits(8, 64)
    logits[:, 5] = logits[:, 9] = logits[:, 40] = 10.0  # a three-way tie
    _, indices = select_kernel.select_top_k(
        jnp.asarray(logits), top_k=4, block_rows=8, renormalize=True
    )
    np.testing.assert_array_equal(np.asarray(indices)[:, :3], [[5, 9, 40]] * 8)


@pytest.mark.cpu_test
@pytest.mark.parametrize("renormalize", [False, True], ids=["raw", "renormalized"])
def test_select_top_k_routes_rows_past_the_valid_count_nowhere(interpret, renormalize):
    """Rows past valid_tokens get weight zero, raw or renormalized, and so
    does a row with a NaN logit under drop_nan_rows (renormalized only);
    the other rows are unchanged."""
    logits = _logits(32, 64)
    logits[3, 7] = np.nan
    valid = 21
    weights, indices = select_kernel.select_top_k(
        jnp.asarray(logits),
        top_k=4,
        block_rows=8,
        renormalize=renormalize,
        drop_nan_rows=renormalize,
        valid_tokens=jnp.asarray([valid], jnp.int32),
    )
    weights = np.asarray(weights)
    assert not weights[valid:].any()
    if renormalize:
        assert not weights[3].any()
    kept = [r for r in range(valid) if r != 3]
    values, want_indices = jax.lax.top_k(jnp.asarray(logits[kept]), 4)
    np.testing.assert_array_equal(np.asarray(indices)[kept], np.asarray(want_indices))
    if renormalize:
        want = jax.nn.softmax(values, axis=-1)
        np.testing.assert_allclose(weights[kept], want, rtol=1e-6, atol=0)
    else:
        np.testing.assert_array_equal(weights[kept], np.asarray(values))


@pytest.mark.cpu_test
@pytest.mark.parametrize("renormalize", [False, True], ids=["raw", "renormalized"])
def test_select_top_k_gives_slots_past_a_rows_finite_scores_weight_zero(
    interpret, renormalize
):
    """A row with two finite scores and top_k 4: its first two slots carry
    the two scores (raw) or their softmax, the other two weight zero."""
    logits = _logits(8, 64)
    logits[5, 2:] = np.nan
    weights, _ = select_kernel.select_top_k(
        jnp.asarray(logits), top_k=4, block_rows=8, renormalize=renormalize
    )
    row = np.asarray(weights)[5]
    assert not row[2:].any()
    top = np.sort(logits[5, :2])[::-1]
    want = np.asarray(jax.nn.softmax(jnp.asarray(top))) if renormalize else top
    np.testing.assert_allclose(row[:2], want, rtol=1e-6, atol=0)


@pytest.mark.cpu_test
def test_select_top_k_refuses_drop_nan_rows_on_raw_scores():
    """drop_nan_rows changes which rows route; raw scores would ignore it."""
    with pytest.raises(ValueError, match="drop_nan_rows"):
        select_kernel.select_top_k(
            jnp.zeros((8, 64), jnp.float32),
            top_k=4,
            block_rows=8,
            renormalize=False,
            drop_nan_rows=True,
        )


@pytest.mark.parametrize(
    "padding_rows,scale_rows", [(0, 0), (4, 4)], ids=["unpadded", "padded"]
)
def test_select_top_k_routing_message_marks_rows_past_the_valid_count(
    interpret_or_chip, padding_rows, scale_rows
):
    """In the routing-message layout the layer uses (`message_rows`; word
    k * rows + r is row r's slot k), rows past valid_tokens carry the
    unrouted index `num_experts` on every slot, and valid rows carry
    lax.top_k's experts. Padded, as the layer pads it: past the index rows,
    pair rows up to `pair_rows` route nowhere and the scale rows after them
    start zero. The first of the two grid steps writes them; both steps'
    index rows land in the same block."""
    rows, experts, top_k, valid = 256, 64, 4, 200
    logits = _logits(rows, experts, seed=3)
    index_rows = rows * top_k // LANES
    pair_rows = index_rows + padding_rows
    message_rows = pair_rows + scale_rows
    weights, message = select_kernel.select_top_k(
        jnp.asarray(logits),
        top_k=top_k,
        block_rows=LANES,
        renormalize=True,
        message_rows=message_rows,
        pair_rows=pair_rows,
        valid_tokens=jnp.asarray([valid], jnp.int32),
    )
    message = np.asarray(message)
    assert message.shape == (message_rows, LANES)
    by_slot = message[:index_rows].reshape(top_k, rows)
    _, want_indices = jax.lax.top_k(jnp.asarray(logits[:valid]), top_k)
    np.testing.assert_array_equal(by_slot[:, :valid].T, np.asarray(want_indices))
    assert (by_slot[:, valid:] == experts).all()
    assert (message[index_rows:pair_rows] == experts).all()
    assert not message[pair_rows:].any()
    assert not np.asarray(weights)[valid:].any()


@pytest.mark.cpu_test
@pytest.mark.parametrize("drop_nan_rows", [False, True], ids=["keep", "drop_nan"])
def test_select_top_k_routing_message_routes_slots_past_finite_scores_nowhere(
    interpret, drop_nan_rows
):
    """In the routing message, a slot with no finite score left carries the
    unrouted index `num_experts`, as a row that routes nowhere does: a real
    expert there would be a second pair of the token for an expert already
    picked, which the tables' one-pair-per-token-and-expert bound does not
    allow for. Row 5 has two finite logits and -inf elsewhere, row 9 two
    finite logits and NaN elsewhere (routed nowhere under drop_nan_rows)."""
    rows, experts, top_k = 128, 64, 4
    logits = _logits(rows, experts, seed=4)
    logits[5, 2:] = -np.inf
    logits[9, 2:] = np.nan
    weights, message = select_kernel.select_top_k(
        jnp.asarray(logits),
        top_k=top_k,
        block_rows=LANES,
        renormalize=True,
        drop_nan_rows=drop_nan_rows,
        message_rows=rows * top_k // LANES,
    )
    by_slot = np.asarray(message).reshape(top_k, rows)
    two_finite = [5] if drop_nan_rows else [5, 9]
    for r in two_finite:
        np.testing.assert_array_equal(by_slot[:2, r], np.argsort(-logits[r, :2]))
        assert (by_slot[2:, r] == experts).all(), by_slot[:, r]
    if drop_nan_rows:
        assert (by_slot[:, 9] == experts).all()
    full = [r for r in range(rows) if r not in (5, 9)]
    _, want_indices = jax.lax.top_k(jnp.asarray(logits[full]), top_k)
    np.testing.assert_array_equal(by_slot[:, full].T, np.asarray(want_indices))
    assert not np.asarray(weights)[two_finite, 2:].any()


# ---- weighted_sum ----


def _pairs(tokens, top_k, hidden, seed=1):
    """Rows in pair order (pair t * top_k + k is token t's slot k), the
    weights with a zero on every token's last slot, and NaN in the rows those
    zero weights point at (a route of weight zero reads a row nobody wrote)."""
    rng = np.random.default_rng(seed)
    rows = rng.standard_normal((tokens * top_k, hidden)).astype(np.float32)
    weights = rng.uniform(0.05, 1.0, (tokens, top_k)).astype(np.float32)
    weights[:, -1] = 0.0
    rows.reshape(tokens, top_k, hidden)[:, -1, :] = np.nan
    return rows, weights


def _k_ordered_sum(rows, coef, tokens, top_k, hidden, dtype=jnp.float32):
    """The k-ordered sum with the products and the sum in `dtype`, the
    coefficient formed in float32 and then cast, as weighted_sum does."""
    terms = jnp.asarray(rows).astype(dtype).reshape(tokens, top_k, hidden)
    coef = jnp.asarray(coef, jnp.float32).astype(dtype)
    acc = None
    for k in range(top_k):
        c = coef[:, k : k + 1]
        term = jnp.where(c != 0, terms[:, k, :] * c, 0.0)
        acc = term if k == 0 else acc + term
    return acc


# (sum dtype, hidden): the default float32 sum; a bfloat16 sum at a width
# whose lane blocks (8) fill no whole bfloat16 staging register, so it stages
# in float32; and one whose 16 lane blocks do, so it stages in bfloat16.
SUM_FORMS = {
    "f32_sum": (jnp.float32, 1024),
    "bf16_sum_f32_staging": (jnp.bfloat16, 1024),
    "bf16_sum_bf16_staging": (jnp.bfloat16, 2048),
}


@pytest.mark.cpu_test
def test_the_sum_forms_cover_both_staging_dtypes():
    stage = {
        name: combine._stage_dtype(dtype, hidden // LANES)
        for name, (dtype, hidden) in SUM_FORMS.items()
    }
    assert stage["bf16_sum_f32_staging"] == jnp.float32
    assert stage["bf16_sum_bf16_staging"] == jnp.bfloat16


@pytest.mark.cpu_test
@pytest.mark.parametrize("form", list(SUM_FORMS))
@pytest.mark.parametrize("fp8_rows", [False, True], ids=["bf16_rows", "fp8_rows"])
def test_weighted_sum_matches_the_k_ordered_sum(interpret, fp8_rows, form):
    dtype, hidden = SUM_FORMS[form]
    tokens, top_k = 128, 4
    rows, weights = _pairs(tokens, top_k, hidden)
    if fp8_rows:
        finite = np.nan_to_num(rows, nan=0.0)
        scales = (np.abs(finite).max(axis=1) / 448.0).astype(np.float32)
        q = jnp.asarray(rows / scales[:, None]).astype(jnp.float8_e4m3fn)
        coef = weights * scales.reshape(tokens, top_k)
        scales_arg = jnp.asarray(scales)
    else:
        q = jnp.asarray(rows).astype(jnp.bfloat16)
        coef = weights
        scales_arg = None
    got = combine.weighted_sum(
        q.reshape(tokens * top_k, hidden // LANES, LANES),
        scales_arg,
        jnp.asarray(weights),
        tokens_per_shard=tokens,
        hidden=hidden,
        out_dtype=jnp.bfloat16,
        dtype=dtype,
    )
    want = _k_ordered_sum(q, coef, tokens, top_k, hidden, dtype)
    got = np.asarray(got.astype(jnp.float32))
    assert np.isfinite(got).all()
    # A fused multiply-add in one of the two programs can move an f32 sum by
    # one ulp and its bf16 rounding by one step on a few values; a slot
    # mixed up or dropped moves most of them (the control below).
    assert _off_fraction(got, want) < 1e-3


@pytest.mark.cpu_test
def test_weighted_sum_drops_a_zero_weight_route_with_a_bad_scale(interpret):
    """A route of weight zero reads a row nobody wrote, whose scale is
    whatever was there: a NaN or Inf scale must not reach the token."""
    tokens, top_k, hidden = 128, 4, 1024
    rows, weights = _pairs(tokens, top_k, hidden)
    finite = np.nan_to_num(rows, nan=0.0)
    scales = (np.abs(finite).max(axis=1) / 448.0).astype(np.float32)
    q = jnp.asarray(finite / np.where(scales > 0, scales, 1)[:, None]).astype(
        jnp.float8_e4m3fn
    )
    bad = scales.reshape(tokens, top_k)
    bad[::2, -1] = np.nan  # the last slot has weight zero on every token
    bad[1::2, -1] = np.inf
    got = combine.weighted_sum(
        q.reshape(tokens * top_k, hidden // LANES, LANES),
        jnp.asarray(bad.reshape(-1)),
        jnp.asarray(weights),
        tokens_per_shard=tokens,
        hidden=hidden,
        out_dtype=jnp.bfloat16,
    )
    got = np.asarray(got.astype(jnp.float32))
    assert np.isfinite(got).all()
    with np.errstate(invalid="ignore"):
        coef = np.where(weights != 0, weights * bad, 0.0)
    assert _off_fraction(got, _k_ordered_sum(q, coef, tokens, top_k, hidden)) < 1e-3


def _off_fraction(got, want):
    """The fraction of values more than one bf16 step from the reference."""
    want = np.asarray(jnp.asarray(want).astype(jnp.bfloat16), np.float32)
    return float((np.abs(got - want) > 2.0**-7 * np.abs(want) + 1e-5).mean())


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "hidden,rows_dtype,rows_sublanes",
    [
        (4096, jnp.float8_e4m3fn, 32),
        (1024, jnp.float8_e4m3fn, 32),
        (1024, jnp.bfloat16, 16),
    ],
    ids=["fp8_4096", "fp8_1024", "bf16_1024"],
)
def test_weighted_sum_counts_its_blocks_as_vmem_holds_them(
    interpret, hidden, rows_dtype, rows_sublanes
):
    """The VMEM limit is twice the step's blocks as TPU7x tiles them: a
    token's rows are lane blocks deep, padded to the dtype's sublane tile,
    so 8 blocks of fp8 (hidden 1024) take 32 sublanes, four times their
    size. At hidden 4096 (32 blocks) nothing is padded."""
    tokens, top_k, f32 = 128, 10, 4
    lane_blocks = hidden // LANES
    stage_dtype = combine._stage_dtype(jnp.float32, lane_blocks)
    assert stage_dtype == jnp.float32
    got = combine._weighted_sum_vmem_bytes(
        tokens_per_tile=tokens,
        top_k=top_k,
        lane_blocks=lane_blocks,
        hidden=hidden,
        rows_dtype=rows_dtype,
        out_dtype=jnp.bfloat16,
        stage_dtype=stage_dtype,
    )
    rows = tokens * top_k * rows_sublanes * LANES * jnp.dtype(rows_dtype).itemsize
    out = tokens * hidden * 2
    staging = tokens * lane_blocks * LANES * f32
    assert got == 2 * rows + 2 * out + staging


@pytest.mark.cpu_test
def test_weighted_sum_rows_are_not_interchangeable(interpret):
    """Control: the same rows with two slots' weights swapped are off the
    reference almost everywhere, so the comparison above sees a slot mix-up."""
    tokens, top_k, hidden = 128, 4, 1024
    rows, weights = _pairs(tokens, top_k, hidden)
    q = jnp.asarray(rows).astype(jnp.bfloat16)
    swapped = weights.copy()
    swapped[:, [0, 1]] = swapped[:, [1, 0]]
    sums = [
        np.asarray(
            combine.weighted_sum(
                q.reshape(tokens * top_k, hidden // LANES, LANES),
                None,
                jnp.asarray(w),
                tokens_per_shard=tokens,
                hidden=hidden,
                out_dtype=jnp.bfloat16,
            ).astype(jnp.float32)
        )
        for w in (weights, swapped)
    ]
    assert _off_fraction(sums[0], sums[1]) > 0.5
