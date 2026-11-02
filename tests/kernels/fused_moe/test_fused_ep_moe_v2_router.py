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
"""Exact selector parity across row tiles, tie rules and separate weight tables.

CPU cases execute the Pallas interpreter; TPU cases compile the same inputs
through Mosaic on one device. Neither mode runs the full MoE layer.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.fused_moe.v2 import router_ops

# Wide enough that a top-k spans several 128-lane blocks at the shapes the
# router actually sees, small enough to check exhaustively against numpy.
EXPERTS = 256
TOPK = 8


@pytest.fixture(
    params=[
        pytest.param(True, marks=pytest.mark.cpu_test, id="cpu"),
        pytest.param(False, id="tpu"),
    ]
)
def interpret(request: pytest.FixtureRequest) -> bool:
    if not request.param and jax.default_backend() != "tpu":
        pytest.skip("Native Pallas lowering requires a TPU")
    return request.param


def _select(
    scores: np.ndarray,
    weight_scores: np.ndarray | None = None,
    topk: int = TOPK,
    block_rows: int | None = None,
    *,
    interpret: bool,
) -> tuple[np.ndarray, np.ndarray]:
    rows = scores.shape[0]
    select = jax.jit(
        functools.partial(
            router_ops.pallas_select,
            topk=topk,
            block_rows=rows if block_rows is None else block_rows,
            interpret=interpret,
        )
    )
    w, i = select(
        jnp.asarray(scores),
        weight_scores=None if weight_scores is None else jnp.asarray(weight_scores),
    )
    assert w.shape == i.shape == (rows, topk)
    assert w.dtype == jnp.float32
    assert i.dtype == jnp.int32
    return np.asarray(w), np.asarray(i)


def _reference(scores, weight_scores, topk=TOPK):
    """Top-k by `scores`, weights gathered from `weight_scores`.

    Ties go to the lowest column, which is the kernel's rule: it takes the
    row max and then the lowest column equal to it. `argsort(-x, stable)`
    is the same rule, and `np.argpartition` is NOT -- it is unordered within
    the partition, so it disagrees on exactly the rows that matter.
    """
    idx = np.argsort(-scores, axis=1, kind="stable")[:, :topk]
    return np.take_along_axis(weight_scores, idx, axis=1), idx


def _scores(rows, seed, scale=1.0, experts=EXPERTS):
    rng = np.random.default_rng(seed)
    return (rng.random((rows, experts), dtype=np.float32) * scale).astype(np.float32)


@pytest.mark.parametrize(
    "rows,experts,topk,block_rows",
    [
        (8, 256, 8, 8),
        (256, 256, 8, 256),
        (512, 256, 8, 256),
        (1024, 256, 8, 256),
        (256, 384, 6, 256),
        pytest.param(2048, 384, 6, 256, id="tuned-2048-rows"),
        pytest.param(512, 384, 6, 64, id="align-partial-tiles"),
        pytest.param(192, 384, 6, 64, id="fallback-to-full-rows"),
        (256, 384, 6, 128),
        pytest.param(128, 128, 1, 128, id="topk-one"),
        pytest.param(8, 16, 16, 8, id="topk-all-experts"),
    ],
)
@pytest.mark.parametrize("separate_weights", [False, True])
def test_selection_matches_reference(
    rows, experts, topk, block_rows, separate_weights, interpret
):
    """Transposition and tiling preserve every token's expert IDs and weights."""
    scores = _scores(rows, seed=rows, experts=experts)
    weights = (
        _scores(rows, seed=rows + 1, experts=experts) if separate_weights else None
    )
    w, i = _select(scores, weights, topk, block_rows, interpret=interpret)
    ref_w, ref_i = _reference(scores, scores if weights is None else weights, topk)
    np.testing.assert_array_equal(i, ref_i)
    np.testing.assert_array_equal(w, ref_w)


def test_the_bias_moves_the_selection_and_not_the_weights(interpret):
    """The `noaux_tc` call as the layer makes it, end to end.

    The bias is large against the score spread, so the biased top-k is a
    genuinely different expert set from the unbiased one -- asserted here,
    because a fixture where it is not would pass against a kernel that
    dropped the bias on the floor.
    """
    raw = _scores(64, seed=7)
    rng = np.random.default_rng(8)
    bias = (rng.random(EXPERTS, dtype=np.float32) * 4.0).astype(np.float32)
    biased = raw + bias[None, :]

    w, i = _select(biased, raw, interpret=interpret)
    ref_w, ref_i = _reference(biased, raw)
    np.testing.assert_array_equal(i, ref_i)
    np.testing.assert_array_equal(w, ref_w)

    # The weights are the unbiased scores at the biased picks: neither the
    # biased scores, nor the weights the unbiased selection would have given.
    np.testing.assert_array_equal(w, np.take_along_axis(raw, i, axis=1))
    assert not np.array_equal(w, np.take_along_axis(biased, i, axis=1))
    _, unbiased_i = _reference(raw, raw)
    assert not np.array_equal(i, unbiased_i), (
        "the fixture's bias does not reorder the top-k, so this test cannot "
        "tell a kernel that applies it from one that ignores it"
    )


@pytest.mark.parametrize("separate_weights", [False, True])
@pytest.mark.parametrize("partial_nan", [False, True], ids=["ties", "partial-nan"])
def test_ties_choose_lowest_expert_across_lane_boundaries(
    separate_weights, partial_nan, interpret
):
    scores = np.zeros((256, 384), dtype=np.float32)
    scores[:, [127, 128, 255, 256, 382, 383]] = 1.0
    if partial_nan:
        scores[:, :127] = np.nan
    weights = _scores(256, seed=11, experts=384) if separate_weights else None
    w, i = _select(scores, weights, topk=6, block_rows=128, interpret=interpret)
    ref_w, ref_i = _reference(
        np.where(np.isnan(scores), -3.0e38, scores),
        scores if weights is None else weights,
        topk=6,
    )
    np.testing.assert_array_equal(i, ref_i)
    np.testing.assert_array_equal(w, ref_w)


# Adapt the every-expert and tiny-score cases from vllm-project/vllm-torchtpu
# PR #2480 to this selector's unnormalized, multi-tile output contract.
@pytest.mark.parametrize("separate_weights", [False, True])
def test_every_expert_id_preserves_token_and_topk_position(separate_weights, interpret):
    """Each output slot visits every expert, with distinct weights per token."""
    rows, experts, topk = 512, 384, 6
    expected_ids = (
        np.arange(rows)[:, None] + np.arange(topk)[None, :] * (experts // topk)
    ) % experts
    for slot in range(topk):
        np.testing.assert_array_equal(
            np.unique(expected_ids[:, slot]), np.arange(experts)
        )
    scores = np.full((rows, experts), -1, dtype=np.float32)
    np.put_along_axis(scores, expected_ids, np.arange(topk, 0, -1)[None, :], axis=1)
    weights = (
        np.arange(rows * experts, dtype=np.float32).reshape(rows, experts)
        if separate_weights
        else None
    )
    w, i = _select(scores, weights, topk, block_rows=128, interpret=interpret)
    np.testing.assert_array_equal(i, expected_ids)
    np.testing.assert_array_equal(
        w,
        np.take_along_axis(
            scores if weights is None else weights, expected_ids, axis=1
        ),
    )


@pytest.mark.parametrize("separate_weights", [False, True])
@pytest.mark.parametrize("case", ["zeros", "uniform", "tiny"])
def test_zero_uniform_and_tiny_scores_preserve_selection(
    case, separate_weights, interpret
):
    """Zero/tied scores select distinct lowest IDs; tiny weights stay unnormalized."""
    scores = _scores(256, seed=17, scale=1e-22, experts=384)
    if case == "zeros":
        scores.fill(0)
    elif case == "uniform":
        scores.fill(1 / 384)
    weights = (
        _scores(256, seed=18, scale=1e-22, experts=384) if separate_weights else None
    )
    w, i = _select(scores, weights, topk=6, block_rows=128, interpret=interpret)
    ref_w, ref_i = _reference(scores, scores if weights is None else weights, topk=6)
    np.testing.assert_array_equal(i, ref_i)
    np.testing.assert_array_equal(w, ref_w)


@pytest.mark.parametrize("separate_weights", [False, True])
def test_all_nan_rows_preserve_sentinel_and_repeated_expert_zero(
    separate_weights, interpret
):
    """The fused router intentionally leaves invalid-row masking to its caller."""
    scores = np.full((8, 384), np.nan, dtype=np.float32)
    weights = _scores(8, seed=12, experts=384) if separate_weights else None
    w, i = _select(scores, weights, topk=6, interpret=interpret)
    np.testing.assert_array_equal(i, np.zeros((8, 6), dtype=np.int32))
    expected = (
        np.full((8, 6), -3.0e38, dtype=np.float32)
        if weights is None
        else np.broadcast_to(weights[:, :1], (8, 6))
    )
    np.testing.assert_array_equal(w, expected)


@pytest.mark.cpu_test
@pytest.mark.parametrize("weight_shape", [(128, 383), (127, 384)])
def test_rejects_mismatched_weight_table(weight_shape):
    with pytest.raises(ValueError, match="must match scores"):
        _select(
            np.zeros((128, 384), dtype=np.float32),
            np.zeros(weight_shape, dtype=np.float32),
            interpret=True,
        )


@pytest.mark.cpu_test
def test_rejects_nondivisible_requested_row_tile():
    with pytest.raises(AssertionError, match=r"\(257, 256\)"):
        _select(
            np.zeros((257, 384), dtype=np.float32),
            block_rows=256,
            interpret=True,
        )


@pytest.mark.parametrize(
    "case", ["biased", "ties", "nonfinite", "tiny", "every_expert"]
)
@pytest.mark.cpu_test
def test_packed_top6_normalizes_unbiased_weights(case):
    """Packing must preserve expert order and normalize the unbiased table."""
    rng = np.random.default_rng(41)
    raw = rng.random((288, 384), dtype=np.float32)
    bias = rng.uniform(-3, 0, 384).astype(np.float32)
    if case == "ties":
        raw[:] = 1
        bias[:] = -100
        bias[[0, 127, 128, 255, 256, 383]] = 0
    elif case == "nonfinite":
        raw[0] = np.nan
        raw[1] = np.inf
        raw[2] = 0
        raw[3, ::3] = np.nan
        raw[4, ::3] = np.inf
        bias[::7] = np.nan
    elif case == "tiny":
        raw *= np.float32(1e-22)
    scores = raw + bias
    if case == "every_expert":
        # Exercise every expert ID, including adjacent IDs above 256.
        expected_ids = (np.arange(288)[:, None] * 6 + np.arange(6)) % 384
        scores[:] = -1
        np.put_along_axis(scores, expected_ids, np.arange(6, 0, -1)[None, :], axis=1)
    original_w, original_i = _select(scores, raw, topk=6, block_rows=32, interpret=True)
    with np.errstate(invalid="ignore"):
        expected = original_w / np.maximum(
            original_w.sum(axis=-1, keepdims=True), np.float32(1e-20)
        )
    weights, indices = router_ops.pallas_select(
        jnp.asarray(scores),
        topk=6,
        block_rows=288,
        weight_scores=jnp.asarray(raw),
        normalize=True,
        interpret=True,
    )
    np.testing.assert_array_equal(indices, original_i)
    np.testing.assert_allclose(weights, expected, rtol=2e-6, atol=1e-6)
    # The fused output must retain the same selection/normalization contract,
    # including nonfinite selected weights and lowest-ID ties.
    gate = jnp.asarray(raw, jnp.bfloat16 if case == "nonfinite" else jnp.float32)
    fused_weights, fused_ids, valid = router_ops.select_finalize_normalized_router(
        jnp.asarray(scores),
        jnp.asarray(raw),
        gate,
        scale=2.5,
        invalid_id=384,
        interpret=True,
    )
    expected_valid = np.isfinite(np.asarray(gate).astype(np.float32)).any(
        axis=-1, keepdims=True
    )
    assert valid.shape == (288, 1) and valid.dtype == jnp.bool_
    np.testing.assert_array_equal(valid, expected_valid)
    np.testing.assert_array_equal(fused_ids, np.where(expected_valid, original_i, 384))
    np.testing.assert_allclose(
        fused_weights, np.where(expected_valid, expected * 2.5, 0), rtol=2e-6, atol=1e-6
    )
    if case == "biased":
        _, unbiased_ids = _reference(raw, raw, topk=6)
        assert not np.array_equal(indices, unbiased_ids)
    elif case == "ties":
        np.testing.assert_array_equal(indices[0], [0, 127, 128, 255, 256, 383])
    elif case == "every_expert":
        np.testing.assert_array_equal(indices, expected_ids)


@pytest.mark.parametrize(
    "change",
    [
        {},
        {"rows": 256},
        {"experts": 128},
        {"topk": 3},
        {"scoring_fn": "softmax"},
        {"scoring_fn": "sigmoid"},
        {"renormalize": False},
        {"has_bias": False},
        {"has_hash": True},
        {"dtype": jnp.float16},
        {"backend": "cpu"},
        {"chip": router_ops.pltpu.ChipVersion.TPU_V6E},
    ],
)
@pytest.mark.cpu_test
def test_normalized_router_only_uses_measured_envelope(monkeypatch, change):
    """Never normalize hash/no-bias routes or widen unmeasured tiles/devices."""
    from types import SimpleNamespace
    from unittest.mock import Mock

    config = dict(
        rows=288,
        experts=384,
        topk=6,
        scoring_fn="sqrtsoftplus",
        renormalize=True,
        has_bias=True,
        has_hash=False,
        dtype=jnp.bfloat16,
        backend="tpu",
        chip=router_ops.pltpu.ChipVersion.TPU_7X,
    )
    config.update(change)
    monkeypatch.setattr(router_ops.jax, "default_backend", lambda: config["backend"])
    monkeypatch.setattr(
        router_ops.pltpu,
        "get_tpu_info",
        lambda: SimpleNamespace(chip_version=config["chip"]),
    )
    monkeypatch.setattr(router_ops, "logger", Mock())
    gate = jax.ShapeDtypeStruct((config["rows"], config["experts"]), config["dtype"])
    tile = router_ops.normalized_router_block_rows(
        gate,
        topk=config["topk"],
        scoring_fn=config["scoring_fn"],
        renormalize=config["renormalize"],
        has_bias=config["has_bias"],
        has_hash=config["has_hash"],
    )
    assert tile == (None if change else 288)


@pytest.mark.parametrize("topk,block_rows", [(3, 288), (6, 32)])
@pytest.mark.cpu_test
def test_normalized_selector_rejects_unsupported_tiles(topk, block_rows):
    scores = jnp.ones((288, 384), jnp.float32)
    with pytest.raises(ValueError, match="Normalized selection requires"):
        router_ops.pallas_select(
            scores,
            topk=topk,
            block_rows=block_rows,
            weight_scores=scores,
            normalize=True,
            interpret=True,
        )


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "adapter_widen,near_ties,scale,mesh_ep_ranks",
    [
        (False, False, 1.0, None),
        (True, False, 2.5, (0,)),
        (False, True, 2.5, (0,)),
        (True, True, 1.0, None),
    ],
)
def test_normalized_route_stage_keeps_scaling_and_invalid_sentinels(
    monkeypatch, adapter_widen, near_ties, scale, mesh_ep_ranks
):
    """Exercise production wiring without running experts or TPU collectives."""
    import functools
    from unittest.mock import Mock, patch

    from vllm_torchtpu.kernels.fused_moe.v2 import host, kernel, layer

    info = router_ops.pltpu.get_tpu_info_for_chip(
        router_ops.pltpu.ChipVersion.TPU_7X, 1
    )
    monkeypatch.setattr(router_ops.pltpu, "get_tpu_info", lambda: info)
    monkeypatch.setattr(kernel, "logger", Mock())
    monkeypatch.setattr(
        layer,
        "pallas_select",
        functools.partial(router_ops.pallas_select, interpret=True),
    )
    fused_select = Mock(
        wraps=functools.partial(
            router_ops.select_finalize_normalized_router, interpret=True
        )
    )
    finalize = Mock(
        wraps=functools.partial(router_ops.finalize_normalized_router, interpret=True)
    )
    monkeypatch.setattr(layer, "select_finalize_normalized_router", fused_select)
    monkeypatch.setattr(layer, "finalize_normalized_router", finalize)
    # Isolate the layer's function cache from other tests and from this A/B.
    monkeypatch.setattr(layer, "_LAYER_SM_CACHE", {})
    with jax.default_device(jax.devices("cpu")[0]):
        rng = np.random.default_rng(43)
        gate = jnp.asarray(rng.normal(size=(288, 384)), dtype=jnp.bfloat16)
        gate = gate.at[:3].set(
            jnp.array([jnp.nan, -jnp.inf, jnp.inf], jnp.bfloat16)[:, None]
        )
        if adapter_widen:
            # fused_moe_ep._build_op.call passes this widened form to the layer.
            gate = gate.astype(jnp.float32)
        if near_ties:
            # Adjacent logits and cancelling bias expose changed Top6 membership.
            ranks = rng.permutation(384).astype(np.float32)
            near = np.float32(0.25) + ranks * np.spacing(np.float32(0.25))
            if not adapter_widen:
                near = np.float32(0.25) + ranks * np.float32(2**-11)
            gate = jnp.broadcast_to(jnp.asarray(near, gate.dtype), (288, 384))
            gate = gate.at[:3].set(
                jnp.array([jnp.nan, -jnp.inf, jnp.inf], gate.dtype)[:, None]
            )
            old_scores = layer._apply_scoring_fn(gate[3], "sqrtsoftplus")
            bias = -old_scores + jnp.asarray((np.arange(384) % 7) * 2**-23, jnp.float32)
        else:
            bias = jnp.asarray(rng.normal(size=(384,)), dtype=jnp.float32)
        mesh = jax.sharding.Mesh(np.array(jax.devices("cpu")[:1]), ("d",))
        args = (
            jnp.zeros((288, 128), jnp.bfloat16),
            jnp.zeros((384, 128, 256), jnp.bfloat16),
            jnp.zeros((384, 128, 128), jnp.bfloat16),
            None,
            None,
            gate,
        )
        selected_tiles = []

        def production_tile(gate, **kwargs):
            with patch.object(router_ops.jax, "default_backend", return_value="tpu"):
                tile = router_ops.normalized_router_block_rows(gate, **kwargs)
            selected_tiles.append(tile)
            return tile

        outputs = []
        for choose_tile in (lambda *a, **kw: None, production_tile):
            layer._LAYER_SM_CACHE.clear()
            monkeypatch.setattr(layer, "normalized_router_block_rows", choose_tile)
            outputs.append(
                tuple(
                    np.asarray(value)
                    for value in layer.fused_ep_moe_v2(
                        *args,
                        topk=6,
                        renormalize=True,
                        mesh=mesh,
                        mesh_ep_ranks=mesh_ep_ranks,
                        capacity=128,
                        weight_format=host.WeightFormat.BF16,
                        scoring_fn="sqrtsoftplus",
                        score_bias=bias,
                        routed_scaling_factor=scale,
                        stage="route",
                    )
                )
            )
        assert selected_tiles and all(tile == 288 for tile in selected_tiles)
        assert fused_select.called
        finalize.assert_not_called()
        (old_wire, old_weights, old_payload), (wire, weights, payload) = outputs
        np.testing.assert_array_equal(wire, old_wire)
        np.testing.assert_array_equal(payload, old_payload)
        np.testing.assert_allclose(weights, old_weights, rtol=2e-6, atol=1e-6)
        np.testing.assert_array_equal(weights[:3], 0)
        np.testing.assert_array_equal(payload[0, : 3 * 6], 384)
        np.testing.assert_allclose(weights[3:].sum(axis=-1), scale, rtol=2e-6)


@pytest.mark.cpu_test
@pytest.mark.parametrize("dtype,scale", [(jnp.bfloat16, 1.0), (jnp.float32, 2.5)])
def test_finalize_masks_invalid_ids_after_mesh_permutation(dtype, scale):
    """The sentinel must not be mapped to expert zero by an EP permutation."""
    from vllm_torchtpu.kernels.fused_moe.v2.layer import (
        _relabel_expert_ids_to_mesh_order,
    )

    permutation = (3, 0, 7, 1, 6, 2, 5, 4)
    ids = (np.arange(288)[:, None] * 6 + np.arange(6)) % 384
    gate = jnp.ones((288, 384), dtype)
    gate = gate.at[:3].set(jnp.array([jnp.nan, -jnp.inf, jnp.inf], dtype)[:, None])
    # A single finite expert is enough to keep a row, including the final column.
    gate = gate.at[3].set(-jnp.inf).at[3, 383].set(0)
    weights = jnp.full((288, 6), 1 / 6, jnp.float32)
    remapped = _relabel_expert_ids_to_mesh_order(
        jnp.asarray(ids, jnp.int32), g_local=48, mesh_ep_ranks=permutation
    )
    expected_ids = np.argsort(permutation)[ids // 48] * 48 + ids % 48
    np.testing.assert_array_equal(remapped, expected_ids)
    out_weights, out_ids, valid = router_ops.finalize_normalized_router(
        gate, weights, remapped, scale=scale, invalid_id=384, interpret=True
    )
    expected_valid = np.ones((288, 1), dtype=bool)
    expected_valid[:3] = False
    assert valid.shape == (288, 1) and valid.dtype == jnp.bool_
    np.testing.assert_array_equal(valid, expected_valid)
    np.testing.assert_array_equal(out_ids, np.where(expected_valid, expected_ids, 384))
    np.testing.assert_allclose(
        out_weights, np.where(expected_valid, np.asarray(weights) * scale, 0), rtol=2e-6
    )


def test_packed_top6_tpu_lowering_matches_numpy():
    """Run the normalized selector through Mosaic on one supported device."""
    if jax.default_backend() != "tpu":
        pytest.skip("requires a TPU7x device")
    if (
        router_ops.pltpu.get_tpu_info().chip_version
        != router_ops.pltpu.ChipVersion.TPU_7X
    ):
        pytest.skip("normalized selector is calibrated on TPU7x")
    rng = np.random.default_rng(42)
    raw = rng.random((288, 384), dtype=np.float32)
    biased = raw + rng.uniform(-3, 0, 384).astype(np.float32)
    biased[0] = 1
    expected_weights, expected_ids = _reference(biased, raw, topk=6)
    expected_weights /= expected_weights.sum(axis=-1, keepdims=True)
    weights, indices = jax.jit(
        lambda scores, values: router_ops.pallas_select(
            scores, topk=6, block_rows=288, weight_scores=values, normalize=True
        )
    )(jnp.asarray(biased), jnp.asarray(raw))
    np.testing.assert_array_equal(indices, expected_ids)
    np.testing.assert_allclose(weights, expected_weights, rtol=2e-6, atol=1e-6)
    fused_weights, fused_ids, valid = jax.jit(
        lambda scores, values: router_ops.select_finalize_normalized_router(
            scores, values, values, scale=2.5, invalid_id=384
        )
    )(jnp.asarray(biased), jnp.asarray(raw))
    np.testing.assert_array_equal(fused_ids, expected_ids)
    np.testing.assert_allclose(
        fused_weights, expected_weights * 2.5, rtol=2e-6, atol=1e-6
    )
    assert valid.shape == (288, 1) and valid.dtype == jnp.bool_
    np.testing.assert_array_equal(valid, True)


@pytest.mark.cpu_test
def test_fp32_sqrtsoftplus_validity_matches_generic_row_check():
    """FP32 admission must preserve invalid rows without rounding logits to BF16."""
    from vllm_torchtpu.kernels.fused_moe.v2.layer import _apply_scoring_fn, _row_routes

    rng = np.random.default_rng(47)
    bits = rng.integers(0, 2**32, (288, 384), dtype=np.uint32)
    values = bits.view(np.float32)
    extremes = np.array(
        [
            np.finfo(np.float32).max,
            -np.finfo(np.float32).max,
            0,
            -100,
            np.nan,
            np.inf,
            -np.inf,
            np.finfo(np.float32).tiny,
        ],
        dtype=np.float32,
    )
    values[:8] = extremes[:, None]
    values[8:16] = np.nan
    values[8:16, 383] = extremes
    with jax.default_device(jax.devices("cpu")[0]):
        gate = jnp.asarray(values)
        scores = _apply_scoring_fn(gate, "sqrtsoftplus")
        original = _row_routes(scores, gate)
        simplified = jnp.any(jnp.isfinite(gate), axis=-1, keepdims=True)
        np.testing.assert_array_equal(simplified, original)
        assert np.isfinite(np.asarray(scores)[0]).all()


@pytest.mark.cpu_test
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
def test_specialized_softplus_preserves_finite_and_nonfinite_scores(dtype):
    """The measured formula must retain overflow safety and invalid-row semantics."""
    from vllm_torchtpu.kernels.fused_moe.v2.layer import _apply_scoring_fn

    rng = np.random.default_rng(48)
    bits = rng.integers(0, 2**32, 4096, dtype=np.uint32)
    edge = np.array(
        [
            0.0,
            -0.0,
            np.nan,
            np.inf,
            -np.inf,
            100,
            -100,
            np.finfo(np.float32).max,
            -np.finfo(np.float32).max,
        ],
        np.float32,
    )
    logits = jnp.asarray(np.concatenate((edge, bits.view(np.float32))), dtype)
    generic = _apply_scoring_fn(logits, "sqrtsoftplus")
    specialized = _apply_scoring_fn(logits, "sqrtsoftplus", specialize_softplus=True)
    np.testing.assert_array_equal(np.isnan(specialized), np.isnan(generic))
    np.testing.assert_array_equal(np.isposinf(specialized), np.isposinf(generic))
    np.testing.assert_allclose(specialized, generic, rtol=2e-6, atol=1e-6)
