# SPDX-License-Identifier: Apache-2.0
"""additional-config qwen4_exp.qsa_prefill_bound_stab: the DCP block-sparse
prefill with a precomputed per-row stabilizer (no online row max / rescale).

Interpret mode on CPU (FP32 dots, so only tolerance-level agreement is
checked here; TPU numerics are gated by the DCP harness):
- with the bound on, (out, lse) match the online-softmax kernel within the
  BF16-probability tolerance, on both ranks;
- the stabilizer really is an upper bound of every row's scores (lse minus
  the stabilizer stays below log(owned keys) + shift);
- an absurd bound pushes every row out of range: the kernel alone then
  underflows (positive control), and the wrapper's guard re-runs the online
  softmax, bit-identical to the flag-off call;
- no bound set: the wrapper is the flag-off call, bit for bit.
"""

import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.qwen4_exp import block_sparse_gqa_prefill as bsp

PAGE, HEADS, DIM, RATIO, D = 512, 3, 256, 4, 2
K_SCALE, V_SCALE = 0.5, 2.0


@pytest.fixture
def interpret(monkeypatch):
    original = pl.pallas_call
    monkeypatch.setattr(
        pl, "pallas_call", lambda *a, **k: original(*a, **{**k, "interpret": True})
    )
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)
    monkeypatch.setattr(envs, "VLLM_TPU_QSA_PREFILL_LANE_SUM", "1")
    monkeypatch.setattr(envs, "VLLM_TPU_QSA_PREFILL_LEAN", True)
    monkeypatch.setattr(envs, "VLLM_TPU_QSA_PREFILL_LEAN_WRAPPER", True)
    monkeypatch.delenv("VLLM_TPU_QSA_SMATTR", raising=False)
    monkeypatch.setattr(bsp, "_KEY_NORM_BOUND", None)


def make_case(rng, pieces, budget):
    lens = [n for _, n in pieces]
    tokens = sum(lens)
    starts = np.concatenate([[0], np.cumsum(lens)]).astype(np.int32)
    cols = max(c + n for c, n in pieces) // (D * PAGE) + 2
    table = np.zeros((tokens, cols), np.int32)
    free = rng.permutation(np.arange(len(pieces) * cols + 4))
    blocks = np.full((tokens, budget), -1, np.int32)
    lengths = np.zeros(tokens, np.int32)
    i = 0
    for r, (ctx, n) in enumerate(pieces):
        table[starts[r] : starts[r + 1]] = free[r * cols : (r + 1) * cols]
        for t in range(n):
            complete = (ctx + t + 1) // RATIO
            pick = np.sort(rng.choice(complete, size=min(budget, complete), replace=False))
            blocks[i, : len(pick)] = pick
            lengths[i] = ctx + t + 1
            i += 1
    # Keys as the model caches them: unit-RMS rows times (1 + w), / k_scale.
    w = rng.uniform(-0.6, 0.8, DIM)
    raw = rng.normal(0, 1, (len(pieces) * cols + 4, PAGE, 1, 4, DIM))
    raw /= np.sqrt(np.mean(raw * raw, axis=-1, keepdims=True))
    raw[..., 0, :] *= 1 + w
    raw[..., 2, :] *= 1 + w
    raw[..., 0, :] /= K_SCALE
    raw[..., 2, :] /= K_SCALE
    cache = jnp.asarray(raw, jnp.float8_e4m3fn)
    query = jnp.asarray(rng.normal(0, 1.5, (tokens, HEADS, DIM)), jnp.bfloat16)
    bound = float(np.sqrt(DIM) * np.abs(1 + w).max())
    return (query, jnp.asarray(blocks), jnp.asarray(lengths), jnp.asarray(table),
            cache, jnp.asarray(starts), bound)


def attend(case, rank, offset):
    query, blocks, lengths, table, cache, starts, _ = case
    gate = jnp.zeros_like(query)
    jax.clear_caches()
    return jax.jit(
        lambda q, c: bsp.block_sparse_gqa_attention(
            q, gate, blocks, lengths, table, c, starts, kv_heads=1, ratio=RATIO,
            k_scale=K_SCALE, v_scale=V_SCALE, query_block=256, kv_tile=PAGE,
            query_tile=256, kv_head_offset=offset,
            dcp=(jnp.asarray([rank], jnp.int32), D), return_lse=True, ascending=True,
        )[:2]
    )(query, cache)


def kernel_only(case, rank, offset):
    query, blocks, lengths, table, cache, starts, _ = case
    jax.clear_caches()
    return jax.jit(
        lambda q, c: bsp.block_sparse_gqa_prefill(
            q, blocks, lengths, table, c, ratio=RATIO, kv_heads=1, query_block=256,
            kv_tile=PAGE, k_scale=K_SCALE, v_scale=V_SCALE, bf16_io=True,
            kv_head_offset=offset, query_tile=256, return_lse=True,
            dcp_rank=jnp.asarray([rank], jnp.int32), dcp_world=D, ascending=True,
            bound_stab=True,
        )
    )(query, cache)


def bits(x):
    return np.asarray(jax.lax.bitcast_convert_type(x, jnp.uint32))


CASES = {"single": [(1500, 256)], "two": [(700, 200), (2047, 56)]}


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("name", sorted(CASES))
def test_bound_stab_matches_online(interpret, monkeypatch, rank, name):
    case = make_case(np.random.default_rng(3), CASES[name], 24)
    offset = 1
    want = attend(case, rank, offset)
    bsp.note_key_norm_bound(case[-1])
    got = attend(case, rank, offset)
    out_w, lse_w = (np.asarray(x) for x in want)
    out_g, lse_g = (np.asarray(x) for x in got)
    assert not np.isnan(out_g).any()
    np.testing.assert_array_equal(np.isfinite(lse_g), np.isfinite(lse_w))
    fin = np.isfinite(lse_w)
    # BF16-rounded probabilities: |d log mass| <= log(1 + 2^-8).
    np.testing.assert_allclose(lse_g[fin], lse_w[fin], atol=4e-3)
    scale = np.maximum(np.abs(out_w).max(-1, keepdims=True), 1e-30)
    assert float((np.abs(out_g - out_w) / scale).max()) < 8e-3
    # Upper bound: lse <= rowmax + log(n) <= stabilizer + shift + log(n).
    stab = np.asarray(bsp.bound_stabilizer(case[0], k_scale=K_SCALE, bound=case[-1]))
    assert np.all(lse_w[fin] - stab[fin] <= bsp.BOUND_STAB_SHIFT + np.log(24 * RATIO))
    # And far from underflow (the guard did not fire on real-shaped data).
    assert np.all(lse_w[fin] - stab[fin] > bsp.BOUND_STAB_GUARD + 10)


@pytest.mark.parametrize("rank", [0, 1])
def test_guard_reruns_online_softmax(interpret, monkeypatch, rank):
    case = make_case(np.random.default_rng(4), CASES["single"], 24)
    want = attend(case, rank, 0)
    bsp.note_key_norm_bound(1e4)  # 100 x too large: every row underflows
    _, lse_kernel = kernel_only(case, rank, 0)
    lse_kernel = np.asarray(lse_kernel)
    fin = np.isfinite(np.asarray(want[1]))
    # Positive control: the bound-stabilized kernel alone loses the rows.
    assert not np.allclose(lse_kernel[fin], np.asarray(want[1])[fin], atol=1e-2)
    got = attend(case, rank, 0)
    for a, b in zip(got, want):
        np.testing.assert_array_equal(bits(a), bits(b))


def test_no_bound_is_flag_off(interpret):
    case = make_case(np.random.default_rng(5), CASES["single"], 24)
    first = attend(case, 0, 0)
    assert bsp.key_norm_bound() is None
    second = attend(case, 0, 0)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(bits(a), bits(b))


def test_note_key_norm_bound_keeps_max(monkeypatch):
    monkeypatch.setattr(bsp, "_KEY_NORM_BOUND", None)
    bsp.note_key_norm_bound(20.0)
    bsp.note_key_norm_bound(10.0)
    assert bsp.key_norm_bound() == 20.0
    with pytest.raises(ValueError):
        bsp.note_key_norm_bound(float("nan"))


def test_debug_marks_do_not_change_outputs(interpret, monkeypatch):
    case = make_case(np.random.default_rng(6), CASES["single"], 24)
    bsp.note_key_norm_bound(case[-1])
    plain = attend(case, 0, 1)
    monkeypatch.setenv("VLLM_TPU_QSA_BOUND_STAB_MARKS", "0,20,40")
    marked = attend(case, 0, 1)
    for a, b in zip(marked, plain):
        np.testing.assert_array_equal(bits(a), bits(b))


@pytest.mark.parametrize(
    "env_value,config_value,want",
    [
        ("", False, False),
        ("", True, True),
        ("1", False, True),
        ("0", True, False),
        ("1", True, True),
        ("0", False, False),
    ],
)
def test_env_takes_precedence_over_additional_config(env_value, config_value, want):
    assert bsp.bound_stab_enabled(env_value, config_value) is want


def test_bad_env_value_rejected():
    with pytest.raises(ValueError):
        bsp.bound_stab_enabled("yes", False)


@pytest.mark.parametrize(
    "env_value,config_value,noted",
    [("0", True, False), ("1", False, True), ("", True, True), ("", False, False)],
)
def test_layer_notes_bound_per_precedence(monkeypatch, env_value, config_value, noted):
    """The QSA layer's finalize hook reads the env first, then the config key."""
    import types

    import torch
    import vllm.config

    from vllm_torchtpu.models.vllm.qwen4_exp import qsa

    monkeypatch.setattr(bsp, "_KEY_NORM_BOUND", None)
    monkeypatch.setattr(envs, "VLLM_TPU_QSA_PREFILL_BOUND_STAB", env_value)
    cfg = types.SimpleNamespace(
        additional_config={"qwen4_exp": {"qsa_prefill_bound_stab": config_value}}
    )
    monkeypatch.setattr(vllm.config, "get_current_vllm_config", lambda: cfg)
    layer = types.SimpleNamespace(
        k_norm=types.SimpleNamespace(weight=torch.full((DIM,), 0.5)),
        head_dim=DIM,
        prefix="layers.3.self_attn",
    )
    qsa.Qwen4ExpQSA._note_key_norm_bound(layer)
    if noted:
        assert bsp.key_norm_bound() == pytest.approx(16 * 1.5)
    else:
        assert bsp.key_norm_bound() is None
