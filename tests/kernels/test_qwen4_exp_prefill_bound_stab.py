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
    monkeypatch.setattr(bsp, "_KEY_NORM_BOUND", None)
    monkeypatch.setattr(bsp, "_TRACE_BOUND", [None])


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
            bound_stab=True, key_norm_bound=bsp.key_norm_bound(),
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
        prefix="model.layers.3.self_attn",
        main_kv_dcp=2,
    )
    qsa.Qwen4ExpQSA._note_key_norm_bound(layer)
    if noted:
        assert bsp.key_norm_bound() == pytest.approx(16 * 1.5)
    else:
        assert bsp.key_norm_bound() is None


@pytest.mark.parametrize("prefix,dcp", [("mtp.layers.48.self_attn", 2), ("mtp.layers.48.self_attn", 1)])
def test_mtp_layer_does_not_note(monkeypatch, prefix, dcp):
    import types

    import torch
    import vllm.config

    from vllm_torchtpu.models.vllm.qwen4_exp import qsa

    monkeypatch.setattr(bsp, "_KEY_NORM_BOUND", None)
    monkeypatch.setattr(envs, "VLLM_TPU_QSA_PREFILL_BOUND_STAB", "1")
    cfg = types.SimpleNamespace(additional_config={})
    monkeypatch.setattr(vllm.config, "get_current_vllm_config", lambda: cfg)
    layer = types.SimpleNamespace(
        k_norm=types.SimpleNamespace(weight=torch.full((DIM,), 6.3)),
        head_dim=DIM, prefix=prefix, main_kv_dcp=dcp,
    )
    qsa.Qwen4ExpQSA._note_key_norm_bound(layer)
    assert bsp.key_norm_bound() is None


@pytest.mark.parametrize("rank", [0, 1])
def test_high_guard_reruns_when_keys_exceed_bound(interpret, monkeypatch, rank):
    case = make_case(np.random.default_rng(7), CASES["single"], 24)
    query = case[0] * 8  # scores far above what a 0.05-scaled bound allows
    case = (query,) + case[1:]
    want = attend(case, rank, 0)
    bsp.note_key_norm_bound(case[-1] * 0.05)
    _, lse_kernel = kernel_only(case, rank, 0)
    stab = np.asarray(bsp.bound_stabilizer(query, k_scale=K_SCALE, bound=case[-1] * 0.05))
    lse_kernel = np.asarray(lse_kernel)
    fin = np.isfinite(np.asarray(want[1]))
    # Positive control: the kernel alone sits above the high guard (or overflowed).
    over = ~np.isfinite(lse_kernel[fin]) | (lse_kernel[fin] - stab[fin] > bsp.BOUND_STAB_GUARD_HIGH)
    assert over.any()
    got = attend(case, rank, 0)
    for a, b in zip(got, want):
        np.testing.assert_array_equal(bits(a), bits(b))


def attend_replicated(case, offset, gated=False):
    query, blocks, lengths, table, cache, starts, _ = case
    gate = jnp.zeros_like(query)
    jax.clear_caches()
    return jax.jit(
        lambda q, c: bsp.block_sparse_gqa_attention(
            q, gate, blocks, lengths, table, c, starts, kv_heads=1, ratio=RATIO,
            k_scale=K_SCALE, v_scale=V_SCALE, query_block=256, kv_tile=PAGE,
            query_tile=256, kv_head_offset=offset,
            **({} if gated else {"return_lse": True}),
        )[: 1 if gated else 2]
    )(query, cache)


@pytest.mark.parametrize("gated", [False, True])
def test_replicated_route_uses_the_bound(interpret, monkeypatch, gated):
    """The non-DCP call (lse and gated forms) runs the bound stabilizer too."""
    case = make_case(np.random.default_rng(8), CASES["two"], 24)
    want = attend_replicated(case, 1, gated)
    bsp.note_key_norm_bound(case[-1])
    got = attend_replicated(case, 1, gated)
    out_w, out_g = np.asarray(want[0], np.float32), np.asarray(got[0], np.float32)
    assert not np.array_equal(out_w, out_g)  # a different rounding, so it ran
    scale = np.maximum(np.abs(out_w).max(-1, keepdims=True), 1e-30)
    assert float((np.abs(out_g - out_w) / scale).max()) < 1e-2
    if not gated:
        fin = np.isfinite(np.asarray(want[1]))
        np.testing.assert_allclose(np.asarray(got[1])[fin], np.asarray(want[1])[fin], atol=4e-3)


def test_replicated_guard_reruns_online(interpret, monkeypatch):
    case = make_case(np.random.default_rng(9), CASES["single"], 24)
    want = attend_replicated(case, 0, gated=True)
    bsp.note_key_norm_bound(1e4)
    got = attend_replicated(case, 0, gated=True)
    np.testing.assert_array_equal(bits(got[0].astype(jnp.float32)), bits(want[0].astype(jnp.float32)))


# DCP-vs-replicated agreement with the bound on both routes needs the paired
# DCP / replicated pool layouts of the dcp2 harness: gate_bs.py arm "bound"
# (TPU, j-03d65bf7: pre-gate 12/12 at 5-6e-6, gated 2 ulp).


def attend_with_layer_bound(case, offset, weight):
    """The QSA step's order in one trace: register the layer bound, then attend."""
    query, blocks, lengths, table, cache, starts, _ = case
    gate = jnp.zeros_like(query)
    jax.clear_caches()

    def step(q, c, w):
        bsp.trace_key_norm_bound(w, DIM)
        return bsp.block_sparse_gqa_attention(
            q, gate, blocks, lengths, table, c, starts, kv_heads=1, ratio=RATIO,
            k_scale=K_SCALE, v_scale=V_SCALE, query_block=256, kv_tile=PAGE,
            query_tile=256, kv_head_offset=offset, return_lse=True,
        )[:2]

    return jax.jit(step)(query, cache, weight)


def test_layer_bound_is_used_over_process_bound(interpret, monkeypatch):
    """The registered layer bound, not the process bound, drives the kernel
    and the guard. With an unusable process bound (100x too large: every row
    under the low guard, so the call re-runs online, bitwise flag-off) the
    layer's own correct bound runs the bound kernel with no re-run. The same
    holds in the MTP direction (layer keys longer than the target bound)."""
    case = make_case(np.random.default_rng(11), CASES["single"], 24)
    want = attend_replicated(case, 0)  # online, flag off
    bsp.note_key_norm_bound(case[-1] * 100)
    fallback = attend_replicated(case, 0)
    for a, b in zip(fallback, want):
        np.testing.assert_array_equal(bits(a), bits(b))
    weight = jnp.full((DIM,), case[-1] / np.sqrt(DIM) - 1.0, jnp.float32)
    got = attend_with_layer_bound(case, 0, weight)
    out_w, out_g = np.asarray(want[0]), np.asarray(got[0])
    assert not np.array_equal(out_w, out_g)  # the bound kernel ran, no re-run
    scale = np.maximum(np.abs(out_w).max(-1, keepdims=True), 1e-30)
    assert float((np.abs(out_g - out_w) / scale).max()) < 8e-3
    fin = np.isfinite(np.asarray(want[1]))
    np.testing.assert_allclose(np.asarray(got[1])[fin], np.asarray(want[1])[fin], atol=4e-3)


def test_stale_layer_bound_falls_back_to_process_bound(interpret, monkeypatch):
    """A bound registered in a finished trace is dropped, not used."""
    case = make_case(np.random.default_rng(12), CASES["single"], 24)
    bsp.note_key_norm_bound(case[-1])
    plain = attend_replicated(case, 0)
    jax.jit(lambda w: (bsp.trace_key_norm_bound(w, DIM), w)[1])(jnp.zeros((DIM,)))
    assert bsp._TRACE_BOUND[0] is not None
    again = attend_replicated(case, 0)
    for a, b in zip(again, plain):
        np.testing.assert_array_equal(bits(a), bits(b))
    assert bsp._TRACE_BOUND[0] is None


def test_no_registration_when_off(monkeypatch):
    monkeypatch.setattr(bsp, "_KEY_NORM_BOUND", None)
    monkeypatch.setattr(bsp, "_TRACE_BOUND", [None])
    bsp.trace_key_norm_bound(jnp.zeros((DIM,)), DIM)
    assert bsp._TRACE_BOUND[0] is None


@pytest.mark.parametrize("vpu", ["1", "2", "mask"])
def test_vpu_path_uses_the_bound(interpret, monkeypatch, vpu):
    """VLLM_TPU_QSA_PREFILL_VPU (no LANE_SUM) takes the bound too: both routes
    match the online softmax within the BF16-probability tolerance, and an
    absurd bound makes the guard re-run the online softmax bit for bit."""
    monkeypatch.setattr(envs, "VLLM_TPU_QSA_PREFILL_VPU", vpu)
    assert bsp.bound_kernel_eligible(
        ratio=RATIO, query_block=256, query_tile=256, kv_tile=PAGE, dcp=True,
        cache_dtype=jnp.float8_e4m3fn,
    )
    case = make_case(np.random.default_rng(13), CASES["two"], 24)
    want_dcp, want_rep = attend(case, 0, 1), attend_replicated(case, 1, gated=True)
    bsp.note_key_norm_bound(case[-1])
    got_dcp, got_rep = attend(case, 0, 1), attend_replicated(case, 1, gated=True)
    out_w, lse_w = (np.asarray(x) for x in want_dcp)
    out_g, lse_g = (np.asarray(x) for x in got_dcp)
    assert not np.array_equal(out_w, out_g)  # a different rounding, so it ran
    np.testing.assert_array_equal(np.isfinite(lse_g), np.isfinite(lse_w))
    fin = np.isfinite(lse_w)
    np.testing.assert_allclose(lse_g[fin], lse_w[fin], atol=4e-3)
    scale = np.maximum(np.abs(out_w).max(-1, keepdims=True), 1e-30)
    assert float((np.abs(out_g - out_w) / scale).max()) < 8e-3
    rep_w = np.asarray(want_rep[0], np.float32)
    rep_g = np.asarray(got_rep[0], np.float32)
    scale = np.maximum(np.abs(rep_w).max(-1, keepdims=True), 1e-30)
    assert float((np.abs(rep_g - rep_w) / scale).max()) < 1e-2
    bsp.note_key_norm_bound(1e4)  # every row underflows: the guard re-runs
    rerun_dcp, rerun_rep = attend(case, 0, 1), attend_replicated(case, 1, gated=True)
    for a, b in zip(rerun_dcp, want_dcp):
        np.testing.assert_array_equal(bits(a), bits(b))
    np.testing.assert_array_equal(
        bits(rerun_rep[0].astype(jnp.float32)), bits(want_rep[0].astype(jnp.float32))
    )


@pytest.mark.parametrize("query_scale,unshifted", [(1.0, True), (1.5, False)])
def test_vpu_unshifted_and_shifted_blocks(
    interpret, monkeypatch, query_scale, unshifted
):
    """A block whose score bounds are within BOUND_STAB_ZERO_LIMIT walks with
    a zero stabilizer (no per-score subtraction), a block above it with the
    bound stabilizer: both match the online softmax within the
    BF16-probability tolerance."""
    monkeypatch.setattr(envs, "VLLM_TPU_QSA_PREFILL_VPU", "2")
    case = make_case(np.random.default_rng(14), CASES["two"], 24)
    case = (case[0] * query_scale,) + case[1:]
    stab = np.asarray(bsp.bound_stabilizer(case[0], k_scale=K_SCALE, bound=case[-1]))
    score_bound = stab.max() + bsp.BOUND_STAB_SHIFT
    assert (score_bound <= bsp.BOUND_STAB_ZERO_LIMIT) == unshifted
    want = attend(case, 0, 1)
    bsp.note_key_norm_bound(case[-1])
    got = attend(case, 0, 1)
    out_w, lse_w = (np.asarray(x) for x in want)
    out_g, lse_g = (np.asarray(x) for x in got)
    assert not np.array_equal(out_w, out_g)  # a different rounding, so it ran
    np.testing.assert_array_equal(np.isfinite(lse_g), np.isfinite(lse_w))
    fin = np.isfinite(lse_w)
    np.testing.assert_allclose(lse_g[fin], lse_w[fin], atol=4e-3)
    scale = np.maximum(np.abs(out_w).max(-1, keepdims=True), 1e-30)
    assert float((np.abs(out_g - out_w) / scale).max()) < 8e-3


@pytest.mark.parametrize("vpu", ["1", "2"])
def test_vpu_bound_block_unit_bitwise(interpret, monkeypatch, vpu):
    """With the bound, a 512-token block of 3 query heads runs as one unit
    per KV tile (BOUND_UNIT_SCORES): the same per-row arithmetic as two
    256-token subtiles, bit for bit here (on TPU the taller PV dot may sum
    its FP32 partials in another order)."""
    monkeypatch.setattr(envs, "VLLM_TPU_QSA_PREFILL_VPU", vpu)
    query, blocks, lengths, table, cache, starts, bound = make_case(
        np.random.default_rng(15), [(700, 600), (2047, 424)], 24
    )
    bsp.note_key_norm_bound(bound)

    def run():
        jax.clear_caches()
        return jax.jit(
            lambda q, c: bsp.block_sparse_gqa_attention(
                q, jnp.zeros_like(q), blocks, lengths, table, c, starts, kv_heads=1,
                ratio=RATIO, k_scale=K_SCALE, v_scale=V_SCALE, query_block=512,
                kv_tile=PAGE, query_tile=256, return_lse=True,
            )[:2]
        )(query, cache)

    whole = run()
    monkeypatch.setattr(bsp, "BOUND_UNIT_SCORES", 768 * PAGE)
    split = run()
    for a, b in zip(whole, split):
        np.testing.assert_array_equal(bits(a), bits(b))


def test_bound_route_on_when_eligible(interpret, monkeypatch):
    monkeypatch.setattr(envs, "VLLM_TPU_QSA_PREFILL_VPU", "")
    assert bsp.bound_kernel_eligible(
        ratio=RATIO, query_block=256, query_tile=256, kv_tile=PAGE, dcp=True,
        cache_dtype=jnp.float8_e4m3fn,
    )
    monkeypatch.setattr(envs, "VLLM_TPU_QSA_PREFILL_LANE_SUM", "")
    assert not bsp.bound_kernel_eligible(
        ratio=RATIO, query_block=256, query_tile=256, kv_tile=PAGE, dcp=True,
        cache_dtype=jnp.float8_e4m3fn,
    )


def test_guarded_gated_output_realigns_bf16(interpret, monkeypatch):
    """Gated call with the bound on: only the guard reads lse, so the row-segment
    DMA back to token order moves bf16(out), bit-identical to the scatter layout."""
    case = make_case(np.random.default_rng(8), CASES["two"], 24)
    bsp.note_key_norm_bound(case[-1])
    copied = []
    original = bsp.copy_row_segments

    def spy(src, dst, *args, **kwargs):
        copied.append(src.dtype)
        return original(src, dst, *args, **kwargs)

    monkeypatch.setattr(bsp, "copy_row_segments", spy)
    monkeypatch.setattr(envs, "VLLM_TPU_QSA_PREFILL_ALIGN_DMA", False)
    want = attend_replicated(case, 1, gated=True)
    assert not copied
    monkeypatch.setattr(envs, "VLLM_TPU_QSA_PREFILL_ALIGN_DMA", True)
    got = attend_replicated(case, 1, gated=True)
    assert copied and set(copied) == {jnp.dtype(jnp.bfloat16)}
    np.testing.assert_array_equal(
        bits(got[0].astype(jnp.float32)), bits(want[0].astype(jnp.float32))
    )
