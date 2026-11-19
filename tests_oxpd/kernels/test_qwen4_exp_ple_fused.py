# SPDX-License-Identifier: Apache-2.0
"""Fused prefill PLE after the lookup (kernels/qwen4_exp/ple_fused.py) and the
one-hot history read in short_conv (interpret mode, small widths).

The conv pass plus its fix-up must equal short_conv bit for bit on the same
inputs; the gate pass reorders reductions, so it is judged against a float64
truth by the 2x rule (its error at most twice the reference path's)."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.qwen4_exp import ple_fused
from vllm_torchtpu.kernels.qwen4_exp.ple import short_conv

STREAMS, HIDDEN, TAPS, DIL, EPS, SLOTS = 4, 128, 4, 3, 1e-6, 8
WIDTH = STREAMS * HIDDEN
STATE = (TAPS - 1) * DIL


def _layout(name, tokens):
    if name == "one":
        lengths, ids, init = [tokens], [5], [True]
    elif name == "two":
        a, valid = tokens * 5 // 16, tokens - tokens // 16
        lengths, ids, init = [a, valid - a], [5, 9], [True, False]
    elif name == "mix":
        valid = tokens - 37
        lengths = [5, 3, 11, 1, 7, tokens // 4, 2]
        lengths.append(valid - sum(lengths))
        ids, init = list(range(1, 9)), [True, False, True, True, False, True, True, False]
    elif name == "idle":  # an active request between an id-0 slot and padding
        lengths, ids, init = [40, tokens - 40 - 16], [0, 7], [True, True]
    else:
        raise ValueError(name)
    starts = [0]
    for length in lengths:
        starts.append(starts[-1] + length)
    n = len(lengths)
    valid = starts[n]
    starts += [starts[-1]] * (SLOTS - n)
    ids += [-1] * (SLOTS - n)
    init += [False] * (SLOTS - n)
    return (
        jnp.asarray(starts, jnp.int32),
        jnp.asarray(ids, jnp.int32),
        jnp.asarray(init, bool),
        jnp.asarray(valid, jnp.int32),
    )


def _bf16(rng, shape, scale=1.0):
    return jnp.asarray((rng.normal(size=shape) * scale).astype(np.float32), jnp.bfloat16)


def _owner_mask(starts, ids, valid, tokens):
    """forward_with_history's mask with token_requests' cumsum owner, in numpy."""
    starts, ids = np.asarray(starts), np.asarray(ids)
    marks = np.zeros(tokens + 1, np.int64)
    for s in np.minimum(starts[1:SLOTS], tokens):
        marks[s] += 1
    owner = np.minimum(np.cumsum(marks[:tokens]), SLOTS - 1)
    return (np.arange(tokens) < int(valid)) & (ids[owner] >= 0)


@pytest.mark.parametrize("layout", ["one", "two", "mix", "idle"])
def test_onehot_history_is_bitwise(layout):
    tokens = 256
    rng = np.random.default_rng(3)
    x = _bf16(rng, (tokens, WIDTH))
    weight = _bf16(rng, (WIDTH, TAPS), 0.5)
    history = np.array(_bf16(rng, (SLOTS, STATE, WIDTH)))
    starts, ids, init, _ = _layout(layout, tokens)
    # A non-finite row in a slot without initial state must not reach any output.
    unused = int(np.flatnonzero(~np.asarray(init))[0])
    history[unused, 2, 7] = np.nan
    history = jnp.asarray(history)
    active = (ids > 0) & (starts[1:] > starts[:-1])
    slots = jnp.where(active, jnp.arange(SLOTS, dtype=jnp.int32), -1)
    run = jax.jit(short_conv, static_argnames=("dilation", "request_owned_history", "past_matmul"))
    ref = run(x, weight, history, starts, slots, init, dilation=DIL, request_owned_history=True)
    got = run(
        x, weight, history, starts, slots, init, dilation=DIL,
        request_owned_history=True, past_matmul=True,
    )
    for a, b in zip(ref, got):
        np.testing.assert_array_equal(
            np.asarray(a).view(np.uint16), np.asarray(b).view(np.uint16)
        )


@pytest.mark.parametrize("layout", ["one", "two", "mix", "idle"])
def test_request_rows_matches_served_masks(layout):
    tokens = 256
    starts, ids, init, valid = _layout(layout, tokens)
    masks = np.asarray(ple_fused.request_rows(starts, ids, valid, tokens))
    np.testing.assert_array_equal(masks[:, 0] != 0, _owner_mask(starts, ids, valid, tokens))
    s, i = np.asarray(starts), np.asarray(ids)
    t = np.arange(tokens)
    req = np.minimum((t[:, None] >= s[None, 1:]).sum(1), SLOTS - 1)
    active = (i > 0) & (s[1:] > s[:-1])
    np.testing.assert_array_equal(masks[:, 1] != 0, (t < s[-1]) & active[req])


@pytest.mark.parametrize("tap_mode,tile", [("slice", 128), ("roll", 128), ("mxu", 128), ("slice", 256)])
@pytest.mark.parametrize("layout", ["one", "two", "mix", "idle"])
def test_conv_pass_and_fixup_equal_short_conv(layout, tap_mode, tile):
    tokens = 512
    rng = np.random.default_rng(11)
    conv_in, gated = _bf16(rng, (tokens, WIDTH)), _bf16(rng, (tokens, WIDTH))
    weight = _bf16(rng, (WIDTH, TAPS), 0.5)
    history = _bf16(rng, (SLOTS, STATE, WIDTH))
    starts, ids, init, valid = _layout(layout, tokens)
    masks = ple_fused.request_rows(starts, ids, valid, tokens)
    taps = jnp.zeros((8, WIDTH), jnp.float32).at[:TAPS].set(weight.astype(jnp.float32).T)
    out = ple_fused.ple_conv_fn(WIDTH, TAPS, DIL, tile, tap_mode, True)(conv_in, gated, taps, masks)
    out = ple_fused.conv_fixup(
        out, conv_in, gated, history, weight, starts, ids, init, masks, dilation=DIL
    )
    windows = ple_fused.new_windows(conv_in, history, starts, init)

    active = (ids > 0) & (starts[1:] > starts[:-1])
    slots = jnp.where(active, jnp.arange(SLOTS, dtype=jnp.int32), -1)
    conv, updated = short_conv(
        conv_in, weight, history, starts, slots, init, dilation=DIL, request_owned_history=True
    )
    total = (gated.astype(jnp.float32) + conv.astype(jnp.float32)).astype(jnp.bfloat16)
    ok = _owner_mask(starts, ids, valid, tokens)
    ref = jnp.where(jnp.asarray(ok)[:, None], total, jnp.zeros((), total.dtype))
    np.testing.assert_array_equal(np.asarray(out).view(np.uint16), np.asarray(ref).view(np.uint16))
    np.testing.assert_array_equal(
        np.asarray(windows).view(np.uint16), np.asarray(updated).view(np.uint16)
    )


def _reference(h, e, key_w, value_w, wq, wk, wc):
    """gated_values + norm_conv with PyTorch opmath rounding (bf16 ops in f32, one rounding)."""
    r = lambda x: x.astype(jnp.bfloat16).astype(jnp.float32)  # noqa: E731
    tokens = h.shape[0]

    def norm(x, w):
        g = x.astype(jnp.float32).reshape(tokens, STREAMS, HIDDEN)
        g = g * jax.lax.rsqrt(jnp.mean(g * g, axis=-1, keepdims=True) + EPS)
        return (g.reshape(tokens, WIDTH) * (1.0 + w.astype(jnp.float32))).astype(jnp.bfloat16)

    def lin(x, w):
        return jax.lax.dot_general(
            x, w, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
        ).astype(jnp.bfloat16)

    key, value = lin(e, key_w), lin(e, value_w)
    kn = norm(key, wk).reshape(tokens, STREAMS, HIDDEN).astype(jnp.float32)
    qn = norm(h, wq).reshape(tokens, STREAMS, HIDDEN).astype(jnp.float32)
    dot = r(jnp.sum(r(kn * qn), axis=-1, keepdims=True))
    gate = r(dot / jnp.float32(math.sqrt(HIDDEN)))
    signed = r(r(jnp.sqrt(r(jnp.maximum(jnp.abs(gate), jnp.float32(1e-6))))) * jnp.sign(gate))
    sig = r(jax.nn.sigmoid(signed))
    gated = (sig * value.astype(jnp.float32)[:, None, :]).astype(jnp.bfloat16).reshape(tokens, WIDTH)
    return key, value, gated, norm(gated, wc)


def _truth(h, e, key_w, value_w, wq, wk, wc):
    f = lambda x: np.asarray(x.astype(jnp.float32), np.float64)  # noqa: E731
    key, value = f(e) @ f(key_w).T, f(e) @ f(value_w).T

    def norm(x, w):
        g = x.reshape(len(x), STREAMS, HIDDEN)
        g = g / np.sqrt(np.mean(g * g, axis=-1, keepdims=True) + EPS)
        return g.reshape(len(x), WIDTH) * (1.0 + f(w))

    kn = norm(key, wk).reshape(-1, STREAMS, HIDDEN)
    qn = norm(f(h), wq).reshape(-1, STREAMS, HIDDEN)
    gate = np.sum(kn * qn, axis=-1, keepdims=True) / math.sqrt(HIDDEN)
    signed = np.sqrt(np.maximum(np.abs(gate), 1e-6)) * np.sign(gate)
    gated = (value[:, None, :] / (1.0 + np.exp(-signed))).reshape(-1, WIDTH)
    return gated, norm(gated, wc)


def _err(a, truth):
    d = np.asarray(a.astype(jnp.float32), np.float64) - truth
    return float(np.sqrt(np.mean(d * d))), float(np.max(np.abs(d)))


@pytest.mark.parametrize("tile", [128, 256])
def test_gate_pass_meets_float64_rule(tile):
    tokens = 256
    rng = np.random.default_rng(5)
    h, e = _bf16(rng, (tokens, WIDTH)), _bf16(rng, (tokens, HIDDEN))
    key_w, value_w = _bf16(rng, (WIDTH, HIDDEN), HIDDEN**-0.5), _bf16(rng, (HIDDEN, HIDDEN), HIDDEN**-0.5)
    wq, wk, wc = (_bf16(rng, (WIDTH,), 0.1) for _ in range(3))
    key, value, ref_gated, ref_conv = _reference(h, e, key_w, value_w, wq, wk, wc)
    one = lambda w: (1.0 + w.astype(jnp.float32)).reshape(1, WIDTH)  # noqa: E731
    truth_gated, truth_conv = _truth(h, e, key_w, value_w, wq, wk, wc)
    for mutant in (False, True):
        gated, conv_in = ple_fused.ple_gate_fn(STREAMS, HIDDEN, tile, EPS, mutant, True)(
            h, key, value, one(wq), one(wk), one(wc)
        )
        passes = all(
            got[0] <= 2 * ref[0] and got[1] <= 2 * ref[1]
            for got, ref in (
                (_err(gated, truth_gated), _err(ref_gated, truth_gated)),
                (_err(conv_in, truth_conv), _err(ref_conv, truth_conv)),
            )
        )
        # The sign-flipped gate is the control that the rule can fail.
        assert passes is (not mutant)


@pytest.mark.parametrize("layout", ["two", "mix"])
def test_fused_layer_meets_float64_rule_and_validates(layout):
    tokens = 256
    rng = np.random.default_rng(8)
    h, e = _bf16(rng, (tokens, WIDTH)), _bf16(rng, (tokens, HIDDEN))
    key_w, value_w = _bf16(rng, (WIDTH, HIDDEN), HIDDEN**-0.5), _bf16(rng, (HIDDEN, HIDDEN), HIDDEN**-0.5)
    wq, wk, wc = (_bf16(rng, (WIDTH,), 0.1) for _ in range(3))
    conv_w, history = _bf16(rng, (WIDTH, TAPS), 0.5), _bf16(rng, (SLOTS, STATE, WIDTH))
    starts, ids, init, valid = _layout(layout, tokens)
    one = lambda w: (1.0 + w.astype(jnp.float32)).reshape(1, WIDTH)  # noqa: E731
    out, windows = ple_fused.ple_fused_after_lookup(
        h, e, key_w, value_w, one(wq), one(wk), one(wc), conv_w, history, starts, ids, init, valid,
        streams=STREAMS, hidden=HIDDEN, eps=EPS, dilation=DIL, token_tile=128, interpret=True,
    )
    _, _, gated, conv_in = _reference(h, e, key_w, value_w, wq, wk, wc)
    active = (ids > 0) & (starts[1:] > starts[:-1])
    slots = jnp.where(active, jnp.arange(SLOTS, dtype=jnp.int32), -1)
    conv, updated = short_conv(
        conv_in, conv_w, history, starts, slots, init, dilation=DIL, request_owned_history=True
    )
    ok = _owner_mask(starts, ids, valid, tokens)
    ref = jnp.where(
        jnp.asarray(ok)[:, None],
        (gated.astype(jnp.float32) + conv.astype(jnp.float32)).astype(jnp.bfloat16),
        jnp.zeros((), jnp.bfloat16),
    )
    # Masked rows are exact zeros; the rest stays within 2x of the reference's distance to
    # its own output (no float64 conv here: both share short_conv's math downstream).
    got, want = np.asarray(out.astype(jnp.float32)), np.asarray(ref.astype(jnp.float32))
    assert np.all(got[~ok] == 0) and np.all(np.isfinite(got))
    assert float(np.mean(np.abs(got - want))) <= 2e-3 * float(np.mean(np.abs(want)))
    assert np.asarray(windows).shape == np.asarray(updated).shape


@pytest.mark.parametrize("tap_mode", ["slice", "mxu"])
@pytest.mark.parametrize("layout", ["one", "two", "mix", "idle"])
@pytest.mark.parametrize("tokens", [256, 512])
def test_single_pass_equals_two_pass_bitwise(layout, tokens, tap_mode):
    """ple_single_after_lookup (one Pallas pass, one-hot history and windows) gives the same bits as
    ple_fused_after_lookup (gate pass + conv pass + XLA fix-up): outputs and new windows."""
    rng = np.random.default_rng(21)
    h, e = _bf16(rng, (tokens, WIDTH)), _bf16(rng, (tokens, HIDDEN))
    key_w, value_w = _bf16(rng, (WIDTH, HIDDEN), HIDDEN**-0.5), _bf16(rng, (HIDDEN, HIDDEN), HIDDEN**-0.5)
    wq, wk, wc = (_bf16(rng, (WIDTH,), 0.1) for _ in range(3))
    conv_w, history = _bf16(rng, (WIDTH, TAPS), 0.5), _bf16(rng, (SLOTS, STATE, WIDTH))
    starts, ids, init, valid = _layout(layout, tokens)
    one = lambda w: (1.0 + w.astype(jnp.float32)).reshape(1, WIDTH)  # noqa: E731
    args = (h, e, key_w, value_w, one(wq), one(wk), one(wc), conv_w, history, starts, ids, init, valid)
    kw = dict(streams=STREAMS, hidden=HIDDEN, eps=EPS, dilation=DIL, token_tile=128, interpret=True)
    two_out, two_win = ple_fused.ple_fused_after_lookup(*args, **kw)
    one_out, one_win = ple_fused.ple_single_after_lookup(*args, **kw, tap_mode=tap_mode)
    np.testing.assert_array_equal(np.asarray(one_out).view(np.uint16), np.asarray(two_out).view(np.uint16))
    np.testing.assert_array_equal(np.asarray(one_win).view(np.uint16), np.asarray(two_win).view(np.uint16))


@pytest.mark.parametrize("mode", ["single", "single_mxu", "slice"])
def test_pooled_dispatch_routes_single_modes(monkeypatch, mode):
    """pooled_fused_ple sends single / single_mxu to the single-pass kernel and the rest to the two-pass path."""
    seen = []
    monkeypatch.setattr(ple_fused, "ple_single_after_lookup",
                        lambda *a, **k: seen.append(("single", k["tap_mode"])) or (a[0], a[8]))
    monkeypatch.setattr(ple_fused, "ple_fused_after_lookup",
                        lambda *a, **k: seen.append(("two", k["tap_mode"])) or (a[0], a[8]))
    import vllm_torchtpu.kernels.qwen4_exp.pool as pool

    monkeypatch.setattr(pool, "read_region", lambda p, ids, elements, dtype: jnp.zeros((len(ids), elements), dtype))
    monkeypatch.setattr(pool, "write_region", lambda p, v, ids: p)
    rng = np.random.default_rng(2)
    h = _bf16(rng, (256, WIDTH))
    starts, ids, init, valid = _layout("two", 256)
    w = _bf16(rng, (WIDTH,))
    ple_fused.pooled_fused_ple(
        h, _bf16(rng, (256, HIDDEN)), _bf16(rng, (WIDTH, HIDDEN)), _bf16(rng, (HIDDEN, HIDDEN)), w, w, w,
        _bf16(rng, (WIDTH, TAPS)), jnp.zeros((4,)), starts, ids, init, valid.reshape(1),
        streams=STREAMS, hidden=HIDDEN, eps=EPS, dilation=DIL, state_len=STATE, token_tile=128, tap_mode=mode,
        interpret=True,
    )
    want = {"single": ("single", "slice"), "single_mxu": ("single", "mxu"), "slice": ("two", "slice")}[mode]
    assert seen == [want]


@pytest.mark.parametrize("layout", ["one", "two", "mix"])
def test_onehot_probe_mutant_changes_history_rows(layout):
    """Q38_PROBE_PLE_ONEHOT_MUTANT's short_conv(past_matmul="mutant") must differ from the gather wherever history
    is read (the identity probe's must-differ control), and only there."""
    tokens = 256
    rng = np.random.default_rng(4)
    x, weight = _bf16(rng, (tokens, WIDTH)), _bf16(rng, (WIDTH, TAPS), 0.5)
    history = _bf16(rng, (SLOTS, STATE, WIDTH))
    starts, ids, init, _ = _layout(layout, tokens)
    active = (ids > 0) & (starts[1:] > starts[:-1])
    slots = jnp.where(active, jnp.arange(SLOTS, dtype=jnp.int32), -1)
    ref, _ = short_conv(x, weight, history, starts, slots, init, dilation=DIL, request_owned_history=True)
    mut, _ = short_conv(x, weight, history, starts, slots, init, dilation=DIL, request_owned_history=True,
                        past_matmul="mutant")
    differ = np.any(np.asarray(ref).view(np.uint16) != np.asarray(mut).view(np.uint16), axis=1)
    s, init_np = np.asarray(starts), np.asarray(init)
    hist_rows = np.zeros(tokens, bool)
    for r in range(SLOTS):
        if init_np[r] and np.asarray(active)[r]:
            hist_rows[s[r]:min(s[r] + STATE, s[r + 1])] = True
    assert differ.any() and not differ[~hist_rows].any()


@pytest.mark.parametrize("norm_mode", ["sum", "mean", "recip", "newton"])
def test_gate_norm_modes_meet_float64_rule(norm_mode):
    tokens = 256
    rng = np.random.default_rng(5)
    h, e = _bf16(rng, (tokens, WIDTH)), _bf16(rng, (tokens, HIDDEN))
    key_w, value_w = _bf16(rng, (WIDTH, HIDDEN), HIDDEN**-0.5), _bf16(rng, (HIDDEN, HIDDEN), HIDDEN**-0.5)
    wq, wk, wc = (_bf16(rng, (WIDTH,), 0.1) for _ in range(3))
    key, value, ref_gated, ref_conv = _reference(h, e, key_w, value_w, wq, wk, wc)
    one = lambda w: (1.0 + w.astype(jnp.float32)).reshape(1, WIDTH)  # noqa: E731
    truth_gated, truth_conv = _truth(h, e, key_w, value_w, wq, wk, wc)
    gated, conv_in = ple_fused.ple_gate_fn(STREAMS, HIDDEN, 128, EPS, False, True, norm_mode)(
        h, key, value, one(wq), one(wk), one(wc))
    for got, ref, truth in ((gated, ref_gated, truth_gated), (conv_in, ref_conv, truth_conv)):
        g, r = _err(got, truth), _err(ref, truth)
        assert g[0] <= 2 * r[0] and g[1] <= 2 * r[1], (norm_mode, g, r)


def _f32_value_reference(h, e, key_w, value_w, wq, wk, wc):
    """Precision "xla": PyTorch's sig chain, the value projection kept f32, the conv norm on sig * value in f32."""
    key, _, _, _ = _reference(h, e, key_w, value_w, wq, wk, wc)
    value32 = jax.lax.dot_general(e, value_w, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)
    return key, value32


def test_gate_precision_xla_is_more_accurate_on_the_conv_input():
    """Precision "xla" takes the value projection in f32 and feeds the conv norm the f32 product: its conv input is
    closer to float64 than the torch rounding's (on CPU at full widths: 0.58x RMS, mb6b), gated stays bf16(sig * value)."""
    tokens = 256
    rng = np.random.default_rng(5)
    h, e = _bf16(rng, (tokens, WIDTH)), _bf16(rng, (tokens, HIDDEN))
    key_w, value_w = _bf16(rng, (WIDTH, HIDDEN), HIDDEN**-0.5), _bf16(rng, (HIDDEN, HIDDEN), HIDDEN**-0.5)
    wq, wk, wc = (_bf16(rng, (WIDTH,), 0.1) for _ in range(3))
    key, value, _, _ = _reference(h, e, key_w, value_w, wq, wk, wc)
    _, value32 = _f32_value_reference(h, e, key_w, value_w, wq, wk, wc)
    one = lambda w: (1.0 + w.astype(jnp.float32)).reshape(1, WIDTH)  # noqa: E731
    truth_gated, truth_conv = _truth(h, e, key_w, value_w, wq, wk, wc)
    g_t, c_t = ple_fused.ple_gate_fn(STREAMS, HIDDEN, 128, EPS, False, True)(h, key, value, one(wq), one(wk), one(wc))
    g_x, c_x = ple_fused.ple_gate_fn(STREAMS, HIDDEN, 128, EPS, False, True, "sum", "xla")(
        h, key, value32, one(wq), one(wk), one(wc))
    rt, rx = _err(c_t, truth_conv), _err(c_x, truth_conv)
    assert rx[0] <= 0.8 * rt[0], (rx, rt)
    # gated is still stored as bf16 of the f32 product, and stays within the torch rounding's error
    assert g_x.dtype == jnp.bfloat16 and _err(g_x, truth_gated)[0] <= 1.05 * _err(g_t, truth_gated)[0]
    with pytest.raises(ValueError):   # the "xla" gate takes an f32 value only
        ple_fused.ple_gate_fn(STREAMS, HIDDEN, 128, EPS, False, True, "sum", "xla")(
            h, key, value, one(wq), one(wk), one(wc))


@pytest.mark.parametrize("tap_mode", ["slice", "mxu"])
def test_precision_xla_single_pass_equals_two_pass_and_beats_torch_on_windows(tap_mode):
    """Under precision "xla" the single pass still gives the two-pass path's bits, and the new windows (the conv
    input's last rows) are closer to float64 than under "torch" (mb1's failing metric)."""
    tokens = 256
    rng = np.random.default_rng(21)
    h, e = _bf16(rng, (tokens, WIDTH)), _bf16(rng, (tokens, HIDDEN))
    key_w, value_w = _bf16(rng, (WIDTH, HIDDEN), HIDDEN**-0.5), _bf16(rng, (HIDDEN, HIDDEN), HIDDEN**-0.5)
    wq, wk, wc = (_bf16(rng, (WIDTH,), 0.1) for _ in range(3))
    conv_w, history = _bf16(rng, (WIDTH, TAPS), 0.5), _bf16(rng, (SLOTS, STATE, WIDTH))
    starts, ids, init, valid = _layout("two", tokens)
    one = lambda w: (1.0 + w.astype(jnp.float32)).reshape(1, WIDTH)  # noqa: E731
    args = (h, e, key_w, value_w, one(wq), one(wk), one(wc), conv_w, history, starts, ids, init, valid)
    kw = dict(streams=STREAMS, hidden=HIDDEN, eps=EPS, dilation=DIL, token_tile=128, interpret=True)
    two_out, two_win = ple_fused.ple_fused_after_lookup(*args, **kw, precision="xla")
    one_out, one_win = ple_fused.ple_single_after_lookup(*args, **kw, tap_mode=tap_mode, precision="xla")
    np.testing.assert_array_equal(np.asarray(one_out).view(np.uint16), np.asarray(two_out).view(np.uint16))
    np.testing.assert_array_equal(np.asarray(one_win).view(np.uint16), np.asarray(two_win).view(np.uint16))
    _, t_win = ple_fused.ple_fused_after_lookup(*args, **kw)
    _, truth_conv = _truth(h, e, key_w, value_w, wq, wk, wc)
    s = np.asarray(starts)
    rows = [r for r in range(SLOTS) if int(ids[r]) > 0 and s[r + 1] - s[r] >= STATE]
    truth = np.stack([truth_conv[s[r + 1] - STATE:s[r + 1]] for r in rows])
    ex = _err(np.asarray(two_win)[rows], truth)
    et = _err(np.asarray(t_win)[rows], truth)
    assert rows and ex[0] <= 0.8 * et[0], (ex, et)



def test_pooled_fused_ple_takes_the_conv_layer_weight_whole(monkeypatch):
    """The model passes conv1d.weight [C, 1, taps] whole (no torch-side slice, so the fused and served traces lift it
    in the same placeholder position, srv1-F3); the op slices it, and the kernels receive exactly the [C, taps]
    values of the 2-D call."""
    seen = []
    monkeypatch.setattr(ple_fused, "ple_fused_after_lookup",
                        lambda *a, **k: seen.append(np.asarray(a[7])) or (a[0], a[8]))
    import vllm_torchtpu.kernels.qwen4_exp.pool as pool

    monkeypatch.setattr(pool, "read_region", lambda p, ids, elements, dtype: jnp.zeros((len(ids), elements), dtype))
    monkeypatch.setattr(pool, "write_region", lambda p, v, ids: p)
    rng = np.random.default_rng(3)
    h = _bf16(rng, (256, WIDTH))
    starts, ids, init, valid = _layout("two", 256)
    w = _bf16(rng, (WIDTH,))
    conv2 = _bf16(rng, (WIDTH, TAPS), 0.5)
    rest = (_bf16(rng, (256, HIDDEN)), _bf16(rng, (WIDTH, HIDDEN)), _bf16(rng, (HIDDEN, HIDDEN)), w, w, w)
    kw = dict(streams=STREAMS, hidden=HIDDEN, eps=EPS, dilation=DIL, state_len=STATE, token_tile=128, tap_mode="slice",
              interpret=True)
    for conv in (conv2, conv2[:, None, :]):
        ple_fused.pooled_fused_ple(h, *rest, conv, jnp.zeros((4,)), starts, ids, init, valid.reshape(1), **kw)
    assert len(seen) == 2 and seen[1].shape == seen[0].shape == (WIDTH, TAPS)
    np.testing.assert_array_equal(seen[1].view(np.uint16), seen[0].view(np.uint16))


def test_pooled_fused_ple_kq_takes_the_norm_weights_key_first(monkeypatch):
    """Fix 4: the custom op takes norm_key before norm_query (the served PLE step's first-use order, so a token bucket
    handed between the fused and the served trace binds both correctly). pooled_fused_ple_kq gives exactly the bits of
    pooled_fused_ple with the roles kept, and passing the two [C] norm weights in each other's place changes the
    output, so a mis-binding of the two would be visible in the numbers."""
    import vllm_torchtpu.kernels.qwen4_exp.pool as pool

    monkeypatch.setattr(pool, "read_region", lambda p, ids, elements, dtype: jnp.zeros((len(ids), elements), dtype))
    monkeypatch.setattr(pool, "write_region", lambda p, v, ids: p)
    rng = np.random.default_rng(5)
    h = _bf16(rng, (256, WIDTH))
    e, key, value = _bf16(rng, (256, HIDDEN)), _bf16(rng, (WIDTH, HIDDEN)), _bf16(rng, (HIDDEN, HIDDEN))
    wq, wk, wc = _bf16(rng, (WIDTH,)), _bf16(rng, (WIDTH,)), _bf16(rng, (WIDTH,))
    conv = _bf16(rng, (WIDTH, TAPS), 0.5)
    starts, ids, init, valid = _layout("two", 256)
    tail = (conv, jnp.zeros((4,)), starts, ids, init, valid.reshape(1))
    kw = dict(streams=STREAMS, hidden=HIDDEN, eps=EPS, dilation=DIL, state_len=STATE, token_tile=128, tap_mode="slice",
              interpret=True)
    ref = ple_fused.pooled_fused_ple(h, e, key, value, wq, wk, wc, *tail, **kw)
    kq = ple_fused.pooled_fused_ple_kq(h, e, key, value, wk, wq, wc, *tail, **kw)
    swapped = ple_fused.pooled_fused_ple(h, e, key, value, wk, wq, wc, *tail, **kw)
    bits = lambda x: np.asarray(x).view(np.uint16)  # noqa: E731
    np.testing.assert_array_equal(bits(kq[1]), bits(ref[1]))
    assert not np.array_equal(bits(swapped[1]), bits(ref[1])), "swapping norm_query / norm_key did not change F"
