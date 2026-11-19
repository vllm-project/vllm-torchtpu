# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_PLE_DECODE_BOUNDED (q38 L1 D07-b2): the speculative PLE conv on a
uniform-window bucket is bitwise the default op.

``pooled_short_conv_speculative(bounded=1)`` reads the conv history for the
bucket's ceil(T / window) leading request rows instead of every max-num-seqs
row; ``bounded=2`` also takes each token's taps and checkpoint window by static
slices. Every case compares the conv output (all T rows, padded ones included)
and the whole pool after the checkpoint writes against ``bounded=0``.

Pure JAX on CPU: the pool's Pallas region read / write are replaced by plain
XLA stand-ins with the same contract (read row ``max(id, 0)``; write only ids
> 0). Both arms call the same stand-ins, so equal pools mean the arms handed
the writer the same (target, value) pairs. The real Pallas I/O kernels are
unchanged by the flag; the bounded arm only passes them fewer read ids.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.qwen4_exp import pool as pool_mod

STATE_LEN, DILATION, TAPS, WINDOW = 9, 3, 4, 4


def _read_region_ref(pool, ids, *, elements, dtype, tok0=0):
    assert tok0 == 0 and dtype == jnp.bfloat16 and pool.dtype == jnp.bfloat16
    _READS.append(int(ids.shape[0]))
    return pool[jnp.maximum(ids, 0)][:, :elements]


def _write_region_ref(pool, values, ids, *, tok0=0):
    assert tok0 == 0
    _WRITES.append(int(ids.shape[0]))
    destination = jnp.where(ids > 0, ids, pool.shape[0])
    return pool.at[destination, : values.shape[1]].set(values, mode="drop")


_READS: list[int] = []
_WRITES: list[int] = []


@pytest.fixture(autouse=True)
def _xla_pool_io(monkeypatch):
    monkeypatch.setattr(pool_mod, "read_region", _read_region_ref)
    monkeypatch.setattr(pool_mod, "write_region", _write_region_ref)
    _READS.clear()
    _WRITES.clear()


def _case(
    seed,
    *,
    count,
    tokens,
    live,
    channels=128,
    offsets=None,
    no_history=(),
    no_slot=(),
    poison_padding=False,
    heavy=False,
):
    """A decode bucket: ``live`` requests of WINDOW tokens first, then padding.

    ``count`` is max-num-seqs (the metadata's request rows), ``tokens`` the
    padded token bucket. ``offsets[r]`` picks request r's read checkpoint
    (its last accepted verify position); ``no_history`` rows have no initial
    state; ``no_slot`` live rows have state id 0 (inactive);
    ``poison_padding`` fills padded token rows and the null page with NaN.
    ``heavy`` draws half of the values as +-2^30 and the weights as +-1, so
    the f32 tap sum absorbs small terms and its order shows in the output
    (a reordered tap sum must fail the identity check, mutant M6).
    """
    rng = np.random.default_rng(seed)
    assert live * WINDOW <= tokens and live <= count
    pages = 1 + count * WINDOW + 7
    ckpt = np.zeros((count, WINDOW), np.int32)
    ckpt[:live] = rng.permutation(np.arange(1, pages))[: live * WINDOW].reshape(live, WINDOW)
    if offsets is None:
        offsets = rng.integers(0, WINDOW, size=count)
    offsets = np.asarray(offsets, np.int32)
    rows = np.arange(count)
    read_ids = np.where(rows < live, ckpt[rows, offsets], 0).astype(np.int32)
    ids = ckpt[:, 0].copy()
    ids[list(no_slot)] = 0
    starts = np.minimum(np.arange(count + 1) * WINDOW, live * WINDOW).astype(np.int32)
    initial = rows < live
    initial[list(no_history)] = False
    windowed = rows < live
    x = rng.normal(size=(tokens, channels)).astype(np.float32)
    pool = rng.normal(size=(pages, STATE_LEN * channels)).astype(np.float32)
    if poison_padding:
        x[live * WINDOW :] = np.nan
        pool[0] = np.nan
    weight = rng.normal(size=(channels, TAPS)).astype(np.float32)
    if heavy:
        big = rng.random(x.shape) < 0.5
        x = np.where(big, np.sign(rng.normal(size=x.shape)) * 2.0**30, x)
        big = rng.random(pool.shape) < 0.5
        pool = np.where(big & ~np.isnan(pool), np.sign(rng.normal(size=pool.shape)) * 2.0**30, pool)
        weight = np.sign(rng.normal(size=weight.shape)).astype(np.float32)
    return dict(
        x=jnp.asarray(x, jnp.bfloat16),
        weight=jnp.asarray(weight, jnp.bfloat16),
        pool=jnp.asarray(pool, jnp.bfloat16),
        starts=jnp.asarray(starts),
        ids=jnp.asarray(ids),
        initial=jnp.asarray(initial),
        read_ids=jnp.asarray(read_ids),
        ckpt_ids=jnp.asarray(ckpt),
        windowed=jnp.asarray(windowed),
    )


def _run(case, bounded):
    fn = functools.partial(
        pool_mod.pooled_short_conv_speculative,
        dilation=DILATION,
        state_len=STATE_LEN,
        bounded=bounded,
    )
    _READS.clear()
    _WRITES.clear()
    pool, output = jax.jit(fn)(
        case["x"],
        case["weight"],
        case["pool"],
        case["starts"],
        case["ids"],
        case["initial"],
        case["read_ids"],
        case["ckpt_ids"],
        case["windowed"],
    )
    return np.asarray(pool.view(jnp.uint16)), np.asarray(output.view(jnp.uint16)), list(_READS)


def _bits(a):
    return np.asarray(a.view(jnp.uint16))


# (count = max-num-seqs, tokens = bucket, live requests, extra case options)
CASES = [
    (48, 16, 4, {}),
    (48, 16, 1, {}),
    (48, 16, 3, {}),
    (48, 8, 2, {}),
    (48, 8, 1, {"offsets": [3] * 48}),
    (48, 32, 8, {}),
    (48, 32, 5, {"offsets": [0] * 48}),
    (48, 48, 12, {}),
    (48, 64, 16, {}),
    (48, 64, 13, {"no_history": (2, 7)}),
    (48, 64, 16, {"poison_padding": True}),
    (48, 32, 6, {"poison_padding": True, "no_slot": (1,)}),
    (48, 16, 4, {"no_slot": (0, 3), "no_history": (1,)}),
    (16, 64, 16, {}),
    (8, 32, 8, {}),
    (128, 64, 9, {}),
    (64, 16, 4, {"channels": 256}),
    # Order-sensitive f32 tap sums (see _case's ``heavy``).
    (48, 16, 4, {"heavy": True}),
    (48, 32, 7, {"heavy": True}),
    (48, 64, 13, {"heavy": True, "no_history": (5,)}),
    # Not a multiple of the window: bounded=2 falls back to the bounded rows.
    (48, 18, 4, {}),
    # Above count * window: the request-major path, never bounded.
    (4, 24, 4, {}),
]


def _id(case):
    count, tokens, live, extra = case
    tags = "-".join(f"{k}" for k in extra)
    return f"c{count}-t{tokens}-n{live}" + (f"-{tags}" if tags else "")


@pytest.mark.parametrize("case", CASES, ids=[_id(c) for c in CASES])
@pytest.mark.parametrize("bounded", [1, 2])
def test_bounded_conv_is_bitwise_the_default(case, bounded):
    count, tokens, live, extra = case
    inputs = _case(count * 1000 + tokens * 10 + live, count=count, tokens=tokens, live=live, **extra)
    before = _bits(inputs["pool"])
    pool0, out0, reads0 = _run(inputs, 0)
    pool1, out1, reads1 = _run(inputs, bounded)

    # Not vacuous: the default op wrote checkpoints and produced live outputs.
    changed = np.any(pool0 != before, axis=1)
    if extra.get("poison_padding"):
        changed[0] = False
    assert changed.sum() > 0, "the default op wrote no checkpoint"
    live_rows = live * WINDOW
    assert np.any(out0[:live_rows] != 0), "the default op produced no live output"
    assert not np.any(out0[live_rows:]), "padded tokens must output zero"

    np.testing.assert_array_equal(out1, out0, err_msg="PLE conv output differs")
    np.testing.assert_array_equal(pool1, pool0, err_msg="pool after checkpoint writes differs")

    # Engagement: the bounded arm read history for the bucket's rows only.
    token_path = tokens <= count * WINDOW
    rows = min(count, -(-tokens // WINDOW)) if token_path else count
    assert reads0 == [count]
    assert reads1 == [rows], (reads1, rows)
    print(f"L1-D07B2 identity c{count} t{tokens} n{live} bounded={bounded}: history rows {reads0[0]} -> {reads1[0]}")


def _sub_jaxprs(value):
    if hasattr(value, "eqns"):
        yield value
    elif hasattr(value, "jaxpr") and hasattr(value.jaxpr, "eqns"):
        yield value.jaxpr
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _sub_jaxprs(item)


def _gathers_on_channels(jaxpr, channels):
    """Shapes of gather operands whose last axis is the channel axis: the
    history [rows, state_len, C] and the step's x [T, C]. The stand-in pool
    read gathers [pages, state_len * C] rows and is not counted."""
    found = []

    def walk(j):
        for eqn in j.eqns:
            if eqn.primitive.name == "gather":
                shape = tuple(eqn.invars[0].aval.shape)
                if shape[-1] == channels:
                    found.append(shape)
            for value in eqn.params.values():
                for sub in _sub_jaxprs(value):
                    walk(sub)

    walk(jaxpr.jaxpr)
    return found


@pytest.mark.parametrize("tokens,live", [(16, 4), (32, 7), (64, 16)])
def test_static_windows_drop_the_channel_gathers(tokens, live):
    """bounded=2 on a whole-window bucket gathers no history / x rows at all;
    bounded=0 and 1 do (the per-token window and the per-tap rows)."""
    channels = 128
    inputs = _case(tokens + live, count=48, tokens=tokens, live=live, channels=channels)
    args = [inputs[k] for k in ("x", "weight", "pool", "starts", "ids", "initial", "read_ids", "ckpt_ids", "windowed")]
    counts = {}
    for bounded in (0, 1, 2):
        fn = functools.partial(
            pool_mod.pooled_short_conv_speculative,
            dilation=DILATION,
            state_len=STATE_LEN,
            bounded=bounded,
        )
        counts[bounded] = _gathers_on_channels(jax.make_jaxpr(fn)(*args), channels)
    print(f"L1-D07B2 channel gathers at {tokens} tokens: " + str({b: len(v) for b, v in counts.items()}))
    assert counts[0] and counts[1]
    assert all(s[0] == 48 for s in counts[0] if len(s) == 3), counts[0]
    assert all(s[0] == tokens // WINDOW for s in counts[1] if len(s) == 3), counts[1]
    assert counts[2] == [], counts[2]


def test_bounded_rows_rule():
    assert pool_mod.bounded_request_rows(16, 48, 4) == 4
    assert pool_mod.bounded_request_rows(8, 48, 4) == 2
    assert pool_mod.bounded_request_rows(18, 48, 4) == 5
    assert pool_mod.bounded_request_rows(64, 48, 4) == 16
    assert pool_mod.bounded_request_rows(256, 48, 4) == 48
