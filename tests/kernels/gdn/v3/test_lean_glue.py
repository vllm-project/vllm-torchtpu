# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_GDN_LEAN_GLUE: the GDN v3 op gives the same bytes with less glue.

Each piece has its own check, so a failure names it.
- zero_fill (CPU, Pallas interpreter, dense states): the BATCHED kernel writes
  zeros to the rows past its sequences instead of reading an XLA zero buffer.
  Unwritten memory reads as NaN in the interpreter, so a row the kernel misses
  shows up. Layouts: decode only, decode then PER_SEQ, both with bucket-padding
  rows past the last sequence, and a bucket with no padding.
- ba_packed (CPU, interpreter): b and a reach the kernels as one array.
- slot_rows (CPU, interpreter): the kernels read their layer's row of stacked
  state indices and read offsets; the other rows hold other layers' slots, so
  a wrong row reads another sequence's state.
- No checkpoint-index operand in LOG mode: the kernels address checkpoint 0
  only, so the flag relies on `ckpt_indices[:, 0] == state_indices`. Checked
  here on the producer of both, the staged metadata walk, in both of its
  gather forms, with idle requests and rows too short for the window. The
  kernel-level byte check of LOG mode runs on TPU
  (test_lean_glue_log_is_byte_identical_tpu): the interpreter rejects the
  LOG kernel's typed loads.
"""

import sys
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import AbstractDevice, AbstractMesh, use_abstract_mesh

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.gdn.v3 import metadata, wrapper

N_KQ, N_V, D, KS = 1, 2, 128, 4
DIM = 2 * N_KQ * D + N_V * D


def _run_dense(lean, lengths, num_decode, rows, slot_rows_row=None):
    """fused_conv1d_gdn on dense states in the interpreter; lengths[i] tokens
    for sequence i (the first num_decode one token each), `rows` >= their sum."""
    num_seqs = len(lengths)
    rngs = iter(jax.random.split(jax.random.key(7), 9))
    qkv = jax.random.normal(next(rngs), (rows, DIM), dtype=jnp.bfloat16)
    b = jax.random.normal(next(rngs), (rows, N_V), dtype=jnp.bfloat16)
    a = jax.random.normal(next(rngs), (rows, N_V), dtype=jnp.bfloat16)
    conv_w = jax.random.normal(next(rngs), (DIM, 1, KS), dtype=jnp.bfloat16)
    conv_b = jax.random.normal(next(rngs), (DIM,), dtype=jnp.bfloat16)
    a_log = jax.random.normal(next(rngs), (N_V,), dtype=jnp.float32)
    dt_bias = jax.random.normal(next(rngs), (N_V,), dtype=jnp.float32)
    init_conv = jax.random.normal(next(rngs), (num_seqs, KS - 1, DIM), dtype=jnp.bfloat16)
    init_rec = jax.random.normal(next(rngs), (num_seqs, N_V, D, D), dtype=jnp.float32)
    q_loc = jnp.asarray(np.concatenate([[0], np.cumsum(lengths)]), jnp.int32)
    state_indices = jnp.arange(num_seqs, dtype=jnp.int32)
    slot_rows = None
    if slot_rows_row is not None:
        # Three layers; the other rows name other sequences' state slots.
        layers = 3
        stack = np.stack([np.roll(np.arange(num_seqs), k + 1) for k in range(layers)])
        stack[slot_rows_row] = np.arange(num_seqs)
        slot_rows = metadata.GdnSlotRows(
            jnp.asarray(stack, jnp.int32),
            jnp.zeros((layers, num_seqs), jnp.int32),
            jnp.asarray([slot_rows_row], jnp.int32),
        )
        # The per-layer arrays give only the shape: poison them.
        state_indices = jnp.full((num_seqs,), num_seqs - 1, jnp.int32)
    mesh = AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
    pltpu.reset_tpu_interpret_mode_state()
    with (
        use_abstract_mesh(mesh),
        mock.patch.object(envs, "TPU_ENABLE_GDN_DYNAMIC_TILING", False),
        pltpu.force_tpu_interpret_mode(
            pltpu.InterpretParams(
                dma_execution_mode="eager",
                skip_floating_point_ops=False,
                uninitialized_memory="nan",
            )
        ),
    ):
        (conv_out, rec_out), out = wrapper.fused_conv1d_gdn(
            qkv=qkv,
            b=b,
            a=a,
            conv_state=init_conv,
            recurrent_state=init_rec,
            conv_weight=conv_w,
            conv_bias=conv_b,
            a_log=a_log,
            dt_bias=dt_bias,
            query_start_loc=q_loc,
            state_indices=state_indices,
            distribution=jnp.asarray([num_decode, num_seqs, num_seqs], jnp.int32),
            seq_lens=jnp.full((num_seqs,), 11, dtype=jnp.int32),
            n_kq=N_KQ,
            n_v=N_V,
            d_k=D,
            d_v=D,
            kernel_size=KS,
            decode_tile_size=2,
            # The interpreter has no ref.reshape for the C > 1 chunked path.
            mixed_tile_size=1,
            lean_glue=lean,
            slot_rows=slot_rows,
        )
        jax.block_until_ready((conv_out, rec_out, out))
    return np.asarray(conv_out), np.asarray(rec_out), np.asarray(out)


@pytest.mark.parametrize(
    "lengths,num_decode,rows",
    [
        ([1] * 6, 6, 9),  # decode only, 3 padding rows
        ([1] * 4 + [1, 1], 4, 10),  # decode, then PER_SEQ, 4 padding rows
        ([1] * 6, 6, 6),  # no padding row
    ],
    ids=["decode_pad", "decode_perseq_pad", "no_pad"],
)
def test_zero_fill_is_byte_identical(lengths, num_decode, rows):
    off = _run_dense(False, lengths, num_decode, rows)
    on = _run_dense(("zero_fill",), lengths, num_decode, rows)
    for name, x, y in zip(("conv", "recurrent", "out"), off, on):
        assert np.array_equal(x.view(np.uint8), y.view(np.uint8)), name
    out = on[2]
    assert np.all(out[sum(lengths) :] == 0)
    assert np.isfinite(out.astype(np.float32)).all()


_LAYOUTS = [([1] * 6, 6, 9), ([1] * 4 + [1, 1], 4, 10)]


def _assert_same(off, on):
    for name, x, y in zip(("conv", "recurrent", "out"), off, on):
        assert np.array_equal(x.view(np.uint8), y.view(np.uint8)), name


@pytest.mark.parametrize("lengths,num_decode,rows", _LAYOUTS, ids=["decode", "decode_perseq"])
def test_ba_packed_is_byte_identical(lengths, num_decode, rows):
    _assert_same(
        _run_dense(False, lengths, num_decode, rows),
        _run_dense(("ba_packed",), lengths, num_decode, rows),
    )


@pytest.mark.parametrize("row", [0, 2])
@pytest.mark.parametrize("lengths,num_decode,rows", _LAYOUTS, ids=["decode", "decode_perseq"])
def test_slot_rows_are_byte_identical(lengths, num_decode, rows, row):
    _assert_same(
        _run_dense(False, lengths, num_decode, rows),
        _run_dense(False, lengths, num_decode, rows, slot_rows_row=row),
    )


def _staged_walk():
    """The staged metadata walk's pure function (attention_metadata imports
    torch and vLLM; skip where they are absent)."""
    try:
        from vllm_torchtpu.layers.core.attention_metadata import _staged_walk_products
    except ImportError as e:
        pytest.skip(f"attention_metadata needs torch and vLLM: {e}")
    return _staged_walk_products


@pytest.mark.parametrize("class_gather", [False, True])
def test_checkpoint_zero_is_the_state_block(class_gather):
    torch = pytest.importorskip("torch")
    walk = _staged_walk()
    g = torch.Generator().manual_seed(3)
    window, reqs = 4, 12
    # Two mamba groups of one geometry and one of another (class gather
    # batches the first two); 7 and 3 blocks a row, the second too short for
    # some requests' windows.
    geoms = [(7, 64), (7, 64), (3, 128)]
    table_geom, offset, parts = [], 0, []
    for t, _ in geoms:
        tbl = torch.randint(1, 10_000, (reqs, t), generator=g, dtype=torch.int64)
        parts.append(tbl.reshape(-1))
        table_geom.append((offset, reqs, t))
        offset += reqs * t
    staged = torch.cat(parts)
    mamba_geom = tuple((i, t, bs) for i, (t, bs) in enumerate(geoms))
    seq_lens = torch.randint(0, 600, (reqs,), generator=g, dtype=torch.int64)
    seq_lens[[0, 5]] = 0  # idle requests
    outs = walk(staged, seq_lens, tuple(table_geom), mamba_geom, window, False, class_gather)
    n = len(geoms)
    states, ckpts = outs[:n], outs[n:]
    assert len(ckpts) == n
    for s, c in zip(states, ckpts):
        assert c.shape == (reqs, window)
        assert torch.equal(c[:, 0], s)
        assert torch.all(s[seq_lens == 0] == 0)


_TPU_CASES = [("no_ckpt",), ("zero_fill",), ("ba_packed",), "slot_rows", "all"]


@pytest.mark.skipif("jax.default_backend() != 'tpu'")
@pytest.mark.parametrize("piece", _TPU_CASES, ids=["no_ckpt", "zero_fill", "ba_packed", "slot_rows", "all"])
def test_lean_glue_log_is_byte_identical_tpu(piece):
    """LOG mode with the factored window at the served Flash decode geometry:
    outputs and every pool byte with one piece (or all) equal those without,
    over a verify step with full and short windows, one-token rows and bucket
    padding, after a fresh step that writes the states."""
    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    from test_window_log import K, LOG_PLAN, _f8

    from vllm_torchtpu.kernels.gdn.v3.input_layout import Conv1DInput

    n_kq, n_v, d, dim, ks, w = 2, 6, 128, 1280, 4, K + 1
    num_seqs, tokens, layers, row = 32, 128, 3, 1
    rng = np.random.default_rng(11)
    num_blocks = 1 + w * num_seqs
    pool = _f8(jnp.asarray(rng.integers(0, 256, (num_blocks, 512, 1, 4, 256), dtype=np.uint8)))
    conv_w = jnp.asarray(rng.normal(0, 0.5, (ks, 1, dim)), jnp.float32)
    a_log = jnp.asarray(np.log(rng.uniform(1, 16, n_v)), jnp.float32)
    dt_bias = jnp.asarray(rng.normal(0, 1, n_v), jnp.bfloat16)
    si = (1 + w * np.arange(num_seqs)).astype(np.int32)
    ck = (si[:, None] + np.arange(w)[None, :]).astype(np.int32)

    def call(lean, slot_rows, pool, lens, base, ro, seed):
        n = len(lens)
        qsl = np.zeros(num_seqs + 1, np.int32)
        qsl[1 : n + 1] = np.cumsum(lens)
        qsl[n + 1 :] = qsl[n]
        sl = np.full(num_seqs, base, np.int32)
        sl[:n] += np.asarray(lens, np.int32)
        ro = np.where(base > 0, ro, 0).astype(np.int32)
        rows = None
        if slot_rows:
            # Other rows: other slots and offsets; only row `row` is this layer's.
            stack_si = np.stack([np.roll(si, k) for k in range(layers)])
            stack_ro = np.stack([np.roll(ro, k + 3) for k in range(layers)])
            stack_si[row], stack_ro[row] = si, ro
            rows = metadata.GdnSlotRows(
                jnp.asarray(stack_si), jnp.asarray(stack_ro), jnp.asarray([row], jnp.int32)
            )
        r = np.random.default_rng(seed)
        qkv = jnp.asarray(r.normal(size=(tokens, 1, dim)), jnp.float32)
        b = jnp.asarray(r.normal(size=(tokens, n_v)), jnp.float32)
        a = jnp.asarray(r.normal(size=(tokens, n_v)), jnp.float32)
        fn = jax.jit(
            lambda rows, *x: wrapper.fused_conv1d_gdn(
                Conv1DInput(x[0], jnp.bfloat16), *x[1:3], None, None, x[4], None, x[5], x[6],
                *x[7:],
                n_kq=n_kq, n_v=n_v, d_k=d, d_v=d, kernel_size=ks,
                state_source=x[3], state_plan=LOG_PLAN, num_spec_tokens=K,
                conv_weight_kmajor=True, factored_window=True, lean_glue=lean,
                slot_rows=rows,
            ),
            donate_argnums=(4,),
        )
        return fn(rows, qkv, b, a, pool, conv_w, a_log, dt_bias, jnp.asarray(qsl), jnp.asarray(si),
                  jnp.asarray([n, n, n], jnp.int32), jnp.asarray(sl), jnp.asarray(ro), jnp.asarray(ck))

    lean = True if piece == "all" else (False if piece == "slot_rows" else piece)
    with_rows = piece in ("slot_rows", "all")
    copy = lambda p: jnp.array(p, copy=True)  # noqa: E731
    pool, _ = call(False, False, pool, [w] * num_seqs, 0, np.zeros(num_seqs, np.int32), 1)
    lens = [4, 1, 3, 4, 2, 1, 4, 4] * 3 + [1, 4]  # 26 sequences, 72 tokens, 56 padding rows
    ro = rng.integers(0, w, num_seqs).astype(np.int32)
    pool_off, out_off = call(False, False, copy(pool), lens, w, ro, 2)
    pool_on, out_on = call(lean, with_rows, copy(pool), lens, w, ro, 2)
    assert np.array_equal(np.asarray(out_off).view(np.uint8), np.asarray(out_on).view(np.uint8))
    assert np.array_equal(np.asarray(pool_off).view(np.uint8), np.asarray(pool_on).view(np.uint8))
    assert np.all(np.asarray(out_on, np.float32)[sum(lens) :] == 0)


def test_stack_rows_follow_the_forward_layers():
    """Every layer of a forward derives the same row order from the same
    metadata dict; layers registered elsewhere (another model) are left out."""
    try:
        from vllm_torchtpu.layers.adapter.custom_ops import gdn_attention_op as op
    except ImportError as e:
        pytest.skip(f"the GDN adapter needs torch and vLLM: {e}")
    saved = set(op._POOLED_GDN_PREFIXES)
    try:
        op._POOLED_GDN_PREFIXES.clear()
        op._POOLED_GDN_PREFIXES.update(
            {"model.layers.2.linear_attn", "model.layers.0.linear_attn", "draft.layers.0.linear_attn"}
        )
        md = {
            "model.layers.0.linear_attn": 0,
            "model.layers.1.self_attn": 0,
            "model.layers.2.linear_attn": 0,
        }
        assert op.stacked_gdn_prefixes(md) == [
            "model.layers.0.linear_attn",
            "model.layers.2.linear_attn",
        ]
    finally:
        op._POOLED_GDN_PREFIXES.clear()
        op._POOLED_GDN_PREFIXES.update(saved)
