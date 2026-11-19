# SPDX-License-Identifier: Apache-2.0
"""q38 L2-META (D49): VLLM_TPU_GDN_HOIST_READ_OFFSETS is bitwise neutral.

With the flag the GDN v3 PER_SEQ metadata stops rolling the layer's own state,
read-offset and checkpoint ids by ``distribution[0]`` and points
``p_id_to_s_idx`` at the absolute request index instead. The kernel only uses
``s_idx`` to index those arrays and ``s_idx_has_initial_state``, so:

1. ``test_per_seq_metadata_kernel_view``: at every tile id (padding included)
   the values the kernel reads are identical with and without the flag, over
   decode, partial verify windows, mixed (windows + prefill + idle slots) and
   all-prefill steps, at 16 / 32 / 64-row buckets.
2. ``test_kernel_outputs_bitwise``: the fused conv1d + GDN kernel (CPU
   interpret mode, dense states, MTP window 4, non-zero read offsets) returns
   identical outputs and states with and without the flag on mixed steps whose
   PER_SEQ segment starts at a non-zero request, plus a decode-only step.
3. ``test_lowered_program``: engagement. The lowered PER_SEQ program loses the
   per-layer rolls of the layer's ids (three ``dynamic_slice`` sites, one per
   rolled id array) and keeps the batched kernel unchanged; with the flag off
   the lowering is unchanged by this patch (same text as without the kwarg).
"""

import re
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import AbstractDevice, AbstractMesh, use_abstract_mesh

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.gdn.v3 import config, metadata, wrapper

WINDOW = 4  # MTP K = 3


def _step(kind: str, num_seqs: int, rows: int, seed: int):
    """Per-step inputs for one GDN layer (its own state / checkpoint ids)."""
    rng = np.random.default_rng(seed)
    qlens = np.zeros(num_seqs, np.int32)
    if kind == "decode":
        n_win, n_pre = num_seqs - 2, 0
        qlens[:n_win] = 1
    elif kind == "verify":
        n_win, n_pre = num_seqs - 1, 0
        qlens[:n_win] = rng.integers(1, WINDOW + 1, n_win)  # partial windows
    elif kind == "mixed":
        n_win = max(1, num_seqs // 2)
        n_pre = max(1, num_seqs // 4)
        qlens[:n_win] = rng.integers(1, WINDOW + 1, n_win)
        budget = rows - int(qlens.sum())
        pre = rng.integers(1, max(2, budget // n_pre), n_pre)
        qlens[n_win : n_win + n_pre] = pre
    elif kind == "prefill":
        n_win, n_pre = 0, max(1, num_seqs // 3)
        qlens[:n_pre] = rng.integers(1, max(2, rows // n_pre), n_pre)
    else:
        raise ValueError(kind)
    assert qlens.sum() <= rows, (kind, qlens.sum(), rows)
    live = qlens > 0
    query_start_loc = np.concatenate([[0], np.cumsum(qlens)]).astype(np.int32)
    ctx = rng.integers(0, 3000, num_seqs)
    seq_lens = np.where(live, qlens + ctx, 0).astype(np.int32)
    if n_pre:
        # Some prefill chunks are fresh (no initial state).
        seq_lens[n_win] = qlens[n_win]
    blocks = rng.permutation(np.arange(1, 4096)).astype(np.int32)
    state_indices = np.where(live, blocks[:num_seqs], 0).astype(np.int32)
    ckpt = blocks[num_seqs : num_seqs + num_seqs * WINDOW].reshape(num_seqs, WINDOW)
    ckpt[:, 0] = state_indices
    ckpt = np.where(live[:, None], ckpt, 0).astype(np.int32)
    read_offsets = np.where(live, rng.integers(0, WINDOW, num_seqs), 0).astype(np.int32)
    end = n_win + n_pre
    distribution = np.array([n_win, end, end], np.int32)
    return dict(
        seq_lens=jnp.asarray(seq_lens),
        query_start_loc=jnp.asarray(query_start_loc),
        state_indices=jnp.asarray(state_indices),
        read_offsets=jnp.asarray(read_offsets),
        ckpt_indices=jnp.asarray(ckpt),
        distribution=jnp.asarray(distribution),
    )


def _cfg(batch_size: int, tile: int) -> config.GDNConfig:
    return config.GDNConfig(
        mode=config.GDNMode.PER_SEQ,
        batch_size=batch_size,
        kernel_size=4,
        tile_size=tile,
        window_size=1,
        dim_size=2 * 128 * 2 + 6 * 128,
        num_kq_heads=2,
        num_v_heads=6,
        kq_head_dim=128,
        v_head_dim=128,
        dtypes=config.Dtypes(
            act_in=jnp.dtype(jnp.float32),
            act_out=jnp.dtype(jnp.bfloat16),
            compute=jnp.dtype(jnp.float32),
            recurrent_state=jnp.dtype(jnp.float32),
            conv_state=jnp.dtype(jnp.float32),
        ),
    )


def _per_seq(s, cfg, absolute):
    return metadata.compute_per_seq_metadata(
        cfg=cfg,
        seq_lens=s["seq_lens"],
        query_start_loc=s["query_start_loc"],
        state_indices=s["state_indices"],
        read_offsets=s["read_offsets"],
        start_seq=s["distribution"][0],
        end_seq=s["distribution"][-1],
        ckpt_indices=s["ckpt_indices"],
        absolute_s_idx=absolute,
    )


def _kernel_view(md, max_tokens):
    """Every value the kernel reads per tile id, for all tile ids."""
    n = lambda x: np.asarray(x)  # noqa: E731
    s_idx = n(md.p_id_to_s_idx.data)
    assert s_idx.shape == (max_tokens,)
    assert (s_idx >= 0).all() and (s_idx < n(md.s_idx_to_state_indices).size).all()
    return dict(
        num_tiles=int(md.num_tiles),
        r_base=n(md.p_id_to_r_base.data),
        r_size=n(md.p_id_to_r_size.data),
        first=n(md.p_id_is_first_tile.data),
        last=n(md.p_id_is_last_tile.data),
        state=n(md.s_idx_to_state_indices)[s_idx],
        has_init=n(md.s_idx_has_initial_state)[s_idx],
        read_offset=n(md.s_idx_to_read_offset)[s_idx],
        ckpt=n(md.s_idx_to_ckpt_indices)[s_idx],
    )


@pytest.mark.parametrize("rows,num_seqs", [(16, 4), (32, 8), (64, 16), (64, 13)])
@pytest.mark.parametrize("kind", ["decode", "verify", "mixed", "prefill"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_per_seq_metadata_kernel_view(rows, num_seqs, kind, seed):
    s = _step(kind, num_seqs, rows, seed)
    cfg = _cfg(rows, min(64, rows))
    base = _kernel_view(_per_seq(s, cfg, False), rows)
    before = metadata.ABSOLUTE_S_IDX_TRACES
    got = _kernel_view(_per_seq(s, cfg, True), rows)
    assert metadata.ABSOLUTE_S_IDX_TRACES == before + 1
    for key in base:
        np.testing.assert_array_equal(got[key], base[key], err_msg=key)
    if kind in ("mixed", "prefill"):
        assert base["num_tiles"] > 0
        # The roll matters: the PER_SEQ segment starts at a non-zero request.
        if kind == "mixed":
            assert int(s["distribution"][0]) > 0


def _run(inputs, flag):
    mesh = AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
    # fused_conv1d_gdn is jitted and reads the flag at trace time: never reuse
    # the other arm's trace.
    jax.clear_caches()
    pltpu.reset_tpu_interpret_mode_state()
    try:
        with (
            use_abstract_mesh(mesh),
            mock.patch.object(envs, "TPU_ENABLE_GDN_DYNAMIC_TILING", False),
            mock.patch.object(envs, "VLLM_TPU_GDN_HOIST_READ_OFFSETS", flag),
            pltpu.force_tpu_interpret_mode(
                pltpu.InterpretParams(
                    dma_execution_mode="eager", skip_floating_point_ops=False
                )
            ),
        ):
            states = {k: jnp.copy(inputs[k]) for k in ("conv_state", "recurrent_state")}
            result = wrapper.fused_conv1d_gdn(
                decode_tile_size=2,
                # CPU interpret mode needs C = 1 tiles on the prefill path.
                mixed_tile_size=1,
                **{**inputs, **states},
            )
            return jax.block_until_ready(result)
    finally:
        pltpu.reset_tpu_interpret_mode_state()


def _kernel_inputs(lengths, num_win, seed, ctx):
    n_kq, n_v, d_k, d_v, kernel_size = 1, 2, 128, 128, 4
    dim = 2 * n_kq * d_k + n_v * d_v
    num_seqs = len(lengths)
    rows = sum(lengths)
    rng = np.random.default_rng(seed)
    rngs = iter(jax.random.split(jax.random.key(seed), 12))
    # Slot groups of WINDOW consecutive slots per request, shuffled; slot 0 null.
    groups = rng.permutation(num_seqs + 2)[:num_seqs]
    state_indices = jnp.asarray(1 + groups * WINDOW, jnp.int32)
    num_blocks = 1 + (num_seqs + 2) * WINDOW
    read_offsets = np.zeros(num_seqs, np.int32)
    read_offsets[:num_win] = rng.integers(0, WINDOW, num_win)
    # A prefill row may also resume from a non-zero checkpoint (prefix resume).
    read_offsets[num_win:] = rng.integers(0, WINDOW, num_seqs - num_win)
    q_loc = jnp.asarray(np.concatenate([[0], np.cumsum(lengths)]), jnp.int32)
    seq_lens = np.asarray(lengths, np.int32) + np.asarray(ctx, np.int32)
    return dict(
        qkv=jax.random.normal(next(rngs), (rows, dim), dtype=jnp.bfloat16),
        b=jax.random.normal(next(rngs), (rows, n_v), dtype=jnp.bfloat16),
        a=jax.random.normal(next(rngs), (rows, n_v), dtype=jnp.bfloat16),
        conv_weight=jax.random.normal(next(rngs), (dim, 1, kernel_size), jnp.bfloat16),
        conv_bias=jax.random.normal(next(rngs), (dim,), dtype=jnp.bfloat16),
        a_log=jax.random.normal(next(rngs), (n_v,), dtype=jnp.float32),
        dt_bias=jax.random.normal(next(rngs), (n_v,), dtype=jnp.float32),
        conv_state=jax.random.normal(
            next(rngs), (num_blocks, kernel_size - 1, dim), dtype=jnp.bfloat16
        ),
        recurrent_state=jax.random.normal(
            next(rngs), (num_blocks, n_v, d_k, d_v), dtype=jnp.float32
        ),
        query_start_loc=q_loc,
        state_indices=state_indices,
        distribution=jnp.asarray([num_win, num_seqs, num_seqs], jnp.int32),
        seq_lens=jnp.asarray(seq_lens),
        read_offsets=jnp.asarray(read_offsets),
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        kernel_size=kernel_size,
        num_spec_tokens=WINDOW - 1,
    )


KERNEL_CASES = {
    # verify windows (full, partial, 1-token) then prefill chunks, one fresh.
    "mixed_a": dict(lengths=[4, 2, 1, 3, 5, 3], num_win=4, ctx=[9, 7, 5, 8, 0, 6]),
    "mixed_b": dict(lengths=[1, 4, 6, 2], num_win=2, ctx=[3, 11, 4, 0]),
    # decode-only verify step: the PER_SEQ segment is empty.
    "verify_only": dict(lengths=[4, 1, 3, 2], num_win=4, ctx=[5, 6, 7, 8]),
}


@pytest.mark.parametrize("case", sorted(KERNEL_CASES))
def test_kernel_outputs_bitwise(case):
    inputs = _kernel_inputs(seed=7, **KERNEL_CASES[case])
    expected = _run(inputs, False)
    before = metadata.ABSOLUTE_S_IDX_TRACES
    got = _run(inputs, True)
    assert metadata.ABSOLUTE_S_IDX_TRACES > before
    (e_conv, e_rec), e_out = expected
    (g_conv, g_rec), g_out = got
    for name, e, g in (
        ("output", e_out, g_out),
        ("conv_state", e_conv, g_conv),
        ("recurrent_state", e_rec, g_rec),
    ):
        np.testing.assert_array_equal(np.asarray(g), np.asarray(e), err_msg=name)
    if case != "verify_only":
        # Positive control: the PER_SEQ rows produced output and moved state.
        num_win = KERNEL_CASES[case]["num_win"]
        start = int(np.cumsum([0] + KERNEL_CASES[case]["lengths"])[num_win])
        assert np.abs(np.asarray(e_out[start:], np.float32)).sum() > 0
        assert not np.array_equal(np.asarray(e_rec), np.asarray(inputs["recurrent_state"]))


def _lowered_per_seq_text(flag, num_seqs=16, rows=64):
    s = _step("mixed", num_seqs, rows, 0)
    cfg = _cfg(rows, 64)

    def f(seq_lens, qsl, si, ro, ck, dist):
        md = metadata.compute_per_seq_metadata(
            cfg=cfg,
            seq_lens=seq_lens,
            query_start_loc=qsl,
            state_indices=si,
            read_offsets=ro,
            start_seq=dist[0],
            end_seq=dist[-1],
            ckpt_indices=ck,
            absolute_s_idx=flag,
        )
        return jax.tree.leaves(md)

    args = (
        s["seq_lens"],
        s["query_start_loc"],
        s["state_indices"],
        s["read_offsets"],
        s["ckpt_indices"],
        s["distribution"],
    )
    return jax.jit(f).lower(*args).as_text()


def _roll_sites(text):
    # jnp.roll by a traced shift lowers to a call of a private `_roll_dynamic*`
    # function (concatenate + dynamic_slice); count the call sites.
    return len(re.findall(r"call @_roll_dynamic\w*\(", text))


def test_lowered_program():
    off = _lowered_per_seq_text(False)
    on = _lowered_per_seq_text(True)
    n_off, n_on = _roll_sites(off), _roll_sites(on)
    print("Q38_L2META_D49_LOWERED roll sites off=%d on=%d" % (n_off, n_on))
    # Off: query_start_loc, seq_lens, state_indices, read_offsets and
    # ckpt_indices are each rolled. On: the two layer-invariant rolls plus the
    # roll back of has_initial_state (layer-invariant too).
    assert (n_off, n_on) == (5, 3), (n_off, n_on)
