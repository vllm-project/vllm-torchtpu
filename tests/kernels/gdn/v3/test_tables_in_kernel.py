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
"""In-kernel routing tables must equal the XLA tables of `metadata.py`.

Runs `tables_in_kernel.fill_and_wrap` in Pallas interpret mode and compares
every entry the pipeline reads: each visited tile's row range and flags, and
its sequence's initial-state flag, state slot, read offset and checkpoint
blocks.
"""

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.gdn.v3 import config, metadata, tables_in_kernel

pytestmark = pytest.mark.cpu_test

# Rows per visited tile: s_idx, r_base, r_size, first, last, has_init,
# state index, read offset, then one per checkpoint block.
_NUM_FIELDS = 8


def _cfg(mode, tile_size, batch_size, window_size=1):
    return config.GDNConfig(
        mode=mode,
        dtypes=config.Dtypes(
            act_in=jnp.float32.dtype,
            act_out=jnp.bfloat16.dtype,
            compute=jnp.float32.dtype,
            recurrent_state=jnp.float32.dtype,
            conv_state=jnp.float32.dtype,
        ),
        batch_size=batch_size,
        dim_size=384,
        kernel_size=4,
        tile_size=tile_size,
        num_kq_heads=1,
        num_v_heads=1,
        kq_head_dim=128,
        v_head_dim=128,
        window_size=window_size,
    )


def _inputs(decode_lens, prefill_lens, context_lens, max_seqs, read_offsets, num_ckpts):
    lens = np.asarray(list(decode_lens) + list(prefill_lens), dtype=np.int32)
    n = lens.size
    query_start_loc = np.zeros((max_seqs + 1,), dtype=np.int32)
    query_start_loc[1 : n + 1] = np.cumsum(lens)
    query_start_loc[n + 1 :] = query_start_loc[n]
    seq_lens = np.zeros((max_seqs,), dtype=np.int32)
    seq_lens[:n] = lens + np.asarray(context_lens, dtype=np.int32)
    # Distinct, non-identity slots so a wrong sequence index shows up.
    state_indices = (np.arange(max_seqs, dtype=np.int32) * 7 + 3) % 1000
    read = np.zeros((max_seqs,), dtype=np.int32)
    read[:n] = read_offsets
    distribution = np.array([len(decode_lens), n, n], dtype=np.int32)
    ckpt_indices = None
    if num_ckpts:
        # Distinct blocks per (sequence, checkpoint), not affine in either.
        ckpt_indices = (
            np.arange(max_seqs * num_ckpts, dtype=np.int32).reshape(max_seqs, -1) * 13
            + 5
        ) % 997
    return query_start_loc, seq_lens, state_indices, read, distribution, ckpt_indices


def _tables_kernel(
    qsl_ref,
    seq_lens_ref,
    state_ref,
    read_ref,
    dist_ref,
    ckpt_ref,
    out_b_ref,
    out_p_ref,
    num_tiles_ref,
    *scratch,
    cfg_b,
    cfg_p,
    max_seqs,
    names,
):
    sc = dict(zip(names, scratch))
    meta_b, meta_p = tables_in_kernel.fill_and_wrap(
        qsl_ref,
        seq_lens_ref,
        state_ref,
        read_ref,
        dist_ref,
        ckpt_ref,
        sc,
        cfg_b,
        cfg_p,
        max_seqs,
    )
    num_ckpts = 0 if ckpt_ref is None else ckpt_ref.shape[1]

    def dump(meta, out_ref, size, seq_tile_size):
        def body(i, carry):
            p_id, idx = i // seq_tile_size, i % seq_tile_size
            s = meta.p_id_to_s_idx[p_id, idx]
            fields = (
                s,
                meta.p_id_to_r_base[p_id, idx],
                meta.p_id_to_r_size[p_id, idx],
                meta.p_id_is_first_tile[p_id, idx].astype(jnp.int32),
                meta.p_id_is_last_tile[p_id, idx].astype(jnp.int32),
                meta.s_idx_has_initial_state[s].astype(jnp.int32),
                meta.s_idx_to_state_indices[s],
                meta.s_idx_to_read_offset[s],
                *(meta.s_idx_to_ckpt_indices[s, t] for t in range(num_ckpts)),
            )
            for k, value in enumerate(fields):
                out_ref[k, i] = value
            return carry

        lax.fori_loop(0, size, body, 0)

    num_tiles_ref[0] = meta_b.num_tiles[...]
    num_tiles_ref[1] = meta_p.num_tiles[...]
    dump(
        meta_b,
        out_b_ref,
        meta_b.num_tiles[...] * cfg_b.seq_tile_size,
        cfg_b.seq_tile_size,
    )
    dump(meta_p, out_p_ref, meta_p.num_tiles[...], 1)


def _in_kernel_tables(cfg_b, cfg_p, max_seqs, raw):
    ckpt_indices = raw[-1]
    num_ckpts = 0 if ckpt_indices is None else ckpt_indices.shape[1]
    num_fields = _NUM_FIELDS + num_ckpts
    scratch = tables_in_kernel.scratch_shapes(
        max_seqs, cfg_p.batch_size, cfg_b.tile_size, num_ckpts
    )
    names = tuple(scratch)
    num_b = tables_in_kernel.batched_table_size(max_seqs, cfg_b.tile_size)
    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    # Interpret mode is the CPU reference; on a TPU backend it reads some
    # SMEM entries back wrong, so pin it to the host.
    cpu = jax.devices("cpu")[0]
    kernel = pl.pallas_call(
        functools.partial(
            _tables_kernel, cfg_b=cfg_b, cfg_p=cfg_p, max_seqs=max_seqs, names=names
        ),
        out_shape=(
            jax.ShapeDtypeStruct((num_fields, num_b), jnp.int32),
            jax.ShapeDtypeStruct((num_fields, cfg_p.batch_size), jnp.int32),
            jax.ShapeDtypeStruct((2,), jnp.int32),
        ),
        in_specs=[None if x is None else smem for x in raw],
        out_specs=(smem, smem, smem),
        scratch_shapes=[scratch[n] for n in names],
        interpret=True,
    )
    with jax.default_device(cpu):
        out_b, out_p, num_tiles = kernel(
            *(None if x is None else jnp.asarray(x) for x in raw)
        )
    return np.asarray(out_b), np.asarray(out_p), np.asarray(num_tiles)


def _xla_tables(meta, size, seq_tile_size):
    def field(wrapper):
        return np.asarray(wrapper.data).astype(np.int32)

    s_idx = field(meta.p_id_to_s_idx)[:size]
    rows = [
        s_idx,
        field(meta.p_id_to_r_base)[:size],
        field(meta.p_id_to_r_size)[:size],
        field(meta.p_id_is_first_tile)[:size],
        field(meta.p_id_is_last_tile)[:size],
        np.asarray(meta.s_idx_has_initial_state).astype(np.int32)[s_idx],
        np.asarray(meta.s_idx_to_state_indices)[s_idx],
        np.asarray(meta.s_idx_to_read_offset)[s_idx],
    ]
    if meta.s_idx_to_ckpt_indices is not None:
        rows += list(np.asarray(meta.s_idx_to_ckpt_indices)[s_idx].T)
    return np.stack(rows)


def _check(
    decode_lens,
    prefill_lens,
    context_lens,
    read_offsets,
    *,
    max_seqs,
    batch_size,
    decode_tile_size=4,
    mixed_tile_size=64,
    num_spec_tokens=0,
):
    raw = _inputs(
        decode_lens,
        prefill_lens,
        context_lens,
        max_seqs,
        read_offsets,
        num_spec_tokens + 1 if num_spec_tokens else 0,
    )
    query_start_loc, seq_lens, state_indices, read, distribution, ckpt = raw
    window_size = num_spec_tokens + 1
    cfg_b = _cfg(config.GDNMode.BATCHED, decode_tile_size, batch_size, window_size)
    cfg_p = _cfg(config.GDNMode.PER_SEQ, mixed_tile_size, batch_size)
    out_b, out_p, num_tiles = _in_kernel_tables(cfg_b, cfg_p, max_seqs, raw)

    meta_b = metadata.compute_batched_seq_metadata(
        cfg=cfg_b,
        seq_lens=jnp.asarray(seq_lens),
        query_start_loc=jnp.asarray(query_start_loc),
        state_indices=jnp.asarray(state_indices),
        read_offsets=jnp.asarray(read),
        end_seq=jnp.asarray(distribution[0]),
        ckpt_indices=None if ckpt is None else jnp.asarray(ckpt),
    )
    meta_p = metadata.compute_per_seq_metadata(
        cfg=cfg_p,
        seq_lens=jnp.asarray(seq_lens),
        query_start_loc=jnp.asarray(query_start_loc),
        state_indices=jnp.asarray(state_indices),
        read_offsets=jnp.asarray(read),
        start_seq=jnp.asarray(distribution[0]),
        end_seq=jnp.asarray(distribution[2]),
        ckpt_indices=None if ckpt is None else jnp.asarray(ckpt),
    )
    num_tiles_b, num_tiles_p = int(meta_b.num_tiles), int(meta_p.num_tiles)
    assert tuple(num_tiles) == (num_tiles_b, num_tiles_p)

    # Batched entries past `max_seqs` exist only in the in-kernel table (the
    # XLA one stops at `max_seqs`); they must be inert.
    size_b = min(num_tiles_b * decode_tile_size, max_seqs)
    np.testing.assert_array_equal(
        out_b[:, :size_b], _xla_tables(meta_b, size_b, decode_tile_size)
    )
    tail = out_b[:, size_b : num_tiles_b * decode_tile_size]
    assert not tail[2].any() and not tail[3].any() and not tail[4].any()
    np.testing.assert_array_equal(
        out_p[:, :num_tiles_p], _xla_tables(meta_p, num_tiles_p, 1)
    )


def test_decode_only():
    _check([1] * 6, [], [5, 0, 9, 3, 1, 2], [0] * 6, max_seqs=8, batch_size=8)


def test_mixed_decode_and_prefill():
    _check(
        [1, 1, 1],
        [130, 7, 64],
        [4, 9, 2, 0, 33, 64],
        [0] * 6,
        max_seqs=8,
        batch_size=208,
    )


def test_prefill_resumes_from_read_offset():
    # A prefix-cache resume may read a non-zero checkpoint in the PER_SEQ pass;
    # the leading decodes shift its sequences by `distribution[0]`.
    _check([1, 1], [40, 3], [5, 6, 16, 4], [0, 0, 2, 1], max_seqs=4, batch_size=48)


def test_padded_rows_and_sequences():
    # Two live decodes, the rest padding (zero-length sequences, extra rows).
    _check([1, 1], [], [3, 8], [0, 0], max_seqs=8, batch_size=16)


def test_batched_tail_past_max_seqs():
    # max_seqs not a multiple of the decode tile: the last tile reads past
    # the sequences.
    _check([1] * 6, [], [2] * 6, [0] * 6, max_seqs=6, batch_size=8)


def test_full_rows():
    n = tables_in_kernel.MAX_ROWS
    _check([1] * n, [], [1] * n, [0] * n, max_seqs=n, batch_size=n)


@pytest.mark.parametrize("num_spec_tokens", [1, 3])
def test_spec_windows_with_partial_acceptance(num_spec_tokens):
    # Verify windows of varying length (<= k + 1) resuming from the checkpoint
    # of the last accepted token.
    window = num_spec_tokens + 1
    lens = [window, 1, window, max(1, window - 1), window]
    reads = [0, num_spec_tokens, 1 % window, num_spec_tokens, 0]
    _check(
        lens,
        [],
        [9, 4, 12, 7, 30],
        reads,
        max_seqs=8,
        batch_size=sum(lens),
        num_spec_tokens=num_spec_tokens,
    )


@pytest.mark.parametrize("num_spec_tokens", [1, 3])
def test_spec_windows_with_prefill(num_spec_tokens):
    # The PER_SEQ pass reads checkpoint rows shifted by `distribution[0]`,
    # including a prefix-cache resume from a non-zero checkpoint.
    window = num_spec_tokens + 1
    _check(
        [window, 1, window],
        [70, 5],
        [8, 3, 11, 30, 0],
        [num_spec_tokens, 0, 1, num_spec_tokens, 0],
        max_seqs=6,
        batch_size=2 * window + 1 + 75,
        num_spec_tokens=num_spec_tokens,
    )


def test_spec_full_rows():
    window = 4
    n = tables_in_kernel.MAX_ROWS // window
    _check(
        [window] * n,
        [],
        [window] * n,
        [i % window for i in range(n)],
        max_seqs=n,
        batch_size=n * window,
        num_spec_tokens=window - 1,
    )


# Wrapper dispatch: which Pallas calls `fused_conv1d_gdn` emits per form.
# Traced on CPU against v7x's static chip info; nothing is executed.
_N_KQ, _N_V, _D_K, _D_V, _KERNEL_SIZE = 2, 8, 128, 128, 4
_DIM = _N_KQ * _D_K * 2 + _N_V * _D_V
_POOL_SPLIT = 3


def _pallas_eqns(closed_jaxpr):
    eqns = []

    def walk(jaxpr):
        for eqn in jaxpr.eqns:
            if eqn.primitive.name == "pallas_call":
                eqns.append(eqn)
            for param in eqn.params.values():
                inner = getattr(param, "jaxpr", None)
                if inner is not None:
                    walk(getattr(inner, "jaxpr", inner))

    walk(closed_jaxpr.jaxpr)
    return eqns


@pytest.fixture
def _v7x_info(monkeypatch):
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)


def _pallas_call_names(rows, gdn_form, num_spec_tokens=0, **kwargs):
    from vllm_torchtpu.kernels import pool_adapters
    from vllm_torchtpu.kernels.gdn.v3 import wrapper

    tok_bytes = 2 * 256 * 2
    ssm_ntok = _N_V * _D_K * _D_V * 4 // tok_bytes
    pool = jax.ShapeDtypeStruct((5 * _POOL_SPLIT, 256, 1, 2, 256), jnp.bfloat16)
    plan = pool_adapters.v3_state_source(
        pool,
        split=_POOL_SPLIT,
        ssm_ntok=ssm_ntok,
        conv_tok0=ssm_ntok,
        conv_ntok=16,
        conv_dim=_DIM,
        n_v=_N_V,
        d_k=_D_K,
        d_v=_D_V,
        kernel_size=_KERNEL_SIZE,
    )
    n = 4
    ckpt = jnp.zeros((n, num_spec_tokens + 1), jnp.int32) if num_spec_tokens else None

    def run(source):
        return wrapper.fused_conv1d_gdn(
            jnp.zeros((rows, _DIM)),
            jnp.zeros((rows, _N_V)),
            jnp.zeros((rows, _N_V)),
            None,
            None,
            jnp.zeros((_DIM, 1, _KERNEL_SIZE)),
            None,
            jnp.zeros((_N_V,)),
            jnp.zeros((_N_V,)),
            jnp.zeros((n + 1,), jnp.int32),
            jnp.zeros((n,), jnp.int32),
            jnp.zeros((3,), jnp.int32),
            jnp.zeros((n,), jnp.int32),
            jnp.zeros((n,), jnp.int32),
            ckpt,
            n_kq=_N_KQ,
            n_v=_N_V,
            d_k=_D_K,
            d_v=_D_V,
            kernel_size=_KERNEL_SIZE,
            state_source=source,
            state_plan=plan,
            num_spec_tokens=num_spec_tokens,
            gdn_form=gdn_form,
            **kwargs,
        )

    return [eqn.params["name"] for eqn in _pallas_eqns(jax.make_jaxpr(run)(pool))]


_TWO_CALLS = ["fused_conv1d_gdn_batched", "fused_conv1d_gdn_per_seq"]


@pytest.mark.usefixtures("_v7x_info")
class TestWrapperForm:
    def test_two_calls_is_the_default_form(self):
        assert _pallas_call_names(8, "two_calls") == _TWO_CALLS

    @pytest.mark.parametrize("rows", [8, tables_in_kernel.MAX_ROWS])
    def test_tables_form_is_one_call(self, rows):
        assert _pallas_call_names(rows, "tables") == [
            "fused_conv1d_gdn_batched_tables_in_kernel"
        ]

    def test_tables_form_falls_back_above_max_rows(self):
        rows = tables_in_kernel.MAX_ROWS + 1
        assert _pallas_call_names(rows, "tables") == _TWO_CALLS

    def test_tables_form_serves_speculation(self):
        assert _pallas_call_names(8, "tables", num_spec_tokens=3) == [
            "fused_conv1d_gdn_batched_w4_tables_in_kernel"
        ]

    def test_tables_form_under_speculation_falls_back_above_max_rows(self):
        rows = tables_in_kernel.MAX_ROWS + 1
        assert _pallas_call_names(rows, "tables", num_spec_tokens=3) == [
            "fused_conv1d_gdn_batched_w4",
            "fused_conv1d_gdn_per_seq",
        ]

    @pytest.mark.parametrize(
        ("flag", "expected"),
        [
            ("batched_only", ["fused_conv1d_gdn_batched"]),
            ("prefill_only", ["fused_conv1d_gdn_per_seq"]),
        ],
    )
    def test_single_pass_requests_keep_their_call(self, flag, expected):
        assert _pallas_call_names(8, "tables", **{flag: True}) == expected

    def test_unknown_form_is_rejected(self):
        with pytest.raises(ValueError, match="gdn_form"):
            _pallas_call_names(8, "fused")


def _per_seq_tile_size(n_v, num_spec_tokens, rows=512):
    from vllm_torchtpu.kernels.gdn.v3 import wrapper

    n = 4
    dim = _N_KQ * _D_K * 2 + n_v * _D_V

    def run():
        return wrapper.fused_conv1d_gdn(
            jnp.zeros((rows, dim)),
            jnp.zeros((rows, n_v)),
            jnp.zeros((rows, n_v)),
            jnp.zeros((n + 1, _KERNEL_SIZE - 1, dim)),
            jnp.zeros((n + 1, n_v, _D_K, _D_V)),
            jnp.zeros((dim, 1, _KERNEL_SIZE)),
            None,
            jnp.zeros((n_v,)),
            jnp.zeros((n_v,)),
            jnp.zeros((n + 1,), jnp.int32),
            jnp.zeros((n,), jnp.int32),
            jnp.zeros((3,), jnp.int32),
            jnp.zeros((n,), jnp.int32),
            jnp.zeros((n,), jnp.int32),
            n_kq=_N_KQ,
            n_v=n_v,
            d_k=_D_K,
            d_v=_D_V,
            kernel_size=_KERNEL_SIZE,
            num_spec_tokens=num_spec_tokens,
        )

    [per_seq] = [
        eqn
        for eqn in _pallas_eqns(jax.make_jaxpr(run)())
        if eqn.params["name"].endswith("per_seq")
    ]
    return per_seq.params["metadata"]["tile_size"]


@pytest.mark.usefixtures("_v7x_info")
@pytest.mark.parametrize(
    ("n_v", "num_spec_tokens", "tile_size"),
    [(8, 0, 128), (16, 0, 64), (8, 3, 64)],
)
def test_mixed_tile_size_default(n_v, num_spec_tokens, tile_size):
    assert _per_seq_tile_size(n_v, num_spec_tokens) == tile_size


@pytest.mark.usefixtures("_v7x_info")
@pytest.mark.parametrize(
    ("rows", "num_spec_tokens", "tile_size"),
    [(72, 0, 64), (120, 0, 112), (128, 0, 128), (20, 0, 16), (8, 0, 8), (40, 3, 32)],
)
def test_mixed_tile_size_default_traces_for_any_rows(rows, num_spec_tokens, tile_size):
    # The chunked pass only accepts tiles of at most 16 rows or a multiple of
    # 16; tracing fails otherwise.
    assert _per_seq_tile_size(8, num_spec_tokens, rows=rows) == tile_size


# Qwen3.5-397B's GDN layer per device: 16 key and 64 value heads split eight
# ways under TP8, whole under DP8, over the HND fp8 attention pool (2 KV heads
# per device under DP8, 1 under TP8).
_PROD = {
    "TP8": dict(n_kq=2, n_v=8, payload=(2, 64, 4)),
    "DP8": dict(n_kq=16, n_v=64, payload=(4, 64, 4)),
}
_PAGE = 128


def _prod_tables_calls(monkeypatch, shard, rows, max_seqs, num_spec_tokens=0):
    """(name, vmem_limit_bytes) of each pallas_call at production geometry."""
    from vllm_torchtpu.kernels import pool_adapters
    from vllm_torchtpu.kernels.gdn.v3 import wrapper

    monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", "HND")
    g = _PROD[shard]
    n_kq, n_v = g["n_kq"], g["n_v"]
    dim = n_kq * _D_K * 2 + n_v * _D_V
    tok_bytes = int(np.prod(g["payload"]))
    ssm_ntok = n_v * _D_K * _D_V * 4 // tok_bytes
    conv_ntok = 1
    while conv_ntok * tok_bytes < (_KERNEL_SIZE - 1) * dim * 2:
        conv_ntok *= 2
    split = pl.cdiv(ssm_ntok + conv_ntok, _PAGE)
    pool = jax.ShapeDtypeStruct((5 * split, *g["payload"], _PAGE), jnp.float8_e4m3fn)
    plan = pool_adapters.v3_state_source(
        pool,
        split=split,
        ssm_ntok=ssm_ntok,
        conv_tok0=ssm_ntok,
        conv_ntok=conv_ntok,
        conv_dim=dim,
        n_v=n_v,
        d_k=_D_K,
        d_v=_D_V,
        kernel_size=_KERNEL_SIZE,
    )
    n = max_seqs
    ckpt = jnp.zeros((n, num_spec_tokens + 1), jnp.int32) if num_spec_tokens else None

    def run(source):
        return wrapper.fused_conv1d_gdn(
            jnp.zeros((rows, dim)),
            jnp.zeros((rows, n_v)),
            jnp.zeros((rows, n_v)),
            None,
            None,
            jnp.zeros((dim, 1, _KERNEL_SIZE)),
            None,
            jnp.zeros((n_v,)),
            jnp.zeros((n_v,)),
            jnp.zeros((n + 1,), jnp.int32),
            jnp.zeros((n,), jnp.int32),
            jnp.zeros((3,), jnp.int32),
            jnp.zeros((n,), jnp.int32),
            jnp.zeros((n,), jnp.int32),
            ckpt,
            n_kq=n_kq,
            n_v=n_v,
            d_k=_D_K,
            d_v=_D_V,
            kernel_size=_KERNEL_SIZE,
            state_source=source,
            state_plan=plan,
            num_spec_tokens=num_spec_tokens,
            gdn_form="tables",
        )

    return [
        (eqn.params["name"], eqn.params["compiler_params"].vmem_limit_bytes)
        for eqn in _pallas_eqns(jax.make_jaxpr(run)(pool))
    ]


def _carry_bytes(shard):
    # f32 carry scratch of the per-sequence pass: the recurrent state, and the
    # conv state's (kernel_size - 1, 1, dim) with each 1-row slab padded to 8
    # sublanes.
    g = _PROD[shard]
    dim = g["n_kq"] * _D_K * 2 + g["n_v"] * _D_V
    return 4 * (g["n_v"] * _D_K * _D_V + (_KERNEL_SIZE - 1) * 8 * dim)


@pytest.mark.usefixtures("_v7x_info")
class TestTablesFormVmem:
    """The one call holds the per-sequence carry scratch through the batched
    pass, so its limit is the batched pass's own plus that scratch. At DP8
    that is above the passes' own limit (the call failed to compile at it)
    and still within v7x's VMEM."""

    @pytest.mark.parametrize(
        ("shard", "rows", "max_seqs"),
        [("DP8", 64, 40), ("DP8", 128, 40), ("TP8", 64, 64), ("TP8", 256, 64)],
    )
    def test_one_call_budgets_both_passes(self, monkeypatch, shard, rows, max_seqs):
        capacity = pltpu.get_tpu_info().vmem_capacity_bytes
        batched = int(config.GDNConfig.DEFAULT_VMEM_FRACTION * capacity)
        [(name, limit)] = _prod_tables_calls(monkeypatch, shard, rows, max_seqs)
        assert name == "fused_conv1d_gdn_batched_tables_in_kernel"
        assert limit == batched + _carry_bytes(shard)
        assert limit <= capacity

    @pytest.mark.parametrize(("rows", "max_seqs"), [(8, 8), (48, 48)])
    def test_split_state_fits_the_one_call_at_tp8(self, monkeypatch, rows, max_seqs):
        # The rows and local v-heads `gdn_pooled_form_for_step` selects the
        # tables form at. Both passes keep split buffers and the decode tile;
        # each pass's buffers fit its own share, the batched pass's beside
        # the carry.
        from vllm_torchtpu.kernels.gdn.v3 import memory_ref

        seen = []
        original = tables_in_kernel.vmem_limit_bytes

        def capture(cfg_b, cfg_p):
            seen.append((cfg_b, cfg_p))
            return original(cfg_b, cfg_p)

        monkeypatch.setattr(tables_in_kernel, "vmem_limit_bytes", capture)
        fits = []
        original_fits = memory_ref.split_state_fits

        def capture_fits(cfg, conv_state, recurrent_state):
            ok = original_fits(cfg, conv_state, recurrent_state)
            fits.append((cfg.mode, cfg.tile_size, ok))
            return ok

        monkeypatch.setattr(memory_ref, "split_state_fits", capture_fits)
        [(name, limit)] = _prod_tables_calls(monkeypatch, "TP8", rows, max_seqs)
        assert name == "fused_conv1d_gdn_batched_tables_in_kernel"
        [(cfg_b, cfg_p)] = seen
        assert cfg_b.split_state and cfg_b.tile_size == 4
        assert cfg_p.split_state
        assert (config.GDNMode.BATCHED, 4, True) in fits
        assert (config.GDNMode.PER_SEQ, cfg_p.tile_size, True) in fits
        assert limit == cfg_b.get_vmem_limit_bytes() + _carry_bytes("TP8")
        assert limit >= cfg_p.get_vmem_limit_bytes()
        assert limit <= pltpu.get_tpu_info().vmem_capacity_bytes

    def test_spec_keeps_one_call(self, monkeypatch):
        [(name, limit)] = _prod_tables_calls(monkeypatch, "DP8", 32, 8, 3)
        assert name == "fused_conv1d_gdn_batched_w4_tables_in_kernel"
        assert limit <= pltpu.get_tpu_info().vmem_capacity_bytes

    def test_falls_back_when_both_passes_do_not_fit(self, monkeypatch):
        # A verify window's batched pass already takes 90% of VMEM; on a chip
        # with 48 MiB the DP8 carry scratch no longer fits beside it.
        info = dataclasses.replace(
            pltpu.get_tpu_info(), vmem_capacity_bytes=48 * 1024 * 1024
        )
        monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)
        # The chip info is read at trace time, so this shape is traced nowhere
        # else in this module.
        calls = _prod_tables_calls(monkeypatch, "DP8", 48, 12, 3)
        names = [name for name, _ in calls]
        assert names == ["fused_conv1d_gdn_batched_w4", "fused_conv1d_gdn_per_seq"]
