# SPDX-License-Identifier: Apache-2.0
"""2D bf16 qkv rows into the GDN v3 PER_SEQ kernel.

`VLLM_TPU_GDN_PREFILL_BF16_QKV_2D` hands prefill buckets' raw bf16
[tokens, dim] qkv to `fused_conv1d_gdn`, whose PER_SEQ kernel DMAs 16-row
aligned tiles and casts the rows to f32 in VMEM instead of reading an f32 copy
written to HBM. The BATCHED kernel keeps f32 rows, cast from the leading
num_seqs * window rows. A bf16 -> f32 cast is exact, so outputs and states
must be bitwise equal to the f32 path; with the flag off nothing changes.
"""

import functools
from types import SimpleNamespace
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import AbstractDevice, AbstractMesh, use_abstract_mesh

from vllm_torchtpu import envs
from vllm_torchtpu import gdn_pool_layout as pool_layout
from vllm_torchtpu.kernels.gdn.v3 import tiling, vmem_ldst, wrapper
from vllm_torchtpu.kernels.gdn.v3.input_layout import Conv1DInput

_TPU7X = AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
_KERNEL = "fused_conv1d_gdn_"


# --------------------------------------------------------------------------
# Eligibility.


_ELIGIBLE = dict(
    enabled=True,
    shape=(6400, 10240),
    dtype=jnp.bfloat16,
    decode_capacity=2,
    mixed_tile_size=64,
    num_lanes=128,
)


@pytest.mark.parametrize(
    "change,expected",
    [
        ({}, True),
        ({"enabled": False}, False),
        ({"dtype": jnp.float32}, False),
        ({"shape": (6400, 1, 10240)}, False),
        # The BATCHED kernel reads f32 rows cast from the leading
        # decode_capacity rows; a bucket no larger than that gains nothing.
        ({"shape": (64, 10240), "decode_capacity": 64}, False),
        ({"shape": (65, 10240), "decode_capacity": 64}, True),
        ({"shape": (15, 10240), "decode_capacity": 1, "mixed_tile_size": 8}, False),
        ({"shape": (16, 10240), "decode_capacity": 1, "mixed_tile_size": 16}, True),
        ({"shape": (6400, 10240 + 128)}, False),
        ({"mixed_tile_size": 20}, False),
        ({"mixed_tile_size": 8}, True),
        ({"is_kda": True}, False),
        ({"batched_only": True}, False),
    ],
)
def test_bf16_qkv_2d_eligibility(change, expected):
    assert wrapper.bf16_qkv_2d_eligible(**{**_ELIGIBLE, **change}) is expected


def test_vmem_estimate_adds_2d_rows_to_prefill_tiles_only():
    shape = dict(
        tile_b=1,
        chunk_sz=64,
        n_kq=16,
        n_v=48,
        d_k=128,
        d_v=128,
        kernel_size=4,
        act_out_bytes=2,
        rec_state_bytes=4,
        num_lanes=128,
        conv_state_dim_size=10240,
    )
    base = tiling.get_vmem_estimate_bytes(**shape)
    assert tiling.get_vmem_estimate_bytes(**shape, qkv_2d=True) == base + 136 * 10240
    decode = dict(shape, chunk_sz=1, tile_b=4, is_decode=True)
    assert tiling.get_vmem_estimate_bytes(
        **decode, qkv_2d=True
    ) == tiling.get_vmem_estimate_bytes(**decode)


# --------------------------------------------------------------------------
# Small dense-state problem shared by the routing and interpret-mode tests.


_N_KQ, _N_V, _D, _K = 1, 2, 128, 4
_DIM = 2 * _N_KQ * _D + _N_V * _D  # 512, a multiple of 2 * 128 lanes.


def _small_case(query_start_loc, num_decode, tokens, num_slots, fresh=(), seed=0):
    """Dense-state inputs; `num_slots` request slots, the first `num_decode`
    of them single-token decodes, sequences in `fresh` without initial state."""
    q_loc = list(query_start_loc)
    num_seqs = len(q_loc) - 1
    q_loc += [q_loc[-1]] * (num_slots - num_seqs)
    q_lens = np.diff(q_loc)
    seq_lens = q_lens + 37
    for s in fresh:
        seq_lens[s] = q_lens[s]
    rngs = iter(jax.random.split(jax.random.key(seed), 10))

    def normal(shape, dtype, scale=1.0):
        return (jax.random.normal(next(rngs), shape, jnp.float32) * scale).astype(
            dtype
        )

    args = dict(
        qkv=normal((tokens, _DIM), jnp.bfloat16),
        b=normal((tokens, _N_V), jnp.bfloat16),
        a=normal((tokens, _N_V), jnp.bfloat16),
        conv_weight=normal((_DIM, 1, _K), jnp.bfloat16),
        conv_bias=normal((_DIM,), jnp.bfloat16),
        a_log=normal((_N_V,), jnp.float32),
        dt_bias=normal((_N_V,), jnp.float32),
        query_start_loc=jnp.asarray(q_loc, jnp.int32),
        # Slot 0 is the null block; padded slots point at it.
        state_indices=jnp.asarray(
            [s + 1 if s < num_seqs else 0 for s in range(num_slots)], jnp.int32
        ),
        distribution=jnp.asarray([num_decode, num_seqs, num_seqs], jnp.int32),
        seq_lens=jnp.asarray(seq_lens, jnp.int32),
    )
    states = dict(
        conv_state=normal((num_slots + 1, _K - 1, _DIM), jnp.float32),
        recurrent_state=normal((num_slots + 1, _N_V, _D, _D), jnp.float32, 0.1),
    )
    static = dict(n_kq=_N_KQ, n_v=_N_V, d_k=_D, d_v=_D, kernel_size=_K)
    return args, states, static


def _gdn_calls(jaxpr, found=None):
    """(kernel name, first floating operand aval) of each GDN v3 pallas_call."""
    found = [] if found is None else found
    for eqn in jaxpr.eqns:
        name = eqn.params.get("name") or ""
        if eqn.primitive.name == "pallas_call" and name.startswith(_KERNEL):
            qkv = next(
                v.aval for v in eqn.invars if jnp.issubdtype(v.aval.dtype, jnp.floating)
            )
            found.append((name, qkv))
            continue
        for param in eqn.params.values():
            sub = getattr(param, "jaxpr", None)
            if sub is not None and hasattr(sub, "eqns"):
                _gdn_calls(sub, found)
    return found


def _routes(args, states, static, **kwargs):
    fn = functools.partial(wrapper.fused_conv1d_gdn, **static, **kwargs)
    with (
        use_abstract_mesh(_TPU7X),
        mock.patch.object(envs, "TPU_ENABLE_GDN_DYNAMIC_TILING", False),
    ):
        closed = jax.make_jaxpr(fn)(**args, **states)
    return [
        (name.split("_")[3], (qkv.dtype.name, qkv.shape))
        for name, qkv in _gdn_calls(closed.jaxpr)
    ]


# 2 decode rows, then prefills starting at rows 2, 23 and 61 (not 16-aligned),
# padding rows past 100, and one unused request slot. 104 tokens pad to 112
# for 2D rows.
_Q_LOC = (0, 1, 2, 23, 61, 100)
_T, _T_2D, _SLOTS = 104, 112, 6


@pytest.mark.parametrize(
    "case,kwargs,expected",
    [
        (
            "on",
            dict(bf16_qkv_2d=True),
            [
                ("batched", ("float32", (_SLOTS, 1, _DIM))),
                ("per", ("bfloat16", (_T_2D, _DIM))),
            ],
        ),
        (
            "off",
            dict(bf16_qkv_2d=False),
            [
                ("batched", ("float32", (_T, 1, _DIM))),
                ("per", ("float32", (_T, 1, _DIM))),
            ],
        ),
        (
            "prefill_only",
            dict(bf16_qkv_2d=True, prefill_only=True),
            [("per", ("bfloat16", (_T_2D, _DIM)))],
        ),
        (
            "batched_only",
            dict(bf16_qkv_2d=True, batched_only=True),
            [("batched", ("float32", (_T, 1, _DIM)))],
        ),
        (
            "chunk_not_multiple_of_8",
            dict(bf16_qkv_2d=True, mixed_tile_size=12),
            [
                ("batched", ("float32", (_T, 1, _DIM))),
                ("per", ("float32", (_T, 1, _DIM))),
            ],
        ),
    ],
)
def test_wrapper_routes_qkv(case, kwargs, expected):
    args, states, static = _small_case(_Q_LOC, 2, _T, _SLOTS)
    kwargs.setdefault("mixed_tile_size", 16)
    assert _routes(args, states, static, decode_tile_size=2, **kwargs) == expected


def test_wrapper_routes_qkv_fallbacks():
    # A bucket within the decode capacity keeps f32 rows.
    args, states, static = _small_case((0, 1, 2, 3, 20), 3, 24, 24)
    routes = _routes(args, states, static, bf16_qkv_2d=True, mixed_tile_size=16)
    assert {dtype for _, (dtype, _) in routes} == {"float32"}

    # Tokens are padded to whole 16-row tiles for 2D rows.
    args, states, static = _small_case((0, 1, 9, 50), 1, 50, 4)
    assert _routes(
        args, states, static, bf16_qkv_2d=True, decode_tile_size=2, mixed_tile_size=16
    ) == [
        ("batched", ("float32", (4, 1, _DIM))),
        ("per", ("bfloat16", (64, _DIM))),
    ]

    # A prepared FP32 input is used as is.
    args, states, static = _small_case(_Q_LOC, 2, _T, _SLOTS)
    args["qkv"] = Conv1DInput(
        args["qkv"].astype(jnp.float32)[:, None, :], args["qkv"].dtype
    )
    assert {
        (dtype, shape)
        for _, (dtype, shape) in _routes(
            args, states, static, bf16_qkv_2d=True, mixed_tile_size=16
        )
    } == {("float32", (_T, 1, _DIM))}


def test_wrapper_reads_flag_from_env(monkeypatch):
    # A token count no other test traces, so no cached trace answers for it.
    args, states, static = _small_case((0, 1, 2, 23, 61, 90), 2, 96, _SLOTS)
    monkeypatch.setattr(envs, "VLLM_TPU_GDN_PREFILL_BF16_QKV_2D", True)
    routes = _routes(args, states, static, decode_tile_size=2, mixed_tile_size=16)
    assert routes[-1] == ("per", ("bfloat16", (96, _DIM)))


# --------------------------------------------------------------------------
# Flag off leaves the program unchanged.


def _lowered(args, states, static, **kwargs):
    """TPU lowering text (Mosaic bodies included), produced without a TPU."""

    def fn(conv_state, recurrent_state, operands):
        return wrapper.fused_conv1d_gdn(
            conv_state=conv_state,
            recurrent_state=recurrent_state,
            **operands,
            **static,
            **kwargs,
        )

    with (
        use_abstract_mesh(_TPU7X),
        mock.patch.object(envs, "TPU_ENABLE_GDN_DYNAMIC_TILING", False),
    ):
        traced = jax.jit(fn).trace(
            states["conv_state"], states["recurrent_state"], args
        )
        return traced.lower(lowering_platforms=("tpu",)).as_text()


def test_flag_off_program_is_unchanged_by_flag_plumbing(monkeypatch):
    """Only an eligible raw bf16 input under the flag changes the program.

    The pre-change baseline itself was compared on the Mac (location-free
    Mosaic bodies, q38 pooled shapes): flag off lowers to identical text.
    """
    args, states, static = _small_case(_Q_LOC, 2, _T, _SLOTS)
    kw = dict(decode_tile_size=2, mixed_tile_size=16)
    monkeypatch.setattr(envs, "VLLM_TPU_GDN_PREFILL_BF16_QKV_2D", False)
    off = _lowered(args, states, static, bf16_qkv_2d=False, **kw)
    assert _lowered(args, states, static, bf16_qkv_2d=None, **kw) == off
    on = _lowered(args, states, static, bf16_qkv_2d=True, **kw)
    assert on != off
    # Ineligible chunk: flag on falls back to exactly the flag-off program.
    kw12 = dict(decode_tile_size=2, mixed_tile_size=12)
    assert _lowered(args, states, static, bf16_qkv_2d=True, **kw12) == _lowered(
        args, states, static, bf16_qkv_2d=False, **kw12
    )
    # A prepared input ignores the flag.
    prepared = dict(
        args,
        qkv=Conv1DInput(args["qkv"].astype(jnp.float32)[:, None, :], jnp.bfloat16),
    )
    assert _lowered(prepared, states, static, bf16_qkv_2d=True, **kw) == _lowered(
        prepared, states, static, bf16_qkv_2d=False, **kw
    )


# --------------------------------------------------------------------------
# Bitwise equality in CPU interpret mode.


def _strided_heads_by_value(vmem_ref, num_heads, head_dim, lane_offset=0):
    """`vmem_ldst.load_strided_heads` without a ref reshape (same values)."""
    lanes = pltpu.get_tpu_info().num_lanes
    rows = vmem_ref[...]
    rows = rows.reshape(rows.shape[0], rows.shape[-1])
    start = lane_offset * lanes
    return jnp.stack(
        [
            rows[:, start + h * head_dim : start + (h + 1) * head_dim]
            for h in range(num_heads)
        ],
        axis=0,
    )


def _compact_to_large_by_value(vmem_ref):
    """`vmem_ldst.load_compact_to_large` without a ref reshape (same values)."""
    assert vmem_ref.dtype.itemsize == 4 and vmem_ref.shape[-2] == 1
    return vmem_ref[...].reshape(vmem_ref.shape[:-2] + (vmem_ref.shape[-1],))


def _interpret(args, states, static, **kwargs):
    pltpu.reset_tpu_interpret_mode_state()
    try:
        with (
            use_abstract_mesh(_TPU7X),
            mock.patch.object(envs, "TPU_ENABLE_GDN_DYNAMIC_TILING", False),
            pltpu.force_tpu_interpret_mode(
                pltpu.InterpretParams(
                    dma_execution_mode="eager", skip_floating_point_ops=False
                )
            ),
        ):
            # The wrapper donates the states; give each run its own.
            fresh = {k: jnp.copy(v) for k, v in states.items()}
            result = wrapper.fused_conv1d_gdn(**args, **fresh, **static, **kwargs)
            return jax.tree.map(np.asarray, jax.block_until_ready(result))
    finally:
        pltpu.reset_tpu_interpret_mode_state()


@pytest.mark.parametrize(
    "q_loc,num_decode,tokens,slots,chunk,fresh",
    [
        # Unaligned starts 2, 23, 61; partial last tiles; padding rows.
        (_Q_LOC, 2, _T, _SLOTS, 16, ()),
        # Chunk 8 (half a row tile) and a sequence without initial state.
        (_Q_LOC, 2, _T, _SLOTS, 8, (3,)),
        # A start one row before a tile boundary and one past it.
        ((0, 15, 47, 81), 0, 96, 4, 16, (0,)),
    ],
)
def test_2d_rows_match_f32_rows_bitwise(
    monkeypatch, q_loc, num_decode, tokens, slots, chunk, fresh
):
    # CPU interpret mode cannot reshape refs, which the chunked (C > 1) head
    # loads do; both runs take these value-level equivalents. The 2D DMA and
    # unpack under test are not replaced.
    monkeypatch.setattr(vmem_ldst, "load_strided_heads", _strided_heads_by_value)
    monkeypatch.setattr(vmem_ldst, "load_compact_to_large", _compact_to_large_by_value)
    args, states, static = _small_case(q_loc, num_decode, tokens, slots, fresh)
    kw = dict(decode_tile_size=2, mixed_tile_size=chunk)
    assert _routes(args, states, static, bf16_qkv_2d=True, **kw)[-1] == (
        "per",
        ("bfloat16", (-(-tokens // 16) * 16, _DIM)),
    )

    expected = _interpret(args, states, static, bf16_qkv_2d=False, **kw)
    got = _interpret(args, states, static, bf16_qkv_2d=True, **kw)

    (exp_conv, exp_rec), exp_out = expected
    (got_conv, got_rec), got_out = got
    assert np.abs(exp_out[: q_loc[-1]].astype(np.float32)).max() > 0
    for name, g, e in (
        ("out", got_out, exp_out),
        ("conv_state", got_conv, exp_conv),
        ("recurrent_state", got_rec, exp_rec),
    ):
        assert g.shape == e.shape and g.dtype == e.dtype, name
        np.testing.assert_array_equal(
            g.view(np.uint8), e.view(np.uint8), err_msg=f"{name} differs"
        )


# --------------------------------------------------------------------------
# prepare_gdn_input.


def test_prepare_gdn_input_keeps_bf16_rows_under_flag(monkeypatch):
    from vllm_torchtpu.models.vllm.qwen4_exp.gdn import prepare_gdn_input

    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("model",))
    qkv = jnp.zeros((256, 10240), jnp.bfloat16)

    monkeypatch.setattr(envs, "VLLM_TPU_GDN_PREFILL_BF16_QKV_2D", True)
    assert prepare_gdn_input(qkv, 4, mesh) is qkv

    monkeypatch.setattr(envs, "VLLM_TPU_GDN_PREFILL_BF16_QKV_2D", False)
    prepared = jax.eval_shape(lambda x: prepare_gdn_input(x, 4, mesh), qkv)
    assert isinstance(prepared, Conv1DInput)
    assert prepared.values.shape == (256, 1, 10240)
    assert prepared.values.dtype == jnp.float32
    # Decode-sized buckets were never converted.
    assert prepare_gdn_input(qkv, 256, mesh) is qkv


# --------------------------------------------------------------------------
# q38 shapes on TPU: pooled state, full input-preparation path.


_Q38 = SimpleNamespace(n_kq=16, n_v=48, d=128, width=10240, kernel_size=4)
_Q38_SLOTS = 4
_Q38_CONTEXT = 100_000
_Q38_CASES = [
    # (bucket tokens, query lengths, leading decode rows, recurrent dtype)
    (3200, (3187,), 0, jnp.float32),
    (3200, (1235, 1950), 0, jnp.float32),  # 2nd request starts at 1235 = 3 mod 16
    (6400, (1, 1, 6390), 2, jnp.float32),  # prefill starts at row 2
    (6400, (3001, 3390), 0, jnp.float32),  # 2nd request starts at 3001 = 9 mod 16
    (6400, (3001, 3390), 0, jnp.bfloat16),
]


def _q38_pool_and_page(recurrent_dtype, num_blocks):
    """fp8 unified pool with 1024-byte token rows; page fits one GDN state."""
    payload, lanes = (1, 4), 256
    layout = pool_layout.derive_pooled_gdn_state_layout(
        ssm_bytes=pool_layout.pooled_gdn_ssm_state_bytes(
            num_v_heads=_Q38.n_v,
            head_k_dim=_Q38.d,
            head_v_dim=_Q38.d,
            dtype=jnp.dtype(recurrent_dtype),
        ),
        conv_bytes=pool_layout.pooled_gdn_conv_state_bytes(
            kernel_size=_Q38.kernel_size,
            conv_dim=_Q38.width,
            dtype=jnp.dtype(jnp.float32),
        ),
        token_bytes=int(np.prod(payload)) * lanes,
    )
    page = layout.required_tokens
    return jnp.zeros((num_blocks, page, *payload, lanes), jnp.float8_e4m3fn), page


def _q38_case(tokens, lengths, num_decode, seed=38):
    """Host-side inputs of one q38 prefill step (CPU-safe)."""
    num_seqs = len(lengths)
    q_loc = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32)
    assert q_loc[-1] <= tokens
    q_loc = np.concatenate([q_loc, [q_loc[-1]] * (_Q38_SLOTS - num_seqs)])
    rng = np.random.default_rng(seed)

    def normal(shape, scale=0.02):
        return jnp.asarray(rng.normal(0, scale, shape), jnp.bfloat16)

    return dict(
        qkv=normal((tokens, _Q38.width), 1.0),
        b=normal((tokens, _Q38.n_v), 1.0),
        a=normal((tokens, _Q38.n_v), 1.0),
        weight=normal((_Q38.width, 1, _Q38.kernel_size), 0.2),
        bias=normal((_Q38.width,)),
        a_log=jnp.asarray(rng.normal(0, 0.5, _Q38.n_v), jnp.float32),
        dt_bias=jnp.asarray(rng.normal(0, 0.5, _Q38.n_v) - 2, jnp.float32),
        query_start_loc=jnp.asarray(q_loc, jnp.int32),
        # Request slot s owns pool block s + 1; padded slots the null block.
        state_indices=jnp.asarray(
            [s + 1 if s < num_seqs else 0 for s in range(_Q38_SLOTS)], jnp.int32
        ),
        distribution=jnp.asarray([num_decode, num_seqs, num_seqs], jnp.int32),
        lengths=np.asarray(lengths + (0,) * (_Q38_SLOTS - num_seqs), np.int32),
    )


@pytest.mark.parametrize("tokens,lengths,num_decode,recurrent_dtype", _Q38_CASES)
def test_q38_case_setup(tokens, lengths, num_decode, recurrent_dtype):
    """The TPU test's host-side setup: shapes, unaligned starts, pool fit."""
    case = _q38_case(tokens, lengths, num_decode)
    starts = np.asarray(case["query_start_loc"])[num_decode : len(lengths)]
    assert tokens > _Q38_SLOTS  # Beyond the decode capacity: a 2D bucket.
    assert any(s % 16 for s in starts) or len(lengths) == 1
    pool, page = _q38_pool_and_page(recurrent_dtype, _Q38_SLOTS + 1)
    assert pool.shape[1] == page
    assert wrapper.bf16_qkv_2d_eligible(
        enabled=True,
        shape=case["qkv"].shape,
        dtype=case["qkv"].dtype,
        decode_capacity=_Q38_SLOTS,
        mixed_tile_size=64,
        num_lanes=128,
    )


@pytest.mark.tpu_test
@pytest.mark.parametrize("tokens,lengths,num_decode,recurrent_dtype", _Q38_CASES)
def test_q38_pooled_prefill_2d_rows_bitwise(
    monkeypatch, tokens, lengths, num_decode, recurrent_dtype
):
    from vllm_torchtpu.layers.adapter.custom_ops import gdn_attention_op
    from vllm_torchtpu.models.vllm.qwen4_exp.gdn import prepare_gdn_input

    monkeypatch.setattr(envs, "TPU_GDN_CONV_QK_PAIR_LAYOUT", False)
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("model",))
    case = _q38_case(tokens, lengths, num_decode)
    pool, page = _q38_pool_and_page(recurrent_dtype, _Q38_SLOTS + 1)

    def step(pool, qkv, b, a, query_start_loc, distribution, seq_lens):
        prepared = prepare_gdn_input(qkv, _Q38_SLOTS, mesh)
        return gdn_attention_op.gdn_attention_pooled_core_tpu(
            prepared,
            b,
            a,
            pool,
            case["weight"],
            case["bias"],
            case["a_log"],
            case["dt_bias"],
            case["state_indices"],
            query_start_loc,
            distribution,
            seq_lens,
            n_kq=_Q38.n_kq,
            n_v=_Q38.n_v,
            d_k=_Q38.d,
            d_v=_Q38.d,
            kernel_size=_Q38.kernel_size,
            pool_block_tokens=page,
            mesh=mesh,
            recurrent_state_dtype=recurrent_dtype,
            conv_state_dtype=jnp.float32,
        )

    # Warm-up: every live slot prefills 512 fresh tokens (flag off, all
    # PER_SEQ), so the measured step resumes from real, nonzero cached state.
    monkeypatch.setattr(envs, "VLLM_TPU_GDN_PREFILL_BF16_QKV_2D", False)
    num_seqs = len(lengths)
    warm = 512 * num_seqs
    warm_loc = np.minimum(np.arange(_Q38_SLOTS + 1) * 512, warm).astype(np.int32)
    warm_lens = np.where(np.arange(_Q38_SLOTS) < num_seqs, 512, 0)
    rng = np.random.default_rng(7)
    pool, _ = jax.jit(step)(
        pool,
        jnp.asarray(rng.normal(0, 1, (warm, _Q38.width)), jnp.bfloat16),
        jnp.asarray(rng.normal(0, 1, (warm, _Q38.n_v)), jnp.bfloat16),
        jnp.asarray(rng.normal(0, 1, (warm, _Q38.n_v)), jnp.bfloat16),
        jnp.asarray(warm_loc, jnp.int32),
        jnp.asarray([0, num_seqs, num_seqs], jnp.int32),
        jnp.asarray(warm_lens, jnp.int32),
    )
    seq_lens = jnp.asarray(
        np.where(case["lengths"] > 0, _Q38_CONTEXT + case["lengths"], 0), jnp.int32
    )
    operands = (
        case["qkv"],
        case["b"],
        case["a"],
        case["query_start_loc"],
        case["distribution"],
        seq_lens,
    )

    results = {}
    for flag in (False, True):
        monkeypatch.setattr(envs, "VLLM_TPU_GDN_PREFILL_BF16_QKV_2D", flag)
        run = jax.jit(step)
        hlo = run.lower(pool, *operands).as_text()
        # The flag must actually change the path: no f32 conversion kernel,
        # bf16 rows into the PER_SEQ kernel.
        assert ("gdn_compact_qkv_input" in hlo) is (not flag)
        assert ('act_in\\22:\\22bfloat16' in hlo) is flag
        new_pool, out = run(jnp.copy(pool), *operands)
        results[flag] = (np.asarray(new_pool), np.asarray(out))

    (pool_off, out_off), (pool_on, out_on) = results[False], results[True]
    live = int(case["query_start_loc"][-1])
    assert np.isfinite(out_off[:live].astype(np.float32)).all()
    assert np.abs(out_off[:live].astype(np.float32)).max() > 0
    np.testing.assert_array_equal(out_on.view(np.uint16), out_off.view(np.uint16))
    np.testing.assert_array_equal(pool_on.view(np.uint8), pool_off.view(np.uint8))


# --------------------------------------------------------------------------
# Token-dense conv (VLLM_TPU_GDN_PREFILL_DENSE_CONV) over the 2D bf16 rows.


def _metadata_of(args, states, static, **kwargs):
    """GDNConfig metadata strings of each GDN v3 pallas_call in the program."""
    text = _lowered(args, states, static, **kwargs)
    return text.count("dense_conv")


def test_dense_conv_flag_off_program_unchanged(monkeypatch):
    """dense_conv=None reads the env (off) and lowers exactly as False; on, the
    PER_SEQ kernel's metadata carries dense_conv and the program changes; the
    f32 path (no 2D rows) ignores the flag; dense_conv never appears off."""
    args, states, static = _small_case(_Q_LOC, 2, _T, _SLOTS)
    kw = dict(decode_tile_size=2, mixed_tile_size=16, bf16_qkv_2d=True)
    monkeypatch.setattr(envs, "VLLM_TPU_GDN_PREFILL_DENSE_CONV", False)
    off = _lowered(args, states, static, dense_conv=False, **kw)
    assert _lowered(args, states, static, dense_conv=None, **kw) == off
    assert "dense_conv" not in off
    on = _lowered(args, states, static, dense_conv=True, **kw)
    assert on != off and "dense_conv" in on
    f32 = dict(kw, bf16_qkv_2d=False)
    assert _lowered(args, states, static, dense_conv=True, **f32) == _lowered(
        args, states, static, dense_conv=False, **f32
    )
    # Ineligible chunk (12, not a multiple of 8): exactly the compact 2D program.
    kw12 = dict(kw, mixed_tile_size=12)
    assert _lowered(args, states, static, dense_conv=True, **kw12) == _lowered(
        args, states, static, dense_conv=False, **kw12
    )


def test_dense_conv_reads_env(monkeypatch):
    args, states, static = _small_case(_Q_LOC, 2, _T, _SLOTS)
    kw = dict(decode_tile_size=2, mixed_tile_size=16, bf16_qkv_2d=True)
    monkeypatch.setattr(envs, "VLLM_TPU_GDN_PREFILL_DENSE_CONV", True)
    # The env is read at trace time; a trace cached by an earlier test under the
    # same static arguments would be reused (a serving process reads it once).
    jax.clear_caches()
    assert "dense_conv" in _lowered(args, states, static, dense_conv=None, **kw)
    monkeypatch.setattr(envs, "VLLM_TPU_GDN_PREFILL_DENSE_CONV", False)
    jax.clear_caches()


@pytest.mark.parametrize(
    "q_loc,num_decode,tokens,slots,chunk,fresh",
    [
        # Unaligned starts 2, 23, 61; partial last tiles; padding rows; decodes.
        (_Q_LOC, 2, _T, _SLOTS, 16, ()),
        # Chunk 8 and a sequence without initial state.
        (_Q_LOC, 2, _T, _SLOTS, 8, (3,)),
        # Starts one row before and one past a 16-row tile boundary; chunk 32.
        ((0, 15, 47, 81), 0, 96, 4, 32, (0,)),
    ],
)
def test_dense_conv_matches_compact_rows_bitwise(
    monkeypatch, q_loc, num_decode, tokens, slots, chunk, fresh
):
    """The dense path's outputs, conv states and recurrent states equal the
    compact 2D path's and the f32 path's bit for bit (same arithmetic, op for
    op; only the VMEM layout differs)."""
    monkeypatch.setattr(vmem_ldst, "load_strided_heads", _strided_heads_by_value)
    monkeypatch.setattr(vmem_ldst, "load_compact_to_large", _compact_to_large_by_value)
    args, states, static = _small_case(q_loc, num_decode, tokens, slots, fresh, seed=5)
    kw = dict(decode_tile_size=2, mixed_tile_size=chunk)
    f32 = _interpret(args, states, static, bf16_qkv_2d=False, dense_conv=False, **kw)
    compact = _interpret(args, states, static, bf16_qkv_2d=True, dense_conv=False, **kw)
    dense = _interpret(args, states, static, bf16_qkv_2d=True, dense_conv=True, **kw)
    assert np.abs(f32[1][: q_loc[-1]].astype(np.float32)).max() > 0
    for ref_name, ref in (("f32", f32), ("compact2d", compact)):
        (rc, rr), ro = ref
        (dc, dr), do = dense
        for name, g, e in (("out", do, ro), ("conv_state", dc, rc), ("recurrent_state", dr, rr)):
            assert g.shape == e.shape and g.dtype == e.dtype, name
            np.testing.assert_array_equal(
                g.view(np.uint8), e.view(np.uint8), err_msg=f"dense vs {ref_name}: {name} differs"
            )


# --------------------------------------------------------------------------
# q|k|v columns straight from the qkvz projection (VLLM_TPU_GDN_PREFILL_QKV_FROM_PROJ).

from vllm_torchtpu.kernels.gdn.v3.input_layout import QKVColumns  # noqa: E402


def _with_z(args, extra=256, seed=9):
    """args with qkv widened by `extra` z columns; returns (qkvz args, qkv width)."""
    qkv = args["qkv"]
    z = jax.random.normal(jax.random.key(seed), (qkv.shape[0], extra), jnp.float32).astype(qkv.dtype)
    return dict(args, qkv=QKVColumns(jnp.concatenate([qkv, z], axis=1), qkv.shape[1])), qkv.shape[1]


def test_qkv_columns_route_whole_projection_to_per_seq():
    """Under 2D rows the PER_SEQ kernel takes the whole [T, qkv + z] bf16
    projection (strided DMA of its q|k|v columns, no copy); off the 2D path the
    wrapper slices, so the kernels see today's operands."""
    args, states, static = _small_case(_Q_LOC, 2, _T, _SLOTS)
    wide, _ = _with_z(args)
    kw = dict(decode_tile_size=2, mixed_tile_size=16)
    routes = _routes(wide, states, static, bf16_qkv_2d=True, **kw)
    assert routes[-1] == ("per", ("bfloat16", (_T_2D, _DIM + 256))), routes
    assert routes[:-1] == _routes(args, states, static, bf16_qkv_2d=True, **kw)[:-1]
    assert _routes(wide, states, static, bf16_qkv_2d=False, **kw) == _routes(
        args, states, static, bf16_qkv_2d=False, **kw
    )


@pytest.mark.parametrize("dense", [False, True])
@pytest.mark.parametrize(
    "q_loc,num_decode,tokens,slots,chunk,fresh",
    [(_Q_LOC, 2, _T, _SLOTS, 16, ()), ((0, 15, 47, 81), 0, 96, 4, 32, (0,))],
)
def test_qkv_columns_match_sliced_qkv_bitwise(monkeypatch, dense, q_loc, num_decode, tokens, slots, chunk, fresh):
    monkeypatch.setattr(vmem_ldst, "load_strided_heads", _strided_heads_by_value)
    monkeypatch.setattr(vmem_ldst, "load_compact_to_large", _compact_to_large_by_value)
    args, states, static = _small_case(q_loc, num_decode, tokens, slots, fresh, seed=11)
    wide, _ = _with_z(args)
    kw = dict(decode_tile_size=2, mixed_tile_size=chunk, dense_conv=dense)
    for two_d in (True, False):
        ref = _interpret(args, states, static, bf16_qkv_2d=two_d, **kw)
        got = _interpret(wide, states, static, bf16_qkv_2d=two_d, **kw)
        (rc, rr), ro = ref
        (gc, gr), go = got
        for name, g, e in (("out", go, ro), ("conv_state", gc, rc), ("recurrent_state", gr, rr)):
            np.testing.assert_array_equal(g.view(np.uint8), e.view(np.uint8), err_msg=f"2d={two_d}: {name}")


def test_prepare_gdn_input_qkv_columns(monkeypatch):
    from vllm_torchtpu.models.vllm.qwen4_exp.gdn import prepare_gdn_input

    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("model",))
    qkvz = jnp.zeros((256, 16384), jnp.bfloat16)
    monkeypatch.setattr(envs, "VLLM_TPU_GDN_PREFILL_BF16_QKV_2D", True)
    got = prepare_gdn_input(qkvz, 4, mesh, 10240)
    assert isinstance(got, QKVColumns) and got.values is qkvz and got.cols == 10240
    # Decode-sized buckets are sliced to the conv width.
    assert prepare_gdn_input(qkvz, 256, mesh, 10240).shape == (256, 10240)
    monkeypatch.setattr(envs, "VLLM_TPU_GDN_PREFILL_BF16_QKV_2D", False)
    prepared = jax.eval_shape(lambda x: prepare_gdn_input(x, 4, mesh, 10240), qkvz)
    assert isinstance(prepared, Conv1DInput) and prepared.values.shape == (256, 1, 10240)
    # Already the conv width: unchanged behaviour.
    qkv = jnp.zeros((256, 10240), jnp.bfloat16)
    monkeypatch.setattr(envs, "VLLM_TPU_GDN_PREFILL_BF16_QKV_2D", True)
    assert prepare_gdn_input(qkv, 4, mesh, 10240) is qkv
