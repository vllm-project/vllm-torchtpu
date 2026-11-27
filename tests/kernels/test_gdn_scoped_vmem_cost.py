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
"""Scoped VMEM and cost estimate coverage for GDN and KDA."""

import contextlib
import math
import re
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu
from kernels.scoped_vmem_test_utils import (
    _abstract_tpu,
    _assert_bitwise_equal,
    _declared_vmem_bytes,
    _eqns,
    _extract_tpu_custom_calls,
    _require_tpu,
    _scoped_vmem_bytes,
    _sds,
    _subjaxprs,
)

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.gdn.v3 import compute_conv1d, compute_gdn, compute_kda
from vllm_torchtpu.kernels.gdn.v3 import config as gdn_config
from vllm_torchtpu.kernels.gdn.v3 import tiling as gdn_tiling
from vllm_torchtpu.kernels.gdn.v3 import wrapper as gdn_wrapper

_MIB = 1024 * 1024
_GDN = gdn_config.AttentionMode.GDN
_KDA = gdn_config.AttentionMode.KDA
_TRANSCENDENTAL_PRIMITIVES = frozenset(
    {"exp", "exp2", "expm1", "log", "log1p", "logistic", "tanh", "rsqrt", "sqrt"}
)
_ELEMENTWISE_FLOP_PRIMITIVES = frozenset({"add", "sub", "mul", "div"})


def _make_gdn_abstract_inputs(
    *,
    n_kq: int,
    n_v: int,
    d_k: int = 128,
    d_v: int = 128,
    kernel_size: int = 4,
    tokens: int = 64,
    max_seqs: int = 64,
    attention_mode: gdn_config.AttentionMode = gdn_config.AttentionMode.GDN,
):
    dim = 2 * n_kq * d_k + n_v * d_v
    is_kda = attention_mode == gdn_config.AttentionMode.KDA
    a_shape = (tokens, n_v * d_k) if is_kda else (tokens, n_v)
    dt_shape = (n_v, d_k) if is_kda else (n_v,)
    return (
        _sds((tokens, dim), jnp.bfloat16),
        _sds((tokens, n_v), jnp.bfloat16),
        _sds(a_shape, jnp.bfloat16),
        _sds((max_seqs + 1, kernel_size - 1, dim), jnp.float32),
        _sds((max_seqs + 1, n_v, d_k, d_v), jnp.float32),
        _sds((dim, 1, kernel_size), jnp.bfloat16),
        _sds((n_v,), jnp.float32),
        _sds(dt_shape, jnp.float32),
        _sds((tokens + 1,), jnp.int32),
        _sds((tokens,), jnp.int32),
        _sds((3,), jnp.int32),
        _sds((tokens,), jnp.int32),
        _sds((tokens,), jnp.int32),
    )


def test_env_flag_defaults_false(monkeypatch):
    monkeypatch.delenv("TPU_ATTN_SCOPED_VMEM_SIZING", raising=False)
    # The getter evaluates the default even if serving enabled the module cache.
    assert envs.environment_variables["TPU_ATTN_SCOPED_VMEM_SIZING"]() is False


@pytest.mark.parametrize("device_kind", ["TPU7x", "TPU v6 lite"])
@pytest.mark.parametrize(
    "n_kq,n_v,tokens,attention_mode,num_spec_tokens",
    [
        (2, 8, 8, gdn_config.AttentionMode.GDN, 0),
        (2, 8, 64, gdn_config.AttentionMode.GDN, 0),
        (16, 64, 64, gdn_config.AttentionMode.GDN, 0),
        (2, 8, 32, gdn_config.AttentionMode.GDN, 3),
        (2, 8, 64, gdn_config.AttentionMode.KDA, 0),
        (8, 8, 32, gdn_config.AttentionMode.KDA, 3),
    ],
)
def test_gdn_v3_scoped_vmem_and_cost_lowering(
    device_kind: str,
    n_kq: int,
    n_v: int,
    tokens: int,
    attention_mode: gdn_config.AttentionMode,
    num_spec_tokens: int,
):
    args = _make_gdn_abstract_inputs(
        n_kq=n_kq,
        n_v=n_v,
        tokens=tokens,
        max_seqs=max(64, tokens),
        attention_mode=attention_mode,
    )

    def make_fn(scoped: bool, cost: bool):
        def fn(
            qkv,
            b,
            a,
            conv_state,
            rec_state,
            conv_w,
            a_log,
            dt_bias,
            query_start_loc,
            state_indices,
            distribution,
            seq_lens,
            read_offsets,
        ):
            return gdn_wrapper.fused_conv1d_gdn(
                qkv,
                b,
                a,
                (conv_state, rec_state),
                conv_w,
                None,
                a_log,
                dt_bias,
                query_start_loc,
                state_indices,
                distribution,
                seq_lens,
                read_offsets=read_offsets if num_spec_tokens > 0 else None,
                n_kq=n_kq,
                n_v=n_v,
                d_k=128,
                d_v=128,
                kernel_size=4,
                num_spec_tokens=num_spec_tokens,
                attention_mode=attention_mode,
                use_scoped_vmem_limit=scoped,
                use_cost_estimate=cost,
            )

        return fn

    with _abstract_tpu(device_kind):
        vmem_cap = pltpu.get_tpu_info().vmem_capacity_bytes
        base_calls = _extract_tpu_custom_calls(make_fn(False, False), args)
        scoped_calls = _extract_tpu_custom_calls(make_fn(True, True), args)

    assert len(base_calls) == len(scoped_calls) >= 2
    any_reduced = False
    for (bname, bcfg), (sname, scfg) in zip(base_calls, scoped_calls):
        # Tile selection and Mosaic kernel codegen must be 100% unchanged.
        assert bname == sname
        bcc = bcfg["custom_call_config"]
        scc = scfg["custom_call_config"]
        assert bcc["body"] == scc["body"]

        # Default path preserves the mode-appropriate VMEM budget (80%, or 90%
        # for windowed speculative verify) and omits cost_estimate.
        launch_window = (num_spec_tokens + 1) if "batched" in bname else 1
        expected_default = gdn_config.GDNConfig.get_vmem_limit_bytes(
            window_size=launch_window,
            vmem_capacity_bytes=vmem_cap,
        )
        base_res = _scoped_vmem_bytes(bcfg)
        assert base_res == expected_default
        assert "cost_estimate" not in bcc

        # Scoped path never exceeds default limit and always covers declared VMEM.
        # Speculative verify windows keep the default limit.
        new_res = _scoped_vmem_bytes(scfg)
        declared = _declared_vmem_bytes(scc["body"])
        assert new_res is not None
        assert declared <= new_res <= expected_default
        if launch_window > 1:
            assert new_res == expected_default
        if new_res < expected_default:
            any_reduced = True

        # CostEstimate must be populated with positive flop/transcendental/byte terms.
        assert "cost_estimate" in scc
        ce = scc["cost_estimate"]
        assert ce["flops"] > 0
        assert ce["transcendentals"] > 0
        assert ce["bytes_accessed"] > 0

    assert any_reduced


@pytest.fixture
def decode_config():
    conv_dim = 2 * 2 * 128 + 8 * 128
    return gdn_config.GDNConfig(
        mode=gdn_config.GDNMode.BATCHED,
        dtypes=gdn_config.Dtypes(
            act_in=jnp.bfloat16,
            act_out=jnp.bfloat16,
            compute=jnp.float32,
            recurrent_state=jnp.float32,
            conv_state=jnp.float32,
        ),
        batch_size=64,
        dim_size=conv_dim,
        kernel_size=4,
        tile_size=4,
        num_kq_heads=2,
        num_v_heads=8,
        kq_head_dim=128,
        v_head_dim=128,
    )


def test_gdn_v3_scoped_vmem_and_cost_unit_properties(decode_config):
    cfg_decode = decode_config
    conv_dim = cfg_decode.dim_size
    default_cap = int(128 * _MIB * gdn_config.GDNConfig.DEFAULT_VMEM_FRACTION)
    limit = gdn_tiling.get_scoped_vmem_limit_bytes(
        cfg_decode,
        conv_state_dim_size=conv_dim,
        num_lanes=128,
        num_sublanes=8,
        default_limit_bytes=default_cap,
    )
    assert limit < default_cap
    assert limit % _MIB == 0

    # Small cap clamps cleanly without exceeding default_limit_bytes.
    tight_cap = 4 * _MIB
    assert (
        gdn_tiling.get_scoped_vmem_limit_bytes(
            cfg_decode,
            conv_state_dim_size=conv_dim,
            num_lanes=128,
            num_sublanes=8,
            default_limit_bytes=tight_cap,
        )
        == tight_cap
    )

    # Verify windows keep the default limit; Mosaic outgrows the estimate.
    windowed = replace(cfg_decode, window_size=4)
    assert (
        gdn_tiling.get_scoped_vmem_limit_bytes(
            windowed,
            conv_state_dim_size=conv_dim,
            num_lanes=128,
            num_sublanes=8,
            default_limit_bytes=default_cap,
        )
        == default_cap
    )

    ce_no_bias = gdn_tiling.get_cost_estimate(
        cfg_decode,
        num_seqs=64,
        conv_state_dim_size=conv_dim,
        has_conv_bias=False,
    )
    ce_with_bias = gdn_tiling.get_cost_estimate(
        cfg_decode,
        num_seqs=64,
        conv_state_dim_size=conv_dim,
        has_conv_bias=True,
    )
    assert ce_with_bias.flops > ce_no_bias.flops
    assert ce_with_bias.bytes_accessed > ce_no_bias.bytes_accessed


def test_gdn_cost_uses_python_integers_before_arithmetic(decode_config):
    # 2,147,483,648 elements in the recurrent state alone.
    cfg = replace(
        decode_config,
        mode=gdn_config.GDNMode.PER_SEQ,
        attention_mode=gdn_config.AttentionMode.KDA,
        batch_size=np.int32(64),
        tile_size=np.int32(64),
        window_size=np.int32(1),
        dim_size=np.int32(1_048_576),
        kernel_size=np.int32(4),
        num_kq_heads=np.int32(32),
        num_v_heads=np.int32(128),
        kq_head_dim=np.int32(4096),
        v_head_dim=np.int32(4096),
    )
    cost = gdn_tiling.get_cost_estimate(
        cfg, np.int32(32), conv_state_dim_size=np.int32(1_048_576)
    )
    assert cost.bytes_accessed > 2**31
    assert cost.flops > 2**31
    for field in (cost.flops, cost.transcendentals, cost.bytes_accessed):
        assert type(field) is int and field > 0


def test_gdn_per_seq_cost_bounds_ragged_tiles_and_checkpoint_traffic(decode_config):
    cfg = replace(decode_config, mode=gdn_config.GDNMode.PER_SEQ, tile_size=64)
    dim = cfg.dim_size
    slot_bytes = (3 * dim + 8 * 128 * 128) * 4
    one_seq = gdn_tiling.get_cost_estimate(cfg, 1, dim)
    ragged = gdn_tiling.get_cost_estimate(cfg, 32, dim)
    checkpointed = gdn_tiling.get_cost_estimate(
        cfg, 32, dim, has_prefill_checkpoint=True
    )
    assert ragged.flops == 32 * one_seq.flops
    assert ragged.transcendentals == 32 * one_seq.transcendentals
    assert ragged.bytes_accessed - one_seq.bytes_accessed == 2 * 31 * slot_bytes
    assert checkpointed.flops == 2 * ragged.flops
    assert checkpointed.bytes_accessed - ragged.bytes_accessed == 32 * slot_bytes


def test_gdn_speculative_cost_bounds_mixed_decode_state_reads(decode_config):
    # 19 ordinary decodes and one four-token verify window.
    cfg = replace(decode_config, batch_size=23, window_size=4)
    dim = cfg.dim_size
    slot_bytes = (3 * dim + 8 * 128 * 128) * 4
    cost = gdn_tiling.get_cost_estimate(cfg, 20, dim)
    assert cost.bytes_accessed >= (20 + 23) * slot_bytes
    assert cost.flops >= 20 * 4 * 7 * 8 * 128 * 128


def _traced_work(jaxpr) -> tuple[int, int]:
    """Transcendental elements and flops of a jaxpr's floating-point math.

    Matches the counting in gdn_tiling.get_cost_estimate: 2 per dot_general
    multiply-add, 1 per output element of an add, sub, mul or div, and 1 per
    element a reduce_sum folds away. Integer, comparison, select, max, min and
    neg ops add nothing.
    """
    transcendentals = flops = 0
    for eqn in jaxpr.eqns:
        out = eqn.outvars[0].aval if eqn.outvars else None
        dtype = getattr(out, "dtype", None)
        if dtype is not None and jnp.issubdtype(dtype, jnp.floating):
            size = math.prod(out.shape)
            name = eqn.primitive.name
            if name in _TRANSCENDENTAL_PRIMITIVES:
                transcendentals += size
            elif name == "dot_general":
                (contracting, _), _ = eqn.params["dimension_numbers"]
                lhs_shape = eqn.invars[0].aval.shape
                flops += 2 * size * math.prod(lhs_shape[d] for d in contracting)
            elif name in _ELEMENTWISE_FLOP_PRIMITIVES:
                flops += size
            elif name == "reduce_sum":
                flops += math.prod(eqn.invars[0].aval.shape) - size
        for inner in _subjaxprs(eqn):
            inner_trans, inner_flops = _traced_work(inner)
            transcendentals += inner_trans
            flops += inner_flops
    return transcendentals, flops


def _tile_config(mode, attention_mode, n_kq, n_v, tile_size, window_size=1, **kw):
    batched = mode == gdn_config.GDNMode.BATCHED
    return gdn_config.GDNConfig(
        mode=mode,
        dtypes=gdn_config.Dtypes(
            act_in=jnp.float32,
            act_out=jnp.bfloat16,
            compute=jnp.float32,
            recurrent_state=jnp.float32,
            conv_state=jnp.float32,
        ),
        batch_size=tile_size * window_size if batched else tile_size,
        dim_size=(2 * n_kq + n_v) * 128,
        kernel_size=4,
        tile_size=tile_size,
        num_kq_heads=n_kq,
        num_v_heads=n_v,
        kq_head_dim=128,
        v_head_dim=128,
        attention_mode=attention_mode,
        window_size=window_size,
        **kw,
    )


def _trace_tile_math(cfg) -> tuple[int, int]:
    """Trace the conv, SiLU and recurrence math one kernel tile executes."""
    seqs, chunk = cfg.seq_tile_size, cfg.chunk_size
    n_kq, n_v, d = cfg.num_kq_heads, cfg.num_v_heads, cfg.kq_head_dim
    lanes = gdn_tiling.align_to(n_v, 128)
    # The recurrent path takes an extra unit axis before the head dim.
    unit = (1,) if cfg.use_recurrent else ()

    def act(heads, width):
        return _sds((seqs, heads, chunk, *unit, width), jnp.float32)

    q = act(n_kq, d)
    gate = act(n_v, d) if cfg.is_kda else act(1, lanes)
    args = (
        _sds((seqs,), jnp.int32),
        _sds((seqs, chunk + cfg.prev_kernel_size, 1, cfg.dim_size), jnp.float32),
        _sds((cfg.kernel_size, 1, cfg.dim_size), jnp.float32),
        q,
        q,
        act(n_v, d),
        act(1, lanes),
        gate,
        _sds((seqs, n_v, d, d), jnp.float32),
        _sds((n_v,) if cfg.is_kda else (lanes,), jnp.float32),
        _sds((n_v, d) if cfg.is_kda else (lanes,), jnp.float32),
    )
    module = compute_kda if cfg.is_kda else compute_gdn
    if cfg.use_recurrent:
        compute = module.recurrent_kda if cfg.is_kda else module.recurrent_gdn
        names = ("q_compact", "k_compact", "v_compact", "b_compact", "a_compact")
    else:
        compute = module.chunked_kda if cfg.is_kda else module.chunked_gdn
        names = ("q_large", "k_large", "v_large", "b_large", "a_large")

    def tile(real_sizes, lhs, conv_weight, q, k, v, b, a, state, a_log, dt_bias):
        conv_out, _ = compute_conv1d.causal_conv1d(
            real_sizes, lhs, conv_weight, None, cfg
        )
        return jax.nn.silu(conv_out), compute(
            real_sizes=real_sizes,
            state_prev=state,
            a_log=a_log,
            dt_bias=dt_bias,
            cfg=cfg,
            **dict(zip(names, (q, k, v, b, a), strict=True)),
        )

    with _abstract_tpu("TPU v6 lite"):
        return _traced_work(jax.make_jaxpr(tile)(*args).jaxpr)


@pytest.mark.parametrize(
    "mode,attention_mode,n_kq,n_v,tile_size,window_size,options",
    [
        # One 64-row KDA prefill tile: 16-row sub-blocks, 6 off-diagonal pairs.
        pytest.param("PER_SEQ", _KDA, 8, 8, 64, 1, {}, id="kda-c64"),
        # 32 heads shrink the sub-block to 8 rows.
        pytest.param("PER_SEQ", _KDA, 32, 32, 64, 1, {}, id="kda-32h-c64"),
        pytest.param("PER_SEQ", _KDA, 32, 32, 16, 1, {}, id="kda-32h-c16"),
        # A chunk shorter than the sub-block clips the block.
        pytest.param("PER_SEQ", _KDA, 8, 8, 8, 1, {}, id="kda-c8"),
        pytest.param(
            "PER_SEQ", _KDA, 8, 8, 16, 1, {"gate_lower_bound": -5.0}, id="kda-bounded"
        ),
        pytest.param(
            "PER_SEQ", _KDA, 8, 8, 16, 1, {"beta_scale": 2.0}, id="kda-beta-scale"
        ),
        pytest.param("PER_SEQ", _GDN, 2, 8, 64, 1, {}, id="gdn-c64"),
        pytest.param("BATCHED", _KDA, 8, 8, 4, 1, {}, id="kda-decode"),
        pytest.param("BATCHED", _KDA, 32, 32, 1, 4, {}, id="kda-verify"),
        pytest.param(
            "BATCHED",
            _KDA,
            13,
            13,
            1,
            3,
            {"gate_lower_bound": -5.0, "beta_scale": 2.0},
            id="kda-bounded-verify",
        ),
        pytest.param("BATCHED", _GDN, 2, 8, 4, 1, {}, id="gdn-decode"),
        pytest.param("BATCHED", _GDN, 16, 64, 2, 4, {}, id="gdn-verify"),
        # 200 value heads pad b and the GDN gate to two lane tiles.
        pytest.param("BATCHED", _GDN, 2, 200, 2, 2, {}, id="gdn-two-lane-tiles"),
    ],
)
def test_gdn_cost_covers_traced_tile_math(
    mode, attention_mode, n_kq, n_v, tile_size, window_size, options
):
    cfg = _tile_config(
        gdn_config.GDNMode[mode],
        attention_mode,
        n_kq,
        n_v,
        tile_size,
        window_size,
        **options,
    )
    transcendentals, flops = _trace_tile_math(cfg)
    # One tile of one sequence group: the launch runs exactly this math.
    cost = gdn_tiling.get_cost_estimate(cfg, cfg.seq_tile_size, cfg.dim_size)
    assert cost.transcendentals == transcendentals
    assert cost.flops == flops


@pytest.mark.parametrize(
    "attention_mode,n_kq,n_v,gate_lanes,conv_weight_dtype,conv_bias",
    [
        pytest.param(_GDN, 2, 8, 128, jnp.float32, False, id="gdn"),
        pytest.param(_KDA, 8, 8, 1024, jnp.float32, False, id="kda"),
        # The wrapper casts conv weight and bias to kernel_conv_weight_dtype.
        pytest.param(_GDN, 2, 8, 128, jnp.bfloat16, True, id="gdn-bf16-conv"),
    ],
)
def test_gdn_cost_counts_lane_padded_dma_operands(
    attention_mode, n_kq, n_v, gate_lanes, conv_weight_dtype, conv_bias
):
    tokens = num_seqs = 16
    (qkv, b, a, conv, rec, conv_w, a_log, dt_bias, *metadata, _) = (
        _make_gdn_abstract_inputs(
            n_kq=n_kq,
            n_v=n_v,
            tokens=tokens,
            max_seqs=num_seqs,
            attention_mode=attention_mode,
        )
    )

    bias = _sds((qkv.shape[-1],), jnp.bfloat16) if conv_bias else None

    def fn(qkv, b, a, conv, rec, conv_w, bias, a_log, dt_bias, *metadata):
        return gdn_wrapper.fused_conv1d_gdn(
            qkv,
            b,
            a,
            (conv, rec),
            conv_w,
            bias,
            a_log,
            dt_bias,
            *metadata,
            n_kq=n_kq,
            n_v=n_v,
            d_k=128,
            d_v=128,
            kernel_size=4,
            attention_mode=attention_mode,
            batched_only=True,
            kernel_conv_weight_dtype=jnp.dtype(conv_weight_dtype),
            use_cost_estimate=True,
        )

    with _abstract_tpu("TPU v6 lite"):
        jaxpr = jax.make_jaxpr(fn)(
            qkv, b, a, conv, rec, conv_w, bias, a_log, dt_bias, *metadata
        )
    (launch,) = _eqns(jaxpr.jaxpr, "pallas_call")
    avals = [v.aval for v in launch.invars]
    # qkv, b and a reach the kernel as [tokens, 1, lanes] HBM operands. The
    # conv weight, optional bias, A_log and dt_bias are the last operands.
    acts = [x for x in avals if x.ndim == 3 and x.shape[:2] == (tokens, 1)]
    assert [x.shape[-1] for x in acts[1:]] == [128, gate_lanes]
    weights = avals[-4:] if conv_bias else avals[-3:]
    assert weights[0].dtype == conv_weight_dtype
    dma_bytes = sum(x.size * x.dtype.itemsize for x in (*acts, *weights))
    out = launch.outvars[0].aval
    dma_bytes += out.size * out.dtype.itemsize
    # Each live decode reads one state slot and writes one checkpoint.
    slot_bytes = 3 * conv.shape[-1] * 4 + n_v * 128 * 128 * 4
    cost = launch.params["cost_estimate"]
    assert cost.bytes_accessed == dma_bytes + 2 * num_seqs * slot_bytes


@pytest.mark.tpu_test
@pytest.mark.parametrize(
    "lengths,batched,num_spec_tokens,attention_mode",
    [
        pytest.param([1] * 8, 8, 0, gdn_config.AttentionMode.GDN, id="batched"),
        pytest.param([1, 2, 3, 58], 0, 0, gdn_config.AttentionMode.GDN, id="per-seq"),
        pytest.param([1, 4, 3], 2, 3, gdn_config.AttentionMode.GDN, id="spec-verify"),
        pytest.param([1, 2, 5], 1, 0, gdn_config.AttentionMode.KDA, id="kda"),
        pytest.param(
            [1, 4, 3], 2, 3, gdn_config.AttentionMode.KDA, id="kda-spec-verify"
        ),
    ],
)
def test_gdn_v3_scoped_vmem_tpu_parity(
    lengths, batched, num_spec_tokens, attention_mode
):
    _require_tpu()
    n_v, d_k, d_v, kernel_size = 8, 128, 128, 4
    is_kda = attention_mode == gdn_config.AttentionMode.KDA
    n_kq = n_v if is_kda else 2
    dim = 2 * n_kq * d_k + n_v * d_v
    rng = np.random.default_rng(0)
    tokens = sum(lengths)
    num_seqs = len(lengths)
    window = num_spec_tokens + 1
    num_slots = 1 + num_seqs * window
    seq_lens = jnp.asarray(lengths, dtype=jnp.int32) + 9
    query_start_loc = jnp.asarray(np.cumsum([0, *lengths]), dtype=jnp.int32)
    state_indices = jnp.arange(1, num_slots, window, dtype=jnp.int32)
    distribution = jnp.array([batched, batched, num_seqs], dtype=jnp.int32)
    read_offsets = (
        jnp.arange(num_seqs, dtype=jnp.int32) % window if num_spec_tokens else None
    )

    def random(shape, dtype):
        return jnp.asarray(rng.normal(scale=0.1, size=shape), dtype)

    qkv = random((tokens, dim), jnp.bfloat16)
    b = random((tokens, n_v), jnp.bfloat16)
    gate_shape = (n_v, d_k) if is_kda else (n_v,)
    a = random((tokens, int(np.prod(gate_shape))), jnp.bfloat16)
    states = (
        random((num_slots, kernel_size - 1, dim), jnp.float32),
        random((num_slots, n_v, d_k, d_v), jnp.float32),
    )
    conv_weight = random((dim, 1, kernel_size), jnp.bfloat16)
    a_log = random((n_v,), jnp.float32)
    dt_bias = random(gate_shape, jnp.float32)

    def run(scoped):
        return gdn_wrapper.fused_conv1d_gdn(
            qkv,
            b,
            a,
            jax.tree.map(lambda x: x.copy(), states),
            conv_weight,
            None,
            a_log,
            dt_bias,
            query_start_loc,
            state_indices,
            distribution,
            seq_lens,
            read_offsets,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            kernel_size=kernel_size,
            num_spec_tokens=num_spec_tokens,
            attention_mode=attention_mode,
            use_scoped_vmem_limit=scoped,
            use_cost_estimate=scoped,
        )

    jax.tree.map(_assert_bitwise_equal, run(True), run(False))


def _tpu_launch(
    attention_mode,
    n_kq,
    n_v,
    *,
    tokens,
    num_seqs,
    num_spec_tokens,
    recurrent_dtype,
    shared,
    launch,
    scoped,
    mixed_tile_size=64,
):
    """A fused launch on abstract inputs: dense or pooled state, one segment."""
    qkv, b, a, conv, rec, conv_w, a_log, dt_bias, *_ = _make_gdn_abstract_inputs(
        n_kq=n_kq,
        n_v=n_v,
        tokens=tokens,
        max_seqs=num_seqs,
        attention_mode=attention_mode,
    )
    dim = qkv.shape[-1]
    num_slots = num_seqs * (num_spec_tokens + 1) + 1
    if shared:
        packing = 4 // jnp.dtype(recurrent_dtype).itemsize
        rows = n_v * 128 * 128 // (128 * packing) + 3 * dim // 128
        states = _sds((num_slots, rows, packing, 128), recurrent_dtype)
    else:
        states = (
            _sds((num_slots, *conv.shape[1:]), jnp.float32),
            _sds((num_slots, *rec.shape[1:]), recurrent_dtype),
        )
    seq_meta = _sds((num_seqs,), jnp.int32)
    args = (qkv, b, a, states, conv_w, a_log, dt_bias)
    args += (_sds((num_seqs + 1,), jnp.int32), seq_meta, _sds((3,), jnp.int32))
    args += (seq_meta, seq_meta)

    def fn(qkv, b, a, states, conv_w, a_log, dt_bias, *metadata):
        return gdn_wrapper.fused_conv1d_gdn(
            qkv,
            b,
            a,
            states,
            conv_w,
            None,
            a_log,
            dt_bias,
            *metadata,
            n_kq=n_kq,
            n_v=n_v,
            d_k=128,
            d_v=128,
            kernel_size=4,
            attention_mode=attention_mode,
            num_spec_tokens=num_spec_tokens,
            conv_dtype=jnp.dtype(jnp.float32) if shared else None,
            recurrent_dtype=jnp.dtype(recurrent_dtype) if shared else None,
            batched_only=launch == "batched",
            prefill_only=launch == "per_seq",
            mixed_tile_size=mixed_tile_size,
            use_scoped_vmem_limit=scoped,
            use_cost_estimate=scoped,
        )

    return fn, args


@pytest.mark.tpu_test
@pytest.mark.parametrize(
    "attention_mode,n_kq,n_v,num_spec_tokens,recurrent_dtype,shared,batched_only",
    [
        # The dense KDA verify windows that OOMed with an estimated limit.
        pytest.param(_KDA, 32, 32, 1, jnp.float32, False, True, id="kda-w2"),
        pytest.param(_KDA, 32, 32, 3, jnp.float32, False, True, id="kda-w4"),
        pytest.param(_KDA, 32, 32, 7, jnp.float32, False, True, id="kda-w8"),
        pytest.param(_KDA, 32, 32, 3, jnp.bfloat16, True, True, id="kda-w4-pool"),
        pytest.param(_KDA, 32, 32, 3, jnp.float32, False, False, id="kda-w4-mixed"),
        pytest.param(_GDN, 16, 64, 1, jnp.bfloat16, True, True, id="gdn-w2-pool"),
        pytest.param(_GDN, 16, 32, 7, jnp.float32, False, False, id="gdn-w8-mixed"),
    ],
)
def test_gdn_v3_scoped_vmem_tpu_verify_compiles(
    attention_mode, n_kq, n_v, num_spec_tokens, recurrent_dtype, shared, batched_only
):
    _require_tpu()
    window = num_spec_tokens + 1
    default_limit = gdn_config.GDNConfig.get_vmem_limit_bytes(window)
    for scoped in (False, True):
        fn, args = _tpu_launch(
            attention_mode,
            n_kq,
            n_v,
            tokens=128,
            num_seqs=16,
            num_spec_tokens=num_spec_tokens,
            recurrent_dtype=recurrent_dtype,
            shared=shared,
            launch="batched" if batched_only else "both",
            scoped=scoped,
        )
        calls = _extract_tpu_custom_calls(fn, args)
        assert len(calls) == (1 if batched_only else 2)
        verify_name, verify_cfg = calls[0]
        assert verify_name.endswith(f"_w{window}")
        assert _scoped_vmem_bytes(verify_cfg) == default_limit
        assert ("cost_estimate" in verify_cfg["custom_call_config"]) == scoped
        jax.jit(fn).lower(*args).compile()


# 64-head KDA, a 512-token bucket and 128 sequence slots: past the 32-head,
# 128-token sweep that calibrated SCOPED_VMEM_MARGIN_BYTES. A 64-row FP32
# prefill tile needs more than TPU7x's 51.2 MiB default, so its estimate is
# capped there; that v5e calibration shape runs on v5e only, and the portable
# prefill case uses a 32-row tile.
_LARGE_KDA_BUCKETS = [
    pytest.param(jnp.bfloat16, True, "batched", 64, None, id="decode-bf16-pool"),
    pytest.param(jnp.float32, False, "per_seq", 32, None, id="prefill-fp32-c32"),
    pytest.param(
        jnp.float32,
        False,
        "per_seq",
        64,
        pltpu.ChipVersion.TPU_V5E,
        id="prefill-fp32-c64-v5e",
    ),
]
_DYNAMIC_TILING = "TPU_ENABLE_GDN_DYNAMIC_TILING"


@contextlib.contextmanager
def _static_tiling(monkeypatch):
    """Static tiling with fused_conv1d_gdn's trace cache isolated around it.

    The override goes into the envs module dict, so undoing it leaves no
    attribute that would shadow envs' lazy lookup. Clearing the cache keeps a
    warm trace of the same signature, traced under other tiling, from being
    reused.
    """
    with monkeypatch.context() as mp:
        mp.setitem(vars(envs), _DYNAMIC_TILING, False)
        gdn_wrapper.fused_conv1d_gdn.clear_cache()
        try:
            yield
        finally:
            gdn_wrapper.fused_conv1d_gdn.clear_cache()


def _large_kda_bucket_limits(
    monkeypatch, recurrent_dtype, shared, launch, mixed_tile_size, compile=False
):
    """Flag-off and flag-on launches and limits under static tiling."""
    expected_chunk = mixed_tile_size if launch == "per_seq" else 1
    launches = []
    with _static_tiling(monkeypatch):
        for scoped in (False, True):
            fn, args = _tpu_launch(
                _KDA,
                64,
                64,
                tokens=512,
                num_seqs=128,
                num_spec_tokens=0,
                recurrent_dtype=recurrent_dtype,
                shared=shared,
                launch=launch,
                scoped=scoped,
                mixed_tile_size=mixed_tile_size,
            )
            ((name, cfg),) = _extract_tpu_custom_calls(fn, args)
            chunk = re.search(r"_c(\d+)(?:_|$)", name)
            assert chunk is not None and int(chunk.group(1)) == expected_chunk, name
            launches.append((fn, args, _scoped_vmem_bytes(cfg)))
        default_limit = gdn_config.GDNConfig.get_vmem_limit_bytes()
        (_, _, off_limit), (_, _, on_limit) = launches
        assert off_limit == default_limit
        assert on_limit < default_limit
        if compile:
            for fn, args, _ in launches:
                jax.jit(fn).lower(*args).compile()
    return launches


@pytest.mark.parametrize("device_kind", ["TPU v5 lite", "TPU v6 lite", "TPU7x"])
@pytest.mark.parametrize(
    "recurrent_dtype,shared,launch,mixed_tile_size,only_on", _LARGE_KDA_BUCKETS
)
def test_gdn_v3_scoped_vmem_large_kda_bucket_lowering(
    monkeypatch,
    device_kind,
    recurrent_dtype,
    shared,
    launch,
    mixed_tile_size,
    only_on,
):
    with _abstract_tpu(device_kind):
        if only_on is not None and pltpu.get_tpu_info().chip_version != only_on:
            pytest.skip(f"calibration shape for {only_on}")
        _large_kda_bucket_limits(
            monkeypatch, recurrent_dtype, shared, launch, mixed_tile_size
        )


def test_large_kda_bucket_ignores_a_warm_dynamic_trace(monkeypatch):
    # Dynamic tiling picks a 32-row chunk for this signature. The static v5e
    # calibration case must still lower its own 64-row chunk afterwards.
    with _abstract_tpu("TPU v5 lite"):
        with monkeypatch.context() as mp:
            mp.setitem(vars(envs), _DYNAMIC_TILING, True)
            for scoped in (False, True):
                fn, args = _tpu_launch(
                    _KDA,
                    64,
                    64,
                    tokens=512,
                    num_seqs=128,
                    num_spec_tokens=0,
                    recurrent_dtype=jnp.float32,
                    shared=False,
                    launch="per_seq",
                    scoped=scoped,
                    mixed_tile_size=64,
                )
                ((name, _),) = _extract_tpu_custom_calls(fn, args)
                assert name.endswith("_c32"), name
        _large_kda_bucket_limits(monkeypatch, jnp.float32, False, "per_seq", 64)


@pytest.mark.parametrize("initial", ["0", "1"])
def test_static_tiling_override_leaves_env_lookup_live(monkeypatch, initial):
    monkeypatch.delitem(vars(envs), _DYNAMIC_TILING, raising=False)
    monkeypatch.setenv(_DYNAMIC_TILING, initial)
    with _abstract_tpu("TPU7x"):
        _large_kda_bucket_limits(monkeypatch, jnp.float32, False, "per_seq", 32)
    assert _DYNAMIC_TILING not in vars(envs)
    flipped = "0" if initial == "1" else "1"
    monkeypatch.setenv(_DYNAMIC_TILING, flipped)
    assert getattr(envs, _DYNAMIC_TILING) is (flipped == "1")


@pytest.mark.tpu_test
@pytest.mark.parametrize(
    "recurrent_dtype,shared,launch,mixed_tile_size,only_on", _LARGE_KDA_BUCKETS
)
def test_gdn_v3_scoped_vmem_tpu_large_kda_bucket_compiles(
    monkeypatch, recurrent_dtype, shared, launch, mixed_tile_size, only_on
):
    _require_tpu()
    chip = pltpu.get_tpu_info().chip_version
    if only_on is not None and chip != only_on:
        pytest.skip(f"calibration shape for {only_on}, running on {chip}")
    _large_kda_bucket_limits(
        monkeypatch, recurrent_dtype, shared, launch, mixed_tile_size, compile=True
    )
