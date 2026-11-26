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

from __future__ import annotations

import inspect
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.sharding import PartitionSpec as P
from torch_tpu._internal.pallas import pallas as pallas_impl

from vllm_torchtpu.kernels.mla import dcp_token_parallel as tp
from vllm_torchtpu.kernels.mla.absorb_q import (
    absorb_q,
    absorb_q_halves,
    absorb_q_split,
    absorb_q_split_halves,
)
from vllm_torchtpu.kernels.mla.fused_qb_absorb import (
    fused_qb_rope_absorb_split_halves,
    fused_qb_rope_absorb_split_qpe,
)
from vllm_torchtpu.layers.adapter import dcp_nonidx_qb_mla_op as nonidx_mod
from vllm_torchtpu.layers.adapter import dcp_unified_idx_qb_mla_op as unified_mod
from vllm_torchtpu.layers.adapter import dcp_unified_indexer_mla_op as ubase_mod
from vllm_torchtpu.layers.adapter.dcp_nonidx_qb_mla_op import InBranchQbInputs


def _test_dtype() -> jnp.dtype:
    return jnp.bfloat16 if jax.default_backend() == "tpu" else jnp.float32


def _bitcast_raw(arr: jax.Array) -> np.ndarray:
    if arr.dtype == jnp.bfloat16:
        return np.asarray(jax.lax.bitcast_convert_type(arr, jnp.uint16))
    return np.asarray(jax.lax.bitcast_convert_type(arr, jnp.uint32))


@pytest.mark.parametrize("num_tokens", [1, 4, 8, 16])
def test_should_match_monolithic_absorb_q_for_split_w_uk_t_halves(
    num_tokens: int,
) -> None:
    rng = np.random.default_rng(100 + num_tokens)
    heads, d_nope, rank = 16, 24, 32
    dt = _test_dtype()
    q_nope = jnp.asarray(rng.normal(0.0, 0.7, (num_tokens, heads, d_nope)), dtype=dt)
    w_uk = jnp.asarray(rng.normal(0.0, 0.4, (heads, d_nope, rank)), dtype=dt)
    ref = absorb_q(q_nope, w_uk)
    got = absorb_q_halves(q_nope, w_uk[: heads // 2], w_uk[heads // 2 :])
    ref_raw = _bitcast_raw(ref)
    assert np.array_equal(_bitcast_raw(got), ref_raw)

    # Positive control: flipping one entry in the upper half breaks parity.
    w_hi_bad = w_uk[heads // 2 :].at[0, 0, 0].set(w_uk[heads // 2, 0, 0] + 10.0)
    bad = absorb_q_halves(q_nope, w_uk[: heads // 2], w_hi_bad)
    assert not np.array_equal(_bitcast_raw(bad), ref_raw)


def test_should_match_monolithic_absorb_q_split_for_decode_tokens() -> None:
    rng = np.random.default_rng(7)
    heads, d_nope, rank = 16, 24, 32
    dt = _test_dtype()
    q_nope = jnp.asarray(rng.normal(0.0, 0.5, (12, heads, d_nope)), dtype=dt)
    w_uk = jnp.asarray(rng.normal(0.0, 0.3, (heads, d_nope, rank)), dtype=dt)
    ref_h, ref_t = absorb_q_split(q_nope, w_uk, split_rows=4)
    got_h, got_t = absorb_q_split_halves(
        q_nope, w_uk[: heads // 2], w_uk[heads // 2 :], split_rows=4
    )
    assert np.array_equal(_bitcast_raw(got_h), _bitcast_raw(ref_h))
    assert np.array_equal(_bitcast_raw(got_t), _bitcast_raw(ref_t))


@pytest.mark.parametrize(
    ("lo_shape", "hi_shape", "hi_dtype"),
    [
        ((8, 192, 512), (4, 192, 512), jnp.bfloat16),
        ((6, 192, 512), (6, 192, 512), jnp.bfloat16),
        ((8, 192, 512), (8, 192, 512), jnp.float32),
    ],
)
def test_should_reject_invalid_split_w_uk_t_halves_in_fused_qb(
    lo_shape: tuple[int, int, int],
    hi_shape: tuple[int, int, int],
    hi_dtype: jnp.dtype,
) -> None:
    q_c = jnp.zeros((192, 128), jnp.bfloat16)
    w_qb = jnp.zeros((128, 16 * 256), jnp.float8_e4m3fn)
    w_s = jnp.ones((16 * 256,), jnp.float32)
    cs = jnp.zeros((192, 64), jnp.float32)
    w_lo = jnp.zeros(lo_shape, jnp.bfloat16)
    w_hi = jnp.zeros(hi_shape, hi_dtype)
    with pytest.raises(ValueError, match="Split w_uk_t halves"):
        fused_qb_rope_absorb_split_qpe(q_c, w_qb, w_s, (w_lo, w_hi), cs, 64)
    with pytest.raises(ValueError, match="Split w_uk_t halves"):
        fused_qb_rope_absorb_split_halves(q_c, w_qb, w_s, w_lo, w_hi, cs, 64)


def test_should_accept_optional_w_uk_t_hi_in_in_branch_qb_inputs() -> None:
    t = torch.zeros((2, 2))
    plain = InBranchQbInputs(q_c=t, w_qb=t, s_qb=t, w_uk_t=t, cos_sin=t)
    assert plain.w_uk_t_hi is None
    w_hi = torch.ones((2, 2))
    split = InBranchQbInputs(q_c=t, w_qb=t, s_qb=t, w_uk_t=t, cos_sin=t, w_uk_t_hi=w_hi)
    assert split.w_uk_t_hi is w_hi


@pytest.mark.parametrize("has_lane", [False, True])
def test_should_register_valid_jax_op_schemas_for_split_w_uk_t_builders(
    monkeypatch: pytest.MonkeyPatch, has_lane: bool
) -> None:
    mesh = object()
    layout = tp.TokenParallelLayout(4, 4, 32, split_rows=8, max_pages=4)

    for mod in (nonidx_mod, unified_mod):
        monkeypatch.setattr(mod, "glm_dsa_v7x_active", lambda *_a, **_kw: True)
        monkeypatch.setattr(mod, "get_tensor_model_parallel_world_size", lambda: 4)
        monkeypatch.setattr(
            mod, "dcp_rank_operand", lambda: torch.zeros((1,), dtype=torch.int32)
        )
    monkeypatch.setattr(
        unified_mod,
        "dcp_lane_operand",
        lambda: torch.zeros((1,), dtype=torch.int32) if has_lane else None,
    )

    nonidx_calls: list[tuple[str, int, tuple]] = []
    unified_calls: list[tuple[str, int, tuple]] = []

    class _FakeOp:
        def __init__(self, name: str, fn):
            self.name = name
            self.fn = fn
            self.fake = None

        def register_fake(self, fn):
            self.fake = fn
            return fn

        def __call__(self, *args):
            assert self.fake is not None
            assert len(args) == len(inspect.signature(self.fn).parameters)
            return self.fake(*args)

    def _make_capture(sink):
        def _cap(
            name,
            fn,
            *,
            donate_argnums,
            mesh,
            input_partition_specs,
            output_partition_specs,
        ):
            del donate_argnums, mesh, output_partition_specs
            sig = inspect.signature(fn)
            pallas_impl._verify_signature(sig)  # noqa: SLF001 -- real schema verify
            assert pallas_impl._infer_static_argnums(sig) == ()  # noqa: SLF001
            pallas_impl._get_torch_signature(sig)  # noqa: SLF001
            assert len(sig.parameters) == len(input_partition_specs)
            sink.append((name, len(sig.parameters), input_partition_specs))
            return _FakeOp(name, fn)

        return _cap

    monkeypatch.setattr(nonidx_mod, "pcp_streaming_jax_op", _make_capture(nonidx_calls))
    monkeypatch.setattr(
        nonidx_mod, "get_shared_jax_op", lambda kind, key, build: build(f"p::{kind}")
    )
    monkeypatch.setattr(
        unified_mod, "pcp_streaming_jax_op", _make_capture(unified_calls)
    )
    monkeypatch.setattr(
        unified_mod, "get_shared_jax_op", lambda kind, key, build: build(f"p::{kind}")
    )

    class _FakeLayer:
        kv_cache_quantized_dtype = None
        scale = 0.135
        num_heads = 16
        v_head_dim = 8
        use_sparse = True
        W_UK_T_dcp_qrep_lo = torch.zeros((8, 12, 16), dtype=torch.bfloat16)
        W_UK_T_dcp_qrep_hi = torch.zeros((8, 12, 16), dtype=torch.bfloat16)

    layer = _FakeLayer()
    impl = types.SimpleNamespace(_dcp_mesh=mesh, dcp_size=4, dcp_interleave_size=4)
    nonidx_op = nonidx_mod.build_in_branch_qb_attention_op(
        layer, impl, layout, q_replicated=True
    )
    assert nonidx_op is not None
    assert len(nonidx_calls) == 2
    for _, arity, specs in nonidx_calls:
        assert arity == 19
        assert specs[-1] == P()

    indexer_op = types.SimpleNamespace(
        topk_tokens=8, k_cache=types.SimpleNamespace(prefix="layer0.idx")
    )
    cb = ubase_mod.UnifiedIndexerCallbacks(
        resolve_kv_cache_mode=lambda *a, **kw: ("xla", None),
        scatter_indexer_k=lambda *a, **kw: a[0],
        streamidx_block_sizes=lambda *a, **kw: ((1, 1, 1), (1, 1, 1)),
        dcp_indexer_a2a_for_bucket=lambda *a: False,
        register_indexer_layer=lambda prefix: 0,
    )
    unified_op = unified_mod.build_unified_indexer_in_branch_qb_op(
        layer, impl, indexer_op, layout, q_replicated=True, callbacks=cb
    )
    assert unified_op is not None
    assert len(unified_calls) == 3
    expected_arity = 25 if has_lane else 24
    for _, arity, specs in unified_calls:
        assert arity == expected_arity
        assert specs[-1] == P()
