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
"""Scoped VMEM and cost estimate coverage for RPA v3."""

from unittest import mock

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
)

from vllm_torchtpu.kernels.ragged_paged_attention.v3 import kernel as rpa_kernel

_MIB = 1024 * 1024


def _make_rpa_abstract_inputs(
    *,
    n_q: int = 4,
    n_kv: int = 1,
    head_dim: int = 256,
    page_size: int = 256,
    max_model_len: int = 10240,
    tokens: int = 64,
    max_seqs: int = 512,
    q_dtype=jnp.bfloat16,
    kv_dtype=jnp.float8_e4m3fn,
):
    pages_per_seq = -(-max_model_len // page_size)
    kv_shape = rpa_kernel.get_kv_cache_shape(
        max_seqs * pages_per_seq + 1, page_size, n_kv, head_dim, kv_dtype
    )
    return (
        _sds((tokens, n_q, head_dim), q_dtype),
        _sds((tokens, n_kv, head_dim), kv_dtype),
        _sds((tokens, n_kv, head_dim), kv_dtype),
        _sds(kv_shape, kv_dtype),
        _sds((max_seqs,), jnp.int32),
        _sds((max_seqs * pages_per_seq,), jnp.int32),
        _sds((max_seqs + 1,), jnp.int32),
        _sds((3,), jnp.int32),
    )


@pytest.mark.parametrize(
    "device_kind,tpu_version",
    [("TPU7x", 7), ("TPU v6 lite", 6)],
)
@pytest.mark.parametrize(
    "n_q,n_kv,tokens,chunk_prefill_size,sliding_window,soft_cap,skip_kv_update",
    [
        (4, 1, 8, None, None, None, False),
        (4, 1, 64, None, None, None, False),
        (32, 2, 64, 32, None, None, False),
        (4, 1, 64, None, 1024, 50.0, True),
    ],
)
def test_rpa_v3_scoped_vmem_and_cost_lowering(
    device_kind: str,
    tpu_version: int,
    n_q: int,
    n_kv: int,
    tokens: int,
    chunk_prefill_size: int | None,
    sliding_window: int | None,
    soft_cap: float | None,
    skip_kv_update: bool,
):
    args = _make_rpa_abstract_inputs(n_q=n_q, n_kv=n_kv, tokens=tokens)

    def make_fn(
        scoped: bool,
        cost: bool,
        vmem_limit_bytes: int | None = None,
    ):
        def fn(q, k, v, kv_cache, kv_lens, page_indices, cu_q_lens, distribution):
            return rpa_kernel.ragged_paged_attention(
                q,
                k,
                v,
                kv_cache,
                kv_lens,
                page_indices,
                cu_q_lens,
                distribution,
                sm_scale=256**-0.5,
                chunk_prefill_size=chunk_prefill_size,
                sliding_window=sliding_window,
                soft_cap=soft_cap,
                skip_kv_update=skip_kv_update,
                vmem_limit_bytes=vmem_limit_bytes,
                use_scoped_vmem_limit=scoped,
                use_cost_estimate=cost,
            )

        return fn

    with (
        _abstract_tpu(device_kind),
        mock.patch.object(rpa_kernel, "get_tpu_version", return_value=tpu_version),
    ):
        vmem_cap = pltpu.get_tpu_info().vmem_capacity_bytes
        base_calls = _extract_tpu_custom_calls(make_fn(False, False), args)
        scoped_calls = _extract_tpu_custom_calls(make_fn(True, True), args)
        explicit_limit = 42 * _MIB
        override_calls = _extract_tpu_custom_calls(
            make_fn(True, True, vmem_limit_bytes=explicit_limit), args
        )

    expected_launches = 3 if chunk_prefill_size is not None else 2
    assert len(base_calls) == len(scoped_calls) == len(override_calls)
    assert len(base_calls) == expected_launches

    for (bname, bcfg), (sname, scfg), (_, ocfg) in zip(
        base_calls, scoped_calls, override_calls
    ):
        # Mosaic kernel body must be byte-identical.
        assert bname == sname
        bcc = bcfg["custom_call_config"]
        scc = scfg["custom_call_config"]
        assert bcc["body"] == scc["body"]

        # Default reserves full VMEM and omits cost_estimate.
        assert _scoped_vmem_bytes(bcfg) == vmem_cap
        assert "cost_estimate" not in bcc

        # Scoped sizing drops well below full VMEM while covering declared VMEM.
        new_res = _scoped_vmem_bytes(scfg)
        declared = _declared_vmem_bytes(scc["body"])
        assert new_res is not None
        assert declared <= new_res < vmem_cap

        # Explicit vmem_limit_bytes always overrides use_scoped_vmem_limit.
        assert _scoped_vmem_bytes(ocfg) == explicit_limit

        # CostEstimate is populated on every launch.
        assert "cost_estimate" in scc
        ce = scc["cost_estimate"]
        assert ce["flops"] > 0
        assert ce["transcendentals"] > 0
        assert ce["bytes_accessed"] > 0


def test_rpa_v3_bundled_cost_is_independent_of_layer_count():
    n_q, n_kv, head_dim, page_size, tokens, max_seqs, num_layers = (
        4,
        1,
        256,
        256,
        8,
        32,
        2,
    )
    pages_per_seq = 8
    # All requests can reference the same physical pages.
    total_pages = pages_per_seq + 1
    flat_shape = rpa_kernel.get_kv_cache_shape(
        total_pages, page_size, n_kv, head_dim, jnp.float8_e4m3fn
    )
    bundle_shape = (total_pages, num_layers, *flat_shape[1:])
    args = (
        _sds((tokens, n_q, head_dim), jnp.bfloat16),
        _sds((tokens, n_kv, head_dim), jnp.float8_e4m3fn),
        _sds((tokens, n_kv, head_dim), jnp.float8_e4m3fn),
        _sds(bundle_shape, jnp.float8_e4m3fn),
        _sds((), jnp.int32),
        _sds((max_seqs,), jnp.int32),
        _sds((max_seqs * pages_per_seq,), jnp.int32),
        _sds((max_seqs + 1,), jnp.int32),
        _sds((3,), jnp.int32),
    )

    def fn(*fn_args):
        return rpa_kernel.ragged_paged_attention_bundled(
            *fn_args,
            sm_scale=head_dim**-0.5,
            use_scoped_vmem_limit=True,
            use_cost_estimate=True,
        )

    with (
        _abstract_tpu("TPU v6 lite"),
        mock.patch.object(rpa_kernel, "get_tpu_version", return_value=6),
    ):
        vmem_cap = pltpu.get_tpu_info().vmem_capacity_bytes
        calls = _extract_tpu_custom_calls(fn, args)
        single_layer_args = (
            *args[:3],
            _sds((total_pages, 1, *flat_shape[1:]), jnp.float8_e4m3fn),
            *args[4:],
        )
        single_layer_calls = _extract_tpu_custom_calls(fn, single_layer_args)

    assert len(calls) == len(single_layer_calls) == 2
    for (_, cfg), (_, single_layer_cfg) in zip(calls, single_layer_calls):
        res = _scoped_vmem_bytes(cfg)
        assert res is not None and res < vmem_cap
        assert (
            cfg["custom_call_config"]["cost_estimate"]
            == single_layer_cfg["custom_call_config"]["cost_estimate"]
        )


def test_rpa_v3_vmem_and_cost_helper_properties():
    # 7-argument call stays backward compatible (defaults out_dtype=f32).
    est_default = rpa_kernel.get_vmem_estimate_bytes(
        2, 16, 256, 32, 1024, jnp.bfloat16, jnp.float8_e4m3fn
    )
    est_f32 = rpa_kernel.get_vmem_estimate_bytes(
        2, 16, 256, 32, 1024, jnp.bfloat16, jnp.float8_e4m3fn, jnp.float32
    )
    est_bf16 = rpa_kernel.get_vmem_estimate_bytes(
        2, 16, 256, 32, 1024, jnp.bfloat16, jnp.float8_e4m3fn, jnp.bfloat16
    )
    assert est_default == est_f32
    assert est_bf16 < est_f32

    # Soft cap / sliding window bump headroom when logits tiles exceed base margin.
    lim_plain = rpa_kernel.get_scoped_vmem_limit_bytes(
        1,
        16,
        128,
        32,
        1024,
        32,
        1024,
        jnp.bfloat16,
        jnp.bfloat16,
        jnp.bfloat16,
        128 * _MIB,
        has_soft_cap=False,
    )
    lim_capped = rpa_kernel.get_scoped_vmem_limit_bytes(
        1,
        16,
        128,
        32,
        1024,
        32,
        1024,
        jnp.bfloat16,
        jnp.bfloat16,
        jnp.bfloat16,
        128 * _MIB,
        has_soft_cap=True,
    )
    assert lim_capped > lim_plain


def test_rpa_v3_scoped_vmem_fp8_kv_unpack_headroom():
    cap = 128 * _MIB
    # n_kv=1, bkv_csz=10240, hd=256: bkv_x2_ref is 20 MiB for both fp8 and bf16.
    # fp8 adds 1 * 10240 * 256 * 4 = 10 MiB unpack spill headroom.
    bf16_nkv1 = rpa_kernel.get_scoped_vmem_limit_bytes(
        1, 4, 256, 1, 10240, 1, 10240, jnp.bfloat16, jnp.bfloat16, jnp.bfloat16, cap
    )
    fp8_nkv1 = rpa_kernel.get_scoped_vmem_limit_bytes(
        1,
        4,
        256,
        1,
        10240,
        1,
        10240,
        jnp.bfloat16,
        jnp.float8_e4m3fn,
        jnp.bfloat16,
        cap,
    )
    assert fp8_nkv1 - bf16_nkv1 == 10 * _MIB

    # n_kv=2 has the same 20 MiB packed bkv_x2_ref (align_to(2*2, 4) == 4),
    # but pipelines 2 heads so unpack spill headroom doubles to 20 MiB.
    fp8_nkv2 = rpa_kernel.get_scoped_vmem_limit_bytes(
        2,
        4,
        256,
        1,
        10240,
        1,
        10240,
        jnp.bfloat16,
        jnp.float8_e4m3fn,
        jnp.bfloat16,
        cap,
    )
    assert fp8_nkv2 - fp8_nkv1 == 10 * _MIB

    # n_kv=4 caps the pipelined unpack spill term at min(2, n_kv) = 2 heads.
    fp8_nkv4 = rpa_kernel.get_scoped_vmem_limit_bytes(
        4,
        4,
        256,
        1,
        8192,
        1,
        8192,
        jnp.bfloat16,
        jnp.float8_e4m3fn,
        jnp.bfloat16,
        cap,
    )
    scratch_nkv4 = rpa_kernel.get_vmem_estimate_bytes(
        4, 4, 256, 1, 8192, jnp.bfloat16, jnp.float8_e4m3fn, jnp.bfloat16
    )
    expected_nkv4 = (
        (
            scratch_nkv4
            + 2 * 8192 * 256 * 4
            + rpa_kernel.SCOPED_VMEM_MARGIN_BYTES
            + _MIB
            - 1
        )
        // _MIB
    ) * _MIB
    assert fp8_nkv4 == expected_nkv4


@pytest.fixture
def decode_cost_args():
    return dict(
        max_num_tokens=64,
        max_num_seqs=64,
        page_size=256,
        pages_per_seq=40,
        actual_num_q_heads=16,
        actual_num_kv_heads=2,
        actual_head_dim=256,
        q_dtype=jnp.bfloat16,
        kv_dtype=jnp.float8_e4m3fn,
        bq_sz=1,
        bq_csz=1,
        bkv_sz=10240,
        bkv_csz=10240,
        static_q_len=1,
    )


def test_rpa_decode_sliding_window_counts_full_compute_tile(decode_cost_args):
    args = {
        key: np.int32(value) if isinstance(value, int) else value
        for key, value in decode_cost_args.items()
    }
    cost = rpa_kernel.get_cost_estimate(**args, sliding_window=np.int32(1024))
    assert cost.flops == 4 * 64 * 16 * 10240 * 256 > 2**31
    assert cost.transcendentals == 64 * 16 * 10240
    for field in (cost.flops, cost.transcendentals, cost.bytes_accessed):
        assert type(field) is int and field > 0


def test_rpa_shared_prefix_reads_scale_with_active_sequences(decode_cost_args):
    args = decode_cost_args | dict(skip_kv_update=True)
    many = rpa_kernel.get_cost_estimate(**args)
    one = rpa_kernel.get_cost_estimate(
        **(args | dict(max_num_tokens=1, max_num_seqs=1))
    )
    kv_bytes_per_request = 10240 * 2 * 2 * 256
    qo_bytes_per_request = 16 * 256 * (2 + 2)
    assert one.bytes_accessed == kv_bytes_per_request + qo_bytes_per_request
    assert many.bytes_accessed == 64 * one.bytes_accessed
    capped = rpa_kernel.get_cost_estimate(**args, soft_cap=50.0)
    assert capped.transcendentals == 2 * many.transcendentals


def test_rpa_cost_charges_query_dtype_for_q_and_output_dma():
    # 8 one-token decodes, 4 query heads, head_dim 128 and one 128-token page.
    # The output block aliases the query block, so FP32 queries move FP32 Q and
    # O blocks whatever the accumulator dtype.
    shapes = dict(n_q=4, n_kv=1, head_dim=128, page_size=128, max_model_len=128)
    shapes |= dict(tokens=8, max_seqs=8, q_dtype=jnp.float32, kv_dtype=jnp.bfloat16)

    def fn(*inputs):
        return rpa_kernel.ragged_paged_attention(
            *inputs, sm_scale=128**-0.5, use_cost_estimate=True
        )

    with (
        _abstract_tpu("TPU v6 lite"),
        mock.patch.object(rpa_kernel, "get_tpu_version", return_value=6),
    ):
        jaxpr = jax.make_jaxpr(fn)(*_make_rpa_abstract_inputs(**shapes))
    launch = next(_eqns(jaxpr.jaxpr, "pallas_call"))
    # Operand 7 is the query block, aliased to output 0.
    q, out = launch.invars[7].aval, launch.outvars[0].aval
    assert q.dtype == out.dtype == jnp.float32
    qo_bytes = (q.size + out.size) * 4
    assert qo_bytes == 2 * 8 * 4 * 128 * 4

    cost = rpa_kernel.get_cost_estimate(
        max_num_tokens=8,
        max_num_seqs=8,
        page_size=128,
        pages_per_seq=1,
        actual_num_q_heads=4,
        actual_num_kv_heads=1,
        actual_head_dim=128,
        q_dtype=jnp.float32,
        kv_dtype=jnp.bfloat16,
        bq_sz=1,
        bq_csz=1,
        bkv_sz=128,
        bkv_csz=128,
        static_q_len=1,
        skip_kv_update=True,
    )
    kv_read_bytes = 8 * 128 * 2 * 128 * 2
    assert cost.bytes_accessed == kv_read_bytes + qo_bytes


@pytest.mark.parametrize("chunk_prefill_size", [np.int32(8), None])
def test_rpa_prefill_cost_includes_full_cached_prefix(
    decode_cost_args, chunk_prefill_size
):
    args = decode_cost_args | dict(
        max_num_seqs=8, bq_sz=8, bq_csz=8, static_q_len=chunk_prefill_size
    )
    cost = rpa_kernel.get_cost_estimate(**args)
    assert cost.flops >= 4 * 64 * 16 * 10240 * 256
    assert type(cost.flops) is int
    assert type(cost.transcendentals) is int
    assert type(cost.bytes_accessed) is int


@pytest.mark.tpu_test
@pytest.mark.parametrize(
    "kv_dtype,n_kv,long_context,bundled",
    [
        pytest.param(jnp.bfloat16, 1, False, False, id="bf16"),
        pytest.param(jnp.float8_e4m3fn, 1, False, False, id="fp8"),
        pytest.param(jnp.float8_e4m3fn, 1, True, False, id="fp8-long-1-head"),
        pytest.param(jnp.float8_e4m3fn, 2, True, False, id="fp8-long-2-heads"),
        pytest.param(jnp.bfloat16, 1, False, True, id="bundled"),
    ],
)
def test_rpa_v3_scoped_vmem_tpu_parity(kv_dtype, n_kv, long_context, bundled):
    _require_tpu()
    n_q = 4 * n_kv
    rng = np.random.default_rng(1)
    kwargs = {}
    if long_context:
        head_dim, page_size, pages_per_seq = 256, 256, 40
        query_lens, lengths, distribution = [1] * 8, [8192] * 8, [8, 8, 8]
        kwargs["d_block_sizes"] = (1, 10240, 1, 10240)
    else:
        head_dim, page_size, pages_per_seq = 128, 16, 8
        query_lens, lengths, distribution = [1, 8, 7], [32, 48, 40], [1, 2, 3]
        kwargs.update(chunk_prefill_size=8, sliding_window=32, soft_cap=30.0)

    num_seqs, tokens = len(lengths), sum(query_lens)
    total_pages = num_seqs * pages_per_seq
    kv_shape = rpa_kernel.get_kv_cache_shape(
        total_pages, page_size, n_kv, head_dim, kv_dtype
    )

    def random(shape, dtype):
        return jnp.asarray(rng.normal(scale=0.1, size=shape), dtype)

    q = random((tokens, n_q, head_dim), jnp.bfloat16)
    k = random((tokens, n_kv, head_dim), kv_dtype)
    v = random((tokens, n_kv, head_dim), kv_dtype)
    # Share the data pattern across pages while giving each sequence its own
    # physical slots for writeback.
    cache = jnp.broadcast_to(random(kv_shape[1:], kv_dtype), kv_shape)
    page_indices = jnp.arange(total_pages, dtype=jnp.int32)
    kv_lens = jnp.asarray(lengths, dtype=jnp.int32)
    cu_q_lens = jnp.asarray(np.cumsum([0, *query_lens]), dtype=jnp.int32)
    dist = jnp.asarray(distribution, dtype=jnp.int32)
    kernel = rpa_kernel.ragged_paged_attention
    extra = ()
    if bundled:
        kernel = rpa_kernel.ragged_paged_attention_bundled
        cache = jnp.stack((cache, cache), axis=1)
        extra = (jnp.asarray(1, dtype=jnp.int32),)

    def run(scoped):
        return kernel(
            q.copy(),
            k.copy(),
            v.copy(),
            cache.copy(),
            *extra,
            kv_lens,
            page_indices,
            cu_q_lens,
            dist,
            sm_scale=head_dim**-0.5,
            use_scoped_vmem_limit=scoped,
            use_cost_estimate=scoped,
            **kwargs,
        )

    jax.tree.map(_assert_bitwise_equal, run(True), run(False))
