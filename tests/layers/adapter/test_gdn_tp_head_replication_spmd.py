# SPDX-License-Identifier: Apache-2.0
"""Real TP checkpoint loading followed by SPMD unified-pool GDN execution.

TP1 is the numerical reference. Each SPMD shard receives the weights loaded
for that vLLM TP rank, including runtime FP8 postprocessing. Head ownership
in the assertions is derived from V heads, independently of head_geometry.
No TorchTPU native partition ordering or service process is involved.

Run with TPU_SKIP_MDS_QUERY=true:
  .venv/bin/python -m pytest -s \
    tests/layers/adapter/test_gdn_tp_head_replication_spmd.py
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.gdn_pool_layout import derive_pooled_gdn_state_layout
from vllm_torchtpu.kernels import pool_adapters
from vllm_torchtpu.layers.adapter.linear_common import _quantized_matmul_jax
from vllm_torchtpu.layers.core.gdn_attention import \
    run_jax_gdn_attention_pooled_local

pytestmark = pytest.mark.multichip

HEAD_DIM = 128
HIDDEN_SIZE = 128
KERNEL_SIZE = 4
LANES = 256
TOKEN_BYTES = 1024
BLOCK_TOKENS = 256
MANAGER_TOKENS = 4096
SPLIT = MANAGER_TOKENS // BLOCK_TOKENS
# Nontrivial slot order catches accidental sequence-index addressing.
STATE_INDICES = np.array([2, 1], dtype=np.int32)


def _head_columns(num_key_heads, num_value_heads, tp_size, rank, include_z):
    v = np.arange(num_value_heads).reshape(tp_size, -1)[rank]
    k = np.unique(v // (num_value_heads // num_key_heads))
    sizes = [num_key_heads, num_key_heads, num_value_heads, num_value_heads]
    offsets = np.cumsum([0, *sizes]) * HEAD_DIM
    heads = [k, k, v, v] if include_z else [k, k, v]
    columns = np.concatenate([
        offsets[i] + (h[:, None] * HEAD_DIM + np.arange(HEAD_DIM)).ravel()
        for i, h in enumerate(heads)
    ])
    return columns, v


def _checkpoint(num_key_heads, num_value_heads, quantization):
    rng = np.random.default_rng(917)
    sizes = [num_key_heads, num_key_heads, num_value_heads, num_value_heads]
    dtype = (torch.bfloat16 if quantization == "bf16" else torch.float8_e4m3fn)
    qkvz = [
        torch.tensor(rng.integers(-16, 17, (h * HEAD_DIM, HIDDEN_SIZE)) / 16,
                     dtype=dtype) for h in sizes
    ]
    if quantization == "bf16":
        qkvz = [w * 0.125 for w in qkvz]
    conv = torch.tensor(rng.normal(
        0, 0.15, (sum(sizes[:3]) * HEAD_DIM, 1, KERNEL_SIZE)),
                        dtype=torch.bfloat16)
    ba = torch.tensor(rng.normal(0, 0.1, (2 * num_value_heads, HIDDEN_SIZE)),
                      dtype=torch.bfloat16)
    a_log = torch.tensor(rng.normal(-1, 0.1, num_value_heads),
                         dtype=torch.float32)
    dt_bias = torch.tensor(rng.normal(-1, 0.1, num_value_heads),
                           dtype=torch.bfloat16)
    return qkvz, conv, ba, a_log, dt_bias


def _load_weights(make_gdn_attention, checkpoint, num_key_heads,
                  num_value_heads, tp_size, rank, quantization, interleaved):
    layer, _ = make_gdn_attention(tp_size,
                                  rank,
                                  quantization,
                                  num_key_heads=num_key_heads,
                                  num_value_heads=num_value_heads,
                                  gqa_interleaved_layout=interleaved)
    qkvz, conv, ba, a_log, dt_bias = checkpoint
    proj = layer.in_proj_qkvz

    def interleave(segments):
        return torch.cat([
            segment.reshape(num_key_heads, -1, *segment.shape[1:])
            for segment in segments
        ],
                         dim=1).flatten(0, 1)

    def load_segments(param, segments):
        if interleaved:
            param.weight_loader(param, interleave(segments))
        else:
            param.weight_loader(param, torch.cat(segments[:3]), (0, 1, 2))
            param.weight_loader(param, segments[3], 3)

    load_segments(proj.weight, qkvz)
    if quantization != "bf16":
        scale = proj.weight_scale
        if quantization == "fp8_tensor":
            if interleaved:
                scale.weight_loader(scale, torch.tensor(0.125))
            else:
                scale.weight_loader(scale, torch.tensor(0.125), (0, 1, 2))
                scale.weight_loader(scale, torch.tensor(0.25), 3)
        else:
            divisor = 128 if quantization == "fp8_block" else 1
            segments = [
                torch.linspace(0.0625, 0.25, w.shape[0] // divisor)[:, None]
                for w in qkvz
            ]
            load_segments(scale, segments)
        proj.quant_method.process_weights_after_loading(proj)

    layer.conv1d.weight.weight_loader(layer.conv1d.weight, conv)
    if interleaved:
        layer.in_proj_ba.weight.weight_loader(layer.in_proj_ba.weight,
                                              interleave(ba.chunk(2)))
    else:
        for shard, segment in enumerate(ba.chunk(2)):
            layer.in_proj_ba.weight.weight_loader(layer.in_proj_ba.weight,
                                                  segment, shard)
    layer.A_log.weight_loader(layer.A_log, a_log)
    layer.dt_bias.weight_loader(layer.dt_bias, dt_bias)
    layer.process_weights_after_loading(torch.bfloat16)
    params = dict(qkvz=proj.weight,
                  conv=layer.conv1d.weight,
                  ba=layer.in_proj_ba.weight,
                  a_log=layer.A_log,
                  dt_bias=layer.dt_bias)
    if quantization != "bf16":
        params["scale"] = proj.weight_scale
    # float32 round-trip preserves every BF16/FP8 value and avoids relying
    # on NumPy support for torch's BF16 and FP8 tensor dtypes.
    dtypes = {
        torch.bfloat16: jnp.bfloat16,
        torch.float8_e4m3fn: jnp.float8_e4m3fn,
        torch.float32: jnp.float32
    }
    return {
        name: np.asarray(value.detach().float().numpy(),
                         dtype=dtypes[value.dtype])
        for name, value in params.items()
    }


def _state_layout(n_v, conv_dim):
    return derive_pooled_gdn_state_layout(
        ssm_bytes=n_v * HEAD_DIM * HEAD_DIM * 4,
        conv_bytes=(KERNEL_SIZE - 1) * conv_dim * 2,
        token_bytes=TOKEN_BYTES)


def _pack_states(conv, recurrent):
    n_seqs, n_v = recurrent.shape[:2]
    layout = _state_layout(n_v, conv.shape[-1])
    pool = jnp.zeros((3 * SPLIT, BLOCK_TOKENS, 1, 4, LANES), jnp.int8)
    pool = pool_adapters.scatter_region(pool,
                                        recurrent.reshape(
                                            n_seqs, n_v * HEAD_DIM, HEAD_DIM),
                                        STATE_INDICES,
                                        tok0=0,
                                        ntok=layout.ssm_tokens,
                                        split=SPLIT)
    rows = conv.reshape(n_seqs, -1, LANES)
    capacity = layout.conv_tokens * TOKEN_BYTES // (2 * LANES)
    rows = jnp.pad(rows, ((0, 0), (0, capacity - rows.shape[1]), (0, 0)))
    return pool_adapters.scatter_region(pool,
                                        rows,
                                        STATE_INDICES,
                                        tok0=layout.ssm_tokens,
                                        ntok=layout.conv_tokens,
                                        split=SPLIT)


def _unpack_states(pool, n_v, conv_dim):
    layout = _state_layout(n_v, conv_dim)
    recurrent = pool_adapters.gather_region(pool,
                                            STATE_INDICES,
                                            tok0=0,
                                            ntok=layout.ssm_tokens,
                                            split=SPLIT,
                                            out_dtype=jnp.float32,
                                            out_lanes=HEAD_DIM).reshape(
                                                2, n_v, HEAD_DIM, HEAD_DIM)
    conv = pool_adapters.gather_region(pool,
                                       STATE_INDICES,
                                       tok0=layout.ssm_tokens,
                                       ntok=layout.conv_tokens,
                                       split=SPLIT,
                                       out_dtype=jnp.bfloat16)
    conv = conv[:, :(KERNEL_SIZE - 1) * conv_dim // LANES].reshape(
        2, KERNEL_SIZE - 1, conv_dim)
    return conv, recurrent


def _build_spmd(devices, weights, conv, recurrent):
    """The rank axis carries already-loaded tensors, never re-shards heads."""
    mesh = Mesh(np.array(devices), ("tp", ))
    rank_sharding = NamedSharding(mesh, P("tp"))
    weights = jax.tree.map(
        lambda *xs: jax.device_put(np.stack(xs), rank_sharding), *weights)
    pool = jax.jit(
        shard_map(lambda c, r: _pack_states(c[0], r[0])[None],
                  mesh=mesh,
                  in_specs=(P("tp"), P("tp")),
                  out_specs=P("tp"),
                  check_rep=False))(jax.device_put(np.stack(conv),
                                                   rank_sharding),
                                    jax.device_put(np.stack(recurrent),
                                                   rank_sharding))

    @jax.jit
    @functools.partial(shard_map,
                       mesh=mesh,
                       in_specs=(P("tp"), P("tp"), P(), P(), P(), P()),
                       out_specs=(P("tp"), P("tp")),
                       check_rep=False)
    def step(pool, weights, hidden, starts, lengths, distribution):
        w = jax.tree.map(lambda x: x[0], weights)
        n_v = w["a_log"].size
        conv_dim = w["conv"].shape[0]
        n_kq = (conv_dim - n_v * HEAD_DIM) // (2 * HEAD_DIM)
        if "scale" in w:
            projected = _quantized_matmul_jax(hidden, w["qkvz"].T, w["scale"])
        else:
            projected = hidden @ w["qkvz"].T
        ba = hidden @ w["ba"].T
        b, a = jnp.split(ba, 2, axis=-1)
        new_pool, output = run_jax_gdn_attention_pooled_local(
            mixed_qkv=projected[:, :conv_dim],
            b=b,
            a=a,
            recurrent_state=pool[0],
            conv_weight=w["conv"],
            conv_bias=None,
            A_log=w["a_log"],
            dt_bias=w["dt_bias"],
            query_start_loc=starts,
            state_indices=STATE_INDICES,
            distribution=distribution,
            seq_lens=lengths,
            n_kq=n_kq,
            n_v=n_v,
            d_k=HEAD_DIM,
            d_v=HEAD_DIM,
            kernel_size=KERNEL_SIZE,
            pool_block_tokens=MANAGER_TOKENS)
        conv_out, recurrent_out = _unpack_states(new_pool, n_v, conv_dim)
        outputs = dict(projection=projected,
                       ba=ba,
                       output=output.reshape(hidden.shape[0], n_v, HEAD_DIM),
                       conv=conv_out,
                       recurrent=recurrent_out)
        return new_pool[None], jax.tree.map(lambda x: x[None], outputs)

    def run(pool, hidden, starts, lengths, distribution):
        return step(pool, weights, hidden, starts, lengths, distribution)

    return run, pool


# The last case has the same local Q/K=1, V=2 geometry and replication=4
# as the full Qwen3.8 K=16, V=128 model under TP64, on eight physical devices.
@pytest.mark.parametrize(
    "num_key_heads,num_value_heads,tp_size", [(4, 32, 4), (4, 32, 8),
                                              (2, 32, 8), (2, 16, 8)],
    ids=["shard", "replicate2", "replicate4", "replicate4_v2"])
@pytest.mark.parametrize("quantization",
                         ["bf16", "fp8_tensor", "fp8_channel", "fp8_block"])
@pytest.mark.parametrize("interleaved", [False, True],
                         ids=["contiguous", "interleaved"])
def test_loaded_tp_gdn_matches_tp1_spmd(make_gdn_attention, monkeypatch,
                                        num_key_heads, num_value_heads,
                                        tp_size, quantization, interleaved):
    devices = jax.local_devices()
    if len(devices) < tp_size or devices[0].platform != "tpu":
        pytest.skip(f"requires {tp_size} local TPU devices")
    monkeypatch.setenv("TPU_GDN_CONV_QK_PAIR_LAYOUT", "0")
    monkeypatch.delenv("REQUANTIZE_BLOCK_SIZE", raising=False)
    monkeypatch.setenv("REQUANTIZE_WEIGHT_DTYPE", "float8_e4m3fn")
    checkpoint = _checkpoint(num_key_heads, num_value_heads, quantization)
    rng = np.random.default_rng(39)
    conv_dim = (2 * num_key_heads + num_value_heads) * HEAD_DIM
    conv = rng.normal(0, 0.1,
                      (2, KERNEL_SIZE - 1, conv_dim)).astype(jnp.bfloat16)
    recurrent = rng.normal(
        0, 0.02, (2, num_value_heads, HEAD_DIM, HEAD_DIM)).astype(np.float32)
    runners = []
    pools = []
    for size in (1, tp_size):
        weights, conv_shards, recurrent_shards = [], [], []
        for rank in range(size):
            weights.append(
                _load_weights(make_gdn_attention, checkpoint, num_key_heads,
                              num_value_heads, size, rank, quantization,
                              interleaved))
            columns, v = _head_columns(num_key_heads, num_value_heads, size,
                                       rank, False)
            conv_shards.append(conv[..., columns])
            recurrent_shards.append(recurrent[:, v])
        run, pool = _build_spmd(devices[:size], weights, conv_shards,
                                recurrent_shards)
        runners.append(run)
        pools.append(pool)

    # First prefill mixes a fresh slot (must discard nonzero initial state)
    # and a cached prefix. Then consume both states in another prefill and
    # three consecutive decode calls, without resetting the pools.
    totals = np.array([0, 13], dtype=np.int32)
    for phase, lengths in [("prefill", [64, 64]), ("continuation", [64, 64]),
                           ("decode1", [1, 1]), ("decode2", [1, 1]),
                           ("decode3", [1, 1])]:
        totals += np.array(lengths, dtype=np.int32)
        hidden = rng.normal(0, 0.5,
                            (sum(lengths), HIDDEN_SIZE)).astype(jnp.bfloat16)
        starts = np.array([0, lengths[0], sum(lengths)], dtype=np.int32)
        distribution = np.array([2 if lengths[0] == 1 else 0, 2, 2],
                                dtype=np.int32)
        outputs = []
        for i, run in enumerate(runners):
            pools[i], result = run(pool=pools[i],
                                   hidden=hidden,
                                   starts=starts,
                                   lengths=totals,
                                   distribution=distribution)
            outputs.append(
                jax.tree.map(lambda x: np.asarray(x, np.float32), result))
        reference = jax.tree.map(lambda x: x[0], outputs[0])
        actual = outputs[1]
        max_errors = {name: 0.0 for name in actual}
        for rank in range(tp_size):
            qkvz_columns, v = _head_columns(num_key_heads, num_value_heads,
                                            tp_size, rank, True)
            conv_columns, _ = _head_columns(num_key_heads, num_value_heads,
                                            tp_size, rank, False)
            expected = dict(projection=reference["projection"][:,
                                                               qkvz_columns],
                            ba=reference["ba"][:, np.r_[v,
                                                        v + num_value_heads]],
                            output=reference["output"][:, v],
                            conv=reference["conv"][..., conv_columns],
                            recurrent=reference["recurrent"][:, v])
            for name, want in expected.items():
                got = actual[name][rank]
                np.testing.assert_allclose(
                    got,
                    want,
                    atol=1e-5,
                    rtol=1e-4,
                    err_msg=
                    f"{quantization} TP{tp_size} {phase} rank{rank} {name}")
                max_errors[name] = max(max_errors[name],
                                       float(np.max(np.abs(got - want))))
        print(
            f"{quantization} K={num_key_heads} V={num_value_heads} TP={tp_size} {phase}: "
            f"max_abs={max_errors}",
            flush=True)
