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
"""TPU_MLA_SEND_LAYOUT_VMEM_CAP, passed to the kernel as
`sparse_ragged_paged_attention(send_vmem_cap=...)`: with it on, the send-layout
decode attention call (`send_world` without the fused exchange) caps its scoped VMEM
(SEND_LAYOUT_VMEM_LIMIT_BYTES, never below its scratch plus headroom) and keeps
its array operands and its send buffer in HBM. With it off, and for calls
without the send layout, the call keeps the default limit and no memory-space
pins. The kernel body is the same either way.

Checked on CPU by lowering `sparse_ragged_paged_attention` for TPU (no device)
and reading the MLA call's tpu_custom_call backend config. The SparseCore
gather, a separate call that cannot be lowered off-TPU, is replaced by zero
buffers of its output shapes; the MLA call only reads them.
"""

import json
import re

import jax
import jax.numpy as jnp
import pytest

from vllm_torchtpu.distributed.dcp import DCP_AXIS_NAME, DCP_LANE_AXIS_NAME
from vllm_torchtpu.kernels.mla.sparse import kernel as sparse_mla

HEADS = 64
FLAG = "TPU_MLA_SEND_LAYOUT_VMEM_CAP"
PLACEMENT = ("input_memory_space_colors", "output_memory_space_colors")
# Lowering off-device has to name the target TPU (Pallas queries its geometry).
TPU7X = jax.sharding.AbstractMesh(
    (1,),
    ("x",),
    abstract_device=jax.sharding.AbstractDevice(
        device_kind="TPU7x", num_cores=1, platform="tpu"
    ),
)
# The 32-token DCP decode call: send layout for 8 ranks, no fused exchange.
SEND_32 = dict(
    tokens=32,
    topk=2048,
    topk_cap=512,
    batch=8,
    fused_window_prep=True,
    decode_row_widths=(128, 256, 384),
    send_world=8,
)
# Calls without the send layout (output and lse returned), one chunk.
LSE = dict(topk=1024, topk_cap=256, batch=16, gather_and_attention_chunk_size=128)


def _stub_gather(
    cache_kv_nope, cache_kv_rope, indices, *, out_size, kv_write=None, **kwargs
):
    per_row = sparse_mla.native_sc_gather.ROPE_PACKING * (
        2 if kwargs.get("packed_out") else 1
    )
    out = (
        jnp.zeros((out_size, 128), jnp.uint32),
        jnp.zeros((out_size // per_row, 128), jnp.uint32),
    )
    return out + ((cache_kv_nope, cache_kv_rope) if kv_write is not None else ())


def _mla_call(monkeypatch, cap, *, tokens, topk, topk_cap, batch, **kwargs):
    """(kernel name, backend config) of the one MLA call in the lowered module."""
    monkeypatch.setattr(
        sparse_mla.native_sc_gather, "dsa_gather_native_sc", _stub_gather
    )
    kwargs["send_vmem_cap"] = cap
    sds = jax.ShapeDtypeStruct
    with jax.sharding.use_abstract_mesh(TPU7X):
        lowered = _lower(sds, tokens, topk, topk_cap, batch, kwargs)
    return _the_mla_call(lowered)


def _the_mla_call(lowered):
    calls = [
        (m.group(1), m.group(2))
        for line in lowered.as_text().splitlines()
        if (
            m := re.search(
                r'backend_config = "(.*?)".*kernel_name = "(MLA-[^"]*)"', line
            )
        )
    ]
    assert len(calls) == 1, [name for _, name in calls]
    config, name = calls[0]
    # MLIR prints the JSON's quotes as \22; the config has no other escapes.
    return name, json.loads(config.replace("\\22", '"'))


def _lower(sds, tokens, topk, topk_cap, batch, kwargs):
    return sparse_mla.sparse_ragged_paged_attention.trace(
        sds((tokens, HEADS, 512), jnp.bfloat16),
        sds((256, 64, 128), jnp.uint32),  # NOPE cache [pages, page_size, 128]
        sds((256, 16, 128), jnp.uint32),  # ROPE cache [pages, page_size / 4, 128]
        sds((tokens, topk), jnp.int32),
        sds((tokens * 4,), jnp.int32),
        sds((tokens + 1,), jnp.int32),
        sds((3,), jnp.int32),
        q_pe=sds((tokens, HEADS, sparse_mla.Q_ROPE_HEAD_DIM), jnp.bfloat16),
        q_rope_table=sds((tokens, 2, 128), jnp.float32),
        attention_kernel_batch_size=batch,
        return_lse=True,
        cache_layout=sparse_mla.SPARSECORE_CACHE_LAYOUT,
        topk_cap=topk_cap,
        is_dcp_mode=True,
        lse_lane_dense=True,
        lse_mask_empty=True,
        decode_body_levers=sparse_mla.DECODE_BODY_DEFAULT,
        **{"lse_in_kernel": True, **kwargs},
    ).lower(lowering_platforms=("tpu",))


def _default_unpinned(config):
    assert config["scoped_memory_configs"] == [
        {"memory_space": 1, "offset": 0, "size": sparse_mla.DEFAULT_VMEM_LIMIT_BYTES}
    ]
    assert not set(PLACEMENT) & set(config["custom_call_config"])


def test_send_layout_call_off_keeps_default(monkeypatch):
    name, config = _mla_call(monkeypatch, False, **SEND_32)
    assert "-xa2a" not in name and "-ov" not in name, name
    _default_unpinned(config)


def test_send_layout_call_on_caps_vmem_and_pins_hbm(monkeypatch):
    name, config = _mla_call(monkeypatch, True, **SEND_32)
    assert "-xa2a" not in name and "-ov" not in name, name
    assert config["scoped_memory_configs"] == [
        {"memory_space": 1, "offset": 0, "size": 32 * 1024 * 1024}
    ]
    call = config["custom_call_config"]
    # Operands 0-3 are the grid bound and the scalar prefetches; 4-8 are q, the
    # gathered NOPE and ROPE KV, q_pe and the q rope table: all in HBM (color 0).
    assert call["input_memory_space_colors"] == [
        {"operand_index": i, "color": 0} for i in range(4, 9)
    ]
    # The attention output (aliased to q) and the send buffer: both in HBM.
    assert call["output_memory_space_colors"] == [
        {"color": 0, "shape_index": [0]},
        {"color": 0, "shape_index": [1]},
    ]


def test_send_layout_flag_changes_placement_only(monkeypatch):
    _, off = _mla_call(monkeypatch, False, **SEND_32)
    _, on = _mla_call(monkeypatch, True, **SEND_32)
    for config in (off, on):
        config.pop("scoped_memory_configs")
        for key in PLACEMENT:
            config["custom_call_config"].pop(key, None)
    # Everything else, the serialized kernel body included, is the same.
    assert on == off


def test_send_layout_cap_rises_with_scratch(monkeypatch):
    # Batch 16 and top-k cap 1024: the call's own VMEM scratch (about 37 MiB)
    # exceeds the 32 MiB cap, so the limit becomes scratch + headroom instead.
    scratch = []

    def spy(scratch_shapes):
        scratch.append(real(scratch_shapes))
        return scratch[-1]

    real = sparse_mla._vmem_scratch_bytes
    monkeypatch.setattr(sparse_mla, "_vmem_scratch_bytes", spy)
    _, config = _mla_call(monkeypatch, True, **dict(SEND_32, topk_cap=1024, batch=16))
    size = config["scoped_memory_configs"][0]["size"]
    assert len(scratch) == 1 and scratch[0] > 32 * 1024 * 1024
    assert size == scratch[0] + sparse_mla.SEND_LAYOUT_VMEM_HEADROOM_BYTES
    assert config["custom_call_config"]["input_memory_space_colors"] == [
        {"operand_index": i, "color": 0} for i in range(4, 9)
    ]


@pytest.mark.parametrize("tokens", [80, 96, 128])
def test_lse_calls_unchanged_with_flag_on(monkeypatch, tokens):
    name, config = _mla_call(monkeypatch, True, tokens=tokens, **LSE)
    assert "-xa2a" not in name and "-ov" not in name, name
    _default_unpinned(config)


def test_out_vmem_call_untouched(monkeypatch):
    # The 64-token -ov call (out_vmem): the same lowering with the flag off and
    # on, its own 32 MiB limit and VMEM outputs.
    ov = dict(
        tokens=64,
        topk=2048,
        topk_cap=512,
        batch=16,
        gather_and_attention_chunk_size=64,
        lse_in_kernel=False,
        out_vmem=32 * 1024 * 1024,
    )
    name, off = _mla_call(monkeypatch, False, **ov)
    _, on = _mla_call(monkeypatch, True, **ov)
    assert name.endswith("-ov"), name
    assert on == off
    assert off["scoped_memory_configs"][0]["size"] == 32 * 1024 * 1024
    assert off["custom_call_config"]["output_memory_space_colors"] == [
        {"color": 1, "shape_index": [0]},
        {"color": 1, "shape_index": [1]},
    ]


def _exchange_call(cap):
    """The 16-token fused-exchange (-xa2a) call with the gathered KV pinned at
    32 MiB, as served: sparse_attention_chunk on pre-gathered KV inside a
    shard_map over the (dcp_lane 2, dcp 8) mesh."""
    tokens, cap_rows, topk, world = 16, 512, 2048, 8
    windows = topk // cap_rows
    mesh = jax.sharding.AbstractMesh(
        (2, world),
        (DCP_LANE_AXIS_NAME, DCP_AXIS_NAME),
        abstract_device=TPU7X.abstract_device,
    )
    spec = jax.sharding.PartitionSpec((DCP_LANE_AXIS_NAME, DCP_AXIS_NAME))
    n = 2 * world
    shapes = (
        ((n, tokens, HEADS, 512), jnp.bfloat16),
        ((n, tokens, HEADS, sparse_mla.Q_ROPE_HEAD_DIM), jnp.bfloat16),
        ((n, tokens, 2, 128), jnp.float32),
        ((n, windows * tokens, cap_rows, 128), jnp.uint32),
        ((n, windows * tokens, cap_rows // 4, 128), jnp.uint32),
        ((n, tokens), jnp.int32),
        ((n,), jnp.int32),
        ((n,), jnp.int32),
    )

    def body(q, q_pe, tab, nope, rope, kv_lens, rank, lane):
        q, q_pe, tab, nope, rope, kv_lens = (
            a[0] for a in (q, q_pe, tab, nope, rope, kv_lens)
        )
        out = sparse_mla.sparse_attention_chunk(
            q,
            nope,
            rope,
            kv_lens,
            None,
            None,
            start_pos=0,
            end_pos=tokens,
            token_end=jnp.int32(tokens),
            sm_scale=1 / 24,
            k_scale=0.1,
            cache_layout=sparse_mla.SPARSECORE_CACHE_LAYOUT,
            small_chunk_tokens=0,
            attention_kernel_batch_size=tokens,
            topk_cap=cap_rows,
            topk_override=topk,
            is_dcp_mode=True,
            q_pe=q_pe,
            lse_lane_dense=True,
            lse_mask_empty=True,
            row_widths=(128, 256, 384),
            row_levers=sparse_mla.DECODE_BODY_DEFAULT,
            q_rope_table=tab,
            lse_fill=True,
            send_world=world,
            pre_gathered=(nope, rope, None),
            send_exchange=(
                DCP_AXIS_NAME,
                DCP_LANE_AXIS_NAME,
                32 * 1024 * 1024,
                ("kv",),
            ),
            send_peer=jnp.stack([rank[0], lane[0]]),
            send_vmem_cap=cap,
        )
        return out[1][None]

    fn = jax.shard_map(
        body, mesh=mesh, in_specs=(spec,) * len(shapes), out_specs=spec, check_vma=False
    )
    sharding = jax.sharding.NamedSharding(mesh, spec)
    with jax.sharding.use_abstract_mesh(mesh):
        lowered = (
            jax.jit(fn)
            .trace(*(jax.ShapeDtypeStruct(s, d, sharding=sharding) for s, d in shapes))
            .lower(lowering_platforms=("tpu",))
        )
    return _the_mla_call(lowered)


def test_fused_exchange_call_untouched():
    name, off = _exchange_call(False)
    _, on = _exchange_call(True)
    assert name.endswith("-xa2a"), name
    assert on == off
    assert off["scoped_memory_configs"][0]["size"] == 32 * 1024 * 1024
    # Its own pins: the gathered NOPE and ROPE KV (operands 4 and 5) in HBM.
    assert off["custom_call_config"]["input_memory_space_colors"] == [
        {"operand_index": 4, "color": 0},
        {"operand_index": 5, "color": 0},
    ]


def test_flag_is_in_the_compile_key(monkeypatch):
    # Every registered env var not in _TPU_COMPILE_ENV_IGNORED is a compile-key
    # factor, so the flag's two values compile into separate caches.
    from vllm_torchtpu.compilation import tpu_compiler

    assert FLAG not in tpu_compiler._TPU_COMPILE_ENV_IGNORED
    monkeypatch.setenv(FLAG, "0")
    off = tpu_compiler._tpu_compile_env_factors()
    monkeypatch.setenv(FLAG, "1")
    on = tpu_compiler._tpu_compile_env_factors()
    assert (off[FLAG], on[FLAG]) == (False, True)
    assert on != off
