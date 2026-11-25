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
"""SEND_LAYOUT_VMEM_LIMIT_BYTES: the send-layout decode attention call
(`send_world` without the fused exchange) caps its scoped VMEM and keeps its
array operands and its send buffer in HBM; a call without the send layout keeps
the default limit and no memory-space pins.

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

from vllm_torchtpu.kernels.mla.sparse import kernel as sparse_mla

HEADS = 64
# Lowering off-device has to name the target TPU (Pallas queries its geometry).
TPU7X = jax.sharding.AbstractMesh(
    (1,),
    ("x",),
    abstract_device=jax.sharding.AbstractDevice(
        device_kind="TPU7x", num_cores=1, platform="tpu"
    ),
)


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


def _mla_call(monkeypatch, *, tokens, topk, topk_cap, batch, **kwargs):
    """(kernel name, backend config) of the one MLA call in the lowered module."""
    monkeypatch.setattr(
        sparse_mla.native_sc_gather, "dsa_gather_native_sc", _stub_gather
    )
    sds = jax.ShapeDtypeStruct
    with jax.sharding.use_abstract_mesh(TPU7X):
        lowered = _lower(sds, tokens, topk, topk_cap, batch, kwargs)
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
        lse_in_kernel=True,
        decode_body_levers=sparse_mla.DECODE_BODY_DEFAULT,
        **kwargs,
    ).lower(lowering_platforms=("tpu",))


def test_send_layout_call_caps_vmem_and_pins_hbm(monkeypatch):
    # The 32-token DCP decode call: send layout for 8 ranks, no fused exchange.
    name, config = _mla_call(
        monkeypatch,
        tokens=32,
        topk=2048,
        topk_cap=512,
        batch=8,
        fused_window_prep=True,
        decode_row_widths=(128, 256, 384),
        send_world=8,
    )
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


def test_send_layout_cap_rises_with_scratch(monkeypatch):
    # Batch 16 and top-k cap 1024: the call's own VMEM scratch (about 37 MiB)
    # exceeds the 32 MiB cap, so the limit becomes scratch + headroom instead.
    _, config = _mla_call(
        monkeypatch,
        tokens=32,
        topk=2048,
        topk_cap=1024,
        batch=16,
        fused_window_prep=True,
        decode_row_widths=(128, 256, 384),
        send_world=8,
    )
    size = config["scoped_memory_configs"][0]["size"]
    assert 32 * 1024 * 1024 < size < sparse_mla.DEFAULT_VMEM_LIMIT_BYTES
    assert size - sparse_mla.SEND_LAYOUT_VMEM_HEADROOM_BYTES > 32 * 1024 * 1024
    assert config["custom_call_config"]["input_memory_space_colors"] == [
        {"operand_index": i, "color": 0} for i in range(4, 9)
    ]


def test_lse_call_keeps_default_vmem_and_no_pins(monkeypatch):
    # A 128-token call without the send layout (output and lse returned).
    name, config = _mla_call(
        monkeypatch,
        tokens=128,
        topk=1024,
        topk_cap=256,
        batch=16,
        gather_and_attention_chunk_size=128,
    )
    assert "-xa2a" not in name and "-ov" not in name, name
    assert config["scoped_memory_configs"] == [
        {"memory_space": 1, "offset": 0, "size": sparse_mla.DEFAULT_VMEM_LIMIT_BYTES}
    ]
    call = config["custom_call_config"]
    assert "input_memory_space_colors" not in call
    assert "output_memory_space_colors" not in call


@pytest.mark.parametrize("tokens", [80, 96])
def test_other_lse_buckets_unpinned(monkeypatch, tokens):
    _, config = _mla_call(
        monkeypatch,
        tokens=tokens,
        topk=1024,
        topk_cap=256,
        batch=16,
        gather_and_attention_chunk_size=128,
    )
    assert config["scoped_memory_configs"][0]["size"] == (
        sparse_mla.DEFAULT_VMEM_LIMIT_BYTES
    )
    assert "input_memory_space_colors" not in config["custom_call_config"]
