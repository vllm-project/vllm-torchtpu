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
"""Scheduling group ids of the chunked StreamIndex top-k pipeline, per layer.

Every layer calls the same jitted kernel with the same shapes, so a layer whose
call hits `jax.jit`'s cache reuses the first layer's trace, ids included. XLA
then sees one group spanning two layers with un-annotated ops in between, and
libtpu 0.0.50 refuses to compile it ("annotation groups with gaps"). Lowered for
TPU on an abstract mesh, so no TPU is needed.
"""

import re

import jax
import jax.numpy as jnp
import pytest
from jax import export
from jax.sharding import AbstractDevice, AbstractMesh, use_abstract_mesh

from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import config, streamindex_topk

_T, _H, _D, _PAGE, _PAGES, _K, _CHUNK = 128, 4, 128, 128, 8, 256, 32
_ARGS = (
    jax.ShapeDtypeStruct((_T, _H, _D), jnp.float32),  # q
    jax.ShapeDtypeStruct((_T, _H), jnp.float32),  # indexer_weights
    jax.ShapeDtypeStruct((_PAGES, _PAGE // 4, 4, 256), jnp.uint8),  # cache_kv
    jax.ShapeDtypeStruct((2,), jnp.int32),  # seq_lens
    jax.ShapeDtypeStruct((_PAGES,), jnp.int32),  # page_indices
    jax.ShapeDtypeStruct((3,), jnp.int32),  # cu_q_lens
    jax.ShapeDtypeStruct((3,), jnp.int32),  # distribution
)


def _layer_group_ids(slots, chunk=_CHUNK):
    """Group ids of each layer's call, each lowered on its own like a per-layer
    `pallas.jax_op`, within one program (one counter reset)."""

    def make_layer(slot):
        def layer(*args):
            return streamindex_topk(
                *args,
                k=_K,
                compression_ratio=1,
                num_kv_pages_per_block=2,
                num_queries_per_block=8,
                enable_early_exit=False,
                chunk_tokens=chunk,
                scheduling_group_slot=slot,
            )

        return layer

    mesh = AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
    config.reset_scheduling_group_ids()
    ids = []
    with use_abstract_mesh(mesh):
        for slot in slots:
            exported = export.export(jax.jit(make_layer(slot)), platforms=("tpu",))(
                *_ARGS
            )
            text = exported.mlir_module()
            ids.append(
                set(map(int, re.findall(r'_scheduling_group_id\s*=\s*"?(\d+)', text)))
            )
    return ids


def test_distinct_slots_give_each_layer_its_own_groups():
    ids = _layer_group_ids([0, 1, 2])
    # 128 tokens in chunks of 32: 4 chunks, 3 groups per layer.
    assert [len(layer) for layer in ids] == [3, 3, 3]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]), ids


def test_without_slots_cached_layers_share_groups():
    """Positive control: the failure the slots exist to prevent."""
    ids = _layer_group_ids([None, None])
    assert ids[0] and ids[0] == ids[1], ids


def test_slot_ids_stay_clear_of_the_counter_and_each_other():
    first = config.slot_scheduling_group_ids(0, 15)
    second = config.slot_scheduling_group_ids(1, 15)
    assert first >= config.SLOT_ID_BASE
    assert second - first == config.MAX_IDS_PER_SLOT
    with pytest.raises(ValueError):
        config.slot_scheduling_group_ids(-1, 1)
    with pytest.raises(ValueError):
        config.slot_scheduling_group_ids(0, config.MAX_IDS_PER_SLOT + 1)
    with pytest.raises(ValueError):
        config.slot_scheduling_group_ids(config.MAX_SCHEDULING_GROUP_SLOT + 1, 1)


@pytest.mark.parametrize(
    "num_tokens, chunk_tokens",
    # Valid chunks (multiples of 4) that cut a large prefill bucket into more
    # than 1024 chunks, the old per-slot limit.
    [(65536, 32), (131072, 64), (262144, 4)],
)
def test_a_slot_holds_every_chunk_of_a_large_bucket(num_tokens, chunk_tokens):
    count = config.num_chunks(num_tokens, chunk_tokens) - 1
    first = config.slot_scheduling_group_ids(0, count)
    assert first == config.SLOT_ID_BASE


def test_slot_ids_fit_in_int32():
    last_slot = config.MAX_SCHEDULING_GROUP_SLOT
    first = config.slot_scheduling_group_ids(last_slot, config.MAX_IDS_PER_SLOT)
    assert first + config.MAX_IDS_PER_SLOT - 1 < 2**31


@pytest.mark.parametrize(
    "num_tokens, chunk_tokens, expected",
    [
        (128, 32, 4),
        (128, None, 1),  # chunking off
        (32, 32, 1),  # the bucket is one chunk
        (16, 32, 1),  # smaller than a chunk
        (96, 64, 1),  # indivisible: one unpipelined pass
    ],
)
def test_only_a_pipelined_call_keeps_its_slot(num_tokens, chunk_tokens, expected):
    assert config.num_chunks(num_tokens, chunk_tokens) == expected
    slot = config.chunked_scheduling_group_slot(5, num_tokens, chunk_tokens)
    assert slot == (5 if expected > 1 else None)
    assert config.chunked_scheduling_group_slot(None, num_tokens, chunk_tokens) is None


def test_unpipelined_calls_have_no_groups_to_share():
    """Why an unchunked call can drop its slot: it carries no group ids."""
    ids = _layer_group_ids([None, None], chunk=_T)
    assert ids == [set(), set()], ids


def test_layer_slot_is_the_layer_index():
    slot = config.layer_scheduling_group_slot
    assert slot("model.layers.7.self_attn.indexer.k_cache") == 7
    # GLM-5.2's MTP layer continues after the target's 78 layers.
    assert slot("model.layers.78.self_attn.indexer.k_cache") == 78
    # The prefix alone decides: building other layers first changes nothing.
    assert slot("model.layers.7.self_attn.indexer.k_cache") == 7


def test_prefixes_without_one_layer_index_get_their_own_slots():
    slot = config.layer_scheduling_group_slot
    no_index = slot("indexer.k_cache")
    two_indices = slot("mtp.0.layers.3.indexer.k_cache")
    assert min(no_index, two_indices) > 1000  # clear of any layer index
    assert no_index != two_indices
    assert slot("indexer.k_cache") == no_index
    assert max(no_index, two_indices) <= config.MAX_SCHEDULING_GROUP_SLOT
