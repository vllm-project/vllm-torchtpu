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
"""Tuned batched RPA prefill tiles: the table entry, the heuristic elsewhere, and
the refusal of tiles known to be wrong."""

from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import pytest

from vllm_torchtpu.kernels.experimental.batched_rpa import configs, wrapper


def _tpu_info():
    m = MagicMock()
    m.generation = 7
    m.chip_version = "v7x"
    m.num_lanes = 128
    m.mxu_column_size = 256
    m.vmem_capacity_bytes = 64 * 1024 * 1024
    m.fp8_ops_per_second = 2
    m.bf16_ops_per_second = 1
    m.num_sublanes = 8
    return m


def _blocks(total_q_tokens=4352, num_q_heads=32, dtype_kv=jnp.float8_e4m3fn):
    # The PP=8 stage geometry: 32 q heads, 2 KV heads, head_dim 256, fp8 KV, page 128.
    model = configs.ModelConfigs(
        num_q_heads=num_q_heads, num_kv_heads=2, head_dim=256, mask_value=-1e30
    )
    serve = configs.ServingConfigs(
        num_seqs=32,
        page_size=128,
        total_q_tokens=total_q_tokens,
        num_page_indices=2048,
        dtype_q=jnp.bfloat16,
        dtype_kv=dtype_kv,
        dtype_out=jnp.bfloat16,
        kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
    )
    with patch("jax.experimental.pallas.tpu.get_tpu_info", return_value=_tpu_info()):
        return wrapper.calculate_block_sizes(
            model, serve, 64 * 1024 * 1024, decode_query_size=4
        )


def _heuristic(**kwargs):
    with patch.object(wrapper, "_TUNED_PREFILL_TILES", {}):
        return _blocks(**kwargs)


def test_tuned_shape_takes_the_table_tiles():
    dec, pre = _heuristic()
    dec2, pre2 = _blocks()
    assert dec2 == dec
    assert pre2 == configs.BlockSizes(bq_sz=256, bq_c_sz=16, bkv_sz=1024, batch_size=1)
    assert pre2 != pre


@pytest.mark.parametrize("kwargs", [{"num_q_heads": 16}, {"dtype_kv": jnp.bfloat16}])
def test_other_shapes_keep_the_heuristic(kwargs):
    assert _blocks(**kwargs) == _heuristic(**kwargs)


def test_table_tiles_clamp_bq_to_tokens():
    with patch.object(
        wrapper, "_TUNED_PREFILL_TILES", {(32, 2, 256, True): (512, 128, 1024, 1)}
    ):
        _, pre = _blocks(total_q_tokens=100)
    assert pre.bq_sz == 256 and 256 % pre.bq_c_sz == 0 and pre.bkv_sz == 1024


def test_unsafe_tiles_are_refused():
    # 128,16,1024,1 scored 78.01% on GSM8K at this shape (j-332a20f2).
    with (
        patch.object(
            wrapper, "_TUNED_PREFILL_TILES", {(32, 2, 256, True): (128, 16, 1024, 1)}
        ),
        pytest.raises(ValueError, match="known to be wrong"),
    ):
        _blocks()


def test_shipped_table_avoids_the_unsafe_tiles():
    assert not set(wrapper._TUNED_PREFILL_TILES.values()) & set(
        wrapper._UNSAFE_PREFILL_TILES
    )
