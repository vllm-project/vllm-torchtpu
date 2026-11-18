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
"""Tuned batched RPA prefill tiles: the table entry, the heuristic elsewhere and
without the tuned tiles, and the refusal of tiles known to be wrong. Every
fallback to the heuristic's tiles is logged once."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
import pytest

from vllm_torchtpu.kernels.experimental.batched_rpa import configs, wrapper


@pytest.fixture(autouse=True)
def warnings(monkeypatch):
    """The wrapper's warning_once messages, formatted."""
    seen = []
    monkeypatch.setattr(
        wrapper,
        "logger",
        SimpleNamespace(warning_once=lambda msg, *args: seen.append(msg % args)),
    )
    return seen


def _tpu_info(chip_version=pltpu.ChipVersion.TPU_7X, mxu_column_size=256):
    m = MagicMock()
    m.generation = 7
    m.chip_version = chip_version
    m.num_lanes = 128
    m.mxu_column_size = mxu_column_size
    m.vmem_capacity_bytes = 64 * 1024 * 1024
    m.fp8_ops_per_second = 2
    m.bf16_ops_per_second = 1
    m.num_sublanes = 8
    return m


def _blocks(
    total_q_tokens=4352,
    num_q_heads=32,
    dtype_kv=jnp.float8_e4m3fn,
    tuned_tiles=True,
    page_size=128,
    vmem_limit_bytes=64 * 1024 * 1024,
    **chip,
):
    # The PP=8 stage geometry: 32 q heads, 2 KV heads, head_dim 256, fp8 KV, page 128.
    model = configs.ModelConfigs(
        num_q_heads=num_q_heads, num_kv_heads=2, head_dim=256, mask_value=-1e30
    )
    serve = configs.ServingConfigs(
        num_seqs=32,
        page_size=page_size,
        total_q_tokens=total_q_tokens,
        num_page_indices=2048,
        dtype_q=jnp.bfloat16,
        dtype_kv=dtype_kv,
        dtype_out=jnp.bfloat16,
        kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
    )
    with patch(
        "jax.experimental.pallas.tpu.get_tpu_info", return_value=_tpu_info(**chip)
    ):
        return wrapper.calculate_block_sizes(
            model,
            serve,
            vmem_limit_bytes,
            decode_query_size=4,
            tuned_tiles=tuned_tiles,
        )


def _heuristic(**kwargs):
    with patch.object(wrapper, "_TUNED_PREFILL_TILES", {}):
        return _blocks(**kwargs)


_KEY = (pltpu.ChipVersion.TPU_7X, 32, 2, 256, True)


def test_tuned_shape_takes_the_table_tiles():
    dec, pre = _heuristic()
    dec2, pre2 = _blocks()
    assert dec2 == dec
    assert pre2 == configs.BlockSizes(bq_sz=256, bq_c_sz=16, bkv_sz=1024, batch_size=1)
    assert pre2 != pre


@pytest.mark.parametrize("kwargs", [{"num_q_heads": 16}, {"dtype_kv": jnp.bfloat16}])
def test_other_shapes_keep_the_heuristic(kwargs):
    assert _blocks(**kwargs) == _heuristic(**kwargs)


def test_the_table_needs_the_tuned_tiles():
    # A model that does not ask for the tuned tiles gets the default heuristic's
    # tiles at the table's layer shape too.
    _, pre = _blocks(tuned_tiles=False)
    assert pre == configs.BlockSizes(bq_sz=256, bq_c_sz=64, bkv_sz=256, batch_size=1)


def test_table_tiles_clamp_bq_to_tokens():
    with patch.object(wrapper, "_TUNED_PREFILL_TILES", {_KEY: (512, 128, 1024, 1)}):
        _, pre = _blocks(total_q_tokens=100)
    assert pre.bq_sz == 256 and 256 % pre.bq_c_sz == 0 and pre.bkv_sz == 1024


def test_unsafe_tiles_are_refused():
    # 128,16,1024,1 scored 78.01% on GSM8K at this shape (gate 97%).
    with (
        patch.object(wrapper, "_TUNED_PREFILL_TILES", {_KEY: (128, 16, 1024, 1)}),
        pytest.raises(ValueError, match="known to be wrong"),
    ):
        _blocks()


def test_shipped_table_avoids_the_unsafe_tiles():
    assert not set(wrapper._TUNED_PREFILL_TILES.values()) & set(
        wrapper._UNSAFE_PREFILL_TILES
    )


def test_a_clamp_onto_unsafe_tiles_takes_the_heuristic(warnings):
    # On a 128-wide MXU, a call of at most 128 tokens clamps the entry's bq to
    # 128 and bq_c to 16: the tiles that scored 78.01% on GSM8K.
    _, pre = _blocks(total_q_tokens=100, mxu_column_size=128)
    _, heuristic = _heuristic(total_q_tokens=100, mxu_column_size=128)
    assert (pre.bq_sz, pre.bq_c_sz, pre.bkv_sz, pre.batch_size) not in (
        wrapper._UNSAFE_PREFILL_TILES
    )
    assert pre == heuristic
    assert any("known to be wrong" in w for w in warnings), warnings


def test_a_page_size_that_does_not_divide_bkv_takes_the_heuristic(warnings):
    _, pre = _blocks(page_size=2048)
    assert pre == _heuristic(page_size=2048)[1]
    assert any("page size" in w and "2048" in w for w in warnings), warnings


def test_another_chip_takes_the_heuristic_and_logs_its_key(warnings):
    # The table was tuned on TPU7x only.
    _, pre = _blocks(chip_version=pltpu.ChipVersion.TPU_V6E)
    assert pre == _heuristic(chip_version=pltpu.ChipVersion.TPU_V6E)[1]
    assert any("no tuned prefill tiles" in w and "v6e" in w for w in warnings), warnings


def test_tiles_that_do_not_fit_the_vmem_limit_take_the_heuristic(warnings):
    # The entry's tiles need about 44 MiB by the heuristic's VMEM model, over
    # 80% of a 48 MiB limit.
    limit = 48 * 1024 * 1024
    _, pre = _blocks(vmem_limit_bytes=limit)
    assert pre == _heuristic(vmem_limit_bytes=limit)[1]
    assert pre.bkv_sz != 1024
    assert any("do not fit the VMEM limit" in w for w in warnings), warnings


def test_the_table_entry_logs_nothing(warnings):
    _blocks()
    assert not [w for w in warnings if "prefill tiles" in w], warnings
