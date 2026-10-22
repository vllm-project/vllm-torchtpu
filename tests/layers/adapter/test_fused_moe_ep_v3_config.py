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
"""TPU_MOE_FUSED_EP_V3_CONFIG / TPU_MOE_FUSED_EP_V3_TILE_ROWS: unset changes nothing; set, the
named fields (and only those) change; a typo or a refused value raises."""
import dataclasses

import pytest

from vllm_torchtpu.kernels.fused_moe.v3 import Config
from vllm_torchtpu.layers.adapter import fused_moe_ep_v3 as A

SHAPE = dict(hidden=4096, inter=1024, experts_per_shard=64, top_k=10)


def test_unset_is_the_shape_default(monkeypatch):
    monkeypatch.delenv("TPU_MOE_FUSED_EP_V3_CONFIG", raising=False)
    monkeypatch.delenv("TPU_MOE_FUSED_EP_V3_TILE_ROWS", raising=False)
    assert A.served_config(**SHAPE) == Config.for_shape(**SHAPE)
    assert A.tile_rows() == A.TILE_ROWS == 128


def test_overrides_only_named_fields(monkeypatch):
    monkeypatch.setenv("TPU_MOE_FUSED_EP_V3_CONFIG", '{"sorted_rows": false, "weight_slots": 4, "weight_prefetch": 3}')
    got, base = A.served_config(**SHAPE), Config.for_shape(**SHAPE)
    assert got.sorted_rows is False and got.weight_slots == 4 and got.weight_prefetch == 3
    changed = {f.name for f in dataclasses.fields(Config) if getattr(got, f.name) != getattr(base, f.name)}
    assert changed == {"sorted_rows", "weight_slots", "weight_prefetch"}


def test_tile_rows(monkeypatch):
    monkeypatch.setenv("TPU_MOE_FUSED_EP_V3_TILE_ROWS", "64")
    assert A.tile_rows() == 64
    monkeypatch.setenv("TPU_MOE_FUSED_EP_V3_TILE_ROWS", "60")
    with pytest.raises(ValueError):
        A.tile_rows()


@pytest.mark.parametrize("bad", ['{"sorted_row": false}', '[1, 2]', '{"weight_slots": 1}'])
def test_refuses(monkeypatch, bad):
    monkeypatch.setenv("TPU_MOE_FUSED_EP_V3_CONFIG", bad)
    with pytest.raises(ValueError):
        A.served_config(**SHAPE)
