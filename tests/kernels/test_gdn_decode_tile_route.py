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
"""GDN decode tile size through TPU_MOE_FUSED_EP_V3_CONFIG's gdn_decode_tile_size (the route that
survives the serving recipe): unset changes nothing; the field sets the override; the explicit env
TPU_GDN_DECODE_TILE_SIZE wins; the fused-EP Config accepts the field and refuses a negative one."""
import pytest

from vllm_torchtpu.kernels.fused_moe.v3 import Config
from vllm_torchtpu.kernels.gdn.v3 import wrapper as W


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for k in ("TPU_GDN_SPEC_WINDOW_TILE1", "TPU_GDN_DECODE_TILE_SIZE", "TPU_GDN_DECODE_TILE_OVERRIDE",
              "TPU_MOE_FUSED_EP_V3_CONFIG"):
        monkeypatch.delenv(k, raising=False)


def test_unset_is_no_override():
    assert W._decode_tile_override() == 0


def test_json_field_sets_it(monkeypatch):
    monkeypatch.setenv("TPU_MOE_FUSED_EP_V3_CONFIG",
                       '{"combine_real_tokens": true, "dense_slots": true, "gdn_decode_tile_size": 1}')
    assert W._decode_tile_override() == 1
    monkeypatch.setenv("TPU_MOE_FUSED_EP_V3_CONFIG", '{"combine_real_tokens": true}')
    assert W._decode_tile_override() == 0


def test_env_wins(monkeypatch):
    monkeypatch.setenv("TPU_MOE_FUSED_EP_V3_CONFIG", '{"gdn_decode_tile_size": 1}')
    monkeypatch.setenv("TPU_GDN_DECODE_TILE_SIZE", "2")
    assert W._decode_tile_override() == 2


def test_config_field():
    assert Config(gdn_decode_tile_size=1).gdn_decode_tile_size == 1
    with pytest.raises(ValueError):
        Config(gdn_decode_tile_size=-1)
