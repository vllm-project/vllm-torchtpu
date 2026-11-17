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
"""Config.compact_message (the routing message's 64-token slots) cannot drop a token.

A top-k slot of a shard's message holds exactly one pair per token of the shard (slot k
= every token's k-th choice), so a slot's entry count is the shard's row count, a
static shape, never a routing-dependent count. pallas_geometry compacts only when the
shard's (padded) rows are at most COMPACT_MESSAGE_TOKENS, so every token keeps its pair
in every slot. Expert skew changes how many pairs land on a shard (the routed-rows
tables, sized to the no-drop bound, untouched by the flag), not the message layout.

test_geometry runs anywhere jax imports (off a TPU it uses TPU7x's published
sparse-core info). That the layer's output does not change with the flag (and with
Config.reorder_chain) is checked on eight TPU7x cores in
tests/kernels/fused_moe/test_fused_ep_moe_v3.py.
"""

from __future__ import annotations

import pytest

from vllm_torchtpu.kernels.fused_moe.v3 import layer as L
from vllm_torchtpu.kernels.fused_moe.v3.config import COMPACT_MESSAGE_TOKENS

TOP_K = 10
SHARDS = 8


@pytest.fixture
def tpu_info_off_device(monkeypatch):
    """Off a TPU (CPU CI), the geometry's sparse-core count comes from the
    published TPU7x info, the chip the layer serves on; on a TPU, from the
    device as in serving."""
    import jax
    from jax.experimental.pallas import tpu as pltpu

    if jax.default_backend() != "tpu":
        import importlib

        info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
        monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)
        # plsc.VectorSubcoreMesh reads the info through jax's own modules.
        for name in ("jax._src.pallas.mosaic.tpu_info", "jax._src.tpu_info"):
            try:
                mod = importlib.import_module(name)
            except ImportError:
                continue
            if hasattr(mod, "get_tpu_info"):
                monkeypatch.setattr(mod, "get_tpu_info", lambda: info)


@pytest.mark.parametrize("per_shard", [8, 48, 56, 64, 72, 112, 128, 256, 1024])
def test_geometry(per_shard, tpu_info_off_device):
    bps = 8
    rows, tokens, pairs = L.pallas_geometry(per_shard, TOP_K, SHARDS, bps, compact=True)
    rows0, tokens0, pairs0 = L.pallas_geometry(per_shard, TOP_K, SHARDS, bps)
    assert rows == rows0
    # the bound that makes the flag lossless: every row of the shard has its pair in
    # every slot
    assert rows <= tokens
    if rows <= COMPACT_MESSAGE_TOKENS:
        assert tokens == COMPACT_MESSAGE_TOKENS
        assert pairs == 1024 and pairs0 == 2048  # DP4xTP2 decode at 48-64 rows per core
    else:
        assert (tokens, pairs) == (
            tokens0,
            pairs0,
        )  # inactive: the uncompacted geometry
