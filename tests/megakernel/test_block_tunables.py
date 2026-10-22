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
"""Block `Tunables`: per-field numerics, the composed registry, perf-flag defaults."""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import pytest
from conftest import EXACT_FLIPS, tile_field

from vllm_torchtpu import megakernel as mk
from vllm_torchtpu.megakernel import exec_spec
from vllm_torchtpu.megakernel.blocks import GDNBlock, GQABlock, MoEBlock

# Block and cache names the core spec modules must not mention (in any case).
_NON_CORE_NAMES = ("gqa", "gdn", "moe", "pool", "fused", "paged", "ckpt")

# Opt-in performance flags (core `field` / `<block>.<field>`), default off.
_PERF_FLAGS_OFF = {
    "moe.ecopy_opt": 0,
    "moe.pipe_ep": False,
    "moe.r2_direct_store": False,
    "moe.route_vec": False,
    "dense_acc_pair": False,
    "gqa.kv_ering": False,
    "gqa.pair_kv": False,
    "dmm_unroll": 0,
    "gqa.gq_pack": False,
    "moe.topk_lanes": False,
    "gdn.conv_stage": False,
    "gdn.gdn_head_major": False,
    "gqa.gqa_qv": False,
    "moe.moe_unroll": False,
    "rel_fetch": False,
}
_LIBRARY_BLOCKS = (GQABlock, GDNBlock, MoEBlock)


def _library_numerics() -> dict[str, str]:
    """`exec_spec.tile_numerics()` restricted to core and the library's blocks."""
    names = {b.name for b in _LIBRARY_BLOCKS}
    return {
        k: v
        for k, v in exec_spec.tile_numerics().items()
        if "." not in k or k.split(".", 1)[0] in names
    }


@pytest.mark.parametrize("block", _LIBRARY_BLOCKS, ids=lambda b: b.name)
def test_block_tunables_declare_numerics_for_every_field(block: type[mk.Block]):
    cls = block.Tunables
    assert cls is not None and exec_spec.tunables_class(block.name) is cls
    numerics = dict(cls.NUMERICS)
    fields = {f.name for f in dataclasses.fields(cls)}
    assert set(numerics) == fields
    assert set(numerics.values()) <= {"exact", "approx"}
    assert {f"{block.name}.{f}" for f in fields} <= set(exec_spec.tile_numerics())


def test_tile_numerics_classes_from_evidence():
    numerics = _library_numerics()
    for k in (
        "gqa.kv_sub",
        "moe.route_vec",
        "dense_acc_pair",
        "moe.moe_unroll",
        "gdn.gdn_head_major",
        "gqa.gq_pack",
    ):
        assert numerics[k] == "approx", k
    for k in ("gqa.pair_kv", "gqa.kv_ering", "gqa.gqa_qv"):
        assert numerics[k] == "exact", k


def test_exact_flip_list_covers_registry():
    exact = {k for k, v in _library_numerics().items() if v == "exact"}
    # kv_scatter_vec_ns only affects the non-paged path (tested separately).
    assert exact == {f for _, f, _, _, _ in EXACT_FLIPS} | {"gqa.kv_scatter_vec_ns"}


def test_tiles_field_defaults_turn_perf_flags_off():
    t = mk.Tiles()
    for k, v in _PERF_FLAGS_OFF.items():
        assert tile_field(t, k) == v, k
    assert t.block("gqa").kv_sub == 0
    d = mk.default_tiles(8)
    for k, v in _PERF_FLAGS_OFF.items():
        assert tile_field(d, k) == v, k


def test_core_spec_mentions_no_block_or_cache_names():
    """`exec_spec.py` and `spec.py` are core: no block or pool name in them."""
    for module in ("exec_spec.py", "spec.py"):
        path = Path(exec_spec.__file__).parent / module
        text = path.read_text(encoding="utf-8")
        for name in _NON_CORE_NAMES:
            hits = [
                f"{i}: {line.strip()}"
                for i, line in enumerate(text.splitlines(), 1)
                if re.search(name, line, flags=re.IGNORECASE)
            ]
            assert not hits, (module, name, hits)
