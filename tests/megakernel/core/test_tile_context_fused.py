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
"""`TileContext.fused_pool_io`: the bucket's fused pool I/O decision reaches
the tile rule, and rules that ignore it resolve exactly as before."""

from __future__ import annotations

import dataclasses

import pytest

import vllm_torchtpu.megakernel.core.experimental  # noqa: F401  (registers fused I/O)
from tests.megakernel.core.megakernel_synth import TINY_TILES, synth_hybrid_spec
from vllm_torchtpu.megakernel.core.exec_spec import ExecSpec
from vllm_torchtpu.megakernel.core.geometry import ModelGeometry
from vllm_torchtpu.megakernel.core.pool.config import CacheExec
from vllm_torchtpu.megakernel.core.profile import (
    Profile,
    TileContext,
    approx_features,
)
from vllm_torchtpu.megakernel.core.profiles.default import default_tiles
from vllm_torchtpu.megakernel.core.runtime import bucket_exec_spec

_FUSED_NBD = 5


def _geometry() -> ModelGeometry:
    return ModelGeometry(
        L=4,
        att_every=4,
        H=256,
        NH=4,
        NKV=1,
        HD=256,
        ROT=64,
        KH=2,
        VH=4,
        KD=128,
        VD=128,
        E=128,
        TOPK=2,
        I=256,
        IS=256,
        moe_tp=False,
        e_slots=0,
        theta=10_000_000.0,
        zero_centered_norm=True,
    )


def _base_ex() -> ExecSpec:
    return ExecSpec(
        R=1,
        B=4,
        T=1,
        ctx=256,
        interpret=True,
        stack_e=False,
        cache=CacheExec(paged_kv=True, num_pages=96, max_pages_per_seq=2),
    )


def _rule(seen: list[TileContext]):
    base = dataclasses.replace(TINY_TILES, bank=512 << 10)

    def rule(tc: TileContext):
        seen.append(tc)
        return dataclasses.replace(base, nbd=_FUSED_NBD) if tc.fused_pool_io else base

    return rule


def test_tile_context_defaults_to_split_pool_io():
    assert TileContext(B=8).fused_pool_io is False


def test_exec_for_passes_the_fused_decision_to_the_rule():
    seen: list[TileContext] = []
    prof = Profile(name="test-fused-ctx", numerics="approx", tiles=_rule(seen))
    g = _geometry()
    split = prof.exec_for(_base_ex(), B=8, T=4, g=g)
    fused = prof.exec_for(_base_ex(), B=8, T=4, g=g, fused_pool_io=True)
    assert [tc.fused_pool_io for tc in seen] == [False, True]
    assert split.tiles.nbd != _FUSED_NBD
    assert fused.tiles.nbd == _FUSED_NBD


@pytest.mark.parametrize("fused", [False, True])
def test_bucket_exec_spec_resolves_tiles_with_its_fused_decision(fused: bool):
    seen: list[TileContext] = []
    prof = Profile(name="test-fused-bucket", numerics="approx", tiles=_rule(seen))
    ms = synth_hybrid_spec(_geometry(), R=1)
    ex, use_fused, _ = bucket_exec_spec(
        ms,
        _base_ex(),
        prof,
        bucket_t=8,
        tokens_per_seq=4,
        fused_pool_io=fused,
    )
    assert use_fused is fused
    assert ex.cache.fused_pool_io is fused
    assert (ex.tiles.nbd == _FUSED_NBD) is fused
    assert seen[-1].fused_pool_io is fused


def test_rules_that_ignore_the_flag_resolve_as_before():
    prof = Profile(
        name="test-fused-agnostic",
        numerics="approx",
        tiles=lambda tc: dataclasses.replace(TINY_TILES, nbd=7),
    )
    ms = synth_hybrid_spec(_geometry(), R=1)
    ex_split, _, _ = bucket_exec_spec(
        ms, _base_ex(), prof, bucket_t=8, tokens_per_seq=4, fused_pool_io=False
    )
    ex_fused, _, _ = bucket_exec_spec(
        ms, _base_ex(), prof, bucket_t=8, tokens_per_seq=4, fused_pool_io=True
    )
    assert ex_split.tiles == ex_fused.tiles


def test_approx_features_sees_fused_only_rule_changes():
    """A rule that differs from its reference only on fused buckets, by an
    approx knob, is reported by the numerics grid (so an `exact` label fails)."""

    def rule(tc: TileContext):
        t = default_tiles(tc)
        return dataclasses.replace(t, rsag=not t.rsag) if tc.fused_pool_io else t

    prof = Profile(name="test-fused-approx", numerics="approx", tiles=rule)
    assert approx_features(prof) == ("tiles.rsag",)
    with pytest.raises(ValueError, match="approx features"):
        Profile(name="test-fused-exact", numerics="exact", tiles=rule).tiles_for(
            TileContext(B=8, T=4, R=8, fused_pool_io=True)
        )


def test_numerics_grid_skips_the_fused_half_for_rules_that_ignore_the_flag():
    """The fused half of the grid runs only when a rule reads the flag (a rule
    that never reads it on the split half cannot depend on it)."""
    seen: list[bool] = []

    def ignoring(tc: TileContext):
        seen.append(object.__getattribute__(tc, "fused_pool_io"))
        return dataclasses.replace(default_tiles(tc), rsag=True)

    approx_features(Profile(name="test-ignores", numerics="approx", tiles=ignoring))
    assert seen and not any(seen)  # split half only
    seen.clear()

    def reading(tc: TileContext):
        seen.append(object.__getattribute__(tc, "fused_pool_io"))
        return dataclasses.replace(default_tiles(tc), rsag=tc.fused_pool_io)

    approx_features(Profile(name="test-reads", numerics="approx", tiles=reading))
    assert any(seen) and not all(seen)  # both halves
