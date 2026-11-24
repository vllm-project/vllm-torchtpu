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
"""4-bit MSA prefill with levers from separate branches on together: nested
adaptive tiers (TPU_MINIMAX_M3_MSA_PREFILL_ADAPTIVE), a shared scale page
(TPU_MINIMAX_M3_KV4_SHARED_SCALES) and max-free softmax
(TPU_MINIMAX_M3_MSA_MAX_FREE). A layer reading its scale rows at ``row0`` of
a shared page gets the bits of the same call over its own page, in every
tier, with and without max-free."""

import jax
import numpy as np
import pytest

from tests.kernels import test_minimax_m3_msa_prefill_adaptive as ad
from vllm_torchtpu.models.minimax_m3.kernels import msa_attention_lse as lse_mod
from vllm_torchtpu.models.minimax_m3.kernels import msa_fp4

TIERS = "20:8:32;x30:8:48:32"


def _shared(args, row0):
    """`args` with the scale page moved to rows ``row0..`` of a wider page
    whose other rows hold different values."""
    *rest, sc = args
    rows = sc.shape[1]
    other = sc * 3.0 + 0.01
    page = np.concatenate([np.asarray(other)] * 2, axis=1)
    page[:, row0 : row0 + rows] = np.asarray(sc)
    view = msa_fp4.with_row0(jax.numpy.asarray(page), row0)
    return (*rest, view)


@pytest.mark.parametrize("max_free", [False, True], ids=["runmax", "maxfree"])
def test_tiers_with_shared_scale_page(ad_config, max_free):
    ad_config.setenv(lse_mod.ADAPTIVE_ENV, TIERS)
    ad_config.setattr(lse_mod, "MAX_FREE", max_free)
    row0 = 2  # DCP 2: one KV head, rows 2..3 belong to this layer
    for union, seed in ((16, 1), (28, 3), (40, 2)):
        args = ad._inputs(1, union=union, seed=seed)
        jax.clear_caches()
        want = ad._attend(args)
        jax.clear_caches()
        got = ad._attend(_shared(args, row0))
        np.testing.assert_array_equal(got[0], want[0])
        np.testing.assert_array_equal(got[1], want[1])
        assert np.abs(want[0]).max() > 0


@pytest.fixture
def ad_config(monkeypatch):
    """`test_minimax_m3_msa_prefill_adaptive.prefill_config`'s settings."""
    monkeypatch.setattr(ad.msa_dcp, "_MIN_CANDIDATE_TOKENS", 64)
    monkeypatch.setattr(ad.msa_dcp, "DCP_SMALL_BATCH_TOKENS", 64)
    monkeypatch.setattr(lse_mod, "PREFILL_PAGE_CACHE_SLOTS", 512)
    monkeypatch.setattr(lse_mod, "PREFILL_PAGES_PER_STEP", 32)
    monkeypatch.setattr(lse_mod, "PREFILL_STEP_WIDTHS", (2, 4, 8, 12, 16, 20, 24))
    monkeypatch.setattr(lse_mod, "PREFILL_BLOCK_Q", 16)
    monkeypatch.setattr(lse_mod, "PAGE_CACHE_TILE_PAGES", 64)
    return monkeypatch
