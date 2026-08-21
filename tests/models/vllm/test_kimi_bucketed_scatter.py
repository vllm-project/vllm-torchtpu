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
"""MoonViT's bucketed scatter must land inside what boot precompiles.

``MMEncoderManager._capture_budget_glue`` primes ``postprocess_encoder_output``
at every ``(packed rows, bucket)`` pair it can enumerate from the model's
declared ``mm_embed_bucket`` ladder. That enumeration is only worth running if
the model's actual gather never reaches a pair outside it -- one that escapes
compiles at runtime, on the first request that hits it.

``tests/runner/test_mm_encoder_precompile_glue.py`` checks the enumeration
against the duck-typed contract. This checks Kimi's real implementation against
the same property, which is the half that can drift when the model changes.
"""

import types

import pytest
import torch

from vllm_torchtpu.models.vllm.kimi_k3.model import \
    KimiK3ForConditionalGeneration as Kimi
from vllm_torchtpu.runner.mm_encoder_manager import MMEncoderManager

HIDDEN = 8
BUDGET = 2048


def gathered_shapes(packed_rows, per_item):
    """The ``(source rows, gathered rows)`` pairs one batch reaches.

    That pair is the XLA program key for the gather, so two batches sharing a
    pair share a program.
    """
    model = types.SimpleNamespace(
        mm_embed_bucket=Kimi.mm_embed_bucket,
        MM_EMBED_BUCKET_MIN=Kimi.MM_EMBED_BUCKET_MIN,
    )
    out = torch.randn(packed_rows, HIDDEN)
    dest: dict = {}
    Kimi.postprocess_encoder_output(model, {"default": out},
                                    list(range(len(per_item))),
                                    per_item,
                                    dest,
                                    clone=True)
    return {(packed_rows, dest[idx].shape[0]) for idx in dest}


def primed_shapes(packed_rows):
    """What ``_capture_budget_glue`` compiles at boot for this packed size."""
    record = set()

    def postprocess(outputs, indices, per_item_out_tokens, dest, **kw):
        rows = outputs["default"].shape[0]
        for idx, n in zip(indices, per_item_out_tokens):
            dest[idx] = torch.zeros(Kimi.mm_embed_bucket(n), HIDDEN)
            record.add((rows, dest[idx].shape[0]))

    mgr = object.__new__(MMEncoderManager)
    mgr.model = types.SimpleNamespace(
        mm_embed_bucket=Kimi.mm_embed_bucket,
        MM_EMBED_BUCKET_MIN=Kimi.MM_EMBED_BUCKET_MIN,
        postprocess_encoder_output=postprocess,
    )
    mgr.device = torch.device("cpu")
    mgr.dtype = torch.float32
    mgr.config = types.SimpleNamespace(padding_logics={})
    mgr._host_stage = {}
    mgr._host_stage_dirty = {}
    mgr.budget_templates = {"default": {BUDGET: {}}}
    mgr._capture_budget_glue(torch.randn(packed_rows, HIDDEN), BUDGET)
    return record


@pytest.mark.parametrize("packed_rows", [64, 256, 1024, 4096])
def test_real_gather_stays_inside_the_primed_set(packed_rows):
    primed = primed_shapes(packed_rows)
    assert primed, "the warmup primed nothing"

    # Batches a request can actually produce: one big item, a mix of sizes,
    # counts that are not powers of two, and exactly-on-a-bucket counts.
    for per_item in ([1], [7], [64], [65], [1000], [300, 200,
                                                    130], [1, 1, 1,
                                                           1], [500,
                                                                129], [1024]):
        if sum(per_item) > packed_rows:
            continue
        reached = gathered_shapes(packed_rows, per_item)
        assert reached <= primed, (
            f"per_item={per_item} reaches shapes boot never compiled: "
            f"{sorted(reached - primed)}")


def test_bucket_ladder_is_powers_of_two_above_the_floor():
    assert Kimi.mm_embed_bucket(0) == Kimi.MM_EMBED_BUCKET_MIN
    assert Kimi.mm_embed_bucket(1) == Kimi.MM_EMBED_BUCKET_MIN
    for n, expect in ((64, 64), (65, 128), (128, 128), (129, 256),
                      (1024, 1024), (1025, 2048)):
        assert Kimi.mm_embed_bucket(n) == expect, n


def test_gather_reads_the_real_tokens_and_clamps_the_pad():
    """Bucket padding is junk rows, but the real tokens must be exact."""
    model = types.SimpleNamespace(
        mm_embed_bucket=Kimi.mm_embed_bucket,
        MM_EMBED_BUCKET_MIN=Kimi.MM_EMBED_BUCKET_MIN,
    )
    # 180 rows for 170 real tokens, so the second item's 128-row bucket runs
    # off the end of the packed buffer and has to clamp.
    out = torch.arange(180 * HIDDEN, dtype=torch.float32).view(180, HIDDEN)
    dest: dict = {}
    Kimi.postprocess_encoder_output(model, {"default": out}, [0, 1], [70, 100],
                                    dest,
                                    clone=True)

    assert dest[0].shape[0] == 128 and dest[1].shape[0] == 128
    torch.testing.assert_close(dest[0][:70], out[:70])
    torch.testing.assert_close(dest[1][:100], out[70:170])
    # Rows past the end clamp to the last row rather than reading out of
    # bounds; they are junk either way, but they must not fault.
    torch.testing.assert_close(dest[1][-1], out[-1])
