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
"""Unit tests for generalized verify_tokens in BucketPolicy and step eligibility."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_torchtpu.megakernel.profile import Profile
from vllm_torchtpu.megakernel.runtime.checks import build_warmup_plan
from vllm_torchtpu.megakernel.runtime.eligibility import check_step_eligibility
from vllm_torchtpu.megakernel.runtime.policy import BucketPolicy
from vllm_torchtpu.megakernel.runtime.stats import BatchShape, inspect_batch_shape
from vllm_torchtpu.megakernel.spec import LayerType, ModelSpec


def test_verify_tokens_eight_bucket_policy_and_eligibility() -> None:
    """Verify generalized verify_tokens in BucketPolicy and check_step_eligibility."""
    prof_8 = Profile(name="test_8", buckets=(2,), verify_buckets=(8, 16))
    pol_8 = BucketPolicy.resolve(prof_8, verify_tokens=(8,))
    assert pol_8.verify_tokens == (8,)
    assert BucketPolicy.resolve(
        prof_8, speculative_tokens=7, verify_tokens=(8,)
    ).verify_buckets == (8, 16)

    pol_4_8 = BucketPolicy.resolve(
        Profile(name="test_4_8", buckets=(2,), verify_buckets=(4, 8, 16)),
        verify_tokens=(4, 8),
    )
    assert pol_4_8.verify_tokens == (4, 8)
    # Verify buckets must divide by every verify token, so 4 drops for (4, 8).
    assert list(pol_4_8.iter_buckets()) == [(1, 2), (4, 8), (4, 16), (8, 8), (8, 16)]
    pol_def_8 = BucketPolicy.resolve(
        Profile(name="test_def", buckets=None, verify_buckets=None), verify_tokens=(8,)
    )
    assert pol_def_8.verify_buckets == (8, 16, 24, 32, 40, 48, 56, 64)

    b_shape_8 = BatchShape(eligible=True, reason="ok", num_reqs=1, tokens_per_seq=8)
    dec_8 = check_step_eligibility(
        b_shape_8,
        pol_8,
        compiled_buckets={(8, 8)},
        bucket_out_rows={(8, 8): 8},
        input_rows=8,
    )
    assert (dec_8.eligible, dec_8.tokens_per_seq) == (True, 8)

    pol_def = BucketPolicy()
    dec_def = check_step_eligibility(
        b_shape_8,
        pol_def,
        compiled_buckets={(1, 8)},
        bucket_out_rows={},
        input_rows=8,
    )
    assert dec_def.eligible is False
    assert dec_def.reason == "stock_q8"


def test_resolve_explicit_verify_buckets_validation() -> None:
    """Verify explicit verify_buckets raise ValueError when not divisible by vt."""
    prof = Profile(name="test_exp", buckets=(2,), verify_buckets=None)
    with pytest.raises(ValueError, match=r"verify_buckets \[4\] not divisible"):
        BucketPolicy.resolve(prof, verify_buckets=(4, 8), verify_tokens=(8,))
    pol_non_exp = BucketPolicy.resolve(
        prof,
        verify_buckets=(4, 8),
        verify_buckets_explicit=False,
        verify_tokens=(8,),
    )
    assert pol_non_exp.verify_buckets == (8,)


def test_verify_tokens_eight_from_qsl() -> None:
    """Verify QSL batch inspection and step eligibility for 8-token verify steps."""
    qsl = [0, 8, 16, 24, 32]
    bs = inspect_batch_shape(has_roff=True, qsl=qsl, num_reqs=4)
    pol = BucketPolicy(verify_tokens=(4, 8))
    dec = check_step_eligibility(
        bs,
        pol,
        compiled_buckets={(8, 32)},
        bucket_out_rows={(8, 32): 32},
        input_rows=32,
    )
    assert (dec.eligible, dec.tokens_per_seq) == (True, 8)


def test_build_warmup_plan_generalized_verify_tokens() -> None:
    """Verify build_warmup_plan warms verify_tokens=(4, 8) and scales slot stride."""
    layout = SimpleNamespace(pages_per_slot=2, slots_per_pool=3, rec_pages_per_slot=2)
    cb = SimpleNamespace(recurrent_slots_per_pool=3, num_pools=2, layout=layout)
    spec = ModelSpec(
        name="test_spec",
        config=None,
        layer_type=LayerType(name="l", slots=()),
        cache_backend=cb,
    )
    steps, warmed, n_save = build_warmup_plan(
        spec,
        buckets=(2,),
        verify_buckets=(8, 16),
        bucket_fused_io={
            (1, 2): True,
            (4, 8): True,
            (4, 16): True,
            (8, 8): True,
            (8, 16): False,
        },
        bucket_gpr={},
        num_pages=512,
        max_pages_per_seq=16,
        verify_tokens=(4, 8),
    )
    assert (warmed, n_save) == ([(1, 2), (4, 8), (4, 16), (8, 8), (8, 16)], 198)
    assert [s.sentinel.gdn_base for s in steps if s.sentinel] == [3, 27, 51, 75]
    _, warmed_empty, _ = build_warmup_plan(
        spec,
        buckets=(2,),
        verify_buckets=(8,),
        bucket_fused_io={(1, 2): True},
        bucket_gpr={},
        num_pages=256,
        max_pages_per_seq=16,
        verify_tokens=(),
    )
    assert warmed_empty == [(1, 2)]
