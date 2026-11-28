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
"""QSA decode with slot reads.

FP8 main KV rows pack four head slots (a paired page holds layer A's K, V at
slots 0, 1 and layer B's at 2, 3). Batched RPA then attends one KV head at
the layer's slot (``kv_slot``) and unpacks only that K/V pair, instead of
presenting both heads with zero queries for the other layer. On CPU the
kernel is the decode-only reference reading the requested slot; the route is
also lowered for TPU7x, where both slots must share one decode schedule. The
kernel is compared bit for bit on hardware by
scripts/bench_qsa_rpa_decode.py (``test_slot_read``).
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from tests.models.qwen4_exp.experimental.kernels.qsa_decode_cases import (  # noqa: F401
    Q_HEADS,
    RATIO,
    TPU7X,
    WIDTH,
    _attend,
    _lower,
    _schedules,
    _verify_step,
    kernels,
)
from tests.models.qwen4_exp.experimental.kernels.test_paired_main_kv import (
    reference_rpa,
)
from vllm_torchtpu.models.qwen4_exp.experimental.kernels import qsa_rpa


@pytest.mark.parametrize("offset", [0, 1])
def test_slot_read_matches_padded_heads(
    kernels,  # noqa: F811
    monkeypatch,
    offset,
):
    from vllm_torchtpu.models.qwen4_exp.experimental.kernels.batched_rpa import (
        wrapper,
    )

    calls = []

    def rpa(queries, keys, values, kv_cache, *args, **kwargs):
        slot = kwargs.pop("kv_slot", None)
        calls.append((queries.shape[1], keys.shape[1], slot))
        view = kv_cache
        if slot is not None:
            # This slot's K, V pair as the row's only head.
            rows = kv_cache.reshape(*kv_cache.shape[:2], -1, kv_cache.shape[-1])
            pair = rows[:, :, 2 * slot : 2 * slot + 2]
            view = pair.reshape(*kv_cache.shape[:2], 1, 2, kv_cache.shape[-1])
        out, _ = reference_rpa(queries, keys, values, view, *args, **kwargs)
        return out, kv_cache

    monkeypatch.setattr(wrapper, "ragged_paged_attention", rpa)
    rng = np.random.default_rng(23 + offset)
    args, starts = _verify_step(rng, [300, 0, 41, 129, 7], 4, 20, 16, 128, 3)
    options = dict(max_query_len=4, **({"kv_head_offset": 1} if offset else {}))
    # The padded route: both heads of the row, zero queries for the other slot.
    expected, _ = jax.jit(
        functools.partial(
            qsa_rpa.rpa_attention,
            kv_heads=1,
            ratio=RATIO,
            k_scale=0.375,
            v_scale=2.5,
            **({"kv_head_offset": 1} if offset else {}),
        )
    )(*args)
    actual, _ = _attend(args, starts, **options)
    # One KV head, the layer's three queries, its own slot.
    assert calls[-1] == (Q_HEADS, 1, offset)
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert np.abs(expected).max() > 0
    np.testing.assert_array_equal(actual, expected)


def test_bf16_rows_keep_the_padded_route(kernels, monkeypatch):  # noqa: F811
    """Two-sublane BF16 paired rows are not slot-readable."""
    cache = jnp.zeros((4, 128, 2, 2, WIDTH), jnp.bfloat16)
    assert not qsa_rpa.slot_readable(cache, 1, 1)
    assert qsa_rpa.slot_readable(jnp.zeros((4, 128, 1, 4, WIDTH)), 1, 1)


def test_slots_lower_one_schedule_for_tpu7x(monkeypatch):
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: TPU7X)
    rows, columns = 32, 64
    shape = jax.ShapeDtypeStruct
    specs = (
        shape((rows, Q_HEADS, WIDTH), jnp.bfloat16),
        shape((rows, Q_HEADS, WIDTH), jnp.bfloat16),
        shape((rows, 512), jnp.int32),
        shape((rows,), jnp.int32),
        shape((rows, columns), jnp.int32),
        shape((4096, 512, 1, 4, WIDTH), jnp.float8_e4m3fn),
        shape((rows // 4 + 1,), jnp.int32),
    )

    def lower(offset):
        options = dict(kv_heads=1, ratio=RATIO, k_scale=0.375, v_scale=2.5)
        if offset:
            options["kv_head_offset"] = offset
        text = _lower(
            functools.partial(qsa_rpa.qsa_attention, max_query_len=4, **options),
            *specs,
        )
        return text, _schedules(text)

    (_, (first, kernels0)), (text1, (second, _)) = lower(0), lower(1)
    assert len(first) == 1 and first == second
    assert all(k.startswith("RPAd") for k in kernels0)
    # Three queries per row: no zero queries for the other slot.
    assert "tensor<32x6x256" not in text1


@pytest.mark.parametrize("bkv,lanes", [(256, 8), (1024, 8), (1152, 4)])
def test_compact_schedule_lowers_for_tpu7x(monkeypatch, bkv, lanes):
    """The compact KV schedule fits SMEM at KV blocks the per-page one cannot."""
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: TPU7X)
    for name, value in (
        ("VLLM_TPU_QSA_DECODE_COMPACT_SCHEDULE", True),
        ("VLLM_TPU_QSA_DECODE_RPA_BKV", bkv),
        ("VLLM_TPU_QSA_DECODE_RPA_LANES", lanes),
    ):
        monkeypatch.setitem(qsa_rpa._OPTIONS, name, value)
    rows, columns = 32, 64
    shape = jax.ShapeDtypeStruct
    specs = (
        shape((rows, Q_HEADS, WIDTH), jnp.bfloat16),
        shape((rows, Q_HEADS, WIDTH), jnp.bfloat16),
        shape((rows, 512), jnp.int32),
        shape((rows,), jnp.int32),
        shape((rows, columns), jnp.int32),
        shape((4096, 512, 1, 4, WIDTH), jnp.float8_e4m3fn),
        shape((rows // 4 + 1,), jnp.int32),
    )
    text = _lower(
        functools.partial(
            qsa_rpa.qsa_attention,
            kv_heads=1,
            ratio=RATIO,
            k_scale=0.375,
            v_scale=2.5,
            max_query_len=4,
            kv_head_offset=1,
        ),
        *specs,
    )
    assert f'kernel_name = "RPAd-snh-p4-b{lanes}-q1-k{bkv}' in text
