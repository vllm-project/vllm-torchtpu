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

from __future__ import annotations

import jax.numpy as jnp
import pytest

from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3 import config as k3_config
from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3.kernels import (
    decode_megakernel,
    mla_dcp,
)


@pytest.mark.parametrize("batch", [1, 2, 4, 8, 16, 32, 48, 64])
def test_should_allocate_smem_lookup_tables_for_power_of_two_and_b48_batches(
    batch: int,
) -> None:
    assert batch in mla_dcp.SMEM_LOOKUP_BATCHES
    scoped = mla_dcp.scoped_scratch(
        batch,
        heads=mla_dcp.HEADS_PER_RANK,
        group=32,
        cache_dtype=jnp.float8_e4m3fn,
        page_tokens=512,
    )
    bounds_spec = scoped[mla_dcp.SCOPED_NAMES.index("dcp lookup bounds")]
    counts_spec = scoped[mla_dcp.SCOPED_NAMES.index("dcp lookup counts")]
    assert bounds_spec.shape == (batch,)
    assert counts_spec.shape == (batch,)


def test_should_keep_smem_lookup_stub_outside_smem_lookup_batches() -> None:
    assert 3 not in mla_dcp.SMEM_LOOKUP_BATCHES
    scoped = mla_dcp.scoped_scratch(
        3,
        heads=mla_dcp.HEADS_PER_RANK,
        group=32,
        cache_dtype=jnp.float8_e4m3fn,
        page_tokens=512,
    )
    assert scoped[mla_dcp.SCOPED_NAMES.index("dcp lookup bounds")].shape == (1,)
    assert scoped[mla_dcp.SCOPED_NAMES.index("dcp lookup counts")].shape == (1,)


def test_should_enable_fp8_query_wire_and_native_pages_at_b8_plain_dp1() -> None:
    pers = mla_dcp.persistent_scratch(
        8,
        group=32,
        heads=mla_dcp.HEADS_PER_RANK,
        cache_dtype=jnp.float8_e4m3fn,
        sequence_rows=1,
    )
    assert pers[0].dtype == jnp.uint8
    assert mla_dcp.native_pages(8, mla_dcp.HEADS_PER_RANK, 1, jnp.float8_e4m3fn)
    assert mla_dcp.staged_folds(8, mla_dcp.HEADS_PER_RANK, 1, jnp.float8_e4m3fn) == 1
    plan = mla_dcp.page_plan(8, mla_dcp.HEADS_PER_RANK, 1, jnp.float8_e4m3fn)
    scoped = mla_dcp.scoped_scratch(
        8,
        span=plan["span"],
        buffers=plan["buffers"],
        heads=mla_dcp.HEADS_PER_RANK,
        group=32,
        cache_dtype=jnp.float8_e4m3fn,
        page_tokens=512,
        sequence_rows=1,
    )
    pages_spec = scoped[mla_dcp.SCOPED_NAMES.index("dcp latent pages")]
    assert len(pages_spec.shape) == 3
    assert pages_spec.shape[1:] == (1024, mla_dcp.LANES)


@pytest.mark.parametrize(
    ("batch", "group", "heads", "cache_dtype", "sequence_rows"),
    [
        (4, 32, 3, jnp.bfloat16, 1),
        (4, 32, 3, jnp.float8_e4m3fn, 2),
        (8, 32, 3, jnp.bfloat16, 1),
        (8, 32, 3, jnp.float8_e4m3fn, 2),
        (8, 16, 6, jnp.float8_e4m3fn, 1),
    ],
)
def test_should_keep_bf16_wire_and_packed_pages_outside_dp1_fp8_plain(
    batch: int,
    group: int,
    heads: int,
    cache_dtype,
    sequence_rows: int,
) -> None:
    pers = mla_dcp.persistent_scratch(
        batch,
        group=group,
        heads=heads,
        cache_dtype=cache_dtype,
        sequence_rows=sequence_rows,
    )
    assert pers[0].dtype == jnp.bfloat16
    assert not mla_dcp.native_pages(batch, heads, sequence_rows, cache_dtype)


@pytest.mark.parametrize("batch_size", [32, 64])
@pytest.mark.parametrize("attention_dp", [1, 2])
def test_should_enable_kda_inplace_writes_for_plain_decode_above_b16(
    batch_size: int, attention_dp: int
) -> None:
    assert decode_megakernel.kda_inplace_writes(
        batch_size=batch_size,
        attention_dp=attention_dp,
        sequence_rows=1,
        slot_reads=True,
        state_row=None,
    )


@pytest.mark.parametrize(
    ("batch_size", "attention_dp", "sequence_rows", "slot_reads", "state_row"),
    [
        (1, 1, 1, True, None),
        (4, 1, 1, True, None),
        (8, 1, 1, True, None),
        (12, 1, 1, True, None),
        (16, 1, 1, True, None),
        (8, 2, 1, True, None),
        (16, 2, 1, True, None),
        (32, 1, 4, True, None),
        (32, 1, 1, False, None),
        (32, 1, 1, True, 0),
        (32, 4, 1, True, None),
    ],
)
def test_should_disable_kda_inplace_writes_when_preconditions_fail(
    batch_size: int,
    attention_dp: int,
    sequence_rows: int,
    slot_reads: bool,
    state_row: int | None,
) -> None:
    assert not decode_megakernel.kda_inplace_writes(
        batch_size=batch_size,
        attention_dp=attention_dp,
        sequence_rows=sequence_rows,
        slot_reads=slot_reads,
        state_row=state_row,
    )


def test_should_respect_kda_inplace_config_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        k3_config, "_ACTIVE", k3_config.KernelConfig(kda_inplace_writes=False)
    )
    assert not decode_megakernel.kda_inplace_writes(
        batch_size=32,
        attention_dp=1,
        sequence_rows=1,
        slot_reads=True,
        state_row=None,
    )
