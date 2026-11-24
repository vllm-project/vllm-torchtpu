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

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3.kernels import (
    decode_megakernel,
    paged,
)


@pytest.mark.parametrize("batch_size", [2, 4, 6, 8, 16, 32, 64, 96, 128])
def test_should_enable_grouped_expert_routes_for_b2_and_above_when_relaxed(
    batch_size: int,
) -> None:
    assert decode_megakernel.use_grouped_expert_routes(batch_size, "relaxed")


@pytest.mark.parametrize(
    ("batch_size", "residual_reduction_order"),
    [
        (1, "relaxed"),
        (2, "reference"),
        (4, "reference"),
        (4, "row"),
        (8, "reference"),
    ],
)
def test_should_disable_grouped_expert_routes_at_b1_or_non_relaxed_order(
    batch_size: int,
    residual_reduction_order: str,
) -> None:
    assert not decode_megakernel.use_grouped_expert_routes(
        batch_size, residual_reduction_order
    )


@pytest.mark.parametrize(
    ("batch_size", "attention_dp", "sequence_rows"),
    [
        (1, 1, 1),
        (2, 1, 1),
        (4, 1, 1),
        (8, 1, 1),
        (16, 1, 1),
        (32, 1, 1),
        (128, 2, 1),
    ],
)
def test_should_enable_overlap_router_latent_for_plain_dp1_b1_to_b32_and_dp2_b128(
    batch_size: int,
    attention_dp: int,
    sequence_rows: int,
) -> None:
    assert decode_megakernel.use_overlap_router_latent(
        batch_size, attention_dp, sequence_rows
    )


@pytest.mark.parametrize(
    ("batch_size", "attention_dp", "sequence_rows"),
    [
        (1, 1, 2),
        (4, 1, 4),
        (6, 1, 1),
        (64, 1, 1),
        (4, 2, 1),
        (32, 2, 1),
        (128, 2, 2),
        (32, 4, 1),
    ],
)
def test_should_disable_overlap_router_latent_outside_supported_regimes(
    batch_size: int,
    attention_dp: int,
    sequence_rows: int,
) -> None:
    assert not decode_megakernel.use_overlap_router_latent(
        batch_size, attention_dp, sequence_rows
    )


@pytest.mark.parametrize(
    ("batch_size", "attention_dp", "residual_reduction_order"),
    [
        (1, 1, "relaxed"),
        (2, 1, "relaxed"),
        (4, 1, "relaxed"),
        (8, 1, "reference"),
        (8, 1, "relaxed"),
        (1, 2, "reference"),
        (4, 2, "reference"),
    ],
)
def test_should_enable_batched_decoder_boundaries_for_relaxed_b124_and_scatter_or_dp(
    batch_size: int,
    attention_dp: int,
    residual_reduction_order: str,
) -> None:
    assert decode_megakernel.use_batched_decoder_boundaries(
        batch_size, attention_dp, residual_reduction_order
    )


@pytest.mark.parametrize(
    ("batch_size", "residual_reduction_order"),
    [
        (1, "reference"),
        (2, "reference"),
        (4, "reference"),
        (4, "row"),
        (6, "relaxed"),
    ],
)
def test_should_preserve_per_row_decoder_boundaries_for_reference_b124_and_b6(
    batch_size: int,
    residual_reduction_order: str,
) -> None:
    assert not decode_megakernel.use_batched_decoder_boundaries(
        batch_size, 1, residual_reduction_order
    )


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_should_pass_un_gathered_kda_slot_banks_to_decode_local_for_dp1_b124(
    monkeypatch: pytest.MonkeyPatch,
    batch_size: int,
) -> None:
    captured: list[tuple[tuple[int, ...], tuple[int, ...], np.ndarray | None]] = []
    committed: list[tuple[int | None, np.ndarray]] = []

    def stub_decode_local(weights, states, tokens, positions, *, batch_size, **kwargs):
        del weights, tokens, positions
        conv_in, rec_in = states[:2]
        state_slots = kwargs.get("state_slots")
        captured.append(
            (
                conv_in.shape,
                rec_in.shape,
                None if state_slots is None else np.asarray(state_slots),
            )
        )
        logits = jnp.zeros((1, batch_size, 16), jnp.float32)
        conv_out = jnp.zeros((batch_size, 2, 1, 1, 1, 128), jnp.bfloat16)
        rec_out = jnp.zeros((batch_size, 2, 1, 8, 128), jnp.float32)
        new_keys = jnp.zeros((batch_size, 1, 3, 256), jnp.bfloat16)
        new_vals = jnp.zeros((batch_size, 1, 1, 128), jnp.bfloat16)
        return logits, (conv_out, rec_out, new_keys, new_vals)

    def spy_commit(
        conv_slots, rec_slots, conv, rec, slots, source_rows=None, null_period=None
    ):
        del conv, rec, source_rows
        committed.append((null_period, np.asarray(slots)))
        return conv_slots, rec_slots

    monkeypatch.setattr(paged, "decode_local", stub_decode_local)
    monkeypatch.setattr(paged, "commit_kda_states", spy_commit)
    monkeypatch.setattr(paged, "commit_latent_rows", lambda pools, *a, **k: pools)

    num_slots = 17
    conv_slots = jnp.zeros((num_slots, 2, 1, 1, 1, 128), jnp.bfloat16)
    rec_slots = jnp.zeros((num_slots, 2, 1, 8, 128), jnp.float32)
    pool = (jnp.zeros((2, 128, 2, 640), jnp.bfloat16),)
    tokens = jnp.arange(10, 10 + batch_size, dtype=jnp.int32)
    positions = jnp.arange(1, 1 + batch_size, dtype=jnp.int32)
    block_tables = jnp.zeros((batch_size, 4), jnp.int32)
    slots = jnp.asarray([13, 0, 9, 4][:batch_size], dtype=jnp.int32)

    paged.decode_paged(
        weights={},
        conv_slots=conv_slots,
        recurrent_slots=rec_slots,
        pool=pool,
        tokens=tokens,
        positions=positions,
        block_tables=block_tables,
        slots=slots,
        batch_size=batch_size,
        dcp_pool_map=(0,),
        attention_dp=1,
        sequence_parallel_size=1,
    )

    assert len(captured) == 1
    conv_shape, rec_shape, passed_slots = captured[0]
    assert conv_shape == conv_slots.shape
    assert rec_shape == rec_slots.shape
    assert passed_slots is not None
    np.testing.assert_array_equal(passed_slots, np.asarray(slots))
    assert not decode_megakernel.kda_inplace_writes(
        batch_size=batch_size,
        attention_dp=1,
        sequence_rows=1,
        slot_reads=True,
        state_row=None,
    )
    assert len(committed) == 1
    assert committed[0][0] == num_slots
    np.testing.assert_array_equal(committed[0][1], np.asarray(slots))


def test_should_keep_gathered_kda_inputs_for_dp4() -> None:
    batch_size = 8
    num_slots = 32
    conv_slots = jnp.zeros((num_slots, 2, 1, 1, 1, 128), jnp.bfloat16)
    rec_slots = jnp.zeros((num_slots, 2, 1, 8, 128), jnp.float32)
    pool = (jnp.zeros((2, 128, 2, 640), jnp.bfloat16),)
    tokens = jnp.arange(batch_size, dtype=jnp.int32)
    positions = jnp.arange(batch_size, dtype=jnp.int32)
    block_tables = jnp.zeros((batch_size, 4), jnp.int32)
    slots = jnp.arange(1, batch_size + 1, dtype=jnp.int32)

    captured_state_slots: list[object] = []

    def stub_decode_local(weights, states, tokens, positions, *, batch_size, **kwargs):
        del weights, tokens, positions
        captured_state_slots.append((states[0].shape[0], kwargs.get("state_slots")))
        logits = jnp.zeros((1, batch_size, 16), jnp.float32)
        conv_out = jnp.zeros((batch_size, 2, 1, 1, 1, 128), jnp.bfloat16)
        rec_out = jnp.zeros((batch_size, 2, 1, 8, 128), jnp.float32)
        new_keys = jnp.zeros((batch_size, 1, 3, 256), jnp.bfloat16)
        new_vals = jnp.zeros((batch_size, 1, 1, 128), jnp.bfloat16)
        return logits, (conv_out, rec_out, new_keys, new_vals)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(paged, "decode_local", stub_decode_local)
        mp.setattr(paged, "commit_latent_rows", lambda pools, *a, **k: pools)
        mp.setattr(jax.lax, "axis_index", lambda _axis: jnp.int32(0))
        paged.decode_paged(
            weights={},
            conv_slots=conv_slots,
            recurrent_slots=rec_slots,
            pool=pool,
            tokens=tokens,
            positions=positions,
            block_tables=block_tables,
            slots=slots,
            batch_size=batch_size,
            dcp_pool_map=(0,),
            attention_dp=4,
            sequence_parallel_size=1,
        )

    assert captured_state_slots == [(batch_size, None)]


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_should_match_per_row_decoder_boundaries_when_batched_exit_enabled(
    monkeypatch: pytest.MonkeyPatch,
    batch_size: int,
) -> None:
    rng = np.random.default_rng(124 + batch_size)
    vocab_shard = 16
    hidden = 32
    layers = 25
    active_blocks = (layers - 1) // 12 + 1
    total_blocks = active_blocks + 2

    emb = jnp.asarray(rng.normal(0.0, 0.5, (vocab_shard, hidden)), jnp.bfloat16)
    out_res = jnp.asarray(rng.normal(0.0, 0.3, (1, hidden)), jnp.float32)
    final_norm = jnp.asarray(rng.normal(1.0, 0.1, (hidden,)), jnp.float32)
    lm_head = jnp.asarray(rng.normal(0.0, 0.2, (hidden, vocab_shard)), jnp.bfloat16)
    blocks_stub = jnp.asarray(
        rng.normal(0.0, 0.4, (batch_size, total_blocks, hidden)), jnp.bfloat16
    )
    blocks_stub = blocks_stub.at[:, active_blocks:, :].set(jnp.bfloat16(999.0))

    tokens = jnp.asarray([3, 19, 11, 27][:batch_size], jnp.int32)
    positions = jnp.arange(1, 1 + batch_size, dtype=jnp.int32)
    weights = {
        "embedding": emb,
        "output_res": out_res,
        "final_norm": final_norm,
        "lm_head": lm_head,
    }
    dummy_states = (
        jnp.zeros((1,), jnp.bfloat16),
        jnp.zeros((1,), jnp.float32),
        jnp.zeros((1,), jnp.bfloat16),
    )

    def stub_stack(prefix, embedding, *args, **kwargs):
        del args, kwargs
        assert prefix.shape == (batch_size, hidden)
        assert embedding.shape == (batch_size, hidden)
        return (
            prefix,
            blocks_stub,
            dummy_states[0],
            dummy_states[1],
            dummy_states[2],
            dummy_states[2],
            None,
        )

    monkeypatch.setattr(decode_megakernel, "stack", stub_stack)
    monkeypatch.setattr(jax.lax, "axis_index", lambda _axis: jnp.int32(0))
    monkeypatch.setattr(decode_megakernel.kimi, "tp_sum", lambda x: x)

    batched_logits, _ = jax.jit(
        lambda w, s, t, p: decode_megakernel.decode_local(
            w,
            s,
            t,
            p,
            batch_size=batch_size,
            layers=layers,
            residual_reduction_order="relaxed",
            sequence_parallel_pre_attention=False,
        )
    )(weights, dummy_states, tokens, positions)

    per_row_logits, _ = jax.jit(
        lambda w, s, t, p: decode_megakernel.decode_local(
            w,
            s,
            t,
            p,
            batch_size=batch_size,
            layers=layers,
            residual_reduction_order="reference",
            sequence_parallel_pre_attention=False,
        )
    )(weights, dummy_states, tokens, positions)

    assert batched_logits.shape == (1, batch_size, vocab_shard)
    np.testing.assert_allclose(
        np.asarray(batched_logits),
        np.asarray(per_row_logits),
        rtol=1e-4,
        atol=1e-4,
    )
