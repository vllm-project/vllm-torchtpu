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
"""Native fused state updates must preserve the existing byte-level contract."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch


@pytest.mark.tpu_test
@pytest.mark.parametrize("requests", [16, 32])
@torch.inference_mode()
def test_axial_decode_state_continuation_reset_and_mixed_windows(requests):
    torch._dynamo.reset()
    from torch_tpu._internal import pallas

    from vllm_torchtpu.kernels.qwen4_exp.decode_state import update_decode_state
    from vllm_torchtpu.kernels.qwen4_exp.qsa import _indexer_updates, _token_metadata
    from vllm_torchtpu.kernels.qwen4_exp.qsa_state import (
        read_ring_and_tags,
        write_ring_and_tags,
    )

    def make_run(fused):
        def core(
            pool: jax.Array,
            raw: jax.Array,
            pos: jax.Array,
            ids: jax.Array,
            starts: jax.Array,
            context: jax.Array,
            norm: jax.Array,
            trig: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            active = starts[1:] > starts[:-1]
            slots = jnp.where(active, jnp.arange(requests), -1)
            if fused:
                pool, keys = update_decode_state(
                    pool,
                    raw,
                    pos,
                    ids,
                    starts,
                    context,
                    norm,
                    trig,
                    ratio=4,
                    norm_eps=1e-6,
                    rotary_dim=64,
                    rope_axes=tuple([0, 1, 2] * 10 + [0, 1]),
                )
                _, logical, valid = _token_metadata(16, starts, context, slots)
                complete = valid & (logical % 4 == 3)
            else:
                ring, tags = read_ring_and_tags(pool, ids, 1024, 32)
                keys, complete, _, _, ring, tags = _indexer_updates(
                    raw,
                    pos,
                    ring.reshape(requests, 8, 128),
                    tags.reshape(requests, 8, 4),
                    norm,
                    trig,
                    starts,
                    context,
                    slots,
                    4,
                    1e-6,
                    64,
                    tuple([0, 1, 2] * 10 + [0, 1]),
                )
                pool = write_ring_and_tags(
                    pool,
                    ring.reshape(requests, -1),
                    tags.reshape(requests, -1),
                    jnp.where(active, ids, 0),
                )
            return pool, jnp.where(complete[:, None], keys, 0)

        op = pallas.jax_op(
            f"qsa_decode_state_test::continuation_bits_{fused}_{requests}",
            core,
            donate_argnums=(0,),
        )

        @op.register_fake
        def fake(pool, raw, pos, ids, starts, context, norm, trig):
            return torch.empty_like(pool), torch.empty_like(raw)

        return torch.compile(op, backend="tpu", fullgraph=True, dynamic=False)

    generator = torch.Generator().manual_seed(9317)
    pages = requests + 4
    pool = torch.randn(pages, 208, 1, 2, 128, generator=generator).bfloat16()
    # Arbitrary high position bits and nonzero flag words exercise typed packing.
    words = np.zeros((pages, 128), dtype=np.uint32)
    words[:, :32:4] = 67003
    words[:, 1:32:4] = 67017
    words[:, 2:32:4] = 67031
    words[:, 3:32:4] = 7
    planes = np.stack((words & 65535, words >> 16), axis=1).astype(np.uint16)
    pool[:, 4].view(torch.uint16).copy_(torch.from_numpy(planes[:, None]))
    original = pool.view(torch.uint16).clone()
    previous = original.clone()
    owners = [pool.to("tpu"), pool.to("tpu")]
    ids = torch.arange(1, requests + 1, dtype=torch.int32)
    ids[0], ids[2] = 0, -1
    context = torch.arange(requests, dtype=torch.int32) + 66048
    norm = (torch.randn(128, generator=generator) * 0.1).bfloat16().to("tpu")
    angles = torch.outer(torch.arange(67584).float(), 1e7 ** (-torch.arange(32) / 32))
    trig = torch.nn.functional.pad(
        torch.cat((angles.cos(), angles.sin()), 1).bfloat16(), (0, 64)
    ).to("tpu")
    runs = [make_run(False), make_run(True)]
    for step, counts in enumerate(
        (
            [1] * 8 + [0] * 8,
            [3, 0, 5, 0, 1, 0, 2, 1] + [0] * 8,
            [0, 16] + [0] * 14,
            [0] * 16,
            [1] * 8 + [0] * 8,
            [0] * 8 + [1] * 8,
        )
    ):
        # Interleave inactive slots and put live rows in the second half when
        # metadata capacity exceeds the query bucket. Never truncate to 16.
        if requests == 32:
            counts = [value for count in counts for value in (0, count)]
        counts = torch.tensor(counts, dtype=torch.int32)
        starts = torch.cat((torch.zeros(1, dtype=torch.int32), counts.cumsum(0).int()))
        if step == 2:
            context[3 if requests == 32 else 1] = 0
        raw = torch.randn(16, 128, generator=generator).bfloat16()
        raw[1, 0] = -0.0
        raw = raw.to("tpu")
        pos = torch.stack(
            [
                torch.arange(16, dtype=torch.int32) + 67001 + step * 19 + axis * 67
                for axis in range(3)
            ]
        ).to("tpu")
        outputs = []
        for index, run in enumerate(runs):
            owners[index], keys = run(
                owners[index],
                raw,
                pos,
                ids.to("tpu"),
                starts.to("tpu"),
                context.to("tpu"),
                norm,
                trig,
            )
            outputs.append((owners[index].cpu().view(torch.uint16), keys.cpu()))
        assert torch.equal(outputs[0][0], outputs[1][0]), step
        torch.testing.assert_close(outputs[0][1], outputs[1][1], rtol=0, atol=0)
        untouched = [0, requests + 1, requests + 2, requests + 3]
        assert torch.equal(outputs[1][0][untouched], original[untouched])
        inactive_ids = ids[(counts == 0) & (ids > 0)]
        # Inactive owners must retain their values from the previous step.
        assert torch.equal(outputs[1][0][inactive_ids], previous[inactive_ids])
        # Each owned page's payload beyond the state prefix must remain intact.
        assert torch.equal(outputs[1][0][:, 5:], original[:, 5:])
        context += counts
        previous = outputs[1][0].clone()


def test_mtp_3d_mrope_cross_chunk_closure_and_rollback():
    """Verify k=3 (ring_capacity=8) with 3D MRoPE across chunk closure and rollback."""
    from vllm_torchtpu.kernels.qwen4_exp.qsa import _indexer_updates

    rng = np.random.default_rng(2026)
    axes = tuple([0, 1, 2] * 10 + [0, 1])
    norm = jnp.asarray(rng.normal(0, 0.1, (128,)), jnp.bfloat16)
    angles = np.outer(np.arange(4096, dtype=np.float32), 1e7 ** (-np.arange(32) / 32))
    trig = jnp.asarray(
        np.concatenate(
            (np.cos(angles), np.sin(angles), np.zeros((4096, 64), np.float32)), axis=1
        ),
        jnp.bfloat16,
    )

    # Generate 6 tokens at logical positions 1664..1669 with distinct 3D axes.
    raw_all = jnp.asarray(rng.normal(0, 0.2, (6, 128)), jnp.bfloat16)
    pos_all = jnp.stack(
        [jnp.arange(1664, 1670, dtype=jnp.int32) + axis * 113 for axis in range(3)],
        axis=0,
    )
    assert not bool(jnp.array_equal(pos_all[0], pos_all[1]))
    assert not bool(jnp.array_equal(pos_all[1], pos_all[2]))

    ring_init = jnp.zeros((1, 8, 128), jnp.bfloat16)
    tags_init = jnp.zeros((1, 8, 4), jnp.int32)
    slots = jnp.array([0], jnp.int32)

    # Oracle: feed 1664..1667 in one non-speculative step starting from 1664.
    oracle_keys, oracle_complete, _, _, _, _ = _indexer_updates(
        raw_all[:4],
        pos_all[:, :4],
        ring_init,
        tags_init,
        norm,
        trig,
        jnp.array([0, 4], jnp.int32),
        jnp.array([1664], jnp.int32),
        slots,
        4,
        1e-6,
        64,
        axes,
        ring_capacity=8,
    )
    np.testing.assert_array_equal(
        np.asarray(oracle_complete), [False, False, False, True]
    )
    expected_closed_key = np.asarray(oracle_keys[3], np.float32)

    # Step 1: Prefill chunk ending at 1666 (writes 1664, 1665 into ring slots 0, 1).
    _, comp1, _, _, ring1, tags1 = _indexer_updates(
        raw_all[:2],
        pos_all[:, :2],
        ring_init,
        tags_init,
        norm,
        trig,
        jnp.array([0, 2], jnp.int32),
        jnp.array([1664], jnp.int32),
        slots,
        4,
        1e-6,
        64,
        axes,
        ring_capacity=8,
    )
    assert not bool(jnp.any(comp1))
    np.testing.assert_array_equal(
        np.asarray(tags1[0, 0, :3]), np.asarray(pos_all[:, 0])
    )
    np.testing.assert_array_equal(
        np.asarray(tags1[0, 1, :3]), np.asarray(pos_all[:, 1])
    )

    # Step 2: MTP verify 4 tokens at 1666, 1667, 1668, 1669 (k=3 -> ring capacity 8).
    # Note: 1668 and 1669 land in ring slots 4 and 5 (% 8), preserving slots 0 and 1!
    keys2, comp2, _, _, ring2, tags2 = _indexer_updates(
        raw_all[2:6],
        pos_all[:, 2:6],
        ring1,
        tags1,
        norm,
        trig,
        jnp.array([0, 4], jnp.int32),
        jnp.array([1666], jnp.int32),
        slots,
        4,
        1e-6,
        64,
        axes,
        ring_capacity=8,
    )
    np.testing.assert_array_equal(np.asarray(comp2), [False, True, False, False])
    np.testing.assert_array_equal(np.asarray(keys2[1], np.float32), expected_closed_key)
    # Verify slots 0, 1 (positions 1664, 1665) were NOT overwritten by 1668, 1669!
    np.testing.assert_array_equal(np.asarray(ring2[0, :2]), np.asarray(ring1[0, :2]))
    np.testing.assert_array_equal(np.asarray(tags2[0, :2]), np.asarray(tags1[0, :2]))

    # Step 3: Suppose draft token at 1666 was accepted, but 1667..1669 were rejected!
    # Next step rolls back context to 1667 and verifies 4 new tokens at 1667..1670.
    # Replace token at 1667 with raw_all[3] so the closed group 1664..1667 must still
    # match `expected_closed_key` after rollback!
    keys3, comp3, _, _, _, _ = _indexer_updates(
        jnp.concatenate([raw_all[3:6], raw_all[:1]], axis=0),
        jnp.concatenate([pos_all[:, 3:6], pos_all[:, 5:6] + 1], axis=1),
        ring2,
        tags2,
        norm,
        trig,
        jnp.array([0, 4], jnp.int32),
        jnp.array([1667], jnp.int32),
        slots,
        4,
        1e-6,
        64,
        axes,
        ring_capacity=8,
    )
    np.testing.assert_array_equal(np.asarray(comp3), [True, False, False, False])
    np.testing.assert_array_equal(np.asarray(keys3[0], np.float32), expected_closed_key)
