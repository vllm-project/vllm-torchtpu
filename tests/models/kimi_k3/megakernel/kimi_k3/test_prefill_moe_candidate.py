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

"""Production dispatch and rank-uniform metadata capacity regression checks."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3 import config, prefill
from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3.kernels import (
    prefill_megakernel as kernels,
)
from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3.kernels.prefill_moe import fused
from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3.kernels.prefill_moe import (
    moe_down_valu as _moe_dv,
)

down_valu_compact = _moe_dv.down_valu_compact

_SUPPORTED_FUSED_WINDOWS = (
    512,
    1024,
    2048,
    4096,
    6144,
    8192,
    10240,
    12288,
    14336,
    16384,
)


@pytest.mark.parametrize(
    "window", [512, 1024, 2048, 3072, 4096, 6144, 8192, 10240, 12288, 14336, 16384]
)
@pytest.mark.parametrize("ep", [8, 16, 32])
@pytest.mark.parametrize("enabled", [False, True])
def test_factory_preserves_unsupported_shapes(monkeypatch, window, ep, enabled):
    cfg = config.KernelConfig(
        prefill_expert_parallel=ep,
        prefill_fused_moe=enabled,
    )
    monkeypatch.setattr(config, "active", lambda: cfg)
    monkeypatch.setattr(
        kernels,
        "_make_indexed_moe_sp_ring_reference",
        lambda mesh, *, seq_len: ("reference", seq_len),
    )
    monkeypatch.setattr(
        fused,
        "make_indexed_moe_sp_ring",
        lambda mesh, *, seq_len: ("fused", seq_len),
    )
    result = kernels.make_indexed_moe_sp_ring(None, seq_len=window)
    expected = "reference"
    if enabled and ep in (8, 16) and window in _SUPPORTED_FUSED_WINDOWS:
        expected = "fused"
    assert result == (expected, window)


@pytest.mark.parametrize(
    "window", [512, 1024, 2048, 3072, 4096, 6144, 8192, 10240, 12288, 14336, 16384]
)
@pytest.mark.parametrize("ep", [8, 16, 32])
@pytest.mark.parametrize("enabled", [False, True])
def test_mixture_moe_factory_preserves_unsupported_shapes(
    monkeypatch, window, ep, enabled
):
    cfg = config.KernelConfig(
        prefill_expert_parallel=ep,
        prefill_fused_moe=enabled,
    )
    monkeypatch.setattr(config, "active", lambda: cfg)
    monkeypatch.setattr(
        kernels,
        "make_mixture_norm_sp",
        lambda mesh, *, seq_len: lambda *a: jnp.int32(seq_len),
    )
    monkeypatch.setattr(
        kernels,
        "make_indexed_moe_sp_ring",
        lambda mesh, *, seq_len: lambda h, *a: jnp.int32(100000) + h,
    )
    monkeypatch.setattr(
        fused,
        "make_indexed_mixture_moe_sp_ring",
        lambda mesh, *, seq_len: ("fused_mix", seq_len),
    )
    result = kernels.make_indexed_mixture_moe_sp_ring(None, seq_len=window)
    if enabled and ep in (8, 16) and window in _SUPPORTED_FUSED_WINDOWS:
        assert result == ("fused_mix", window)
    else:
        assert callable(result)
        z = jnp.int32(0)
        assert int(result(z, z, z, z, z, z, z, z)) == 100000 + window


def test_rollback_has_distinct_compile_cache_key():
    cfg = config.KernelConfig()
    assert cfg.prefill_fused_moe
    assert cfg.key() != cfg.replace(prefill_fused_moe=False).key()


@pytest.mark.parametrize("peak", [0, 32768, 32769, 131072])
def test_metadata_overflow_is_uniform_and_uncapped(monkeypatch, peak):
    from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3.kernels.prefill_moe import (
        moe_down_valu,
    )

    # No TPU work: exercise the real branch wrapper with a stand-in consumer.
    # The stand-in reports capacity, activation rows and streaming mode.
    def consume(x, rhs, scales, counts, destinations, probabilities, tables, **kw):
        return jnp.array(
            [
                x.shape[0],
                destinations.shape[0],
                probabilities.shape[0],
                kw["stream_metadata"],
                kw.get("metadata_row_capacity", 0),
            ]
        )

    monkeypatch.setattr(moe_down_valu, "down_valu", consume)
    x = jnp.zeros((131072, 1), dtype=jnp.bfloat16)
    routes = jnp.arange(131072, dtype=jnp.int32)
    # Local counts differ across ranks, but the identical global peak determines
    # which collective-containing branch all ranks execute.
    results = []
    for local_count in [0, 17, peak]:
        results.append(
            down_valu_compact(
                x,
                None,
                None,
                jnp.array([local_count]),
                routes,
                routes,
                None,
                metadata_prefix_rows=32768,
                slice_input=False,
                global_route_max=jnp.int32(peak),
            )
        )
    for result in results[1:]:
        np.testing.assert_array_equal(result, results[0])
    compact = peak <= 32768
    np.testing.assert_array_equal(
        results[0],
        [
            131072,
            32768 if compact else 131072,
            32768 if compact else 131072,
            not compact,
            32768 if compact else 0,
        ],
    )


@pytest.mark.parametrize("tokens", [512, 1024, 2048, 4096, 8192, 16384])
def test_fused_validity_matches_sequence_parallel_ownership(tokens):
    rows = tokens // 32
    stripe = min(rows, 64)
    for live in (0, 1, 17, tokens - 17, tokens):
        for rank in (0, 1, 15, 31):
            owned = np.arange(rank * stripe, tokens, 32 * stripe)[:, None]
            owned = (owned + np.arange(stripe)[None, :]).reshape(rows, 1)
            actual = fused._stripe_validity(rows, jnp.int32(live), jnp.int32(rank))
            np.testing.assert_array_equal(actual, owned < live)


class _ArrayRef:
    def __init__(self, array: np.ndarray, sl: tuple | None = None):
        self._buf = array
        self._sl = sl

    @staticmethod
    def _norm(idx) -> tuple:
        if not isinstance(idx, tuple):
            idx = (idx,)
        return tuple(
            slice(int(s.start), int(s.start) + int(s.size))
            if hasattr(s, "start") and hasattr(s, "size") and not isinstance(s, slice)
            else s
            for s in idx
        )

    @property
    def shape(self) -> tuple[int, ...]:
        return self._buf[self._sl].shape if self._sl is not None else self._buf.shape

    @property
    def at(self):
        parent = self

        class _At:
            def __getitem__(self, idx):
                assert parent._sl is None
                return _ArrayRef(parent._buf, parent._norm(idx))

        return _At()

    def __getitem__(self, idx):
        view = self._buf[self._sl] if self._sl is not None else self._buf
        if idx is Ellipsis:
            return jnp.asarray(view)
        return jnp.asarray(view[self._norm(idx)])

    def __setitem__(self, idx, val):
        arr = np.asarray(val)
        if self._sl is not None:
            assert idx is Ellipsis
            self._buf[self._sl] = arr
        elif idx is Ellipsis:
            self._buf[...] = arr
        else:
            self._buf[self._norm(idx)] = arr


@pytest.mark.parametrize("tokens", [512, 2048, 4096])
@pytest.mark.parametrize("local_experts", [56, 112])
def test_metadata_steps_ep16_and_ep8(tokens, local_experts):
    import jax
    from jax.experimental import pallas as pl

    rows = tokens // 32
    rng = np.random.default_rng(1030 + local_experts + tokens)
    ids_by_rank = rng.integers(0, 896, size=(32, rows, 16), dtype=np.int32)
    probs_by_rank = rng.uniform(0.01, 1.0, size=(32, rows, 16)).astype(np.float32)
    probs_by_rank[-1, -5:, :] = 0.0
    ids_all = ids_by_rank.reshape(tokens, 16)
    probs_all = probs_by_rank.reshape(tokens, 16)
    payloads = []
    for r in range(32):
        ids_r, probs_r = ids_by_rank[r], probs_by_rank[r]
        if rows < 128:
            w = np.concatenate([ids_r, probs_r.view(np.int32)], axis=1)
        else:
            w = np.concatenate(
                [
                    ids_r.reshape(rows // 8, 128),
                    probs_r.view(np.int32).reshape(rows // 8, 128),
                ],
                axis=0,
            )
        payloads.append(w.reshape(rows // 4, 128))
    wire_np = np.concatenate(payloads, axis=0)
    expert_tp = 32 // (896 // local_experts)
    live_mask = probs_all != 0
    hist_896 = np.bincount(ids_all[live_mask], minlength=896)
    expected_max_group = max(
        int(hist_896[g * local_experts : (g + 1) * local_experts].sum())
        for g in range(896 // local_experts)
    )

    if tokens == 512:
        wire = jnp.asarray(wire_np)

        def kern(wire_ref, rank_ref, keys_ref, probs_ref, tids_ref, counts_ref):
            for _ in fused._metadata_steps(
                wire_ref,
                keys_ref,
                probs_ref,
                tids_ref,
                counts_ref,
                None,
                tokens=tokens,
                rank=rank_ref[0],
                counts_ready=lambda: None,
                local_experts=local_experts,
            ):
                pass

        out_shapes = (
            jax.ShapeDtypeStruct((rows * 4, 128), jnp.int32),
            jax.ShapeDtypeStruct((tokens // 8, 128), jnp.float32),
            jax.ShapeDtypeStruct((tokens // 8, 128), jnp.int32),
            jax.ShapeDtypeStruct((local_experts,), jnp.int32),
        )
        fn = jax.jit(pl.pallas_call(kern, out_shape=out_shapes, interpret=True))
        ranks_to_check = (0, 3, 4, 17, 31)
    else:
        fn = None
        ranks_to_check = (0, 17, 31)

    for rank in ranks_to_check:
        if fn is not None:
            _, probs_out, tids_out, counts_out = fn(
                wire, jnp.array([rank], dtype=jnp.int32)
            )
        else:
            keys_buf = np.zeros((rows * 4, 128), dtype=np.int32)
            probs_buf = np.zeros((tokens // 8, 128), dtype=np.float32)
            tids_buf = np.zeros((tokens // 8, 128), dtype=np.int32)
            counts_buf = np.zeros((local_experts,), dtype=np.int32)
            max_buf = np.zeros((1,), dtype=np.int32)
            for _ in fused._metadata_steps(
                _ArrayRef(wire_np),
                _ArrayRef(keys_buf),
                _ArrayRef(probs_buf),
                _ArrayRef(tids_buf),
                _ArrayRef(counts_buf),
                _ArrayRef(max_buf),
                tokens=tokens,
                rank=jnp.int32(rank),
                counts_ready=lambda: None,
                local_experts=local_experts,
            ):
                pass
            assert int(max_buf[0]) == expected_max_group
            probs_out, tids_out, counts_out = probs_buf, tids_buf, counts_buf
        first_exp = (rank // expert_tp) * local_experts
        expected_counts = hist_896[first_exp : first_exp + local_experts]
        np.testing.assert_array_equal(np.asarray(counts_out), expected_counts)
        total_local = int(expected_counts.sum())
        expected_routes = []
        for e in range(first_exp, first_exp + local_experts):
            tok_idx, slot_idx = np.where((ids_all == e) & live_mask)
            for idx in np.lexsort((slot_idx, tok_idx)):
                expected_routes.append(
                    (tok_idx[idx], probs_all[tok_idx[idx], slot_idx[idx]])
                )
        np.testing.assert_array_equal(
            np.asarray(tids_out).reshape(-1)[:total_local],
            [t for t, _ in expected_routes],
        )
        np.testing.assert_array_equal(
            np.asarray(probs_out).reshape(-1)[:total_local],
            np.array([p for _, p in expected_routes], dtype=np.float32),
        )


def test_mla_projections_head_major_matches_transposed_masked():
    rng = np.random.default_rng(20261106)
    tokens, hidden_dim, q_rank, heads = 16, 128, 64, 3
    hidden = jnp.asarray(
        rng.standard_normal((tokens, hidden_dim), dtype=np.float32) * 0.1,
        dtype=jnp.bfloat16,
    )
    weights = kernels.ProjectionWeights(
        query_down_projection=jnp.asarray(
            rng.standard_normal((hidden_dim, q_rank), dtype=np.float32) * 0.1,
            dtype=jnp.bfloat16,
        ),
        query_normalization=jnp.ones((q_rank,), dtype=jnp.bfloat16),
        query_up_projection=jnp.asarray(
            rng.standard_normal((q_rank, heads * 256), dtype=np.float32) * 0.1,
            dtype=jnp.bfloat16,
        ),
        key_value_down_projection=jnp.asarray(
            rng.standard_normal((hidden_dim, 576), dtype=np.float32) * 0.1,
            dtype=jnp.bfloat16,
        ),
        key_value_normalization=jnp.ones((512,), dtype=jnp.bfloat16),
        key_value_up_projection=jnp.asarray(
            rng.standard_normal((512, heads * 256), dtype=np.float32) * 0.1,
            dtype=jnp.bfloat16,
        ),
        output_gate_projection=jnp.asarray(
            rng.standard_normal((hidden_dim, heads * 128), dtype=np.float32) * 0.1,
            dtype=jnp.bfloat16,
        ),
    )
    length = jnp.asarray([11], dtype=jnp.int32)
    q_ref, k_ref, v_ref, g_ref, lat_ref = kernels.mla_projections(
        hidden, weights, return_latent=True, query_scale=kernels.SCALE_LOG2
    )
    real = (jnp.arange(tokens) < length.reshape(()))[None, :, None]
    k_ref = jnp.where(real, k_ref.transpose(1, 0, 2), 0)
    v_ref = jnp.where(real, v_ref.transpose(1, 0, 2), 0)
    q_ref = q_ref.transpose(1, 0, 2)

    q_hm, k_hm, v_hm, g_hm, lat_hm = kernels.mla_projections(
        hidden,
        weights,
        return_latent=True,
        query_scale=kernels.SCALE_LOG2,
        head_major=True,
        length=length,
    )
    np.testing.assert_array_equal(np.asarray(lat_hm), np.asarray(lat_ref))
    np.testing.assert_array_equal(np.asarray(g_hm), np.asarray(g_ref))
    np.testing.assert_allclose(
        np.asarray(q_hm, dtype=np.float32),
        np.asarray(q_ref, dtype=np.float32),
        rtol=1e-2,
        atol=2e-3,
    )
    np.testing.assert_allclose(
        np.asarray(k_hm, dtype=np.float32),
        np.asarray(k_ref, dtype=np.float32),
        rtol=1e-2,
        atol=2e-3,
    )
    np.testing.assert_allclose(
        np.asarray(v_hm, dtype=np.float32),
        np.asarray(v_ref, dtype=np.float32),
        rtol=1e-2,
        atol=2e-3,
    )
    np.testing.assert_array_equal(np.asarray(k_hm[:, 11:, :]), 0)
    np.testing.assert_array_equal(np.asarray(v_hm[:, 11:, :]), 0)


def test_intermediate_window_elides_logits_and_sampler():
    import types

    import torch

    from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3.prefill import KimiPrefiller
    from vllm_torchtpu.models.kimi_k3.megakernel.program import RequestInput, StepInput

    prefiller = KimiPrefiller.__new__(KimiPrefiller)
    prefiller._writer = object()
    prefiller.draft_latents = None
    prefiller.program = types.SimpleNamespace(
        windows=(512, 1024, 2048, 4096, 8192, 16384),
        capacity=98304,
    )
    prefiller.slots = 16
    prefiller.page_tokens = 256
    prefiller.compute_logits = True
    prefiller._requests = {}
    prefiller._kda_groups = []
    prefiller.config = types.SimpleNamespace(max_num_seqs=4)

    window_calls = []
    first_token_calls = []

    def fake_read_hit_states(rows, hits):
        pass

    def fake_run_window(rows, by_row, window, members, *, need_logits=True):
        window_calls.append((window, len(members), need_logits))
        if not need_logits:
            return None
        out = torch.zeros((len(members), 8), dtype=torch.float32)
        for s, (piece, _) in enumerate(members):
            out[s, 0] = float(100 + piece.row)
        return out

    def fake_first_tokens(rows, logits, done):
        first_token_calls.append((dict(logits), set(done)))
        if not done:
            return torch.full((max(len(rows), 1), 1), -1, dtype=torch.int32)
        return torch.tensor(
            [
                [int(logits[i][0, 0].item()) if i in done else -1]
                for i in range(len(rows))
            ],
            dtype=torch.int32,
        )

    prefiller._read_hit_states = fake_read_hit_states
    prefiller._run_window = fake_run_window
    prefiller._first_tokens = fake_first_tokens

    tokens = np.zeros(80128, dtype=np.int32)
    blocks = (np.zeros(400, dtype=np.int32),)

    r0 = RequestInput("r0", 0, 16384, tokens, 80128, 80128, blocks)
    out1 = prefiller.step(StepInput(requests=[r0]))
    assert window_calls == [(16384, 1, False)]
    assert first_token_calls == [({}, set())]
    assert out1.tokens.tolist() == [[-1]]

    window_calls.clear()
    first_token_calls.clear()
    prefiller._requests["r0"].start = 65536
    r0_end = RequestInput("r0", 65536, 14592, tokens, 80128, 80128, blocks)
    r1_beg = RequestInput("r1", 0, 1792, tokens, 80128, 80128, blocks)
    out2 = prefiller.step(StepInput(requests=[r0_end, r1_beg]))
    assert window_calls == [(16384, 2, True)]
    assert len(first_token_calls) == 1
    logits_map, done_set = first_token_calls[0]
    assert done_set == {0}
    assert set(logits_map.keys()) == {0}
    assert out2.tokens.tolist() == [[100], [-1]]


def test_intermediate_windows_and_two_slot_packing():
    assert prefill.SERIAL_SLOTS == 2
    assert prefill.WINDOWS == (
        512,
        1024,
        2048,
        4096,
        8192,
        10240,
        16384,
    )
    wins = prefill.prefill_windows(16448, 163840)
    assert wins == prefill.WINDOWS
    for length, expected_win in (
        (5375, 8192),
        (9983, 10240),
        (11519, 16384),
        (13055, 16384),
        (14591, 16384),
    ):
        packed = prefill.pack_windows(
            [prefill.Piece(0, 0, length, None, 512)], wins, slots=2, capacity=163840
        )
        assert len(packed) == 1
        assert packed[0][0] == expected_win
    packed_pair = prefill.pack_windows(
        [
            prefill.Piece(0, 65536, 80128, None, 512),
            prefill.Piece(1, 0, 1536, None, 512),
        ],
        wins,
        slots=2,
        capacity=163840,
    )
    assert len(packed_pair) == 1
    assert packed_pair[0][0] == 16384
    assert [(m[0].row, m[1]) for m in packed_pair[0][1]] == [(0, 0), (1, 14848)]


def test_mixture_tile_divides_all_serving_windows():
    import types

    from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3 import prefill

    expected_tiles = {
        512: 16,
        1024: 32,
        2048: 64,
        4096: 128,
        6144: 64,
        8192: 128,
        10240: 64,
        12288: 128,
        14336: 64,
        16384: 128,
    }
    for w in prefill.WINDOWS:
        rows = w // 32
        tile = kernels._mixture_tile(rows)
        assert tile == expected_tiles[w]
        assert rows % tile == 0

    dummy_prefix = types.SimpleNamespace(shape=(192, 7168))
    dummy_blocks = types.SimpleNamespace(shape=(8, 192, 7168))
    with pytest.raises(ValueError, match="multiple of tile"):
        kernels.mixture_norm_tiled_body(
            None,
            dummy_prefix,
            dummy_blocks,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            tile=128,
            eps=1e-5,
        )


def _build_mla_projection_fixture(
    rng: np.random.Generator,
    *,
    tokens: int,
    hidden_dim: int,
    q_rank: int,
    live_tokens: int | None,
):
    heads = 3
    hidden_np = rng.standard_normal((tokens, hidden_dim), dtype=np.float32) * 0.2
    if live_tokens is not None:
        hidden_np[live_tokens:] = 999.0
    hidden_bf16 = jnp.asarray(hidden_np, dtype=jnp.bfloat16)
    weights = kernels.ProjectionWeights(
        query_down_projection=jnp.asarray(
            rng.standard_normal((hidden_dim, q_rank), dtype=np.float32) * 0.1,
            dtype=jnp.bfloat16,
        ),
        query_normalization=jnp.asarray(
            0.75 + 0.5 * rng.random((q_rank,), dtype=np.float32),
            dtype=jnp.bfloat16,
        ),
        query_up_projection=jnp.asarray(
            rng.standard_normal((q_rank, heads * 256), dtype=np.float32) * 0.1,
            dtype=jnp.bfloat16,
        ),
        key_value_down_projection=jnp.asarray(
            rng.standard_normal((hidden_dim, 576), dtype=np.float32) * 0.1,
            dtype=jnp.bfloat16,
        ),
        key_value_normalization=jnp.asarray(
            0.75 + 0.5 * rng.random((512,), dtype=np.float32),
            dtype=jnp.bfloat16,
        ),
        key_value_up_projection=jnp.asarray(
            rng.standard_normal((512, heads * 256), dtype=np.float32) * 0.1,
            dtype=jnp.bfloat16,
        ),
        output_gate_projection=jnp.asarray(
            rng.standard_normal((hidden_dim, heads * 128), dtype=np.float32) * 0.1,
            dtype=jnp.bfloat16,
        ),
    )
    return hidden_bf16, weights


@pytest.mark.parametrize(
    ("tokens", "return_latent", "live_tokens"),
    [
        (512, True, 380),
        (768, False, None),
        (768, True, 611),
    ],
)
def test_should_match_dense_mla_projections_when_full_pallas_tile_streams(
    tokens: int,
    return_latent: bool,
    live_tokens: int | None,
):
    rng = np.random.default_rng(20261112 + tokens + int(return_latent))
    hidden_bf16, weights = _build_mla_projection_fixture(
        rng,
        tokens=tokens,
        hidden_dim=256,
        q_rank=256,
        live_tokens=live_tokens,
    )
    length = (
        None if live_tokens is None else jnp.asarray([live_tokens], dtype=jnp.int32)
    )
    q_ref, k_ref, v_ref, g_ref, lat_ref = kernels.mla_projections(
        hidden_bf16,
        weights,
        return_latent=True,
        query_scale=kernels.SCALE_LOG2,
        head_major=False,
    )
    q_ref = q_ref.transpose(1, 0, 2)
    k_ref = k_ref.transpose(1, 0, 2)
    v_ref = v_ref.transpose(1, 0, 2)
    if live_tokens is not None:
        real = (jnp.arange(tokens) < live_tokens)[None, :, None]
        k_ref = jnp.where(real, k_ref, 0)
        v_ref = jnp.where(real, v_ref, 0)

    outs = kernels.mla_projections(
        hidden_bf16,
        weights,
        return_latent=return_latent,
        query_scale=kernels.SCALE_LOG2,
        head_major=True,
        length=length,
        interpret=True,
    )
    if return_latent:
        q_hm, k_hm, v_hm, g_hm, lat_hm = outs
        np.testing.assert_allclose(
            np.asarray(lat_hm, dtype=np.float32),
            np.asarray(lat_ref, dtype=np.float32),
            rtol=1e-2,
            atol=2e-2,
        )
    else:
        q_hm, k_hm, v_hm, g_hm = outs

    np.testing.assert_allclose(
        np.asarray(g_hm, dtype=np.float32),
        np.asarray(g_ref, dtype=np.float32),
        rtol=1e-2,
        atol=2e-2,
    )
    for actual, expected in ((q_hm, q_ref), (k_hm, k_ref), (v_hm, v_ref)):
        np.testing.assert_allclose(
            np.asarray(actual, dtype=np.float32),
            np.asarray(expected, dtype=np.float32),
            rtol=1e-2,
            atol=2e-2,
        )
    np.testing.assert_array_equal(np.asarray(k_hm[:, :, 192:]), 0)
    if live_tokens is not None:
        np.testing.assert_array_equal(np.asarray(k_hm[:, live_tokens:, :]), 0)
        np.testing.assert_array_equal(np.asarray(v_hm[:, live_tokens:, :]), 0)


@pytest.mark.parametrize(
    ("tokens", "return_latent", "live_tokens"),
    [
        (16, True, 11),
        (512, False, None),
        (768, True, 611),
    ],
)
def test_should_match_dense_mla_projections_when_sp_axis_pallas_streams(
    tokens: int,
    return_latent: bool,
    live_tokens: int | None,
):
    rng = np.random.default_rng(20261201 + tokens + int(return_latent))
    hidden_bf16, weights = _build_mla_projection_fixture(
        rng,
        tokens=tokens,
        hidden_dim=256,
        q_rank=256,
        live_tokens=live_tokens,
    )
    length = (
        None if live_tokens is None else jnp.asarray([live_tokens], dtype=jnp.int32)
    )
    q_ref, k_ref, v_ref, g_ref, lat_ref = kernels.mla_projections(
        hidden_bf16,
        weights,
        return_latent=True,
        query_scale=kernels.SCALE_LOG2,
        head_major=False,
    )
    q_ref = q_ref.transpose(1, 0, 2)
    k_ref = k_ref.transpose(1, 0, 2)
    v_ref = v_ref.transpose(1, 0, 2)
    if live_tokens is not None:
        real = (jnp.arange(tokens) < live_tokens)[None, :, None]
        k_ref = jnp.where(real, k_ref, 0)
        v_ref = jnp.where(real, v_ref, 0)

    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:1]), ("k3_sp",))
    rep = jax.sharding.PartitionSpec()

    @jax.shard_map(
        mesh=mesh,
        in_specs=(rep, rep),
        out_specs=(rep,) * (5 if return_latent else 4),
        check_vma=False,
    )
    def run_sp(h, w):
        return kernels.mla_projections(
            h,
            w,
            return_latent=return_latent,
            query_scale=kernels.SCALE_LOG2,
            sp_axis="k3_sp",
            head_major=True,
            length=length,
            interpret=True,
        )

    outs = run_sp(hidden_bf16, weights)
    if return_latent:
        q_sp, k_sp, v_sp, g_sp, lat_sp = outs
        np.testing.assert_allclose(
            np.asarray(lat_sp, dtype=np.float32),
            np.asarray(lat_ref, dtype=np.float32),
            rtol=1e-2,
            atol=2e-2,
        )
    else:
        q_sp, k_sp, v_sp, g_sp = outs

    np.testing.assert_allclose(
        np.asarray(g_sp, dtype=np.float32),
        np.asarray(g_ref, dtype=np.float32),
        rtol=1e-2,
        atol=2e-2,
    )
    for actual, expected in ((q_sp, q_ref), (k_sp, k_ref), (v_sp, v_ref)):
        np.testing.assert_allclose(
            np.asarray(actual, dtype=np.float32),
            np.asarray(expected, dtype=np.float32),
            rtol=1e-2,
            atol=2e-2,
        )
    if live_tokens is not None:
        np.testing.assert_array_equal(np.asarray(k_sp[:, live_tokens:, :]), 0)
        np.testing.assert_array_equal(np.asarray(v_sp[:, live_tokens:, :]), 0)


def _pack_8k_route_wire(
    ids_by_rank: np.ndarray, probs_by_rank: np.ndarray
) -> np.ndarray:
    rows = ids_by_rank.shape[1]
    payloads = []
    for rank_idx in range(32):
        ids_r, probs_r = ids_by_rank[rank_idx], probs_by_rank[rank_idx]
        packed = np.concatenate(
            [
                ids_r.reshape(rows // 8, 128),
                probs_r.view(np.int32).reshape(rows // 8, 128),
            ],
            axis=0,
        )
        payloads.append(packed.reshape(rows // 4, 128))
    return np.concatenate(payloads, axis=0)


def test_should_match_exact_routes_when_8k_ep8_keep_half_metadata_steps():
    tokens, local_experts = 8192, 112
    rows = tokens // 32
    rng = np.random.default_rng(20261112)
    ids_by_rank = rng.integers(0, 896, size=(32, rows, 16), dtype=np.int32)
    ids_by_rank[:11, :, :16] = rng.integers(
        0, local_experts, size=(11, rows, 16), dtype=np.int32
    )
    probs_by_rank = rng.uniform(0.01, 1.0, size=(32, rows, 16)).astype(np.float32)
    probs_by_rank[-1, -37:, :] = 0.0
    wire_np = _pack_8k_route_wire(ids_by_rank, probs_by_rank)
    ids_all = ids_by_rank.reshape(tokens, 16)
    probs_all = probs_by_rank.reshape(tokens, 16)
    live_mask = probs_all != 0
    hist_896 = np.bincount(ids_all[live_mask], minlength=896)
    expert_tp = 32 // (896 // local_experts)

    for rank in (0, 17, 31):
        keys_buf = np.zeros((rows * 4, 128), dtype=np.int32)
        probs_buf = np.zeros((tokens // 8, 128), dtype=np.float32)
        tids_buf = np.zeros((tokens // 8, 128), dtype=np.int32)
        counts_buf = np.zeros((local_experts,), dtype=np.int32)
        for _ in fused._metadata_steps(  # noqa: SLF001 -- direct Pallas helper spec
            _ArrayRef(wire_np),
            _ArrayRef(keys_buf),
            _ArrayRef(probs_buf),
            _ArrayRef(tids_buf),
            _ArrayRef(counts_buf),
            None,
            tokens=tokens,
            rank=jnp.int32(rank),
            counts_ready=lambda: None,
            local_experts=local_experts,
        ):
            pass
        first_exp = (rank // expert_tp) * local_experts
        expected_counts = hist_896[first_exp : first_exp + local_experts]
        np.testing.assert_array_equal(counts_buf, expected_counts)
        total_local = int(expected_counts.sum())
        assert total_local <= 65536
        expected_tids = []
        expected_probs = []
        for expert in range(first_exp, first_exp + local_experts):
            tok_idx, slot_idx = np.where((ids_all == expert) & live_mask)
            order = np.lexsort((slot_idx, tok_idx))
            expected_tids.append(tok_idx[order])
            expected_probs.append(probs_all[tok_idx[order], slot_idx[order]])
        np.testing.assert_array_equal(
            tids_buf.reshape(-1)[:total_local], np.concatenate(expected_tids)
        )
        np.testing.assert_array_equal(
            probs_buf.reshape(-1)[:total_local], np.concatenate(expected_probs)
        )


def test_should_sort_all_routes_when_8k_ep16_maximum_present():
    tokens, local_experts = 8192, 56
    rows = tokens // 32
    rng = np.random.default_rng(20261113)
    ids_by_rank = rng.integers(0, 896, size=(32, rows, 16), dtype=np.int32)
    ids_by_rank[:18, :, :] = rng.integers(
        0, local_experts, size=(18, rows, 16), dtype=np.int32
    )
    probs_by_rank = rng.uniform(0.05, 1.0, size=(32, rows, 16)).astype(np.float32)
    probs_by_rank[-1, -13:, :] = 0.0
    wire_np = _pack_8k_route_wire(ids_by_rank, probs_by_rank)
    ids_all = ids_by_rank.reshape(tokens, 16)
    probs_all = probs_by_rank.reshape(tokens, 16)
    live_mask = probs_all != 0
    hist_896 = np.bincount(ids_all[live_mask], minlength=896)
    expected_max = max(
        int(hist_896[g * local_experts : (g + 1) * local_experts].sum())
        for g in range(896 // local_experts)
    )
    assert expected_max > 65536

    keys_buf = np.zeros((rows * 4, 128), dtype=np.int32)
    probs_buf = np.zeros((tokens // 8, 128), dtype=np.float32)
    tids_buf = np.zeros((tokens // 8, 128), dtype=np.int32)
    counts_buf = np.zeros((local_experts,), dtype=np.int32)
    max_buf = np.zeros((1,), dtype=np.int32)
    for _ in fused._metadata_steps(  # noqa: SLF001 -- direct Pallas helper spec
        _ArrayRef(wire_np),
        _ArrayRef(keys_buf),
        _ArrayRef(probs_buf),
        _ArrayRef(tids_buf),
        _ArrayRef(counts_buf),
        _ArrayRef(max_buf),
        tokens=tokens,
        rank=jnp.int32(0),
        counts_ready=lambda: None,
        local_experts=local_experts,
    ):
        pass
    assert int(max_buf[0]) == expected_max
    total_local = int(hist_896[:local_experts].sum())
    assert total_local > 65536
    expected_tids = []
    expected_probs = []
    for expert in range(local_experts):
        tok_idx, slot_idx = np.where((ids_all == expert) & live_mask)
        order = np.lexsort((slot_idx, tok_idx))
        expected_tids.append(tok_idx[order])
        expected_probs.append(probs_all[tok_idx[order], slot_idx[order]])
    np.testing.assert_array_equal(
        tids_buf.reshape(-1)[:total_local], np.concatenate(expected_tids)
    )
    np.testing.assert_array_equal(
        probs_buf.reshape(-1)[:total_local], np.concatenate(expected_probs)
    )


@pytest.mark.parametrize("n_ranks", [8, 32])
def test_should_cover_all_ranks_uniquely_when_latent_half_ring_layout(n_ranks):
    layout = fused._latent_half_stream_layout(256, n_ranks)  # noqa: SLF001 -- geometry invariant
    assert len(layout) == 6
    covered_rows: list[int] = []
    for family in range(3):
        pos_stream, neg_stream = layout[2 * family], layout[2 * family + 1]
        assert pos_stream[0] == family and neg_stream[0] == family
        assert pos_stream[1] == 1 and neg_stream[1] == -1
        assert pos_stream[2] == neg_stream[2] and pos_stream[3] == neg_stream[3]
        covered_rows.extend(range(pos_stream[2], pos_stream[2] + pos_stream[3]))
        cycles = (
            [list(range(n_ranks))]
            if n_ranks == 8
            else [
                [
                    chip * 2 + core
                    for chip in kernels.CYCLES[family]
                    for core in range(2)
                ]
            ]
        )
        for cyc in cycles:
            pos_of = {r: idx for idx, r in enumerate(cyc)}
            for rank in range(n_ranks):
                pos = pos_of[rank]
                pos_owners = [
                    cyc[(pos - step) & (n_ranks - 1)]
                    for step in range(1, pos_stream[4] + 1)
                ]
                neg_owners = [
                    cyc[(pos + step) & (n_ranks - 1)]
                    for step in range(1, neg_stream[4] + 1)
                ]
                received = [rank, *pos_owners, *neg_owners]
                assert len(received) == n_ranks
                assert set(received) == set(range(n_ranks))
        for _, _, _, _, max_hop in (pos_stream, neg_stream):
            assert max_hop + 2 < fused._HALF_LATENT_META_STRIDE  # noqa: SLF001 -- layout cap
            sent = sum(1 for step in range(1, max_hop + 1) if step < max_hop - 1)
            consumed = sum(1 for step in range(1, max_hop) if step >= 2)
            assert sent == consumed
    assert covered_rows == list(range(256))


def test_should_match_vmem_epilogue_bitwise_and_zero_masked_tail_when_staged_hbm():
    rows, width, hidden = 64, 3584, 7168
    live_tokens = rows * 32 - 37
    rng = np.random.default_rng(20261119)
    reduced_np = rng.standard_normal((rows, width)).astype(np.float32) * 0.2
    shared_np = rng.standard_normal((rows, hidden)).astype(np.float32) * 0.1
    reduced_np[-1, :] = 50.0
    shared_np[-1, :] = 999.0
    reduced_bf16 = jnp.asarray(reduced_np, jnp.bfloat16)
    shared_bf16 = jnp.asarray(shared_np, jnp.bfloat16)
    norm_f32 = jnp.asarray(0.75 + 0.5 * rng.random((1, 1, width)), jnp.float32)
    up_bf16 = jnp.asarray(rng.standard_normal((1, width, hidden)) * 0.02, jnp.bfloat16)
    prefix_bf16 = jnp.asarray(rng.standard_normal((rows, hidden)) * 0.3, jnp.bfloat16)
    control = jnp.array([live_tokens, 0], dtype=jnp.int32)

    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:1]), ("k3_sp",))
    rep = jax.sharding.PartitionSpec()
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)

    def make_epilogue_call(*, staged_hbm: bool):
        def body(ctrl_r, red_r, shr_r, nrm_r, up_r, pfx_r, out_r):
            fused._epilogue(  # noqa: SLF001 -- direct Pallas helper spec
                ctrl_r,
                red_r,
                shr_r,
                nrm_r,
                up_r,
                pfx_r,
                out_r,
                staged_hbm=staged_hbm,
            )

        rs_spec = hbm if staged_hbm else vmem
        return pl.pallas_call(
            body,
            out_shape=jax.ShapeDtypeStruct((rows, hidden), jnp.bfloat16),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=1,
                in_specs=(rs_spec, rs_spec, hbm, hbm, hbm),
                out_specs=hbm,
            ),
            interpret=True,
        )

    @jax.shard_map(
        mesh=mesh, in_specs=(rep,) * 6, out_specs=(rep, rep), check_vma=False
    )
    def run_pair(ctrl, red, shr, nrm, up, pfx):
        return (
            make_epilogue_call(staged_hbm=False)(ctrl, red, shr, nrm, up, pfx),
            make_epilogue_call(staged_hbm=True)(ctrl, red, shr, nrm, up, pfx),
        )

    out_vmem, out_hbm = run_pair(
        control, reduced_bf16, shared_bf16, norm_f32, up_bf16, prefix_bf16
    )
    np.testing.assert_array_equal(np.asarray(out_hbm), np.asarray(out_vmem))
    out_vmem_masked, out_hbm_masked = run_pair(
        jnp.array([17, 0], dtype=jnp.int32),
        reduced_bf16,
        shared_bf16,
        norm_f32,
        up_bf16,
        prefix_bf16,
    )
    np.testing.assert_array_equal(
        np.asarray(out_hbm_masked), np.asarray(out_vmem_masked)
    )
    np.testing.assert_array_equal(
        np.asarray(out_hbm_masked)[17:], np.asarray(prefix_bf16)[17:]
    )


def test_should_route_collapsed_compound_boundaries_and_clamp_max_model_len():
    wins = prefill.prefill_windows(16448, 163840)
    assert wins == (512, 1024, 2048, 4096, 8192, 10240, 16384)
    for length, expected_win in (
        (4096, 4096),
        (4097, 8192),
        (6144, 8192),
        (8192, 8192),
        (8193, 10240),
        (10240, 10240),
        (10241, 16384),
        (12288, 16384),
        (14336, 16384),
        (16384, 16384),
    ):
        packed = prefill.pack_windows(
            [prefill.Piece(0, 0, length, None, 512)], wins, slots=2, capacity=163840
        )
        assert [w for w, _ in packed] == [expected_win]

    packed_8k = prefill.pack_windows(
        [
            prefill.Piece(0, 0, 4352, None, 512),
            prefill.Piece(1, 0, 3584, None, 512),
        ],
        wins,
        slots=2,
        capacity=163840,
    )
    assert [w for w, _ in packed_8k] == [8192]
    assert [(m[0].row, m[1]) for m in packed_8k[0][1]] == [(0, 0), (1, 4608)]

    staged_8k = prefill.pack_windows(
        [
            prefill.Piece(0, 0, 4352, None, 512),
            prefill.Piece(1, 0, 12033, None, 512),
        ],
        wins,
        slots=2,
        capacity=163840,
    )
    assert [w for w, _ in staged_8k] == [8192, 16384]

    assert prefill.prefill_windows(16448, 6000) == (512, 1024, 2048, 4096, 8192)
    assert prefill.prefill_windows(16448, 9000) == (
        512,
        1024,
        2048,
        4096,
        8192,
        10240,
    )
    assert prefill.prefill_windows(16448, 12000) == (
        512,
        1024,
        2048,
        4096,
        8192,
        10240,
        16384,
    )


def test_should_preserve_bitwise_kda_outputs_when_padded_into_collapsed_window():
    rng = np.random.default_rng(20261114)
    seq_len, live_len, hidden = 512, 317, 256
    kda_fn = kernels.make_kda_tiled(seq_len=seq_len, tile=64, interpret=True)

    hidden_clean = (rng.standard_normal((seq_len, hidden)) * 0.1).astype(np.float32)
    hidden_poison = hidden_clean.copy()
    hidden_poison[live_len:] = 999.0
    hidden_mut = hidden_clean.copy()
    hidden_mut[live_len - 1] += 2.0

    gate_w = jnp.asarray(rng.standard_normal((hidden, 128)) * 0.05, jnp.bfloat16)
    fb_w = jnp.asarray(rng.standard_normal((128, 384)) * 0.05, jnp.bfloat16)
    proj_w = jnp.asarray(rng.standard_normal((hidden, 1664)) * 0.05, jnp.bfloat16)
    conv_w = jnp.asarray(rng.standard_normal((4, 3, 3, 128)) * 0.05, jnp.float32)
    alog = jnp.full((3, 1), -1.5, jnp.float32)
    dt = jnp.zeros((3, 128), jnp.float32)
    norm = jnp.ones((8, 128), jnp.float32)
    rec0 = jnp.asarray(rng.standard_normal((3, 128, 128)) * 0.05, jnp.float32)
    hist0 = jnp.asarray(rng.standard_normal((3, 3, 3, 128)) * 0.05, jnp.bfloat16)
    length = jnp.asarray([live_len], jnp.int32)

    out_c, rec_c, hist_c = kda_fn(
        jnp.asarray(hidden_clean, jnp.bfloat16),
        gate_w,
        fb_w,
        proj_w,
        conv_w,
        alog,
        dt,
        norm,
        rec0,
        hist0,
        length,
    )
    out_p, rec_p, hist_p = kda_fn(
        jnp.asarray(hidden_poison, jnp.bfloat16),
        gate_w,
        fb_w,
        proj_w,
        conv_w,
        alog,
        dt,
        norm,
        rec0,
        hist0,
        length,
    )
    _, rec_m, _ = kda_fn(
        jnp.asarray(hidden_mut, jnp.bfloat16),
        gate_w,
        fb_w,
        proj_w,
        conv_w,
        alog,
        dt,
        norm,
        rec0,
        hist0,
        length,
    )

    np.testing.assert_array_equal(
        np.asarray(out_c)[:live_len], np.asarray(out_p)[:live_len]
    )
    np.testing.assert_array_equal(
        np.asarray(out_p)[live_len:],
        np.zeros((seq_len - live_len, 384), dtype=jnp.bfloat16),
    )
    np.testing.assert_array_equal(np.asarray(rec_c), np.asarray(rec_p))
    np.testing.assert_array_equal(np.asarray(hist_c), np.asarray(hist_p))
    assert float(np.max(np.abs(np.asarray(rec_c) - np.asarray(rec_m)))) > 1e-4
