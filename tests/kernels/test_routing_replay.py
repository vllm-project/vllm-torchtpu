# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the captured-routing replay (pure torch, no TPU)."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_torchtpu.layers.adapter.routing_replay import RoutingReplay

pytestmark = pytest.mark.cpu_test

LAYERS, WINDOWS, POSITIONS, TOPK, EXPERTS = 3, 64, 4, 10, 512


@pytest.fixture
def capture(tmp_path):
    # ids encode (layer, window, position) so each replayed row is traceable.
    rng = np.random.default_rng(0)
    ids = np.stack(
        [
            np.stack(
                [
                    np.stack(
                        [
                            rng.choice(EXPERTS, TOPK, replace=False)
                            for _ in range(POSITIONS)
                        ]
                    )
                    for _ in range(WINDOWS)
                ]
            )
            for _ in range(LAYERS)
        ]
    ).astype(np.int16)
    path = tmp_path / "routes.npz"
    np.savez(path, ids=ids)
    return str(path), ids


def _layer(index, experts_start, ep_size=32):
    return SimpleNamespace(
        layer_name=f"model.layers.{index}.mlp.experts",
        _experts_start=torch.tensor(experts_start, dtype=torch.int32),
        global_num_experts=EXPERTS,
        moe_config=SimpleNamespace(ep_size=ep_size),
    )


def test_rows_replay_one_window_per_group(capture):
    path, ids = capture
    replay = RoutingReplay(path)
    torch.manual_seed(0)
    num_tokens = 12 * POSITIONS
    logits = torch.randn(num_tokens, EXPERTS)
    hidden = torch.zeros(num_tokens, 8, dtype=torch.bfloat16)
    weights, got = replay.route(_layer(4, 0), hidden, logits, TOPK, tp_size=4)
    assert got.shape == (num_tokens, TOPK) and got.dtype == torch.int32
    assert weights.shape == (num_tokens, TOPK) and weights.dtype == hidden.dtype
    table = ids[4 % LAYERS]  # layer 4 replays capture layer 1
    got = got.numpy()
    for group in range(12):
        rows = got[group * POSITIONS : (group + 1) * POSITIONS]
        hits = [w for w in range(WINDOWS) if (table[w] == rows).all()]
        assert hits, f"group {group} is not one captured window"
    torch.testing.assert_close(
        weights.float().sum(-1), torch.ones(num_tokens), atol=1e-2, rtol=0
    )
    # Served layers carry the table as a buffer registered at weight load.
    layer = _layer(4, 0)
    layer._routing_replay_table = replay.table(layer, torch.device("cpu"))
    _, again = replay.route(layer, hidden, logits, TOPK, tp_size=4)
    np.testing.assert_array_equal(again.numpy(), got)


def test_deterministic_and_salted_by_dp_rank(capture):
    path, _ = capture
    replay = RoutingReplay(path)
    torch.manual_seed(1)
    logits = torch.randn(48, EXPERTS)
    hidden = torch.zeros(48, 8)
    # DP x TP4, 16 experts per rank: TP ranks of one engine route alike.
    _, a = replay.route(_layer(0, 0), hidden, logits, TOPK, tp_size=4)
    _, b = replay.route(_layer(0, 48), hidden, logits, TOPK, tp_size=4)
    _, c = replay.route(_layer(0, 64), hidden, logits, TOPK, tp_size=4)
    _, again = replay.route(_layer(0, 0), hidden, logits, TOPK, tp_size=4)
    assert torch.equal(a, again)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
    # New inputs (the next step) pick other windows.
    _, d = replay.route(_layer(0, 0), hidden, torch.randn(48, EXPERTS), TOPK, tp_size=4)
    assert not torch.equal(a, d)


def test_top_k_mismatch_raises(capture):
    path, _ = capture
    with pytest.raises(ValueError, match="top-10"):
        RoutingReplay(path).route(
            _layer(0, 0), torch.zeros(4, 8), torch.randn(4, EXPERTS), 8, tp_size=4
        )
