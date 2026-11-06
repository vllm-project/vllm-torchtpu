# SPDX-License-Identifier: Apache-2.0
"""TPU test for the captured-routing replay: compiled on the device, the
replayed ids equal the CPU (float32-exact) ones for ids of every size."""

from types import SimpleNamespace

import numpy as np
import torch

from vllm_torchtpu.layers.adapter.routing_replay import RoutingReplay

WINDOWS, POSITIONS, TOPK, EXPERTS = 64, 4, 10, 512


def _layer(replay, device):
    layer = SimpleNamespace(
        layer_name="model.layers.0.mlp.experts",
        _experts_start=torch.tensor(0, dtype=torch.int32, device=device),
        global_num_experts=EXPERTS,
        moe_config=SimpleNamespace(ep_size=32),
    )
    layer._routing_replay_table = replay.table(layer, device)
    return layer


def test_replayed_ids_match_cpu_on_tpu(tmp_path):
    rng = np.random.default_rng(0)
    # Every id above 256 that bfloat16 cannot hold, and the rest below.
    pool = np.concatenate([np.arange(257, EXPERTS, 2), np.arange(0, 257)])
    ids = np.stack(
        [
            np.stack([rng.choice(pool, TOPK, replace=False) for _ in range(POSITIONS)])
            for _ in range(WINDOWS)
        ]
    )[None].astype(np.int16)
    path = tmp_path / "routes.npz"
    np.savez(path, ids=ids)
    replay = RoutingReplay(str(path))
    torch.manual_seed(0)
    num_tokens = 24 * POSITIONS
    logits = torch.randn(num_tokens, EXPERTS)
    hidden = torch.zeros(num_tokens, 8, dtype=torch.bfloat16)

    cpu_layer = _layer(replay, torch.device("cpu"))
    _, want = replay.route(cpu_layer, hidden, logits, TOPK, tp_size=4)
    tpu_layer = _layer(replay, torch.device("tpu"))

    @torch.compile(fullgraph=True)
    def route(h, lg):
        return replay.route(tpu_layer, h, lg, TOPK, tp_size=4)

    _, got = route(hidden.to("tpu"), logits.to("tpu"))
    got = got.cpu()
    assert (want >= 256).any()
    np.testing.assert_array_equal(got.numpy(), want.numpy())
