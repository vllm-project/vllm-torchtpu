# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Expert placement plans in the EP8 decode and prefill loaders."""

import json

import numpy as np
import pytest

from vllm_torchtpu.megakernel.kimi_k3.config import KernelConfig
from vllm_torchtpu.megakernel.kimi_k3.kernels import rank_loader

LAYERS = 3  # two MoE layers


def _expert_value(rank, layer, expert):
    return rank * 1e6 + layer * 1e3 + expert


def _write_rank(root, rank):
    path = root / f"rank{rank:02d}"
    path.mkdir(parents=True)
    families, blob, offset = {}, [], 0
    dense = {
        name: np.full((2, 3), rank, np.float32)
        for name in rank_loader.PREFILL_IO
        + rank_loader.PREFILL_COMMON
        + rank_loader.PREFILL_KDA
        + rank_loader.PREFILL_MLA
    }
    dense.update(
        k_o=np.full((2, 768, 3), rank, np.float32),
        m_o=np.full((2, 768, 3), rank, np.float32),
        latent_norm=np.full((2, 1, 8), rank, np.float32),
        # Column e of the router holds logical expert e.
        router=np.broadcast_to(np.arange(896, dtype=np.float32), (2, 5, 896)).copy(),
        router_bias=np.broadcast_to(-np.arange(896, dtype=np.float32), (2, 1, 896)).copy(),
    )
    for name, value in dense.items():
        families[name] = {
            "file": "dense.bin",
            "offset": offset,
            "shape": list(value.shape),
            "dtype": "float32",
        }
        blob.append(value.tobytes())
        offset += value.nbytes
    (path / "dense.bin").write_bytes(b"".join(blob))
    for name in rank_loader.EXPERT_FAMILIES:
        value = np.empty((2, 112, 2, 4), np.float32)
        for layer in range(2):
            for local in range(112):
                value[layer, local] = _expert_value(rank, layer, local)
        files = []
        for index, (lo, hi) in enumerate(((0, 1), (1, 2))):
            filename = f"{name}_{index}.bin"
            (path / filename).write_bytes(value[lo:hi].tobytes())
            files.append(filename)
        families[name] = {
            "files": files,
            "layer_ranges": [[0, 1], [1, 2]],
            "shape": [2, 112, 2, 4],
            "dtype": "float32",
        }
    (path / "index.json").write_text(
        json.dumps({"rank": rank, "layers": LAYERS, "families": families})
    )


@pytest.fixture(scope="module")
def preshard(tmp_path_factory):
    root = tmp_path_factory.mktemp("preshard")
    for rank in range(32):
        _write_rank(root, rank)
    cache = root / "prefill-sp-cache"
    cache.mkdir()
    families = {}
    for name in ("latent_down", "latent_up", "shared_gu_0", "shared_gu_1",
                 "shared_down_0", "shared_down_1"):
        np.zeros((2, 2), np.float32).tofile(cache / f"{name}.bin")
        families[name] = {"file": f"{name}.bin", "shape": [2, 2], "dtype": "float32"}
    (cache / "layout.json").write_text(json.dumps({"layers": LAYERS, "families": families}))
    rng = np.random.default_rng(0)
    order = np.stack([rng.permutation(896) for _ in range(LAYERS - 1)]).astype(np.int32)
    plan = root / "plan.npz"
    np.savez(plan, order=order)
    identity = root / "identity.npz"
    np.savez(identity, order=np.tile(np.arange(896, dtype=np.int32), (LAYERS - 1, 1)))
    return root, cache, plan, order, identity


def _decode(monkeypatch, root, rank, **config):
    monkeypatch.setattr(rank_loader, "_config", lambda: KernelConfig(**config))
    out = {}
    for _, name, value in rank_loader.iter_decode_rank(root, rank, layers=LAYERS, split_experts=True):
        family, _, part = name.partition("@")
        out.setdefault(family, []).append(np.asarray(value))
    return {name: np.concatenate(parts) if len(parts) > 1 else parts[0] for name, parts in out.items()}


def _prefill(monkeypatch, root, cache, rank, **config):
    monkeypatch.setattr(
        rank_loader, "_config", lambda: KernelConfig(prefill_expert_parallel=8, **config)
    )
    out = {}
    for _, name, value in rank_loader.iter_prefill_rank(
        root, cache, rank, layers=LAYERS, split_experts=True
    ):
        family, _, part = name.partition("@")
        out.setdefault(family, []).append(np.asarray(value))
    return {name: np.concatenate(parts) if len(parts) > 1 else parts[0] for name, parts in out.items()}


@pytest.mark.parametrize("rank", [0, 5, 31])
def test_decode_identity_plan_is_the_preshard(monkeypatch, preshard, rank):
    root, _, _, _, identity = preshard
    base = _decode(monkeypatch, root, rank)
    placed = _decode(monkeypatch, root, rank, decode_expert_plan=str(identity))
    for name in rank_loader.EXPERT_FAMILIES + ("router", "router_bias"):
        np.testing.assert_array_equal(placed[name], base[name])


@pytest.mark.parametrize("rank", [0, 6, 13, 31])
def test_decode_plan_places_experts_and_router(monkeypatch, preshard, rank):
    root, _, plan, order, _ = preshard
    placed = _decode(monkeypatch, root, rank, decode_expert_plan=str(plan))
    group, quarter = rank // 4, rank % 4
    for layer in range(LAYERS - 1):
        experts = order[layer, 112 * group:112 * group + 112]
        expected = np.array(
            [_expert_value(4 * (e // 112) + quarter, layer, e % 112) for e in experts], np.float32
        )
        for name in rank_loader.EXPERT_FAMILIES:
            assert placed[name].shape == (2, 112, 2, 4)
            np.testing.assert_array_equal(placed[name][layer, :, 0, 0], expected)
        np.testing.assert_array_equal(placed["router"][layer, 0], order[layer].astype(np.float32))
        np.testing.assert_array_equal(placed["router_bias"][layer, 0], -order[layer].astype(np.float32))


@pytest.mark.parametrize("rank", [3, 22])
def test_prefill_ep8_plan_matches_decode_plan(monkeypatch, preshard, rank):
    root, cache, plan, _, _ = preshard
    decode = _decode(monkeypatch, root, rank, decode_expert_plan=str(plan))
    prefill = _prefill(monkeypatch, root, cache, rank, prefill_expert_plan=str(plan))
    for name in rank_loader.EXPERT_FAMILIES + ("router", "router_bias"):
        np.testing.assert_array_equal(prefill[name], decode[name])


def test_decode_plan_rejects_hot_plans_and_bad_orders(monkeypatch, preshard, tmp_path):
    root, _, plan, order, _ = preshard
    monkeypatch.setattr(rank_loader, "hot_plan", lambda rank, layers: ("hot",))
    with pytest.raises(ValueError, match="hot or replica"):
        _decode(monkeypatch, root, 0, decode_expert_plan=str(plan))
    monkeypatch.undo()
    bad = order.copy()
    bad[0, 0] = bad[0, 1]
    np.savez(tmp_path / "bad.npz", order=bad)
    with pytest.raises(ValueError, match="permutations"):
        _decode(monkeypatch, root, 0, decode_expert_plan=str(tmp_path / "bad.npz"))
