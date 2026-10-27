# SPDX-License-Identifier: Apache-2.0
"""Prefill EP16 expert placement (K3_PREFILL_EXPERT_PLAN): the planned loader
reproduces the pre-shard's regroup under the identity plan and places each
planned expert's rows under a random plan; the shipped serving plan is valid."""

import json
import os

import numpy as np
import pytest

from vllm_torchtpu.megakernel.kimi_k3.kernels import rank_loader as rl

pytestmark = pytest.mark.cpu_test

LAYERS = 3  # two MoE layers
SHAPES = {
    "expert_gu": (2, 112, 4, 8),
    "expert_gus": (2, 112, 1, 8),
    "expert_down": (2, 112, 6, 4),
    "expert_ds": (2, 112, 1, 4),
}
RANGES = [(0, 1), (1, 2)]


@pytest.fixture(scope="module")
def preshard(tmp_path_factory):
    root = tmp_path_factory.mktemp("preshard")
    rng = np.random.default_rng(0)
    data = {}
    for rank in range(32):
        directory = root / f"rank{rank:02d}"
        directory.mkdir()
        families = {}
        for name, shape in SHAPES.items():
            files = []
            for index, (lo, hi) in enumerate(RANGES):
                array = rng.standard_normal((hi - lo, *shape[1:])).astype(np.float32)
                data[(rank, name, index)] = array
                filename = f"{name}_{index}.bin"
                array.tofile(directory / filename)
                files.append(filename)
            families[name] = {
                "shape": list(shape),
                "dtype": "float32",
                "files": files,
                "layer_ranges": [list(r) for r in RANGES],
            }
        (directory / "index.json").write_text(
            json.dumps({"rank": rank, "layers": LAYERS, "families": families})
        )
    return str(root), data


def _planned(root, rank, name, order):
    rank_dir = rl.RankDir(root, rank, LAYERS)
    return np.concatenate(
        [
            read()
            for _, _, read in rl._planned_expert_parts(
                root, rank_dir, name, rank, order, LAYERS
            )
        ]
    )


@pytest.mark.parametrize("rank", [0, 1, 6, 31])
@pytest.mark.parametrize("name", list(SHAPES))
def test_identity_plan_is_the_regroup(preshard, rank, name):
    root, _ = preshard
    order = np.tile(np.arange(896), (LAYERS - 1, 1))
    rank_dir = rl.RankDir(root, rank, LAYERS)
    group = 4 * (rank // 4)
    quarters = (2 * (rank % 2), 2 * (rank % 2) + 1)
    quarter_dirs = [rl.RankDir(root, group + q, LAYERS) for q in quarters]
    expected = np.concatenate(
        [
            read()
            for _, _, read in rank_dir.regrouped_expert_parts(
                name, quarter_dirs, (rank // 2) % 2, 2
            )
        ]
    )
    np.testing.assert_array_equal(_planned(root, rank, name, order), expected)


@pytest.mark.parametrize("rank", [0, 3, 30])
@pytest.mark.parametrize("name", list(SHAPES))
def test_random_plan_places_each_expert(preshard, rank, name):
    root, data = preshard
    rng = np.random.default_rng(rank)
    order = np.stack([rng.permutation(896) for _ in range(LAYERS - 1)])
    got = _planned(root, rank, name, order)
    quarters = (2 * (rank % 2), 2 * (rank % 2) + 1)
    for layer in range(LAYERS - 1):
        for slot, expert in enumerate(
            order[layer, 56 * (rank // 2) : 56 * (rank // 2) + 56]
        ):
            parts = [
                data[(4 * (expert // 112) + q, name, layer)][0, expert % 112]
                for q in quarters
            ]
            if name in ("expert_gu", "expert_gus"):
                half = parts[0].shape[-1] // 2
                want = np.concatenate(
                    [p[..., :half] for p in parts] + [p[..., half:] for p in parts], -1
                )
            else:
                want = np.concatenate(parts, 0)
            np.testing.assert_array_equal(got[layer, slot], want)


def test_shipped_serving_plan_is_valid():
    path = os.path.join(
        os.path.dirname(rl.__file__), "..", "data", "plan_prefill_serve.npz"
    )
    order = np.load(path)["order"]
    assert order.shape == (92, 896)
    assert all(sorted(row) == list(range(896)) for row in order.tolist())
