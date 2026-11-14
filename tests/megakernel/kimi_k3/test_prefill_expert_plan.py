# SPDX-License-Identifier: Apache-2.0
"""EP16 expert placement (KernelConfig.expert_plan): the planned loader
reproduces the pre-shard's regroup under the identity plan and places each
planned expert's rows under a random plan; the default plan is valid."""

import json
import os

import numpy as np
import pytest

from vllm_torchtpu.megakernel.kimi_k3.config import KernelConfig
from vllm_torchtpu.megakernel.kimi_k3.kernels import rank_loader as rl

pytestmark = pytest.mark.cpu_test

LAYERS = 3  # two MoE layers
SHAPES = {  # gate/up: 2 x 256 columns, two interleave chunks per quarter
    "expert_gu": (2, 112, 2, 512),
    "expert_gus": (2, 112, 1, 512),
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
            if name in rl.GATE_UP_FAMILIES:
                want = np.concatenate(
                    [rl.interleave_gate_up(p.copy()) for p in parts], -1
                )
            else:
                want = np.concatenate(parts, 0)
            np.testing.assert_array_equal(got[layer, slot], want)


def test_default_plan_is_valid():
    path = os.path.join(
        os.path.dirname(rl.__file__), "..", "data", KernelConfig().expert_plan
    )
    order = np.load(path)["order"]
    assert order.shape == (92, 896)
    assert all(sorted(row) == list(range(896)) for row in order.tolist())


def test_planned_gate_up_reads_refuse_columns_spanning_more_than_e5m2_holds(tmp_path):
    # uint8 E8M0 gate/up scales, one block of one column of pre-shard rank 4's
    # expert 5 (chip 2's under the identity plan) a span plus one above the rest,
    # in the second MoE layer: rank 4's read of that layer raises, the first
    # layer's passes.
    shape = (2, 112, 2, 512)
    for rank in range(32):
        directory = tmp_path / f"rank{rank:02d}"
        directory.mkdir()
        files = []
        for index, (lo, hi) in enumerate(RANGES):
            scales = np.full((hi - lo, *shape[1:]), 120, np.uint8)
            if rank == 4 and index == 1:
                scales[0, 5, 1, 7] = 120 + rl.E5M2_COLUMN_SPAN + 1
            scales.tofile(directory / f"expert_gus_{index}.bin")
            files.append(f"expert_gus_{index}.bin")
        spec = {"shape": list(shape), "dtype": "uint8", "files": files}
        spec["layer_ranges"] = [list(r) for r in RANGES]
        index_json = {"rank": rank, "layers": LAYERS, "families": {"expert_gus": spec}}
        (directory / "index.json").write_text(json.dumps(index_json))
    order = np.tile(np.arange(896), (LAYERS - 1, 1))
    rank_dir = rl.RankDir(str(tmp_path), 4, LAYERS)
    (_, _, first), (_, _, second) = rl._planned_expert_parts(
        str(tmp_path), rank_dir, "expert_gus", 4, order, LAYERS
    )
    assert first().max() == 120
    with pytest.raises(ValueError, match="span more"):
        second()
