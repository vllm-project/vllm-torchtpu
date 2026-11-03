# SPDX-License-Identifier: Apache-2.0
"""EP16/TP2 decode experts (KernelConfig.decode_expert_parallel=16): the decode
loader holds the prefill's EP16 regroup, each local expert half is an EP8
TP4 quarter of the pre-shard, and routes expand to both halves."""

import json

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.megakernel.kimi_k3.config import KernelConfig
from vllm_torchtpu.megakernel.kimi_k3.kernels import decode_megakernel as dm
from vllm_torchtpu.megakernel.kimi_k3.kernels import rank_loader as rl

pytestmark = pytest.mark.cpu_test

LAYERS = 3  # two MoE layers
SHAPES = {
    "expert_gu": (2, 112, 4, 8),
    "expert_gus": (2, 112, 1, 8),
    "expert_down": (2, 112, 6, 4),
    "expert_ds": (2, 112, 2, 4),
}
RANGES = [(0, 1), (1, 2)]


@pytest.fixture(scope="module")
def preshard(tmp_path_factory):
    root = tmp_path_factory.mktemp("preshard")
    rng = np.random.default_rng(0)
    data = {}
    for rank in range(8):
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


def _decode_family(root, rank, name):
    rank_dir = rl.RankDir(root, rank, LAYERS)
    parts = {
        key: read()
        for key, _, read in rl._ep16_expert_jobs(root, rank_dir, rank, LAYERS)
    }
    return np.concatenate(
        [parts[f"{name}@{i}/{len(RANGES)}"] for i in range(len(RANGES))]
    )


@pytest.mark.parametrize("rank", [0, 1, 2, 7])
@pytest.mark.parametrize("name", list(SHAPES))
def test_decode_loader_is_the_prefill_regroup(preshard, rank, name):
    root, _ = preshard
    rank_dir = rl.RankDir(root, rank, LAYERS)
    quarters = (2 * (rank % 2), 2 * (rank % 2) + 1)
    quarter_dirs = [rl.RankDir(root, 4 * (rank // 4) + q, LAYERS) for q in quarters]
    expected = np.concatenate(
        [
            read()
            for _, _, read in rank_dir.regrouped_expert_parts(
                name, quarter_dirs, (rank // 2) % 2, 2
            )
        ]
    )
    np.testing.assert_array_equal(_decode_family(root, rank, name), expected)


@pytest.mark.parametrize("rank", [0, 1, 6])
@pytest.mark.parametrize("name", list(SHAPES))
def test_each_half_is_an_ep8_quarter(preshard, rank, name):
    """Local expert 2e + h of rank r: TP4 quarter 2 (r % 2) + h of global expert
    56 (r // 2) + e, as the decode slot holds it."""
    root, data = preshard
    family = _decode_family(root, rank, name)
    slot = dm.expert_slot_shape(name, family.shape[1:], halves=2)
    for layer in range(LAYERS - 1):
        for expert in (0, 17, 55):
            for half in range(2):
                got = np.zeros(slot, np.float32)
                for axis, start, size, offset in dm.expert_half_windows(
                    name, slot, half
                ):
                    if axis == 1:
                        got[:, offset : offset + size] = family[layer, expert][
                            :, start : start + size
                        ]
                    else:
                        got[:] = family[layer, expert][start : start + size]
                global_expert = 56 * (rank // 2) + expert
                source = 4 * (global_expert // 112) + 2 * (rank % 2) + half
                want = data[(source, name, layer)][0, global_expert % 112]
                np.testing.assert_array_equal(got, want)


def test_routes_expand_to_both_halves():
    experts = jnp.array([[3, 9, 0, 0], [55, 0, 0, 0]], jnp.int32)
    probabilities = jnp.array(
        [[0.5, 0.25, 0.0, 0.0], [0.75, 0.0, 0.0, 0.0]], jnp.float32
    )
    counts = jnp.array([2, 1], jnp.int32)
    local, local_probabilities, local_counts = dm._expert_half_routes(
        experts, probabilities, counts
    )
    np.testing.assert_array_equal(local[0, :4], [6, 7, 18, 19])
    np.testing.assert_array_equal(local[1, :2], [110, 111])
    np.testing.assert_array_equal(local_probabilities[0, :4], [0.5, 0.5, 0.25, 0.25])
    np.testing.assert_array_equal(local_probabilities[1, :2], [0.75, 0.75])
    np.testing.assert_array_equal(local_counts, [4, 2])
    assert local.shape == (2, 8)


def test_config_rejects_hot_plans_at_ep16():
    with pytest.raises(ValueError, match="decode_expert_parallel"):
        KernelConfig(decode_expert_parallel=12)
    with pytest.raises(ValueError, match="hot-expert plans"):
        KernelConfig(decode_expert_parallel=16, kd_hot_plan="hot_plan_p8c72_plain8.npz")


def test_host_joined_parts_match_the_device_join(preshard):
    root, _ = preshard
    rank_dir = rl.RankDir(root, 3, LAYERS)
    jobs = list(rl._ep16_expert_jobs(root, rank_dir, 3, LAYERS))
    joined = {key: (nbytes, read()) for key, nbytes, read in rl._host_joined(jobs)}
    assert set(joined) == set(SHAPES)
    for name in SHAPES:
        np.testing.assert_array_equal(joined[name][1], _decode_family(root, 3, name))
        assert joined[name][0] == sum(n for k, n, _ in jobs if k.startswith(name + "@"))
