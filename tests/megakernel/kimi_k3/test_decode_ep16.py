# SPDX-License-Identifier: Apache-2.0
"""EP16/TP2 decode experts (KernelConfig.decode_expert_parallel=16): the decode
loader holds the prefill's EP16 families (placed by prefill_expert_plan, router
columns to match, each family joined on the host), each local expert half is an
EP8 TP4 quarter of the pre-shard, and routes expand to both halves."""

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


def _write_dense(directory, rank):
    """The dense families both loaders read; router column e holds logical
    expert e (bias: -e)."""
    dense = {
        name: np.full((2, 3), rank, np.float32)
        for name in rl.PREFILL_IO + rl.PREFILL_COMMON + rl.PREFILL_KDA + rl.PREFILL_MLA
    }
    dense.update(
        k_o=np.full((2, 768, 3), rank, np.float32),
        m_o=np.full((2, 768, 3), rank, np.float32),
        latent_norm=np.full((2, 1, 8), rank, np.float32),
        router=np.broadcast_to(np.arange(896, dtype=np.float32), (2, 5, 896)).copy(),
        router_bias=np.broadcast_to(
            -np.arange(896, dtype=np.float32), (2, 1, 896)
        ).copy(),
    )
    families, blob, offset = {}, [], 0
    for name, value in dense.items():
        families[name] = {
            "file": "dense.bin",
            "offset": offset,
            "shape": list(value.shape),
            "dtype": "float32",
        }
        blob.append(value.tobytes())
        offset += value.nbytes
    (directory / "dense.bin").write_bytes(b"".join(blob))
    return families


@pytest.fixture(scope="module")
def preshard(tmp_path_factory):
    root = tmp_path_factory.mktemp("preshard")
    rng = np.random.default_rng(0)
    data = {}
    for rank in range(32):
        directory = root / f"rank{rank:02d}"
        directory.mkdir()
        families = _write_dense(directory, rank)
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
    cache = root / "prefill-sp-cache"
    cache.mkdir()
    layout = {}
    for name in (
        "latent_down",
        "latent_up",
        "shared_gu_0",
        "shared_gu_1",
        "shared_down_0",
        "shared_down_1",
    ):
        np.zeros((2, 2), np.float32).tofile(cache / f"{name}.bin")
        layout[name] = {"file": f"{name}.bin", "shape": [2, 2], "dtype": "float32"}
    (cache / "layout.json").write_text(
        json.dumps({"layers": LAYERS, "families": layout})
    )
    order = np.stack([rng.permutation(896) for _ in range(LAYERS - 1)]).astype(np.int32)
    np.savez(root / "plan.npz", order=order)
    identity = np.tile(np.arange(896, dtype=np.int32), (LAYERS - 1, 1))
    np.savez(root / "identity.npz", order=identity)
    return root, data, order


def _ep16_entry(data, rank, name, layer, expert):
    """Logical expert ``expert``'s EP16 entry on ``rank``: its TP4 quarters
    2 (rank % 2) and 2 (rank % 2) + 1 (gate quarters then up quarters along
    the columns; down and its scales along the rows)."""
    quarters = [
        data[(4 * (expert // 112) + q, name, layer)][0, expert % 112]
        for q in (2 * (rank % 2), 2 * (rank % 2) + 1)
    ]
    if name in ("expert_gu", "expert_gus"):
        half = quarters[0].shape[-1] // 2
        return np.concatenate(
            [q[..., :half] for q in quarters] + [q[..., half:] for q in quarters], -1
        )
    return np.concatenate(quarters, 0)


def _held_experts(rank, order, layer):
    chip = rank // 2
    if order is None:
        return range(56 * chip, 56 * chip + 56)
    return order[layer, 56 * chip : 56 * chip + 56]


def _decode(monkeypatch, root, rank, **config):
    monkeypatch.setattr(rl, "_config", lambda: KernelConfig(**config))
    return {
        name: value
        for _, name, value in rl.iter_decode_rank(
            root, rank, layers=LAYERS, split_experts=True
        )
    }


@pytest.mark.parametrize("rank", [0, 1, 2, 7, 30])
@pytest.mark.parametrize("name", list(SHAPES))
def test_ep16_families_hold_the_chips_experts(preshard, rank, name):
    root, data, order = preshard
    rank_dir = rl.RankDir(root, rank, LAYERS)
    for plan in (None, order.astype(np.int64)):
        parts = [
            read()
            for key, _, read in rl._ep16_expert_jobs(root, rank_dir, rank, LAYERS, plan)
            if key.startswith(name + "@")
        ]
        family = np.concatenate(parts)
        assert family.shape[:2] == (LAYERS - 1, 56)
        for layer in range(LAYERS - 1):
            for slot, expert in enumerate(_held_experts(rank, plan, layer)):
                np.testing.assert_array_equal(
                    family[layer, slot], _ep16_entry(data, rank, name, layer, expert)
                )


@pytest.mark.parametrize("rank", [0, 1, 6])
@pytest.mark.parametrize("name", list(SHAPES))
def test_each_half_is_an_ep8_quarter(preshard, rank, name):
    """Local expert 2e + h of rank r: TP4 quarter 2 (r % 2) + h of global expert
    56 (r // 2) + e, as the decode slot holds it."""
    root, data, _ = preshard
    rank_dir = rl.RankDir(root, rank, LAYERS)
    family = np.concatenate(
        [
            read()
            for key, _, read in rl._ep16_expert_jobs(root, rank_dir, rank, LAYERS, None)
            if key.startswith(name + "@")
        ]
    )
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


def test_config_checks_decode_expert_parallel():
    assert KernelConfig().decode_expert_parallel == 8
    with pytest.raises(ValueError, match="decode_expert_parallel"):
        KernelConfig(decode_expert_parallel=12)


@pytest.mark.parametrize("ep16", [False, True])
def test_host_join_matches_the_device_join(preshard, ep16):
    """Each family's parts read into one host array: the bytes the per-part
    uploads' device join (a concatenation along the layer axis) holds."""
    root, _, _ = preshard
    rank_dir = rl.RankDir(root, 3, LAYERS)
    if ep16:
        jobs = list(rl._ep16_expert_jobs(root, rank_dir, 3, LAYERS, None))
    else:
        jobs = list(rl._expert_jobs(rank_dir, True))
    jobs.append(("dense", 0, lambda: "dense"))
    joined = rl._host_joined(jobs)
    families = list(dict.fromkeys(key.partition("@")[0] for key, _, _ in jobs))
    assert [key for key, _, _ in joined] == families
    for key, nbytes, read in joined:
        if key == "dense":
            assert read() == "dense"
            continue
        parts = [(n, r()) for k, n, r in jobs if k.partition("@")[0] == key]
        value = read()
        np.testing.assert_array_equal(value, np.concatenate([p for _, p in parts]))
        assert value.flags.c_contiguous
        assert value.nbytes == nbytes == sum(n for n, _ in parts)


def test_host_join_needs_consecutive_parts(preshard):
    root, _, _ = preshard
    jobs = list(rl._expert_jobs(rl.RankDir(root, 0, LAYERS), True))
    with pytest.raises(ValueError, match="consecutive"):
        rl._host_joined([jobs[1], jobs[0], *jobs[2:]])


@pytest.mark.parametrize("rank", [0, 7, 30])
def test_ep16_decode_yields_whole_families_in_plan_order(monkeypatch, preshard, rank):
    root, data, order = preshard
    base = _decode(monkeypatch, root, rank, decode_expert_parallel=16)
    assert not any("@" in name for name in base)
    same = _decode(
        monkeypatch,
        root,
        rank,
        decode_expert_parallel=16,
        prefill_expert_plan=str(root / "identity.npz"),
    )
    placed = _decode(
        monkeypatch,
        root,
        rank,
        decode_expert_parallel=16,
        prefill_expert_plan=str(root / "plan.npz"),
    )
    for name in (*SHAPES, "router", "router_bias"):
        np.testing.assert_array_equal(same[name], base[name])
    for layer in range(LAYERS - 1):
        for name in SHAPES:
            for slot, expert in enumerate(_held_experts(rank, order, layer)):
                np.testing.assert_array_equal(
                    placed[name][layer, slot],
                    _ep16_entry(data, rank, name, layer, expert),
                )
        # Physical column p routes to logical expert order[l, p].
        np.testing.assert_array_equal(placed["router"][layer, 0], order[layer])
        np.testing.assert_array_equal(placed["router_bias"][layer, 0], -order[layer])


@pytest.mark.parametrize("rank", [3, 22])
def test_prefill_ep16_loader_matches_the_ep16_decode(monkeypatch, preshard, rank):
    """The aggregate's prefill can use the EP16 decode's experts and router as
    they are: its own EP16 loader produces the same arrays."""
    root, _, _ = preshard
    plan = str(root / "plan.npz")
    decode = _decode(
        monkeypatch, root, rank, decode_expert_parallel=16, prefill_expert_plan=plan
    )
    monkeypatch.setattr(rl, "_config", lambda: KernelConfig(prefill_expert_plan=plan))
    parts: dict[str, list] = {}
    for _, name, value in rl.iter_prefill_rank(
        root, root / "prefill-sp-cache", rank, layers=LAYERS, split_experts=True
    ):
        parts.setdefault(name.partition("@")[0], []).append(value)
    for name in (*SHAPES, "router", "router_bias"):
        np.testing.assert_array_equal(np.concatenate(parts[name]), decode[name])


def test_plan_must_be_layer_permutations(monkeypatch, preshard, tmp_path):
    root, _, order = preshard
    repeated = order.copy()
    repeated[0, 0] = repeated[0, 1]
    np.savez(tmp_path / "repeated.npz", order=repeated)
    np.savez(tmp_path / "short.npz", order=order[:, :448])
    for bad in ("repeated.npz", "short.npz"):
        with pytest.raises(ValueError, match="permutations"):
            _decode(
                monkeypatch,
                root,
                0,
                decode_expert_parallel=16,
                prefill_expert_plan=str(tmp_path / bad),
            )
