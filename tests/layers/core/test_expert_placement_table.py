# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the static expert placement builder and file format."""

import numpy as np
import pytest

from vllm_torchtpu.layers.core import expert_placement as ep

pytestmark = pytest.mark.cpu_test

E, EP = 512, 32


def _skewed_counts(seed=0):
    rng = np.random.default_rng(seed)
    counts = rng.gamma(2.0, 100.0, size=E)
    counts[:16] *= 3  # hot experts on linear rank 0
    return counts


def _rank_loads(slots, counts, ep_size=EP):
    local = len(slots) // ep_size
    return np.bincount(slots // local, weights=counts, minlength=ep_size)


def test_balance_is_a_full_placement():
    counts = _skewed_counts()
    slots = ep.balance(counts, EP)
    assert slots.dtype == np.int32
    np.testing.assert_array_equal(np.sort(slots), np.arange(E))
    np.testing.assert_array_equal(np.bincount(slots // (E // EP)), np.full(EP, 16))
    # Experts keep their logical order within a rank.
    for r in range(EP):
        on_rank = np.flatnonzero(slots // 16 == r)
        assert (np.diff(slots[on_rank]) > 0).all()
    np.testing.assert_array_equal(ep.balance(counts, EP), slots)


def test_balance_evens_the_busiest_rank():
    counts = _skewed_counts(1)
    linear = counts.reshape(EP, -1).sum(1)
    balanced = _rank_loads(ep.balance(counts, EP), counts)
    assert linear.max() / linear.mean() > 2.0
    # The largest-first fill ends within one expert's count of the mean.
    assert balanced.max() - balanced.mean() <= counts.max()
    assert balanced.max() / balanced.mean() < 1.1
    np.testing.assert_allclose(balanced.sum(), counts.sum())


def test_balance_uniform_counts_and_bad_split():
    slots = ep.balance(np.ones(64), 8)
    np.testing.assert_array_equal(np.bincount(slots // 8), np.full(8, 8))
    with pytest.raises(ValueError, match="do not split"):
        ep.balance(np.ones(10), 4)


def test_save_load_round_trip(tmp_path):
    slots = np.stack([ep.balance(_skewed_counts(i), EP) for i in range(3)])
    names = [f"model.layers.{i}.mlp.experts" for i in range(3)]
    path = str(tmp_path / "p.npz")
    ep.save(path, slots, names, EP)
    table, ep_size = ep.load(path)
    assert ep_size == EP and list(table) == names
    for i, name in enumerate(names):
        np.testing.assert_array_equal(table[name], slots[i])


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda s, n: (np.concatenate([s[:, :1], s[:, :-1]], 1), n), "permutation"),
        (lambda s, n: (s, [n[0], n[0]]), "repeat"),
        (lambda s, n: (s, n[:1]), "one name per layer"),
    ],
)
def test_save_rejects_bad_files(tmp_path, mutate, match):
    slots = np.stack([np.arange(E), np.arange(E)[::-1]]).astype(np.int32)
    names = ["a", "b"]
    slots, names = mutate(slots, names)
    with pytest.raises(ValueError, match=match):
        ep.save(str(tmp_path / "p.npz"), slots, names, EP)


def test_counts_from_ids_and_counts(tmp_path):
    ids = np.array([[[0, 1], [1, -1]], [[3, 3], [2, 1]]], dtype=np.int16)
    np.savez(tmp_path / "ids.npz", ids=ids)
    np.testing.assert_array_equal(
        ep.counts_from_routes(str(tmp_path / "ids.npz"), 4),
        [[1, 2, 0, 0], [0, 1, 1, 2]],
    )
    np.savez(tmp_path / "counts.npz", counts=np.ones((2, 4)))
    assert ep.counts_from_routes(str(tmp_path / "counts.npz")).shape == (2, 4)


def test_cli_tiles_capture_layers(tmp_path, capsys):
    rng = np.random.default_rng(0)
    ids = rng.integers(0, 64, size=(3, 50, 4, 2)).astype(np.int16)
    np.savez(tmp_path / "routes.npz", ids=ids)
    out = str(tmp_path / "placement.npz")
    ep.main(
        [str(tmp_path / "routes.npz"), out, "--ep-size", "8", "--num-experts", "64"]
        + ["--num-layers", "7"]
    )
    table, ep_size = ep.load(out)
    assert ep_size == 8
    assert list(table) == [f"model.layers.{i}.mlp.experts" for i in range(7)]
    np.testing.assert_array_equal(
        table["model.layers.4.mlp.experts"], table["model.layers.1.mlp.experts"]
    )
    assert "wrote" in capsys.readouterr().out
    ep.main([str(tmp_path / "routes.npz"), out, "--ep-size", "8", "--num-layers", "2"])
    assert len(ep.load(out)[0]) == 2
