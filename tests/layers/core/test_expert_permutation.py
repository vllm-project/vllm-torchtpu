# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the static expert placement builder and file format."""

import numpy as np
import pytest

from vllm_torchtpu.layers.core import expert_permutation as ep

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
    table, ep_size, local = ep.load(path)
    assert ep_size == EP and local == E // EP and list(table) == names
    for i, name in enumerate(names):
        np.testing.assert_array_equal(table[name], slots[i][:, None])


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
    table, ep_size, _ = ep.load(out)
    assert ep_size == 8
    assert list(table) == [f"model.layers.{i}.mlp.experts" for i in range(7)]
    np.testing.assert_array_equal(
        table["model.layers.4.mlp.experts"], table["model.layers.1.mlp.experts"]
    )
    assert "wrote" in capsys.readouterr().out
    ep.main([str(tmp_path / "routes.npz"), out, "--ep-size", "8", "--num-layers", "2"])
    assert len(ep.load(out)[0]) == 2


def _hot_counts(seed=0):
    """A few experts far above a rank's mean load (the floor that copies lift)."""
    counts = _skewed_counts(seed)
    counts[[5, 77, 300]] = counts.sum() / EP * np.array([3.0, 1.5, 1.2])
    return counts


def _copy_loads(slots, counts, local, ep_size=EP):
    copies = (slots >= 0).sum(axis=1)
    loads = np.zeros(ep_size)
    for e, row in enumerate(slots):
        for slot in row[row >= 0]:
            loads[slot // local] += counts[e] / copies[e]
    return loads


def test_replicate_without_extra_slots_is_balance():
    counts = _skewed_counts(2)
    slots = ep.replicate(counts, EP, 0)
    assert slots.shape == (E, 1)
    np.testing.assert_array_equal(slots[:, 0], ep.balance(counts, EP))


@pytest.mark.parametrize("extra,copies", [(1, None), (1, 8), (2, None)])
def test_replicate_fills_slots_with_copies_of_the_busiest_experts(extra, copies):
    counts = _hot_counts()
    slots = ep.replicate(counts, EP, extra, copies)
    local = E // EP + extra
    present = slots >= 0
    num_copies = present.sum(axis=1)
    assert slots.dtype == np.int32 and present[:, 0].all()
    assert num_copies.sum() == E + (copies if copies is not None else EP * extra)
    used = slots[present]
    assert len(np.unique(used)) == used.size and used.max() < EP * local
    for row in slots:
        ranks = row[row >= 0] // local
        assert len(np.unique(ranks)) == len(ranks)
    # Water-filling: the hottest expert has the most copies, and each copy
    # went to an expert whose load per copy was the largest at the time.
    assert num_copies[5] == num_copies.max() > num_copies[77] >= num_copies[300] > 1
    copied = num_copies > 1
    before_last = counts[copied] / (num_copies[copied] - 1)
    assert before_last.min() >= (counts / num_copies).max()
    linear = counts.reshape(EP, -1).sum(1)
    balanced = _rank_loads(ep.balance(counts, EP), counts)
    replicated = _copy_loads(slots, counts, local)
    np.testing.assert_allclose(replicated.sum(), counts.sum())
    assert balanced.max() / balanced.mean() > 2.5  # one expert sets the floor
    assert replicated.max() / replicated.mean() < 1.25 < linear.max() / linear.mean()
    np.testing.assert_array_equal(ep.replicate(counts, EP, extra, copies), slots)


def test_save_load_with_copies(tmp_path):
    rows = [ep.replicate(_hot_counts(i), EP, 1) for i in range(2)]
    width = max(r.shape[1] for r in rows)
    slots = np.full((2, E, width), -1, dtype=np.int32)
    for i, r in enumerate(rows):
        slots[i, :, : r.shape[1]] = r
    names = ["a", "b"]
    path = str(tmp_path / "c.npz")
    ep.save(path, slots, names, EP, E // EP + 1)
    table, ep_size, local = ep.load(path)
    assert (ep_size, local, list(table)) == (EP, E // EP + 1, names)
    np.testing.assert_array_equal(table["b"], slots[1])


LOCAL = E // EP + 1  # one spare slot per rank: slot r * LOCAL + LOCAL - 1


def _copy_file():
    """One layer, every expert on its linear rank, second copies empty."""
    slots = np.full((1, E, 2), -1, dtype=np.int32)
    slots[0, :, 0] = np.arange(E) // (E // EP) * LOCAL + np.arange(E) % (E // EP)
    return slots


@pytest.mark.parametrize(
    "first,second,match",
    [
        (None, LOCAL - 1, "one rank"),  # rank 0's spare slot; rank 0 holds expert 3
        (-1, 2 * LOCAL - 1, "copies first"),
        (None, LOCAL, "permutation"),  # expert 16's slot
        (None, EP * LOCAL, "permutation"),
    ],
)
def test_save_rejects_bad_copies(tmp_path, first, second, match):
    slots = _copy_file()
    if first is not None:
        slots[0, 3, 0] = first
    slots[0, 3, 1] = second
    with pytest.raises(ValueError, match=match):
        ep.save(str(tmp_path / "p.npz"), slots, ["a"], EP, LOCAL)


def test_save_accepts_a_copy_on_another_rank(tmp_path):
    slots = _copy_file()
    slots[0, 3, 1] = 2 * LOCAL - 1  # rank 1's spare slot
    ep.save(str(tmp_path / "p.npz"), slots, ["a"], EP, LOCAL)
    with pytest.raises(ValueError, match="cannot hold"):
        ep.save(str(tmp_path / "q.npz"), slots, ["a"], EP, E // EP - 1)


def test_cli_extra_slots(tmp_path, capsys):
    rng = np.random.default_rng(0)
    ids = rng.integers(0, 64, size=(2, 50, 4, 4)).astype(np.int16)
    ids[0, :, :, 0] = 7  # one expert in every route of capture layer 0
    np.savez(tmp_path / "routes.npz", ids=ids)
    out = str(tmp_path / "placement.npz")
    ep.main(
        [str(tmp_path / "routes.npz"), out, "--ep-size", "8", "--num-experts", "64"]
        + ["--extra-slots", "1", "--num-layers", "3"]
    )
    table, ep_size, local = ep.load(out)
    assert (ep_size, local) == (8, 9)
    copies = (table["model.layers.0.mlp.experts"] >= 0).sum(axis=1)
    assert copies.sum() == 64 + 8 and copies[7] == copies.max() > 1
    assert "slots per rank" in capsys.readouterr().out
