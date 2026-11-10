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


def _write_counts(path, names, counts):
    ep.save_counts(str(path), np.asarray(counts), names)


def test_counts_files_sum_by_layer_name(tmp_path):
    rng = np.random.default_rng(0)
    names = [f"model.layers.{i}.mlp.experts" for i in range(2)]
    first, second = rng.integers(0, 100, size=(2, 2, 64))
    engines = tmp_path / "counts"
    engines.mkdir()
    _write_counts(engines / "routing-counts-dp0-pp0-a.npz", names, first)
    _write_counts(engines / "routing-counts-dp1-pp0-a.npz", names, second)
    # A placement file written into the same directory is not read.
    ep.save(str(engines / "placement.npz"), np.stack([np.arange(64)] * 2), names, 8)
    stage = tmp_path / "routing-counts-dp0-pp1-b.npz"
    stage_names = ["model.layers.2.mlp.experts", "mtp.layers.0.mlp.experts"]
    _write_counts(stage, stage_names, np.ones((2, 64), dtype=np.int64))
    counts, got = ep.load_counts([str(engines), str(stage)])
    assert got == names + stage_names
    assert counts.dtype == np.int64
    np.testing.assert_array_equal(counts[:2], first + second)
    np.testing.assert_array_equal(counts[2:], np.ones((2, 64)))
    assert list(tmp_path.rglob("*.tmp")) == []


def test_load_counts_rejects_bad_inputs(tmp_path):
    _write_counts(tmp_path / "a.npz", ["x"], np.ones((1, 64)))
    _write_counts(tmp_path / "b.npz", ["x"], np.ones((1, 32)))
    with pytest.raises(ValueError, match="experts"):
        ep.load_counts([str(tmp_path / "a.npz"), str(tmp_path / "b.npz")])
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="routing-counts"):
        ep.load_counts([str(tmp_path / "empty")])
    data = (tmp_path / "a.npz").read_bytes()
    for name, content in (("short.npz", data[: len(data) // 2]), ("zero.npz", b"")):
        (tmp_path / name).write_bytes(content)
        with pytest.raises(ValueError, match=name):
            ep.load_counts([str(tmp_path / name)])


def test_cli_balances_counts_files(tmp_path, capsys):
    rng = np.random.default_rng(1)
    names = [f"model.layers.{i}.mlp.experts" for i in range(3)]
    files = tmp_path / "counts"
    files.mkdir()
    rows = rng.integers(1, 1000, size=(2, 3, 64))
    for engine, counts in enumerate(rows):
        _write_counts(files / f"routing-counts-dp{engine}-pp0-a.npz", names, counts)
    out = str(tmp_path / "placement.npz")
    ep.main([str(files), out, "--ep-size", "8"])
    table, ep_size = ep.load(out)
    assert ep_size == 8 and list(table) == names
    for name, row in zip(names, rows.sum(0)):
        np.testing.assert_array_equal(table[name], ep.balance(row, 8))
    lines = capsys.readouterr().out.splitlines()
    assert sum("linear ->" in line for line in lines) == 3 and "wrote" in lines[-1]

    rows[:, 1] = 0
    for engine, counts in enumerate(rows):
        _write_counts(files / f"routing-counts-dp{engine}-pp0-a.npz", names, counts)
    with pytest.raises(ValueError, match=r"no routes counted for 1 of 3 .*layers\.1\."):
        ep.main([str(files), out, "--ep-size", "8"])
    _write_counts(files / "routing-counts-dp0-pp0-a.npz", names, np.zeros((3, 64)))
    _write_counts(files / "routing-counts-dp1-pp0-a.npz", names, np.zeros((3, 64)))
    with pytest.raises(ValueError, match="no routes counted"):
        ep.main([str(files), out, "--ep-size", "8"])


def _eplb_replicate(counts, extra):
    """The public replication rule without the guard: each redundant slot to
    the expert with the largest count per replica, ties to the lower id."""
    replicas = np.ones(len(counts), dtype=np.int64)
    for _ in range(extra):
        replicas[int(np.argmax(counts / replicas))] += 1
    return replicas


def _skewed_layer(seed, ep_size=EP, hot=8):
    counts = _skewed_counts(seed)
    counts[:hot] *= 6  # a few experts at several times the mean
    return counts, ep.balance(counts, ep_size)


def test_replicate_is_the_eplb_greedy_above_the_guard():
    counts, _ = _skewed_layer(2)
    got = ep.replicate(counts, EP)
    assert got.sum() == E + EP  # skewed counts fill every redundant slot
    np.testing.assert_array_equal(got, _eplb_replicate(counts, EP))
    # The guard stops where the next replica would carry < 0.7x the mean.
    floor = ep.REPLICA_GUARD * counts.mean()
    assert (counts[got > 1] / got[got > 1] >= floor).all()
    many = ep.replicate(counts, 4 * E)
    assert many.sum() < E + 4 * E
    nxt = np.argmax(counts / many)
    assert counts[nxt] / (many[nxt] + 1) < floor


@pytest.mark.parametrize("max_replicas", [1, 2, 7])
def test_replicate_caps_the_replicas_per_expert(max_replicas):
    counts, _ = _skewed_layer(2)
    counts[0] *= 50  # one expert would take most redundant slots
    capped = ep.replicate(counts, EP, max_replicas)
    assert capped.max() <= max_replicas + 1
    assert ep.replicate(counts, EP).max() > max_replicas + 1
    # Without the cap the result is the uncapped rule's.
    np.testing.assert_array_equal(ep.replicate(counts, EP, 0), ep.replicate(counts, EP))


def test_guard_makes_no_replicas_from_flat_counts():
    flat = np.random.default_rng(3).uniform(0.9, 1.1, size=E) * 1000
    np.testing.assert_array_equal(ep.replicate(flat, EP), np.ones(E))
    slots = ep.balance(flat, EP)
    redundant = ep.place_replicas(flat, slots, EP, 1, ep.replicate(flat, EP))
    assert redundant.dtype == np.int32 and (redundant == -1).all()


@pytest.mark.parametrize("redundant_per_rank", [1, 2])
def test_place_replicas_keeps_primaries_and_splits_hot_experts(redundant_per_rank):
    counts, slots = _skewed_layer(4)
    replicas = ep.replicate(counts, EP * redundant_per_rank)
    redundant = ep.place_replicas(counts, slots, EP, redundant_per_rank, replicas)
    assert redundant.shape == (EP * redundant_per_rank,) and redundant.dtype == np.int32
    names = ["model.layers.0.mlp.experts"]
    ep.validate(slots[None], names, EP, redundant[None])
    # Every placed replica is one the greedy asked for; none is invented.
    placed = np.bincount(redundant[redundant >= 0], minlength=E)
    assert (placed <= replicas - 1).all() and placed.sum() > 0
    # No rank holds an expert twice, primary included.
    local = E // EP
    for q, e in enumerate(redundant):
        if e >= 0:
            assert slots[e] // local != q // redundant_per_rank
    for r in range(EP):
        mine = redundant[r * redundant_per_rank : (r + 1) * redundant_per_rank]
        mine = mine[mine >= 0]
        assert len(set(mine.tolist())) == len(mine)
    # Splitting the hot experts evens the busiest rank further.
    before = _rank_loads(slots, counts)
    after = ep._busiest_with_replicas(counts, slots, redundant, EP)
    assert after < before.max() / before.mean()
    np.testing.assert_array_equal(
        ep.place_replicas(counts, slots, EP, redundant_per_rank, replicas), redundant
    )


def test_place_replicas_drops_a_replica_no_rank_can_take():
    # Two ranks, one redundant slot each: expert 0 asks for two more
    # replicas, but only the rank without its primary can take one.
    counts = np.array([100.0, 1, 1, 1])
    slots = np.array([0, 1, 2, 3])
    redundant = ep.place_replicas(counts, slots, 2, 1, np.array([3, 1, 1, 1]))
    np.testing.assert_array_equal(redundant, [-1, 0])


def test_replica_table_and_physical_slots():
    # 4 ranks x 2 experts: expert 3 (on rank 3) has replicas on ranks 0 and 1,
    # expert 1 (on rank 0) one on rank 3; rank 2's redundant slot is empty.
    slots = np.array([5, 0, 3, 6, 1, 7, 2, 4])
    redundant = np.array([3, 3, -1, 1])
    ep.validate(slots[None], ["a"], 4, redundant[None])
    np.testing.assert_array_equal(
        ep.physical_slots(slots, 4, 1), [7, 0, 4, 9, 1, 10, 3, 6]
    )
    np.testing.assert_array_equal(ep.physical_slots(slots, 4, 0), slots)
    table = ep.replica_table(slots, redundant, 4)
    assert table.dtype == np.int32
    # Each rank's redundant slot follows its 2 primaries.
    np.testing.assert_array_equal(table[0], [2, 5, 8, 11])
    np.testing.assert_array_equal(table[1], [9, 9, -1, 0])
    np.testing.assert_array_equal(table[2], [1, 2, 0, 1])


def test_validate_rejects_bad_redundant_slots(tmp_path):
    slots = np.stack([np.arange(E), np.arange(E)[::-1]]).astype(np.int32)
    names = ["a", "b"]
    redundant = np.full((2, EP), -1, np.int32)
    ep.validate(slots, names, EP, redundant)
    bad = redundant.copy()
    bad[0, 0] = 0  # rank 0 holds expert 0 as its primary already
    with pytest.raises(ValueError, match="twice"):
        ep.save(str(tmp_path / "p.npz"), slots, names, EP, bad)
    ok = redundant.copy()
    ok[0, 1] = 5  # rank 1 holds a replica of expert 5 (primary on rank 0)
    ep.validate(slots, names, EP, ok)
    wide = np.full((2, 2 * EP), -1, np.int32)
    wide[1, 2:4] = 0  # rank 1 holds expert 0 in both of its redundant slots
    with pytest.raises(ValueError, match="twice"):
        ep.validate(slots, names, EP, wide)
    for shape in ((2, EP + 1), (1, EP), (2, 0)):
        with pytest.raises(ValueError, match="ep_size"):
            ep.validate(slots, names, EP, np.full(shape, -1, np.int32))
    with pytest.raises(ValueError, match="expert ids"):
        ep.validate(slots, names, EP, np.full((2, EP), E, np.int32))
    # The packed routing holds slot ids below 2**11.
    with pytest.raises(ValueError, match="packed routing"):
        ep.validate(np.arange(2048)[None], ["a"], 256, np.full((1, 256), -1, np.int32))


def test_redundant_file_round_trip_and_old_reader(tmp_path):
    rows = [_skewed_layer(i) for i in range(3)]
    slots = np.stack([s for _, s in rows])
    names = [f"model.layers.{i}.mlp.experts" for i in range(3)]
    redundant = np.stack(
        [ep.place_replicas(c, s, EP, 1, ep.replicate(c, EP)) for c, s in rows]
    )
    path = str(tmp_path / "redundant.npz")
    ep.save(path, slots, names, EP, redundant)
    # load() reads the same slots as from the file without redundant slots.
    table, ep_size = ep.load(path)
    assert ep_size == EP and list(table) == names
    for i, name in enumerate(names):
        np.testing.assert_array_equal(table[name], slots[i])
    got, redundant_per_rank = ep.load_redundant(path)
    assert redundant_per_rank == 1 and list(got) == names
    for i, name in enumerate(names):
        np.testing.assert_array_equal(got[name], redundant[i])
    plain = str(tmp_path / "plain.npz")
    ep.save(plain, slots, names, EP)
    assert ep.load_redundant(plain) == ({}, 0)
    with np.load(plain) as data:
        assert sorted(data.files) == ["ep_size", "layer_names", "slots"]


def test_cli_without_redundant_slots_writes_the_same_file(tmp_path):
    rng = np.random.default_rng(5)
    names = [f"model.layers.{i}.mlp.experts" for i in range(2)]
    files = tmp_path / "counts"
    files.mkdir()
    _write_counts(
        files / "routing-counts-dp0-pp0-a.npz", names, rng.integers(1, 99, (2, 64))
    )
    ep.main([str(files), str(tmp_path / "a.npz"), "--ep-size", "8"])
    ep.main(
        [
            str(files),
            str(tmp_path / "b.npz"),
            "--ep-size",
            "8",
            "--redundant-per-rank",
            "0",
        ]
    )
    counts, got = ep.load_counts([str(files)])
    slots = np.stack([ep.balance(row, 8) for row in counts])
    ep.save(str(tmp_path / "c.npz"), slots, got, 8)
    data = (tmp_path / "c.npz").read_bytes()
    assert (tmp_path / "a.npz").read_bytes() == data
    assert (tmp_path / "b.npz").read_bytes() == data


def test_cli_plans_replicas_around_kept_slots(tmp_path, capsys):
    names = [f"model.layers.{i}.mlp.experts" for i in range(2)]
    files = tmp_path / "counts"
    files.mkdir()
    counts = np.stack([_skewed_layer(i, ep_size=8)[0][:64] for i in range(2)])
    _write_counts(
        files / "routing-counts-dp0-pp0-a.npz", names, counts.astype(np.int64)
    )
    # Slots kept from another file: planned on other counts.
    kept = np.stack([np.random.default_rng(i).permutation(64) for i in range(2)])
    ep.save(str(tmp_path / "served.npz"), kept, names, 8)
    out = str(tmp_path / "redundant.npz")
    ep.main(
        [
            str(files),
            out,
            "--ep-size",
            "8",
            "--redundant-per-rank",
            "1",
            "--slots-from",
            str(tmp_path / "served.npz"),
        ]
    )
    table, _ = ep.load(out)
    redundant, redundant_per_rank = ep.load_redundant(out)
    assert redundant_per_rank == 1
    for i, name in enumerate(names):
        np.testing.assert_array_equal(table[name], kept[i])
        row = counts[i].astype(np.int64)
        want = ep.place_replicas(row, kept[i], 8, 1, ep.replicate(row, 8))
        np.testing.assert_array_equal(redundant[name], want)
    lines = capsys.readouterr().out.splitlines()
    assert sum("linear ->" in line and "replicas" in line for line in lines) == 2
    assert lines[-2].startswith("replicas: ") and "wrote" in lines[-1]
    with pytest.raises(ValueError, match="EP 8"):
        ep.main(
            [
                str(files),
                out,
                "--ep-size",
                "4",
                "--slots-from",
                str(tmp_path / "served.npz"),
            ]
        )
