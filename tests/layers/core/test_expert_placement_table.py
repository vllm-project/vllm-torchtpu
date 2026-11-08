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
