# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for scripts/vllm/benchmark/inferencex/qwen3.5/eplb and its tables."""

import importlib.util
import json
import pathlib
import sys

import numpy as np
import pytest
import torch

from vllm_torchtpu import expert_placement

pytestmark = pytest.mark.cpu_test

_QWEN = (
    pathlib.Path(__file__).resolve().parents[2]
    / "scripts/vllm/benchmark/inferencex/qwen3.5"
)
_EPLB = _QWEN / "eplb"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def profiler(tmp_path_factory):
    sys.path.insert(0, str(_EPLB / "profiler"))
    mp = pytest.MonkeyPatch()
    mp.setenv("VLLM_TPU_MOE_PROFILE_RING_SLOTS", "8")
    mp.setenv("VLLM_TPU_MOE_PROFILE_RING_STRIDE", "2")
    hook = _load("hook_patch", _EPLB / "profiler/hook_patch.py")
    prof = _load("moe_profiler", _EPLB / "profiler/moe_profiler.py")
    yield hook, prof
    mp.undo()
    sys.path.remove(str(_EPLB / "profiler"))


E, K = 32, 4


class _Padding:
    """Rows at index >= n_valid are padding."""

    n_valid = 0

    @staticmethod
    def zero_routing_weights_for_padding(ids, weights, is_local_tensor=False):
        mask = (torch.arange(ids.shape[0]) < _Padding.n_valid).float().reshape(-1, 1)
        return ids, weights * mask


class _Spy(torch.nn.Module):
    """Records which _eplb_ buffers the accumulators touch."""

    def __getattr__(self, name):
        if name.startswith("_eplb_"):
            object.__getattribute__(self, "__dict__").setdefault("seen", set()).add(
                name
            )
        return super().__getattr__(name)


def _layer(hook, cls=torch.nn.Module):
    layer = cls()
    for name in ("total", "prefill", "decode", "valid", "padded"):
        layer.register_buffer(
            f"_eplb_hist_{name}", torch.zeros(E, dtype=torch.int32), persistent=False
        )
    hook._register_decode_buffers(layer, E, torch.device("cpu"))
    return layer


def _route(rng, rows):
    return np.stack([rng.choice(E, K, replace=False) for _ in range(rows)])


def test_decode_statistics_match_a_numpy_oracle(profiler, monkeypatch, tmp_path):
    hook, prof = profiler
    monkeypatch.setitem(hook._PADDING, "token_padding", _Padding)
    monkeypatch.setitem(hook._PADDING, "is_local_tensor", lambda: False)
    layer = _layer(hook)
    rng = np.random.default_rng(0)
    steps, ring = 0, {}
    cohit, coact = np.zeros((E, E)), np.zeros((E, E))
    step_hit, pad_hit, rows_hist, pad_rows = np.zeros(E), np.zeros(E), np.zeros(129), 0
    for _ in range(23):
        rows = int(rng.choice([4, 8, 16]))
        n_valid = int(rng.integers(0, rows + 1))
        _Padding.n_valid = n_valid
        ids = _route(rng, rows)
        hook._accumulate_topk_ids(layer, torch.tensor(ids, dtype=torch.int32), rows)
        multi = np.zeros((rows, E))
        multi[np.arange(rows)[:, None], ids] = 1
        hit = (multi[:n_valid].sum(0) > 0).astype(float)
        hit_all = (multi.sum(0) > 0).astype(float)
        if n_valid:
            steps += 1
            if steps % 2 == 0:
                ring[(steps // 2) % 8] = (hit, n_valid)
        step_hit += hit
        cohit += np.outer(hit, hit)
        coact += multi[:n_valid].T @ multi[:n_valid]
        pad_hit += (hit_all - hit) > 0
        rows_hist[n_valid] += 1
        pad_rows += rows - n_valid

    assert layer._eplb_dec_steps.item() == steps
    np.testing.assert_array_equal(layer._eplb_dec_step_hit.numpy(), step_hit)
    np.testing.assert_array_equal(layer._eplb_dec_cohit.numpy(), cohit)
    np.testing.assert_array_equal(layer._eplb_dec_tok_coact.numpy(), coact)
    np.testing.assert_array_equal(layer._eplb_dec_padhit.numpy(), pad_hit)
    np.testing.assert_array_equal(layer._eplb_dec_rows_hist.numpy(), rows_hist)
    assert layer._eplb_dec_pad_rows.item() == pad_rows
    for slot, (hit, n_valid) in ring.items():
        np.testing.assert_array_equal(layer._eplb_dec_ring[slot].numpy(), hit)
        assert layer._eplb_dec_ring_rows[slot].item() == n_valid

    # A prefill-sized forward leaves the decode statistics alone.
    before = layer._eplb_dec_cohit.clone()
    _Padding.n_valid = 200
    hook._accumulate_topk_ids(
        layer, torch.tensor(_route(rng, 200), dtype=torch.int32), 200
    )
    assert torch.equal(before, layer._eplb_dec_cohit)

    profiler_obj = prof.MoeProfiler(output_dir=str(tmp_path))
    profiler_obj.register_layer(3, layer, E)
    dump = np.load(profiler_obj.dump("test"))
    assert dump["dec_cohit"].shape == (4, E, E)
    np.testing.assert_array_equal(dump["dec_cohit"][3], cohit)
    assert dump["dec_ring"].dtype == np.uint8


def test_both_row_count_sides_touch_the_same_buffers(profiler, monkeypatch):
    """The runner hands a bucket between the <=128-row and >128-row traces only
    if both lift the same buffers."""
    hook, _ = profiler
    monkeypatch.setitem(hook._PADDING, "token_padding", _Padding)
    monkeypatch.setitem(hook._PADDING, "is_local_tensor", lambda: False)
    rng = np.random.default_rng(1)

    def touched(rows):
        spy = _layer(hook, _Spy)
        _Padding.n_valid = rows
        hook._accumulate_topk_ids(
            spy, torch.tensor(_route(rng, rows), dtype=torch.int32), rows
        )
        return spy.__dict__["seen"]

    assert touched(16) == touched(200)


def _synthetic_profile(path, layers=2, steps=2048, seed=0):
    """A ring of correlated step hit sets: each request favours an expert cluster."""
    rng = np.random.default_rng(seed)
    experts = 512
    popularity = rng.zipf(1.6, experts).astype(float)
    popularity /= popularity.sum()
    cluster = rng.integers(0, 32, experts)
    ring = np.zeros((layers, steps, experts), np.uint8)
    rows = np.zeros((layers, steps), np.int32)
    for layer in range(layers):
        for step in range(steps):
            n = int(rng.choice([4, 8, 16]))
            rows[layer, step] = n
            p = popularity * (
                1 + 8 * np.isin(cluster, rng.integers(0, 32, max(1, n // 4)))
            )
            p /= p.sum()
            for _ in range(n):
                ring[layer, step, rng.choice(experts, 10, replace=False, p=p)] = 1
    np.savez(
        path / "moe_profile_rank_0_pid_1_latest.npz",
        dec_ring=ring,
        dec_ring_rows=rows,
        dec_steps=np.full((layers, 1), steps),
        dec_step_hit=ring.sum(1),
    )


@pytest.mark.parametrize("copies", [0, 16])
def test_solver_beats_contiguous_and_writes_a_loadable_table(
    tmp_path, monkeypatch, copies
):
    solver = _load("solve_placement", _EPLB / "solve_placement.py")
    _synthetic_profile(tmp_path)
    out = tmp_path / "table.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "solve_placement.py",
            str(tmp_path),
            "--copies",
            str(copies),
            "--iters",
            "3000",
            "--out",
            str(out),
        ],
    )
    solver.main()
    table = expert_placement.parse(json.loads(out.read_text()), str(out))
    assert (table.ranks, table.experts, table.owned) == (8, 512, 64)
    assert table.slots == 64 + copies // 8
    samples, _ = solver.load_samples(str(tmp_path))
    for layer, (hits, _) in samples.items():
        owner = np.empty(512, int)
        for rank, slots in enumerate(table.cores[layer]):
            owner[list(slots[: table.owned])] = rank
        copied = [
            (e, r)
            for r, slots in enumerate(table.cores[layer])
            for e in slots[table.owned :]
        ]
        hot = {e: r for e, r in copied}
        solved, _ = solver.replay(owner, [(e, hot[e]) for e in table.hot[layer]], hits)
        contiguous, _ = solver.replay(np.arange(512) // 64, [], hits)
        assert solved < 0.9 * contiguous


@pytest.mark.parametrize("copies", [0, 32])
def test_shipped_agentx_tables_load(copies):
    path = _QWEN / f"expert_placement_tp8_ep8_agentx_corr_c{copies}.json"
    table = expert_placement.parse(json.loads(path.read_text()), str(path))
    assert (table.ranks, table.experts, table.owned) == (8, 512, 64)
    assert table.slots == 64 + copies // 8
    assert sorted(table.cores) == list(range(60))
    assert all(len(table.hot[layer]) == copies for layer in table.cores)
