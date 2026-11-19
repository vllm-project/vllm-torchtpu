# SPDX-License-Identifier: Apache-2.0
"""q38 DEC-HOST-4 D1-norm L-GDNCOPY (VLLM_TPU_GDN_SEED_COPY_PACKED): the packed seed-copy
uploads must hand every consumer bitwise the same index tensors as the per-tensor path.

Each scenario runs ``_collect_mamba_state_seed_copies`` twice on identical fake runners
(flag off, flag on with the CHECK mode) and compares: the staged (src, dst) copies, the
read-offset buffer after the step, and every index tensor handed to the read-offset seed /
migrate programs. The flag-on run must take the packed branch (counter assert) and the
CHECK must compare every packed crossing step; the number compared is printed."""

import copy
from types import SimpleNamespace

import pytest
import torch

import vllm_torchtpu.runner.tpu_runner as runner_mod
from tests.runner.test_mamba_state_seed_copies import (
    BLOCK_SIZE,
    NUM_BLOCKS,
    _make_self,
    _sched,
)
from vllm_torchtpu import envs
from vllm_torchtpu.core import d1n_ab
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

COMPARED = {"n": 0}


def _bind(fake):
    for name in ("_gdn_seed_uploads", "_gdn_packed_upload", "_apply_gdn_seed_copies_packed"):
        setattr(fake, name, getattr(TPUModelRunner, name).__get__(fake))
    return fake


def _scenarios():
    """(name, builder) pairs; each builder returns (fake, sched) for one crossing step."""
    def spec(window=3, split=1, groups=1, offset=1):
        def build():
            offsets = torch.zeros(NUM_BLOCKS, dtype=torch.int32)
            offsets[10] = offset
            rows = [[[10, 11, 12, 13]] for _ in range(groups)]
            fake, _ = _make_self(["a"], [BLOCK_SIZE], rows, state_pos={"a": 0},
                                 ckpt_window=window, read_offsets=offsets, split=split)
            return fake, _sched({"a": 1})
        return build

    def nonspec(with_offsets=True, split=1, groups=2):
        def build():
            offsets = torch.arange(NUM_BLOCKS, dtype=torch.int32) % 3 if with_offsets else None
            rows = [[[5 + g, 6 + g, 7 + g, 8 + g]] for g in range(groups)]
            fake, _ = _make_self(["a"], [2 * BLOCK_SIZE], rows, state_pos={"a": 1},
                                 ckpt_window=1, read_offsets=offsets, split=split)
            return fake, _sched({"a": BLOCK_SIZE})
        return build

    def many(n=9):
        def build():
            req_ids = [f"r{i}" for i in range(n)]
            rows = [[[2 + i, 12 + i, 0, 0] for i in range(n)]]
            fake, _ = _make_self(req_ids, [BLOCK_SIZE] * n, rows,
                                 state_pos={r: 0 for r in req_ids}, ckpt_window=1,
                                 read_offsets=torch.zeros(NUM_BLOCKS, dtype=torch.int32))
            return fake, _sched({r: 1 for r in req_ids})
        return build

    return [
        ("spec_window3", spec()),
        ("spec_two_groups", spec(groups=2)),
        ("spec_split2", spec(split=2)),
        ("spec_window4_offset2", spec(window=4, offset=2)),
        ("nonspec_migrate_two_groups", nonspec()),
        ("nonspec_no_offsets", nonspec(with_offsets=False)),
        ("nonspec_split2", nonspec(split=2)),
        ("many_crossings_padded", many()),
    ]


def _run(build, packed, monkeypatch):
    fake, sched = build()
    _bind(fake)
    calls = []

    def rec(name, orig):
        def f(*args):
            calls.append((name, [a.clone() if torch.is_tensor(a) else a for a in args[1:]]))
            return orig(*args)
        return f

    monkeypatch.setattr(runner_mod, "_rollback_offsets_seed", rec("seed", ORIG["seed"]))
    monkeypatch.setattr(runner_mod, "_rollback_offsets_migrate", rec("migrate", ORIG["migrate"]))
    monkeypatch.setattr(envs, "VLLM_TPU_GDN_SEED_COPY_PACKED", packed, raising=False)
    monkeypatch.setattr(envs, "VLLM_TPU_GDN_SEED_COPY_PACKED_CHECK", 1000 if packed else 0,
                        raising=False)
    TPUModelRunner._collect_mamba_state_seed_copies(fake, sched, 0, sum(r is not None for r in fake.input_batch.req_ids))
    staged = [(tuple(id(r) for r in raws), (str(src.dtype), src.tolist()), (str(dst.dtype), dst.tolist()))
              for raws, src, dst in fake._pending_mamba_state_copies]
    raws_order = {}
    norm = []
    for key, src, dst in staged:
        raws_order.setdefault(key, len(raws_order))
        norm.append((raws_order[key], src, dst))
    offsets = (fake.mamba_slot_read_offsets.tolist()
               if fake.mamba_slot_read_offsets is not None else None)
    return fake, norm, offsets, [(n, [(str(a.dtype), a.tolist()) if torch.is_tensor(a) else a for a in args])
                                 for n, args in calls]


ORIG = {"seed": runner_mod._rollback_offsets_seed, "migrate": runner_mod._rollback_offsets_migrate}


@pytest.fixture(autouse=True)
def _fresh_ab(monkeypatch):
    monkeypatch.setattr(d1n_ab, "PERIOD_MS", 0)
    monkeypatch.setattr(d1n_ab, "D1E_PERIOD_MS", 0)
    d1n_ab._resolve()
    yield
    d1n_ab._resolve()


@pytest.mark.parametrize("name,build", _scenarios())
def test_packed_matches_per_tensor_bitwise(name, build, monkeypatch):
    _, legacy, off_l, calls_l = _run(build, False, monkeypatch)
    fake, packed, off_p, calls_p = _run(build, True, monkeypatch)
    assert legacy, f"{name}: the scenario staged no copy (would be vacuous)"
    assert packed == legacy, name
    assert off_p == off_l, name
    assert calls_p == calls_l, name
    # the packed branch was taken: one upload per dtype, the CHECK compared this step
    assert getattr(fake, "_gdn_packed_uploads", 0) >= 1, f"{name}: packed branch not taken"
    assert getattr(fake, "_gdn_packed_checked", 0) == 1, f"{name}: CHECK did not compare"
    COMPARED["n"] += fake._gdn_packed_checked


def test_check_compared_every_scenario(capsys):
    n = COMPARED["n"]
    with capsys.disabled():
        print(f"\nQ38_GDN_CHECK compared_crossing_steps={n} scenarios={len(_scenarios())}")
    assert n == len(_scenarios()), "CHECK compared fewer crossing steps than scenarios"


def test_one_upload_per_dtype(monkeypatch):
    fake, _, _, _ = _run(_scenarios()[1][1], True, monkeypatch)  # spec, two groups: all int64
    assert fake._gdn_packed_uploads == 1
    fake, _, _, _ = _run(_scenarios()[4][1], True, monkeypatch)  # migrate (int64) + raws (int32)
    assert fake._gdn_packed_uploads == 2


def test_check_catches_a_corrupted_view(monkeypatch):
    real = TPUModelRunner._gdn_packed_upload

    def corrupt(self, ups):
        out = real(self, ups)
        key = next(iter(out))
        bad = out[key].clone().reshape(-1)
        bad[0] += 1
        out[key] = bad.view(out[key].shape)
        return out

    monkeypatch.setattr(TPUModelRunner, "_gdn_packed_upload", corrupt)
    with pytest.raises(RuntimeError, match="L-GDNCOPY CHECK mismatch"):
        _run(_scenarios()[0][1], True, monkeypatch)


def test_flag_off_never_takes_the_packed_branch(monkeypatch):
    fake, _, _, _ = _run(_scenarios()[0][1], False, monkeypatch)
    assert getattr(fake, "_gdn_packed_uploads", 0) == 0


def test_syncs_and_vectorized_counted(monkeypatch):
    """A crossing near a block edge with drafts waits for the previous step's tokens:
    the window counters record the wait (syncs) and the packed step (vectorized)."""
    monkeypatch.setattr(runner_mod, "get_pcp_group", lambda: SimpleNamespace(world_size=1))
    fake, _ = _make_self(["a"], [381], [[[5, 6, 7, 8, 9]]], state_pos={"a": 0},
                         state_block_size=384, ckpt_window=4,
                         read_offsets=torch.zeros(NUM_BLOCKS, dtype=torch.int32))
    _bind(fake)
    sampled = torch.tensor([[77] * 4])
    fake.scheduler_config = SimpleNamespace(async_scheduling=True)
    fake.speculative_config = SimpleNamespace(num_speculative_tokens=3)
    fake._pre_async_results = SimpleNamespace(
        req_id_to_index_copy={"a": 0}, num_draft_per_req={0: 3},
        spec_decode_num_rejected_tokens=torch.tensor([0]), wait_for_copy=lambda: sampled)
    monkeypatch.setattr(envs, "VLLM_TPU_GDN_SEED_COPY_PACKED", True, raising=False)
    monkeypatch.setattr(envs, "VLLM_TPU_GDN_SEED_COPY_PACKED_CHECK", 10, raising=False)
    TPUModelRunner._collect_mamba_state_seed_copies(fake, _sched({"a": 4}), 0, 1)
    c = d1n_ab.counts()
    assert c["syncs"] == 1 and c["steps"] == 1 and c["vectorized"] == 1, c
