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
"""The model's runner subclass (runner/runner.py).

Ported from the tests of #2605 #2607 #2609 #2650 #2651 #2723, pointed at
`Qwen38Runner`; the reference results come from main's `TPUModelRunner`.
"""

import inspect
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from vllm_torchtpu.layers.adapter import token_padding
from vllm_torchtpu.models.qwen4_exp.experimental.runner import runner as qr
from vllm_torchtpu.models.qwen4_exp.experimental.runner.runner import (
    Qwen38Runner,
    SinkStageTiming,
    SkipUnchangedTokenPaddingState,
)
from vllm_torchtpu.runner import stage_timing
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner
from vllm_torchtpu.runner.tpu_runner_async_output import INVALID_TOKEN_ID

# Methods Qwen38Runner adds; every other method it defines overrides one of
# TPUModelRunner's.
NEW_METHODS = {
    "_num_reqs_tensor",
    "_upload_keyed",
    "_padded_sampling_params",
    "_spec_seed_copies",
    "_precompile_spec_tail",
}


def _own_methods():
    return {
        name: fn
        for name, fn in vars(Qwen38Runner).items()
        if inspect.isfunction(fn) and not name.startswith("__")
    }


def test_every_override_names_a_runner_method():
    """Guards the copies against a rename or signature change on main: each
    override must still replace a TPUModelRunner method with the same
    parameters, and each added method must not shadow one."""
    own = _own_methods()
    assert set(own) >= NEW_METHODS
    for name, fn in own.items():
        base = getattr(TPUModelRunner, name, None)
        if name in NEW_METHODS:
            assert base is None, f"TPUModelRunner now defines {name}"
            continue
        assert inspect.isfunction(base), f"TPUModelRunner has no method {name}"
        assert list(inspect.signature(fn).parameters) == list(
            inspect.signature(base).parameters
        ), name


def test_overridden_methods_are_called_by_the_runner():
    """Every override and added method is reached from main's runner (or
    from another override), so none is dead after a refactor on main."""
    sources = inspect.getsource(TPUModelRunner) + inspect.getsource(qr)
    for name in _own_methods():
        if name in ("load_model", "sample_tokens"):
            continue  # entry points the worker calls
        assert f"self.{name}(" in sources, name


# --- #2605 + #2609: post-verify read-offset update ---------------------------


def _offsets_runner(num_blocks=8):
    runner = MagicMock(spec=Qwen38Runner)
    runner.mamba_slot_read_offsets = torch.zeros(num_blocks, dtype=torch.int32)
    runner._update_mamba_slot_read_offsets = (
        Qwen38Runner._update_mamba_slot_read_offsets.__get__(runner)
    )
    return runner


@pytest.mark.parametrize("num_groups", [1, 2])
@pytest.mark.parametrize(
    "width,rows,num_reqs", [(4, 2, 2), (4, 3, 2), (4, 4, 4), (4, 6, 3)]
)
def test_update_read_offsets_impl_matches_eager(num_groups, width, rows, num_reqs):
    """The fused update matches main's eager scatter across 1 vs 2 groups,
    rows < / == / > width, and num_reqs < width."""
    num_blocks = 24
    raw_slots = [[2, 5, 8, 11], [14, 17, 19, 22]]
    groups = [
        torch.tensor(
            [raw_slots[g][i] if i < num_reqs else 0 for i in range(width)],
            dtype=torch.int32,
        )
        for g in range(num_groups)
    ]
    token_rows = [
        [10, 11, 12, 13],
        [20, INVALID_TOKEN_ID, INVALID_TOKEN_ID, INVALID_TOKEN_ID],
        [30, 31, INVALID_TOKEN_ID, INVALID_TOKEN_ID],
        [40, 41, 42, INVALID_TOKEN_ID],
        [50, 51, 52, 53],
        [INVALID_TOKEN_ID] * 4,
    ]
    next_tokens = torch.tensor(token_rows[:rows], dtype=torch.int32)

    main = MagicMock(spec=TPUModelRunner)
    main.mamba_slot_read_offsets = torch.full((num_blocks,), 7, dtype=torch.int32)
    main.mamba_slot_read_offsets[0] = 0
    TPUModelRunner._update_mamba_slot_read_offsets(main, groups, next_tokens, num_reqs)
    expected = main.mamba_slot_read_offsets

    for fn in (
        qr._update_read_offsets_impl,
        torch.compile(qr._update_read_offsets_impl, backend="eager", fullgraph=True),
    ):
        actual = torch.full((num_blocks,), 7, dtype=torch.int32)
        actual[0] = 0
        fn(actual, groups, next_tokens, torch.tensor(num_reqs, dtype=torch.int64))
        assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    "device_type,has_next_tokens,fused",
    [("tpu", True, True), ("tpu", False, False), ("cpu", True, False)],
)
def test_use_fused_read_offset_update(device_type, has_next_tokens, fused):
    read_offsets = SimpleNamespace(device=SimpleNamespace(type=device_type))
    next_tokens = torch.tensor([[1, 2]]) if has_next_tokens else None
    assert qr._use_fused_read_offset_update(read_offsets, next_tokens) is fused


def test_cpu_buffer_takes_eager_scatter(monkeypatch):
    compiled = MagicMock()
    monkeypatch.setattr(qr, "_update_read_offsets_compiled", compiled)
    runner = _offsets_runner(num_blocks=8)
    next_tokens = torch.tensor(
        [[10, 11, INVALID_TOKEN_ID], [20, INVALID_TOKEN_ID, INVALID_TOKEN_ID]]
    )
    runner._update_mamba_slot_read_offsets(
        [torch.tensor([1, 4], dtype=torch.int32)], next_tokens, 2
    )
    compiled.assert_not_called()
    assert runner.mamba_slot_read_offsets.tolist() == [0, 1, 0, 0, 0, 0, 0, 0]


def test_fused_update_passes_a_cached_num_reqs_scalar(monkeypatch):
    seen = []
    monkeypatch.setattr(qr, "_use_fused_read_offset_update", lambda *a: True)
    monkeypatch.setattr(
        qr, "_update_read_offsets_compiled", lambda *a: seen.append(a[3]) or a[3]
    )
    runner = SimpleNamespace(
        mamba_slot_read_offsets=torch.zeros(8, dtype=torch.int32), device="cpu"
    )
    runner._num_reqs_tensor = Qwen38Runner._num_reqs_tensor.__get__(runner)
    for _ in range(2):
        Qwen38Runner._update_mamba_slot_read_offsets(
            runner, [torch.tensor([1, 4])], torch.tensor([[1, 2], [3, 4]]), 2
        )
    assert seen[0] is seen[1] and int(seen[0]) == 2


# --- #2609: sampling-parameter upload reuse ----------------------------------


def test_upload_keyed_reuses_until_values_change():
    runner = SimpleNamespace(_h2d_reuse={}, device="cpu")
    upload = Qwen38Runner._upload_keyed
    temps = torch.tensor([[0.7], [1.0]], dtype=torch.bfloat16)
    first = upload(runner, "t", temps, temps.float().numpy().tobytes())
    again = upload(runner, "t", temps.clone(), temps.float().numpy().tobytes())
    assert again is first
    changed = torch.tensor([[0.5], [1.0]], dtype=torch.bfloat16)
    third = upload(runner, "t", changed, changed.float().numpy().tobytes())
    assert third is not first and torch.equal(third, changed)


def test_padded_sampling_params_reuses_until_inputs_change():
    input_batch = SimpleNamespace(
        temperature_cpu_tensor=torch.tensor([0.7, 1.0, 0.9], dtype=torch.float32),
        top_k_cpu_tensor=torch.tensor([20, 1024, 5], dtype=torch.int32),
        top_p_cpu_tensor=torch.tensor([0.95, 1.0, 0.8], dtype=torch.float32),
    )
    runner = SimpleNamespace(
        _h2d_reuse={},
        device=torch.device("cpu"),
        vocab_size=512,
        input_batch=input_batch,
    )
    for name in (
        "_upload_keyed",
        "_upload_unless_unchanged",
        "_padded_sampling_params",
    ):
        setattr(runner, name, getattr(Qwen38Runner, name).__get__(runner))
    cpu = torch.device("cpu")

    t1, k1, p1 = runner._padded_sampling_params(0, 2, 4, torch.bfloat16, cpu)
    t2, k2, p2 = runner._padded_sampling_params(0, 2, 4, torch.bfloat16, cpu)
    assert t2 is t1 and k2 is k1 and p2 is p1
    # top_k >= vocab_size clamps to 0; inactive slots pad with (1.0, 0, 1.0)
    assert k1.squeeze(1).tolist() == [20, 0, 0, 0]
    # Same values as main's per-call upload.
    ref = TPUModelRunner._build_padded_sampling_params(
        runner, 0, 2, torch.zeros(4, 8, dtype=torch.bfloat16)
    )
    for a, b in zip((t1, k1, p1), ref):
        assert torch.equal(a, b)

    input_batch.temperature_cpu_tensor[0] = 0.3
    t3, k3, p3 = runner._padded_sampling_params(0, 2, 4, torch.bfloat16, cpu)
    assert t3 is not t1 and k3 is k1 and p3 is p1

    t4, k4, p4 = runner._padded_sampling_params(0, 3, 4, torch.bfloat16, cpu)
    assert t4 is not t3 and k4 is not k3 and p4 is not p3
    assert k4.squeeze(1).tolist() == [20, 0, 5, 0]


# --- #2650: one-chunk substitution tail --------------------------------------


@pytest.mark.parametrize("bucketed", [False, True])
def test_spec_tail_matches_main_per_chunk_assembly(bucketed):
    """The fused tail returns main's source (padded to the bound), rejected
    counts, indices and draft counts."""
    k, num_reqs, rows, max_reqs = 3, 3, 4, 8
    main = object.__new__(TPUModelRunner)
    runner = object.__new__(Qwen38Runner)
    for r in (main, runner):
        r.device = torch.device("cpu")
        r.max_num_reqs = max_reqs
        r.speculative_config = SimpleNamespace(num_speculative_tokens=k)
        r.drafter = SimpleNamespace(bucketed_drafts=bucketed)

    next_tokens = torch.tensor(
        [[11, 12, -1, -1], [21, -1, -1, -1], [31, 32, 33, 34], [-1, -1, -1, -1]],
        dtype=torch.int32,
    )
    drafts = (
        torch.arange(100, 100 + (rows if bucketed else num_reqs) * k)
        .reshape(-1, k)
        .to(torch.int32)
    )
    state = SimpleNamespace(
        num_reqs_list=[num_reqs],
        spec_decode_metadata_list=[
            SimpleNamespace(
                draft_lengths=torch.tensor([3, 3, 3, 0], dtype=torch.int32),
                draft_lengths_cpu=np.array([3, 3, 3, 0]),
            )
        ],
    )
    drafts_in = [drafts] if bucketed else drafts
    ref = main._assemble_async_spec_substitution(drafts_in, [next_tokens], state)
    got = runner._assemble_async_spec_substitution(drafts_in, [next_tokens], state)
    bound = max_reqs * (1 + k)
    ref_src = torch.nn.functional.pad(ref[0][0], (0, bound - ref[0][0].shape[0]))
    assert torch.equal(got[0][0], ref_src)
    assert torch.equal(got[2], ref[2])
    assert got[1] == ref[1] and got[3] == ref[3]


def test_spec_tail_precompiles_right_after_the_boundary_copies(monkeypatch):
    calls = []
    monkeypatch.setattr(
        TPUModelRunner,
        "_precompile_spec_boundary_copies",
        lambda self: calls.append("boundary"),
    )
    runner = object.__new__(Qwen38Runner)
    runner._precompile_spec_tail = lambda: calls.append("tail")
    runner._precompile_spec_boundary_copies()
    assert calls == ["boundary", "tail"]


# --- #2607: speculative seed-copy sources in one program ---------------------

NUM_BLOCKS = 32
BLOCK_SIZE = 256


def _table(block_rows, num_rows=16, num_cols=8):
    cpu = torch.zeros((num_rows, num_cols), dtype=torch.int32)
    for row, blocks in enumerate(block_rows):
        cpu[row, : len(blocks)] = torch.tensor(blocks, dtype=torch.int32)
    return SimpleNamespace(get_cpu_tensor=lambda cpu=cpu: cpu)


def _spec_self(cls, offset, *, window=3, row=(10, 11, 12, 13), split=1, new=False):
    """One request crossing from state column 0 to 1 (see
    tests/runner/test_mamba_state_seed_copies.py), methods bound from `cls`."""
    read_offsets = torch.zeros(NUM_BLOCKS, dtype=torch.int32)
    read_offsets[row[0]] = offset
    raw = torch.zeros((NUM_BLOCKS, 8), dtype=torch.float32)
    ns = SimpleNamespace(
        requests={"a": object()},
        _mamba_copy_plan=[(1, [raw])],
        scheduler_config=SimpleNamespace(async_scheduling=False),
        speculative_config=None,
        _pre_async_results=None,
        _mamba_state_pos={"a": 0},
        _mamba_offset_seeded=set() if new else {"a"},
        _mamba_state_block_size=BLOCK_SIZE,
        _pool_block_split=split,
        bm_pool_layout=None,
        _mamba_ckpt_window=window,
        cache_config=SimpleNamespace(block_size=BLOCK_SIZE),
        block_size=BLOCK_SIZE,
        device="cpu",
        mamba_slot_read_offsets=read_offsets,
        _pending_mamba_state_copies=[],
        _unfenced_h2d_sources=[],
        _registered_h2d_sources=set(),
        input_batch=SimpleNamespace(
            req_id_to_index={"a": 0},
            num_computed_tokens_cpu=np.array([BLOCK_SIZE] + [0] * 15),
            req_ids=["a"] + [None] * 15,
            block_table=[_table([]), _table([list(row)])],
        ),
    )
    ns._bucket_len = TPUModelRunner._bucket_len
    ns._pad_to_bucket = TPUModelRunner._pad_to_bucket
    for name in (
        "_pad_dev_to_bucket",
        "_expand_pool_split",
        "_spec_seed_sources",
        "_spec_seed_copies",
        "_copy_mamba_state_blocks",
        "_mamba_column_stride",
    ):
        if hasattr(cls, name):
            setattr(ns, name, getattr(cls, name).__get__(ns))
    return ns


def _real_pairs(staged):
    _raws, src_t, dst_t = staged
    return [
        (int(s), int(d))
        for s, d in zip(src_t.tolist(), dst_t.tolist())
        if (s, d) != (0, 0)
    ]


@pytest.mark.parametrize("offset", [0, 1, 2])
@pytest.mark.parametrize("split", [1, 3])
@pytest.mark.parametrize("new", [False, True])
@pytest.mark.parametrize("row", [(10, 11, 12, 13), (10, 11)])
def test_seed_copies_equal_main_selection(offset, split, new, row):
    sched = SimpleNamespace(num_scheduled_tokens={"a": 1})
    got = _spec_self(Qwen38Runner, offset, split=split, new=new, row=row)
    ref = _spec_self(TPUModelRunner, offset, split=split, new=new, row=row)
    Qwen38Runner._collect_mamba_state_seed_copies(got, sched, 0, 1)
    TPUModelRunner._collect_mamba_state_seed_copies(ref, sched, 0, 1)
    assert [_real_pairs(s) for s in got._pending_mamba_state_copies] == [
        _real_pairs(s) for s in ref._pending_mamba_state_copies
    ]
    assert torch.equal(got.mamba_slot_read_offsets, ref.mamba_slot_read_offsets)


@pytest.mark.parametrize("split", [1, 2, 3])
@pytest.mark.parametrize("num_crossings", [1, 5, 9])
def test_spec_seed_copies_split_lands_on_bucket_ladder(split, num_crossings):
    offsets = torch.zeros(128, dtype=torch.int32)
    offsets[10] = 2
    fake = _spec_self(Qwen38Runner, 0, window=3, split=split)
    fake.mamba_slot_read_offsets = offsets
    rows = []
    expected_pairs = []
    for i in range(num_crossings):
        base = 10 + 4 * i
        new_owner = int(i % 2 == 1)
        dst = base + 3
        ckpts = [base, base + 1, base + 2]
        rows.append([base, new_owner, 1, dst, *ckpts])
        picked = ckpts[0] if new_owner else ckpts[int(offsets[base])]
        for j in range(split):
            expected_pairs.append((picked * split + j, dst * split + j))

    src_t, dst_t = fake._spec_seed_copies(rows, window=3)
    expected_len = TPUModelRunner._bucket_len(num_crossings * split)
    assert src_t.shape == (expected_len,)
    assert dst_t.shape == (expected_len,)
    actual_pairs = [(s, d) for s, d in zip(src_t.tolist(), dst_t.tolist()) if s or d]
    assert actual_pairs == expected_pairs


@pytest.mark.parametrize("split", [1, 2, 3])
def test_spec_seed_rows_impl_fullgraph_compile(split):
    offsets = torch.zeros(32, dtype=torch.int32)
    offsets[5] = 2
    offsets[10] = 1
    rows = [[5, 0, 1, 8, 5, 6, 7], [10, 1, 1, 13, 10, 11, 12]]
    out_len = TPUModelRunner._bucket_len(len(rows) * split)
    in_len = (out_len + split - 1) // split
    packed = torch.tensor(
        rows + [[0, 1, 0, 0, 0, 0, 0]] * (in_len - len(rows)), dtype=torch.long
    )
    ref_src, ref_dst = qr._spec_seed_rows_impl(offsets, packed, split, out_len)
    compiled = torch.compile(
        qr._spec_seed_rows_impl, backend="eager", fullgraph=True, dynamic=False
    )
    got_src, got_dst = compiled(offsets, packed, split, out_len)
    assert torch.equal(got_src, ref_src) and torch.equal(got_dst, ref_dst)


@pytest.mark.parametrize("split", [1, 2, 3])
def test_precompile_mamba_rollback_helpers_warms_spec_seed_ladder(monkeypatch, split):
    warmed: set[tuple[tuple[int, int], int, int]] = set()
    monkeypatch.setattr(qr, "_rollback_offsets_seed_compiled", lambda o, d: None)
    monkeypatch.setattr(qr, "_rollback_offsets_migrate_compiled", lambda o, s, d: None)
    monkeypatch.setattr(
        qr,
        "_spec_seed_rows_compiled",
        lambda o, packed, sp, n: warmed.add((tuple(packed.shape), sp, n)),
    )
    window, max_num_reqs = 3, 6
    fake = _spec_self(Qwen38Runner, 0, window=window, split=split)
    fake.mamba_slot_read_offsets = torch.zeros(128, dtype=torch.int32)
    fake.max_num_reqs = max_num_reqs
    fake.kv_cache_config = None
    fake._precompile_timed = lambda _label: nullcontext()
    Qwen38Runner._precompile_mamba_rollback_helpers(fake)

    observed: set[tuple[tuple[int, int], int, int]] = set()
    monkeypatch.setattr(
        qr,
        "_spec_seed_rows_impl",
        lambda o, packed, sp, n: (
            observed.add((tuple(packed.shape), sp, n))
            or (torch.zeros(n, dtype=torch.int32), torch.zeros(n, dtype=torch.int32))
        ),
    )
    for count in range(1, len(fake._mamba_copy_plan) * max_num_reqs + 1):
        rows = [[10, 0, 1, 13, 10, 11, 12] for _ in range(count)]
        fake._spec_seed_copies(rows, window=window)
    assert observed <= warmed


# --- #2651: token padding mask upload skip -----------------------------------


def test_padding_mask_reuse_skips_unchanged_rows():
    state = SkipUnchangedTokenPaddingState.create(8, "cpu")
    state.update(3, 8)
    assert state.local_padding_mask.tolist() == [False] * 3 + [True] * 5
    state.local_padding_mask[0] = True  # a skipped write leaves this alone
    state.update(3, 8)
    assert state.local_padding_mask[0]
    state.update(2, 8)
    assert state.local_padding_mask.tolist() == [False] * 2 + [True] * 6
    state.update(3, 8)
    assert state.local_padding_mask.tolist() == [False] * 3 + [True] * 5


@pytest.mark.parametrize("with_state", [False, True])
def test_load_model_swaps_in_the_skipping_padding_state(monkeypatch, with_state):
    mask = torch.zeros(8, dtype=torch.bool)

    def load(self):
        self._token_padding_state = (
            token_padding.TokenPaddingState(local_padding_mask=mask)
            if with_state
            else None
        )
        token_padding.set_padding_state(self._token_padding_state)

    monkeypatch.setattr(TPUModelRunner, "load_model", load)
    runner = object.__new__(Qwen38Runner)
    try:
        runner.load_model()
        state = runner._token_padding_state
        if with_state:
            assert isinstance(state, SkipUnchangedTokenPaddingState)
            assert state.local_padding_mask is mask
        else:
            assert state is None
        assert token_padding._padding_state is state
    finally:
        token_padding.set_padding_state(None)


# --- #2723: stage timing only with a KV connector sink -----------------------


def test_timing_skipped_without_sink(monkeypatch):
    monkeypatch.setattr(qr, "has_kv_transfer_group", lambda: False)
    timing = SinkStageTiming(0)
    with timing.stage("main_forward"):
        pass
    with timing.host_stage("cpu_step"):
        pass
    assert timing.take_samples() == ([], [])


def test_timing_recorded_with_sink(monkeypatch):
    monkeypatch.setattr(qr, "has_kv_transfer_group", lambda: True)
    timing = SinkStageTiming(0)
    with timing.host_stage("cpu_step"):
        pass
    cpu, device = timing.take_samples()
    assert [s[:2] for s in cpu] == [("0", "cpu_step")] and device == []


def test_init_swaps_in_the_sink_stage_timing(monkeypatch):
    def init(self, *args, **kwargs):
        self._stage_timing = stage_timing.RunnerStageTiming(3)

    monkeypatch.setattr(TPUModelRunner, "__init__", init)
    runner = Qwen38Runner()
    assert isinstance(runner._stage_timing, SinkStageTiming)
    assert runner._stage_timing._rank == "3"
