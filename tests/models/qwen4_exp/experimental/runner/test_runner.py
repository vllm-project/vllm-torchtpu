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
from typing import Any
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
    "_get_two_slot_gdn_layers",
    "_snapshot_two_slot_gdn_verify_inputs",
    "_plan_two_slot_boundary_copies",
    "_two_slot_dist3",
    "_two_slot_win_and_reqs",
    "_get_compiled_boundary_meta",
    "_post_verify_two_slot_gdn_chunk",
    "_replay_two_slot_gdn_chunk",
    "_export_two_slot_interior_boundaries",
    "_copy_two_slot_boundary_states",
    "_update_ping_pong_mamba_slot_read_offsets",
    "_precompile_two_slot_runtime",
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
    from another override / cache manager), so none is dead after a refactor."""
    from vllm_torchtpu.models.qwen4_exp.experimental import pooled_cache
    from vllm_torchtpu.runner import kv_cache_manager

    sources = inspect.getsource(TPUModelRunner) + inspect.getsource(qr)
    cache_sources = inspect.getsource(pooled_cache) + inspect.getsource(
        kv_cache_manager
    )
    for name in _own_methods():
        if name in ("load_model", "execute_model", "sample_tokens"):
            continue  # entry points the worker calls
        if name == "_build_mamba_copy_plan":
            assert f"runner.{name}(" in cache_sources, name
            continue
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


# --- Two-slot ping-pong GDN runner methods & precompilation ------------------


def _two_slot_runner(*, shared_raw: bool = True, split: int = 1, num_blocks: int = 64):
    """Build a CPU MagicMock(spec=Qwen38Runner) wired with two-slot methods."""
    from vllm_torchtpu.models.qwen4_exp.experimental.runner import two_slot

    raw0 = torch.zeros((num_blocks, 4), dtype=torch.float32)
    raw1 = raw0 if shared_raw else torch.zeros((num_blocks, 4), dtype=torch.float32)

    runner = MagicMock(spec=Qwen38Runner)
    runner.device = torch.device("cpu")
    runner.max_num_reqs = 8
    runner.num_reqs_max_model_len = 4
    runner.num_reqs_most_model_len = 8
    runner._mamba_ckpt_window = 4
    runner._pool_block_split = split
    runner._mamba_state_block_size = 512
    runner._mamba_copy_plan = [(0, [raw0]), (1, [raw1])]
    runner._mamba_copy_ping_pong_gids = {0}
    runner._mamba_copy_group_windows = {0: 2, 1: 4}
    runner.mamba_slot_read_offsets = torch.zeros(num_blocks, dtype=torch.int32)
    runner.cache_config = SimpleNamespace(enable_prefix_caching=True, block_size=512)
    runner.vllm_config = SimpleNamespace(kv_transfer_config=None)
    runner.scheduler_config = SimpleNamespace(async_scheduling=False)
    runner.speculative_config = SimpleNamespace(num_speculative_tokens=3)
    runner.attn_groups = [
        [
            SimpleNamespace(
                kv_cache_group_id=0,
                layer_names=["model.layers.0.linear_attn"],
                kv_cache_spec=None,
            ),
            SimpleNamespace(
                kv_cache_group_id=1,
                layer_names=["model.layers.1.ple"],
                kv_cache_spec=None,
            ),
        ]
    ]
    runner._bucket_len = TPUModelRunner._bucket_len
    for name in (
        "_num_reqs_tensor",
        "_two_slot_dist3",
        "_get_two_slot_gdn_layers",
        "_get_compiled_boundary_meta",
        "_post_verify_two_slot_gdn_chunk",
        "_replay_two_slot_gdn_chunk",
        "_export_two_slot_interior_boundaries",
        "_copy_two_slot_boundary_states",
        "_update_ping_pong_mamba_slot_read_offsets",
        "_plan_two_slot_boundary_copies",
        "_precompile_two_slot_runtime",
        "_expand_pool_split",
        "_pad_dev_to_bucket",
        "_mamba_column_stride",
    ):
        setattr(runner, name, getattr(Qwen38Runner, name).__get__(runner))

    events: list[tuple[str, Any]] = []
    copy_calls: list[tuple[list[torch.Tensor], list[tuple[int, int]]]] = []

    def _record_copy(raws, src_t, dst_t):
        pairs = [
            (int(s), int(d))
            for s, d in zip(src_t.tolist(), dst_t.tolist())
            if s != 0 or d != 0
        ]
        events.append(("boundary_copy", runner.mamba_slot_read_offsets.clone()))
        copy_calls.append((raws, pairs))

    runner._copy_mamba_state_blocks = _record_copy

    replay_calls: list[dict[str, Any]] = []

    def _record_replay(**kwargs):
        tag = (
            "interior_export"
            if torch.all(kwargs["slot_read_offsets"] == 0)
            and kwargs["slot_read_offsets"] is not runner.mamba_slot_read_offsets
            else "replay"
        )
        events.append((tag, runner.mamba_slot_read_offsets.clone()))
        replay_calls.append(
            {
                "tag": tag,
                "state_indices": kwargs["state_indices"].tolist(),
                "ckpt_indices": kwargs["ckpt_indices"].tolist(),
                "commit_lens": kwargs["commit_lens"].tolist(),
                "offsets_at_call": kwargs["slot_read_offsets"].clone(),
                "staged_inputs": kwargs.get("staged_inputs"),
            }
        )

    stub_layer = SimpleNamespace(
        prefix="model.layers.0.linear_attn",
        _two_slot_mtp=True,
        _verify_win_cap=16,
        kv_cache=(raw0,),
        gdn_pooled_replay_op=object(),
        replay_two_slot_state=_record_replay,
    )
    runner.model = SimpleNamespace(
        model=SimpleNamespace(layers=[SimpleNamespace(linear_attn=stub_layer)])
    )
    return runner, stub_layer, events, copy_calls, replay_calls, two_slot


def test_post_verify_two_slot_ordering_replay_copy_flip():
    """Verify exact order: replay -> boundary copy -> interior export -> offset flip."""
    runner, _layer, events, copy_calls, replay_calls, two_slot = _two_slot_runner(
        shared_raw=True, split=1
    )
    # Row 0: interior crossing (510 -> 514, 4 accepted), c_read[11]=0
    # Row 1: partial reject (2 of 4 accepted), exact boundary 512, c_read[30]=0
    # Row 2: prefill (outside num_windowed=2), old offset 1 -> reset to 0 on flip
    runner.mamba_slot_read_offsets[11] = 0
    runner.mamba_slot_read_offsets[30] = 0
    runner.mamba_slot_read_offsets[50] = 1
    initial_offsets = runner.mamba_slot_read_offsets.clone()

    qsl = torch.tensor([0, 4, 8, 16], dtype=torch.int32)
    seq_lens = torch.tensor([514, 512, 16], dtype=torch.int32)
    next_tokens = torch.tensor(
        [
            [1, 2, 3, 4],  # row 0: all 4 accepted -> interior boundary at 512 (k=2)
            [5, 6, -1, -1],  # row 1: 2 accepted < 4 -> replay(2), exact boundary at 512
            [7, -1, -1, -1],  # row 2: non-windowed prefill
        ],
        dtype=torch.int32,
    )
    entries = torch.tensor(
        [
            [0, 512, 4, 10, 11, 12, 0, 0, 20, 21, 22, 23, 24],
            [1, 512, 2, 30, 31, 0, 0, 0, 40, 41, 42, 43, 44],
        ],
        dtype=torch.int32,
    )
    plan = two_slot.SpecBoundaryPlan(
        entries=entries,
        rows=entries[:, 0].long(),
        seq_lens=seq_lens.index_select(0, entries[:, 0].long()),
        block_size=512,
        host_entries=entries,
        chunk_seq_lens=seq_lens,
    )
    layer_md = SimpleNamespace(
        mamba_state_indices=torch.tensor([11, 30, 50], dtype=torch.int32),
        mamba_ckpt_indices=torch.tensor(
            [[11, 12], [30, 31], [50, 51]], dtype=torch.int32
        ),
    )
    chunk = two_slot.PingPongMambaChunkState(
        state_indices=[torch.tensor([11, 30, 50], dtype=torch.int32)],
        attn_metadata={"model.layers.0.linear_attn": layer_md},
        chunk_num_windowed=2,
        num_reqs=3,
        query_start_loc=qsl,
        seq_lens=seq_lens,
        staged_verify_inputs=None,
        staged_ping_pong_stacks=(
            torch.tensor([[11, 30, 50]], dtype=torch.int32),
            torch.tensor([[[11, 12], [30, 31], [50, 51]]], dtype=torch.int32),
        ),
    )
    md = SimpleNamespace(draft_lengths_cpu=np.array([3, 3, 0], dtype=np.int32))

    runner._post_verify_two_slot_gdn_chunk(chunk, md, next_tokens, plan)

    assert [name for name, _ in events] == [
        "replay",
        "boundary_copy",
        "interior_export",
    ]
    # All three pre-flip steps saw the pre-flip read offsets (c_read intact).
    for _, snap_ro in events:
        assert torch.equal(snap_ro, initial_offsets)
    # Replay ran for row 1 with commit_len=2 (row 0 all-accepted -> 0).
    assert replay_calls[0]["commit_lens"] == [0, 2, 0]
    # Boundary copy exported exact GDN write slot 31->30 and PLE 22->20, 41->40.
    assert (31, 30) in copy_calls[0][1]
    assert (22, 20) in copy_calls[0][1]
    # Interior export replayed row 0 from read_blk=11 into dst_b=10 for k=2 tokens.
    assert replay_calls[1]["state_indices"] == [11, 0, 0]
    assert replay_calls[1]["ckpt_indices"] == [[11, 10], [0, 0], [0, 0]]
    assert replay_calls[1]["commit_lens"] == [2, 0, 0]
    # Final flip happened AFTER boundary export: windowed 11,30 flipped 0->1;
    # non-windowed prefill block 50 reset 1->0.
    assert runner.mamba_slot_read_offsets[11].item() == 1
    assert runner.mamba_slot_read_offsets[30].item() == 1
    assert runner.mamba_slot_read_offsets[50].item() == 0


@pytest.mark.parametrize("split", [1, 2])
@pytest.mark.parametrize("verify", [True, False])
def test_copy_two_slot_boundary_states_vectorized_matches_eager_fallback(split, verify):
    """Vectorized shared-pool path and eager distinct-pool fallback agree."""
    entries = torch.tensor(
        [
            [0, 512, 4, 10, 11, 12, 0, 0, 20, 21, 22, 23, 24],
            [1, 512, 1, 30, 31, 0, 0, 0, 40, 41, 42, 43, 44],
            [2, 512, 1, 35, 36, 0, 0, 0, 45, 46, 47, 48, 49],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.int32,
    )
    seq_lens = torch.tensor([514, 512, 512], dtype=torch.int32)
    qsl = torch.tensor([0, 4, 5, 6], dtype=torch.int32)
    next_tokens = (
        torch.tensor(
            [[1, 2, 3, 4], [5, -1, -1, -1], [6, -1, -1, -1]], dtype=torch.int32
        )
        if verify
        else None
    )

    def _run(shared_raw: bool):
        runner, _, _, copy_calls, replay_calls, two_slot = _two_slot_runner(
            shared_raw=shared_raw, split=split
        )
        runner.mamba_slot_read_offsets[11] = 0
        runner.mamba_slot_read_offsets[30] = 0
        runner.mamba_slot_read_offsets[35] = 1
        plan = two_slot.SpecBoundaryPlan(
            entries=entries,
            rows=entries[:, 0].long(),
            seq_lens=seq_lens.index_select(0, entries[:, 0].long()),
            block_size=512,
            host_entries=entries,
            chunk_seq_lens=seq_lens,
        )
        chunk = two_slot.PingPongMambaChunkState(
            state_indices=[torch.tensor([11, 30, 35], dtype=torch.int32)],
            attn_metadata={},
            chunk_num_windowed=3,
            num_reqs=3,
            query_start_loc=qsl,
            seq_lens=seq_lens,
        )
        runner._copy_two_slot_boundary_states(plan, next_tokens, ping_pong_chunk=chunk)
        all_pairs = sorted(p for _, ps in copy_calls for p in ps)
        replays = [
            (r["state_indices"], r["ckpt_indices"], r["commit_lens"])
            for r in replay_calls
        ]
        return all_pairs, replays

    vec_pairs, vec_replays = _run(shared_raw=True)
    eager_pairs, eager_replays = _run(shared_raw=False)
    assert vec_pairs == eager_pairs
    assert vec_replays == eager_replays
    assert len(vec_pairs) > 0
    if verify:
        assert vec_replays == [([11, 0, 0], [[11, 10], [0, 0], [0, 0]], [2, 0, 0])]
    else:
        assert vec_replays == []


def test_plan_two_slot_boundary_copies_candidates_and_fast_exit():
    """Verify fast-exit when no boundary crossed and exact candidate packing."""
    from vllm_torchtpu.models.qwen4_exp.experimental.runner import two_slot

    runner, _, _, _, _, _ = _two_slot_runner(shared_raw=True, split=1)
    runner._unfenced_h2d_sources = []
    runner._registered_h2d_sources = set()
    runner.input_batch = SimpleNamespace(
        num_computed_tokens_cpu=np.array([100, 511, 510], dtype=np.int32),
        req_ids=["r0", "r1", "r2"],
        block_table=[
            _table([[10, 11, 12, 13, 14], [30, 31, 32, 33, 34], [50, 51, 52, 53, 54]]),
            _table([[20, 21, 22, 23, 24], [40, 41, 42, 43, 44], [60, 61, 62, 63, 64]]),
        ],
    )
    runner.query_start_loc_np = np.array([0, 4, 5, 9], dtype=np.int32)
    sched = SimpleNamespace(
        num_scheduled_tokens={"r0": 4, "r1": 1, "r2": 4},
        scheduled_spec_decode_tokens={"r0": [1, 2, 3], "r2": [4, 5, 6]},
    )
    seq_lens = torch.tensor([104, 512, 514], dtype=torch.int32)

    plan = runner._plan_two_slot_boundary_copies(sched, 0, 3, seq_lens)
    assert plan is not None
    assert tuple(plan.entries.shape) == two_slot.two_slot_entries_shape(runner, 2)
    # r1 (plain 1-token decode 511->512) and r2 (spec 510->514) are staged; r0 skipped.
    assert plan.host_entries[0, :3].tolist() == [1, 512, 1]
    assert plan.host_entries[1, :3].tolist() == [2, 512, 4]
    assert plan.host_entries[0, 3:8].tolist() == [30, 31, 32, 33, 34]
    assert plan.host_entries[1, 8:13].tolist() == [60, 61, 62, 63, 64]

    # Fast-exit returns None when no request can reach a boundary.
    runner.input_batch.num_computed_tokens_cpu = np.array([10, 20, 30], dtype=np.int32)
    assert runner._plan_two_slot_boundary_copies(sched, 0, 3, seq_lens) is None


@pytest.mark.parametrize("is_spec_step", [True, False])
def test_sample_tokens_two_slot_verify_and_non_verify_branches(
    monkeypatch, is_spec_step
):
    """Exercise sample_tokens verify + non-verify chunk routing with two-slot state."""
    monkeypatch.setattr(
        qr.AsyncTPUCopyState,
        "from_device_chunks",
        staticmethod(lambda *a, **kw: None),
    )
    runner, _, _, _, _, two_slot = _two_slot_runner(shared_raw=True, split=1)
    runner.scheduler_config.async_scheduling = True
    runner._update_placeholder = lambda *a, **kw: {}
    runner._sync_replicated_drafts_across_tp = lambda d: d
    runner._modify_prev_results = lambda: None
    runner._async_spec_source_rows = lambda: 8
    runner._assemble_async_spec_substitution = lambda *a, **kw: ([], {}, None, None)
    runner.is_pooling_model = False
    runner._pp_is_last = True
    runner._is_async_drafter = False
    runner._is_dspark_pd_producer = False
    runner._adaptive_verification = None
    runner._fence_spec_kv_under_token_copy = False
    runner._dp_step_other_decode = False
    runner.num_prompt_logprobs = {}
    runner.requests = {}
    runner._stage_timing = SinkStageTiming(0)
    runner._stage_timing._sink = False
    runner.input_batch = SimpleNamespace(
        max_num_logprobs=None,
        num_reqs=0,
        req_ids=[],
        req_id_to_index={},
        all_greedy=True,
        generators={},
    )
    runner._min_tokens_batch_state = lambda: None
    runner._get_sampling_generator = lambda: None
    runner._update_num_xla_graphs = lambda _tag: None
    runner.maybe_wait_for_kv_save = lambda _x: None
    runner._build_kv_connector_output = lambda _so: None
    runner.sample_tokens = Qwen38Runner.sample_tokens.__get__(runner)

    calls: list[str] = []
    runner._post_verify_two_slot_gdn_chunk = lambda *a, **kw: calls.append(
        "post_verify"
    )
    runner._copy_two_slot_boundary_states = lambda *a, **kw: calls.append("copy_nv")
    runner._update_mamba_slot_read_offsets = lambda *a, **kw: calls.append("std_off")
    runner._update_ping_pong_mamba_slot_read_offsets = lambda *a, **kw: calls.append(
        "pp_off"
    )
    runner._sample_spec_verify_chunk = lambda *a, **kw: torch.zeros(
        (2, 4), dtype=torch.int32
    )
    runner._sample_plain_chunk = lambda *a, **kw: (
        torch.zeros((2, 1), dtype=torch.int32),
        None,
    )

    chunk = two_slot.PingPongMambaChunkState(
        state_indices=[torch.tensor([1, 2], dtype=torch.int32)],
        attn_metadata={},
        chunk_num_windowed=2,
        num_reqs=2,
        query_start_loc=torch.tensor([0, 4, 8], dtype=torch.int32),
        seq_lens=torch.tensor([10, 12], dtype=torch.int32),
    )
    if is_spec_step:
        # Chunk 0 = verify (md != None), Chunk 1 = non-verify (md is None).
        state = SimpleNamespace(
            scheduler_output=SimpleNamespace(num_scheduled_tokens={}),
            draft_chunks=None,
            top_tokens=False,
            logits_list=[torch.zeros((2, 8)), torch.zeros((2, 8))],
            num_reqs_list=[2, 2],
            spec_decode_metadata_list=[SimpleNamespace(), None],
            mamba_state_indices_list=[[torch.tensor([3, 4])], [torch.tensor([5, 6])]],
            mamba_boundary_plans=[None, None],
            ping_pong_mamba_chunks=[chunk, chunk],
            two_slot_boundary_plans=[None, None],
            kv_save_nan_check=None,
            prompt_logprobs_chunks=[],
            pooler_output_list=[],
        )
    else:
        state = SimpleNamespace(
            scheduler_output=SimpleNamespace(num_scheduled_tokens={}),
            draft_chunks=None,
            top_tokens=False,
            logits_list=[torch.zeros((2, 8))],
            num_reqs_list=[2],
            spec_decode_metadata_list=[None],
            mamba_state_indices_list=[[torch.tensor([3, 4])]],
            mamba_boundary_plans=[None],
            ping_pong_mamba_chunks=[chunk],
            two_slot_boundary_plans=[None],
            kv_save_nan_check=None,
            prompt_logprobs_chunks=[],
            pooler_output_list=[],
        )
    runner.execute_model_state = state
    runner.sample_tokens(None)
    if is_spec_step:
        assert calls == ["post_verify", "std_off", "copy_nv", "std_off", "pp_off"]
    else:
        assert calls == ["copy_nv", "std_off", "pp_off"]


def test_precompile_two_slot_runtime_covers_all_shapes(monkeypatch):
    """Verify _precompile_two_slot_runtime warms all entries/rows/has_win/nt shapes."""
    from vllm_torchtpu.models.qwen4_exp.experimental.runner import two_slot

    runner, _layer, _events, copy_calls, replay_calls, _ = _two_slot_runner(
        shared_raw=True, split=2
    )

    # Pretend TPU device (hashable) for shape/dispatch checks on CPU tensors.
    class _FakeTpuDevice:
        type = "tpu"

    tpu_dev = _FakeTpuDevice()
    orig_zeros = torch.zeros
    orig_full = torch.full
    orig_tensor = torch.tensor

    def _cpu_zeros(*args, **kwargs):
        if kwargs.get("device") is tpu_dev:
            kwargs["device"] = torch.device("cpu")
        return orig_zeros(*args, **kwargs)

    def _cpu_full(*args, **kwargs):
        if kwargs.get("device") is tpu_dev:
            kwargs["device"] = torch.device("cpu")
        return orig_full(*args, **kwargs)

    def _cpu_tensor(*args, **kwargs):
        if kwargs.get("device") is tpu_dev:
            kwargs["device"] = torch.device("cpu")
        return orig_tensor(*args, **kwargs)

    monkeypatch.setattr(torch, "zeros", _cpu_zeros)
    monkeypatch.setattr(torch, "full", _cpu_full)
    monkeypatch.setattr(torch, "tensor", _cpu_tensor)
    monkeypatch.setattr(qr, "synchronize_tensors", lambda *a, **kw: None)

    runner.device = tpu_dev
    runner.max_num_reqs = 48
    runner.num_reqs_max_model_len = 8
    runner.num_reqs_most_model_len = 48
    runner._precompile_timed = lambda _label: nullcontext()

    pp_offset_shapes: set[tuple[str, tuple[int, ...]]] = set()
    nv_calls: set[tuple[tuple[int, int], int, bool, int]] = set()
    v_calls: set[tuple[tuple[int, int], int, tuple[int, int], int]] = set()

    def _spy_offsets(_ro, groups, _qsl, _w, _n):
        if len(groups) == 1 and groups[0].ndim == 2:
            pp_offset_shapes.add(("stack", tuple(groups[0].shape)))
        else:
            pp_offset_shapes.add(("list", (len(groups), groups[0].shape[0])))

    def _spy_nv(
        entries, seq_lens, _ro, _w, _mask, has_win, _bs, _width, _sp, out_bucket
    ):
        nv_calls.add((tuple(entries.shape), seq_lens.shape[0], has_win, out_bucket))
        return orig_zeros(out_bucket, dtype=torch.int32), orig_zeros(
            out_bucket, dtype=torch.int32
        )

    def _spy_v(
        entries, seq_lens, nt, _ro, _w, _mask, _l2p, _bs, _width, _sp, out_bucket
    ):
        v_calls.add(
            (
                tuple(entries.shape),
                seq_lens.shape[0],
                tuple(nt.shape),
                out_bucket,
            )
        )
        z = orig_zeros(out_bucket, dtype=torch.int32)
        return z, z, z, z, z

    copy_buckets: set[int] = set()
    runner._copy_mamba_state_blocks = lambda _raws, s, _d: copy_buckets.add(s.shape[0])

    monkeypatch.setattr(qr, "update_ping_pong_read_offsets_compiled", _spy_offsets)
    monkeypatch.setattr(qr, "compute_boundary_copy_no_verify_compiled", _spy_nv)
    monkeypatch.setattr(qr, "compute_boundary_copy_and_interior_compiled", _spy_v)

    runner._precompile_two_slot_runtime()

    expected_shapes = {two_slot.two_slot_entries_shape(runner, n) for n in range(1, 49)}
    assert len(expected_shapes) == 2  # 32 and 128 candidate rows at max_num_reqs=48
    # Both [G, rows] stack and list-of-G shapes warmed for both row counts.
    assert pp_offset_shapes == {
        ("stack", (1, 8)),
        ("list", (1, 8)),
        ("stack", (1, 48)),
        ("list", (1, 48)),
    }
    # Replay op warmed for both row counts.
    assert {len(r["state_indices"]) for r in replay_calls} == {8, 48}
    # No-verify warmed for every (entries_shape, rows, has_win in {False, True}).
    for sh in expected_shapes:
        ob = runner._bucket_len(len(runner._mamba_copy_plan) * sh[0] * 2)
        assert ob in copy_buckets
        for rows in (8, 48):
            assert (sh, rows, False, ob) in nv_calls
            assert (sh, rows, True, ob) in nv_calls
    # Verify+interior warmed for every (entries_shape, rows, nt_rows bucket).
    for sh in expected_shapes:
        ob = runner._bucket_len(len(runner._mamba_copy_plan) * sh[0] * 2)
        assert (sh, 8, (8, 4), ob) in v_calls
        for nt_rows in (8, 16, 32, 48):
            assert (sh, 48, (nt_rows, 4), ob) in v_calls
