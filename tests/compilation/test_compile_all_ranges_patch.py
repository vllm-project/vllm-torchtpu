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
"""How TPU_PARALLEL_PRECOMPILE splits a subgraph's ranges among a host's ranks.

Each range has one owner on the host, which compiles it; the other ranks
compile their own ranges first, then wait for and replay the owners' artifacts.
The biggest range takes far longer than the others, so startup is bounded by
when it starts: local rank 0 must start on it, not reach it after the smallest.
"""

from types import SimpleNamespace

import pytest
from vllm.compilation import piecewise_backend
from vllm.config.utils import Range
from vllm.distributed import parallel_state

from vllm_torchtpu import _patch_vllm_compile_all_ranges
from vllm_torchtpu.compilation import tpu_compiler

SIZES = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]


@pytest.fixture
def compile_all_ranges(monkeypatch):
    monkeypatch.setenv("TPU_PARALLEL_PRECOMPILE", "1")
    monkeypatch.setenv("TPU_PRECOMPILE_BIGGEST_FIRST", "1")
    monkeypatch.setenv("TPU_PRECOMPILE_PEER_REPLAY", "1")
    # host_slot() takes TPU_LOCAL_RANK_OFFSET off LOCAL_RANK; keep a value from
    # the environment from changing which rank owns which range.
    monkeypatch.delenv("TPU_LOCAL_RANK_OFFSET", raising=False)
    monkeypatch.setattr(
        piecewise_backend, "create_concrete_args", lambda graph, size: [size]
    )
    # Restored on teardown, so the patch does not leak into other tests.
    monkeypatch.setattr(
        piecewise_backend.PiecewiseBackend,
        "compile_all_ranges",
        piecewise_backend.PiecewiseBackend.compile_all_ranges,
    )
    _patch_vllm_compile_all_ranges()
    return piecewise_backend.PiecewiseBackend.compile_all_ranges


def _host(monkeypatch, ranks_per_host):
    """A world of two hosts with `ranks_per_host` ranks each, or an unknown
    one for None."""
    if ranks_per_host is None:

        def get_world_group():
            raise AssertionError("world group is not initialized")

    else:

        def get_world_group():
            return SimpleNamespace(world_size=2 * ranks_per_host)

    monkeypatch.setattr(parallel_state, "get_world_group", get_world_group)
    monkeypatch.setattr(parallel_state, "get_node_count", lambda: 2)


def _compile(
    compile_all_ranges,
    monkeypatch,
    local_rank,
    ranks_per_host=8,
    disable_cache=False,
    fail_at=None,
):
    """(range end, peer wait in seconds) per compile, in order. `fail_at`: the
    range whose compile raises MemoryError."""
    _host(monkeypatch, ranks_per_host)
    monkeypatch.setenv("LOCAL_RANK", str(local_rank))
    calls = []

    def compile(graph, args, *rest, compile_range, **kwargs):
        if compile_range.end == fail_at:
            raise MemoryError("vmem")
        calls.append(
            (compile_range.end, getattr(tpu_compiler._peer_wait, "timeout_s", 0.0))
        )
        return lambda *a: a

    # In vLLM's order: smallest range first.
    backend = SimpleNamespace(
        graph=object(),
        range_entries={
            Range(s, s): piecewise_backend.RangeEntry(compile_range=Range(s, s))
            for s in SIZES
        },
        vllm_backend=SimpleNamespace(
            compiler_manager=SimpleNamespace(
                compile=compile, disable_cache=disable_cache
            ),
            inductor_config={},
            is_encoder=False,
        ),
        compilation_config=None,
        piecewise_compile_index=0,
        total_piecewise_compiles=1,
        _log_compile_start=lambda compile_range: None,
    )
    compile_all_ranges(backend)
    return calls


def _owned(calls):
    return [end for end, wait in calls if wait == 0.0]


def test_local_rank_0_starts_on_the_biggest_range(compile_all_ranges, monkeypatch):
    assert _compile(compile_all_ranges, monkeypatch, 0)[0][0] == 4096


def test_each_range_has_one_owner_per_host(compile_all_ranges, monkeypatch):
    owned = [_owned(_compile(compile_all_ranges, monkeypatch, r)) for r in range(8)]
    # Dealt in a snake: the rank with the smallest first range gets the 9th.
    assert owned == [[4096], [2048], [1024], [512], [256], [128], [64], [32, 16]]


def test_other_ranges_wait_for_their_owner(compile_all_ranges, monkeypatch):
    calls = _compile(compile_all_ranges, monkeypatch, 0)
    assert calls == [(4096, 0.0)] + [
        (end, tpu_compiler.PEER_WAIT_S)
        for end in [32, 64, 128, 256, 512, 1024, 2048, 16]
    ]


def test_every_rank_compiles_every_range(compile_all_ranges, monkeypatch):
    for rank in range(8):
        calls = _compile(compile_all_ranges, monkeypatch, rank)
        assert sorted(end for end, _ in calls) == SIZES


def test_a_smaller_host_splits_the_ranges_among_its_ranks(
    compile_all_ranges, monkeypatch
):
    owned = [
        _owned(_compile(compile_all_ranges, monkeypatch, r, ranks_per_host=4))
        for r in range(4)
    ]
    assert owned == [[4096, 32, 16], [2048, 64], [1024, 128], [512, 256]]


def test_no_wait_when_the_host_size_is_unknown(compile_all_ranges, monkeypatch):
    calls = _compile(compile_all_ranges, monkeypatch, 1, ranks_per_host=None)
    assert all(wait == 0.0 for _, wait in calls)
    assert sorted(end for end, _ in calls) == SIZES


def test_biggest_first_off_deals_in_vllm_order(compile_all_ranges, monkeypatch):
    monkeypatch.setenv("TPU_PRECOMPILE_BIGGEST_FIRST", "0")
    owned = [_owned(_compile(compile_all_ranges, monkeypatch, r)) for r in range(8)]
    # vLLM's order, smallest first: local rank 0 starts on the smallest range.
    assert owned == [[16], [32], [64], [128], [256], [512], [1024], [2048, 4096]]
    for rank in range(8):
        calls = _compile(compile_all_ranges, monkeypatch, rank)
        assert sorted(end for end, _ in calls) == SIZES


@pytest.mark.parametrize(
    "biggest_first, firsts",
    [
        ("1", [4096, 2048, 1024, 512, 256, 128, 64, 32]),
        ("0", SIZES[:8]),
        (None, SIZES[:8]),
    ],
)
@pytest.mark.parametrize("peer_replay", ["0", None])
def test_peer_replay_off_compiles_every_range_without_waiting(
    compile_all_ranges, monkeypatch, peer_replay, biggest_first, firsts
):
    """TPU_PRECOMPILE_PEER_REPLAY off, explicitly or by default (None): no
    owners and no waits; each local rank starts on a different range and
    compiles every range itself."""
    for name, value in [
        ("TPU_PRECOMPILE_PEER_REPLAY", peer_replay),
        ("TPU_PRECOMPILE_BIGGEST_FIRST", biggest_first),
    ]:
        if value is None:
            monkeypatch.delenv(name)
        else:
            monkeypatch.setenv(name, value)
    runs = [_compile(compile_all_ranges, monkeypatch, r) for r in range(8)]
    assert [calls[0][0] for calls in runs] == firsts
    for calls in runs:
        assert all(wait == 0.0 for _, wait in calls)
        assert sorted(end for end, _ in calls) == SIZES


def _host_owners(compile_all_ranges, monkeypatch, local_ranks, **kwargs):
    """Every range a rank compiled without waiting, for each rank present."""
    return {
        r: _owned(_compile(compile_all_ranges, monkeypatch, r, **kwargs))
        for r in local_ranks
    }


def _assert_one_owner_each(owners):
    """The ranks present own every range exactly once between them, so no
    rank waits on a slot no process holds."""
    owned = [end for ends in owners.values() for end in ends]
    assert sorted(owned) == SIZES


def _assert_opted_out(compile_all_ranges, monkeypatch, local_ranks, **kwargs):
    for r in local_ranks:
        calls = _compile(compile_all_ranges, monkeypatch, r, **kwargs)
        assert all(wait == 0.0 for _, wait in calls)
        assert sorted(end for end, _ in calls) == SIZES


def test_dense_dp_engines_compile_alone(compile_all_ranges, monkeypatch):
    """Dense DP replicas keep a world group per engine (one rank at TP=1) and
    all bind LOCAL_RANK=0: with one rank per host they must not wait on slots
    1..7 that no process of theirs holds."""
    _assert_opted_out(compile_all_ranges, monkeypatch, [0] * 8, ranks_per_host=1)


def test_local_rank_offset_is_taken_off(compile_all_ranges, monkeypatch):
    """TPU_LOCAL_RANK_OFFSET=4 with 4 ranks on the host: the binding sets
    LOCAL_RANK 4..7, which are slots 0..3. Without taking the offset off they
    owned no range and replayed nothing."""
    monkeypatch.setenv("TPU_LOCAL_RANK_OFFSET", "4")
    owners = _host_owners(
        compile_all_ranges, monkeypatch, range(4, 8), ranks_per_host=4
    )
    # The same split as ranks 0..3 of a 4-rank host without an offset.
    assert list(owners.values()) == [
        [4096, 32, 16],
        [2048, 64],
        [1024, 128],
        [512, 256],
    ]


def test_a_slot_outside_the_host_opts_out(compile_all_ranges, monkeypatch):
    """A rank whose slot falls outside the host count waits on no one."""
    _assert_opted_out(compile_all_ranges, monkeypatch, [9], ranks_per_host=8)


def test_peer_replay_is_ignored_with_the_compile_cache_disabled(
    compile_all_ranges, monkeypatch
):
    """Peers hand ranges over through vLLM's compile cache, so with the cache
    disabled TPU_PRECOMPILE_PEER_REPLAY=1 is an invalid combination: it warns
    and falls back to every rank compiling every range, without waiting."""
    import vllm_torchtpu

    warnings = []

    class _Logger:
        def warning_once(self, msg, *args, **kwargs):
            warnings.append(msg)

        def __getattr__(self, name):
            return lambda *args, **kwargs: None

    monkeypatch.setattr(vllm_torchtpu, "logger", _Logger())
    runs = [
        _compile(compile_all_ranges, monkeypatch, r, disable_cache=True)
        for r in range(8)
    ]
    for calls in runs:
        assert all(wait == 0.0 for _, wait in calls)
        assert sorted(end for end, _ in calls) == SIZES
    assert warnings and "VLLM_DISABLE_COMPILE_CACHE" in warnings[0]


# --- rank-dependent graphs (rank-salted artifacts: no peer replays them) ---


def test_rank_dependent_graph_compiles_every_range_without_waiting(
    compile_all_ranges, monkeypatch
):
    """A graph calling a rank-dependent op (e.g. the Qwen3.8 megakernel) has no
    owner to wait for: every rank compiles every range at once, then the ranks
    agree on success."""
    monkeypatch.setattr(tpu_compiler, "graph_is_rank_dependent", lambda graph: True)
    agreed = []
    monkeypatch.setattr(
        tpu_compiler, "agree_all_ranks", lambda ok: agreed.append(ok) or ok
    )
    for rank in range(8):
        calls = _compile(compile_all_ranges, monkeypatch, rank)
        assert all(wait == 0.0 for _, wait in calls)
        assert sorted(end for end, _ in calls) == SIZES
    assert agreed == [True] * 8


def test_rank_dependent_failure_on_another_rank_fails_here(
    compile_all_ranges, monkeypatch
):
    monkeypatch.setattr(tpu_compiler, "graph_is_rank_dependent", lambda graph: True)
    monkeypatch.setattr(tpu_compiler, "agree_all_ranks", lambda ok: False)
    with pytest.raises(RuntimeError, match="another rank"):
        _compile(compile_all_ranges, monkeypatch, 3)


def test_rank_dependent_failure_here_still_joins_the_agreement(
    compile_all_ranges, monkeypatch
):
    """The failing rank reaches the agreement (its peers would otherwise block
    in it) and then raises its own error."""
    monkeypatch.setattr(tpu_compiler, "graph_is_rank_dependent", lambda graph: True)
    agreed = []
    monkeypatch.setattr(
        tpu_compiler, "agree_all_ranks", lambda ok: agreed.append(ok) or ok
    )
    with pytest.raises(MemoryError, match="vmem"):
        _compile(compile_all_ranges, monkeypatch, 0, fail_at=512)
    assert agreed == [False]


def test_graph_is_rank_dependent_reads_submodule_code(monkeypatch):
    import torch
    from torch import fx

    monkeypatch.setattr(tpu_compiler, "_RANK_DEPENDENT_OPS", {"qwen38_mk_test"})

    class Inner(torch.nn.Module):
        def forward(self, x):
            return torch.relu(x)

    class Outer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = fx.symbolic_trace(Inner())

        def forward(self, x):
            return self.inner(x) + 1

    graph = fx.symbolic_trace(Outer())
    assert not tpu_compiler.graph_is_rank_dependent(graph)
    tpu_compiler._RANK_DEPENDENT_OPS.add("relu")
    assert tpu_compiler.graph_is_rank_dependent(graph)
    assert not tpu_compiler.graph_is_rank_dependent(object())


def _agree_worker(rank, port, oks, out):
    import torch.distributed as dist

    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=len(oks)
    )
    try:
        out[rank] = tpu_compiler.agree_all_ranks(oks[rank])
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("oks", [[True, True], [True, False], [False, False]])
def test_agree_all_ranks_is_a_min_over_ranks(oks):
    import socket

    import torch.multiprocessing as mp

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    out = mp.Manager().dict()
    mp.spawn(_agree_worker, args=(port, oks, out), nprocs=len(oks), join=True)
    assert [out[r] for r in range(len(oks))] == [all(oks)] * len(oks)


@pytest.mark.parametrize("mode", ["saved", "unserializable", "exception"])
def test_owner_records_progress_for_saved_unserializable_and_failed_buckets(
    compile_all_ranges, monkeypatch, tmp_path, mode
):
    """Owner side of PrecompileProgress: records `start` then `finish` with
    saved buckets when the artifact file exists, `no artifact saved` when
    compile returned without saving one, and the exception repr before
    re-raising when compile fails."""
    _host(monkeypatch, 8)
    monkeypatch.setenv("LOCAL_RANK", "0")
    cache_dir = tmp_path / "hash" / "rank_0_0" / "backbone"
    cache_dir.mkdir(parents=True)

    def compile(graph, args, *rest, compile_range, **kwargs):
        wait = getattr(tpu_compiler._peer_wait, "timeout_s", 0.0)
        if wait == 0.0:
            if mode == "exception":
                raise MemoryError("vmem")
            if mode == "saved":
                start, end = compile_range.start, compile_range.end
                (
                    cache_dir / f"artifact_compile_range_{start}_{end}_subgraph_2"
                ).write_bytes(b"ok")
        return lambda *a: a

    backend = SimpleNamespace(
        graph=object(),
        range_entries={
            Range(s, s): piecewise_backend.RangeEntry(compile_range=Range(s, s))
            for s in SIZES
        },
        vllm_backend=SimpleNamespace(
            compiler_manager=SimpleNamespace(
                compile=compile,
                disable_cache=False,
                compiler=SimpleNamespace(cache_dir=str(cache_dir)),
            ),
            inductor_config={},
            is_encoder=False,
        ),
        compilation_config=None,
        piecewise_compile_index=2,
        total_piecewise_compiles=3,
        _log_compile_start=lambda compile_range: None,
    )
    if mode == "exception":
        with pytest.raises(MemoryError, match="vmem"):
            compile_all_ranges(backend)
    else:
        compile_all_ranges(backend)

    records = tpu_compiler.PrecompileProgress(str(cache_dir), 2).owner_records((0, 0))
    assert [r["event"] for r in records] == ["start", "finish"]
    finish = records[1]
    assert finish["subgraph"] == 2
    assert finish["prefix"] == "backbone"
    if mode == "saved":
        assert finish["saved"] == [4096]
        assert finish["failed"] == {}
    elif mode == "unserializable":
        assert finish["saved"] == []
        assert finish["failed"] == {"4096": "no artifact saved (not serializable)"}
    else:
        assert finish["saved"] == []
        assert finish["failed"] == {"4096": "MemoryError('vmem')"}


@pytest.mark.parametrize(
    (
        "ranks_per_host,local_rank,offset,cache_subdir,"
        "replica_world,dp_size,expected_owners"
    ),
    [
        # DP=1, TP=8 on host 1 (ranks 8..15): slot 2 (rank_10_0) -> (8..15, 0)
        (8, 2, 0, "rank_10_0", 16, 1, [(8 + s, 0) for s in range(8)]),
        # DP=2, TP=4 on 1 host: slot 5 (rank_1_1) -> (0..3, 0) and (0..3, 1)
        (
            8,
            5,
            0,
            "rank_1_1",
            4,
            2,
            [(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1), (3, 1)],
        ),
        # DP=8, TP=1 on 1 host: every rank is rank_0_<dp>; slot 3 -> (0, s)
        (8, 3, 0, "rank_0_3", 1, 8, [(0, s) for s in range(8)]),
        # TPU_LOCAL_RANK_OFFSET=4 with 4 ranks on host 1 (rank_4_0..rank_7_0)
        (4, 5, 4, "rank_5_0", 8, 1, [(4, 0), (5, 0), (6, 0), (7, 0)]),
    ],
)
def test_peer_owner_maps_dp_and_tp_layouts_to_exact_rank_dp_dirs(
    compile_all_ranges,
    monkeypatch,
    tmp_path,
    ranks_per_host,
    local_rank,
    offset,
    cache_subdir,
    replica_world,
    dp_size,
    expected_owners,
):
    _host(monkeypatch, ranks_per_host)
    monkeypatch.setenv("LOCAL_RANK", str(local_rank))
    monkeypatch.setenv("TPU_LOCAL_RANK_OFFSET", str(offset))
    cache_dir = tmp_path / "hash" / cache_subdir / "backbone"
    cache_dir.mkdir(parents=True)

    waited_owners: dict[int, object] = {}

    def compile(graph, args, *rest, compile_range, **kwargs):
        wait = getattr(tpu_compiler._peer_wait, "timeout_s", 0.0)
        if wait > 0.0:
            waited_owners[compile_range.end] = getattr(
                tpu_compiler._peer_wait, "owner", None
            )
        return lambda *a: a

    backend = SimpleNamespace(
        graph=object(),
        range_entries={
            Range(s, s): piecewise_backend.RangeEntry(compile_range=Range(s, s))
            for s in SIZES
        },
        vllm_backend=SimpleNamespace(
            compiler_manager=SimpleNamespace(
                compile=compile,
                disable_cache=False,
                compiler=SimpleNamespace(cache_dir=str(cache_dir)),
            ),
            vllm_config=SimpleNamespace(
                parallel_config=SimpleNamespace(
                    world_size=replica_world,
                    data_parallel_size=dp_size,
                )
            ),
            inductor_config={},
            is_encoder=False,
        ),
        compilation_config=None,
        piecewise_compile_index=0,
        total_piecewise_compiles=1,
        _log_compile_start=lambda compile_range: None,
    )
    compile_all_ranges(backend)

    # Check the snake-dealt owner slot for every non-owned bucket matches
    # expected_owners[owner_slot].
    slot = local_rank - offset
    sorted_sizes = sorted(SIZES, reverse=True)
    for idx, size in enumerate(sorted_sizes):
        turn, pos = divmod(idx, ranks_per_host)
        owner_slot = pos if turn % 2 == 0 else ranks_per_host - 1 - pos
        if owner_slot == slot:
            assert size not in waited_owners
        else:
            assert waited_owners[size] == expected_owners[owner_slot]
