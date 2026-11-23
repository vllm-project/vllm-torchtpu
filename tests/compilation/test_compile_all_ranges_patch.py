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

import itertools
import os
import time
from types import SimpleNamespace

import pytest
from vllm.compilation import piecewise_backend
from vllm.config.utils import Range
from vllm.distributed import parallel_state

from vllm_torchtpu import _patch_vllm_compile_all_ranges
from vllm_torchtpu.compilation import peer_replay, shape_variants, tpu_compiler

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
    monkeypatch.setattr(
        shape_variants, "_deferred_compiler", shape_variants._deferred_compiler
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
):
    """(range end, peer wait in seconds) per compile, in order."""
    _host(monkeypatch, ranks_per_host)
    monkeypatch.setenv("LOCAL_RANK", str(local_rank))
    calls = []

    def compile(graph, args, *rest, compile_range, **kwargs):
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
    monkeypatch.delenv("TPU_PRECOMPILE_PEER_WAIT_S", raising=False)
    calls = _compile(compile_all_ranges, monkeypatch, 0)
    assert calls == [(4096, 0.0)] + [
        (end, 600.0) for end in [32, 64, 128, 256, 512, 1024, 2048, 16]
    ]


@pytest.mark.parametrize("value, wait", [("1800", 1800.0), ("0", 0.0)])
def test_peer_wait_is_read_from_the_env(compile_all_ranges, monkeypatch, value, wait):
    monkeypatch.setenv("TPU_PRECOMPILE_PEER_WAIT_S", value)
    calls = _compile(compile_all_ranges, monkeypatch, 0)
    assert {w for _, w in calls[1:]} == {wait}


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


def _compile_deferred(
    monkeypatch,
    local_rank,
    groups=([16, 32, 64], [128, 256, 512], [1024, 2048, 4096]),
    ranks_per_host=8,
    disable_cache=False,
    num_subgraphs=1,
):
    """Run compile_deferred over several traces' accepted buckets."""
    _host(monkeypatch, ranks_per_host)
    monkeypatch.setenv("LOCAL_RANK", str(local_rank))
    calls = []

    def compile(graph, args, *rest, compile_range, graph_index=0, **kwargs):
        calls.append(
            (
                compile_range.end,
                getattr(tpu_compiler._peer_wait, "timeout_s", 0.0),
                graph_index,
            )
        )
        return lambda *a: a

    backends = []
    for trace_idx, group in enumerate(groups):
        for subgraph_idx in range(num_subgraphs):
            entries = {}
            for s in SIZES:
                entry = piecewise_backend.RangeEntry(compile_range=Range(s, s))
                entry._tpu_deferred = "compile" if s in group else "hand_over"
                entries[Range(s, s)] = entry
            backends.append(
                SimpleNamespace(
                    graph=object(),
                    range_entries=entries,
                    _tpu_trace=trace_idx,
                    vllm_backend=SimpleNamespace(
                        compiler_manager=SimpleNamespace(
                            compile=compile, disable_cache=disable_cache
                        ),
                        inductor_config={},
                        is_encoder=False,
                    ),
                    compilation_config=None,
                    piecewise_compile_index=subgraph_idx,
                    total_piecewise_compiles=num_subgraphs,
                    _log_compile_start=lambda compile_range: None,
                )
            )
    assert shape_variants._deferred_compiler is not None
    shape_variants._deferred_compiler(backends)
    return [(end, wait) for end, wait, _ in calls], calls, backends


def test_deferred_splits_buckets_across_traces_with_host_slot(
    compile_all_ranges, monkeypatch
):
    """compile_deferred deals owners across all traces' buckets at once, taking
    TPU_LOCAL_RANK_OFFSET off LOCAL_RANK, and hands refused buckets only to the
    live trace."""
    monkeypatch.setenv("TPU_LOCAL_RANK_OFFSET", "4")
    owned = []
    for r in range(4, 8):
        calls, _, backends = _compile_deferred(monkeypatch, r, ranks_per_host=4)
        # 9 accepted buckets + 6 live-trace hand-overs (earlier traces get stubs).
        assert len(calls) == 9 + 6
        owned.append(_owned(calls[:9]))
        with pytest.raises(shape_variants.ShapeSpecializationError):
            backends[0].range_entries[Range(4096, 4096)].runnable()
    assert owned == [[4096, 32, 16], [2048, 64], [1024, 128], [512, 256]]


def test_deferred_splits_subgraphs_across_ranks_and_counts_distinct_traces(
    compile_all_ranges, monkeypatch
):
    """With splitting_ops (multiple PiecewiseBackends per trace), every
    subgraph's accepted buckets are dealt across the host's ranks, and the
    summary log counts distinct traces rather than backends."""
    import vllm_torchtpu

    logged = []

    class _Logger:
        def info(self, msg, *args):
            logged.append(msg % args if args else msg)

        def __getattr__(self, name):
            return lambda *args, **kwargs: None

    monkeypatch.setattr(vllm_torchtpu, "logger", _Logger())
    owned_pairs = []
    for r in range(8):
        logged.clear()
        _, full_calls, backends = _compile_deferred(
            monkeypatch, r, ranks_per_host=8, num_subgraphs=2
        )
        # 18 accepted (subgraph, bucket) compiles + 12 live-trace hand-overs.
        assert len(full_calls) == 18 + 12
        owned_pairs.extend(
            (sub, end) for end, wait, sub in full_calls[:18] if wait == 0.0
        )
        assert any(
            "Deferred precompile: 3 traces, 18 buckets to compile, 36 to hand over"
            in line
            for line in logged
        )
        for sub in (0, 1):
            with pytest.raises(shape_variants.ShapeSpecializationError):
                backends[sub].range_entries[Range(4096, 4096)].runnable()
    assert sorted(owned_pairs) == sorted((sub, s) for sub in (0, 1) for s in SIZES)


def test_deferred_falls_back_when_cache_disabled_or_host_unknown(
    compile_all_ranges, monkeypatch
):
    calls, _, _ = _compile_deferred(monkeypatch, 1, ranks_per_host=None)
    assert all(wait == 0.0 for _, wait in calls[:9])
    assert sorted(end for end, _ in calls[:9]) == SIZES

    calls_nocache, _, _ = _compile_deferred(monkeypatch, 1, disable_cache=True)
    assert all(wait == 0.0 for _, wait in calls_nocache[:9])
    assert sorted(end for end, _ in calls_nocache[:9]) == SIZES


def test_peer_replay_claims_list_schedules_and_respects_claim_wait(
    compile_all_ranges, monkeypatch, tmp_path
):
    """peer_replay_scheduler='claims' claims unclaimed ranges via O_EXCL files
    (recording rank_<r>_<dp> dir names and subgraph index) and replays
    peer-claimed ranges with bounded wait."""
    from vllm_torchtpu import profiler_trace

    monkeypatch.setenv("TPU_PRECOMPILE_BIGGEST_FIRST", "1")
    monkeypatch.setenv("TPU_PRECOMPILE_PEER_WAIT_S", "42")
    monkeypatch.setattr(tpu_compiler, "_graph_fingerprint", lambda graph: "fp" * 16)
    cache_dir = str(tmp_path / "hash1" / "rank_0_1" / "backbone")
    vllm_config = SimpleNamespace(additional_config={"peer_replay_scheduler": "claims"})

    def _run_claim_rank(pre_claim: dict[int, str] | None = None):
        _host(monkeypatch, 8)
        monkeypatch.setenv("LOCAL_RANK", "0")
        monkeypatch.setattr(peer_replay, "_claim_call_counter", itertools.count(1))
        calls = []

        def compile(graph, args, *rest, compile_range, **kwargs):
            calls.append(
                (
                    compile_range.end,
                    getattr(tpu_compiler._peer_wait, "timeout_s", 0.0),
                    getattr(tpu_compiler._peer_wait, "allowed_dirs", None),
                )
            )
            return lambda *a: a

        backend = SimpleNamespace(
            graph=object(),
            vllm_config=vllm_config,
            range_entries={
                Range(s, s): piecewise_backend.RangeEntry(compile_range=Range(s, s))
                for s in [16, 64, 256]
            },
            vllm_backend=SimpleNamespace(
                compiler_manager=SimpleNamespace(
                    compile=compile,
                    disable_cache=False,
                    compiler=SimpleNamespace(cache_dir=cache_dir),
                ),
                inductor_config={},
                is_encoder=False,
            ),
            compilation_config=None,
            piecewise_compile_index=0,
            total_piecewise_compiles=1,
            _log_compile_start=lambda compile_range: None,
        )
        if pre_claim:
            cdir = (
                tmp_path
                / "hash1"
                / "peer_claims"
                / profiler_trace.profile_session_id(include_starttime=True)
                / "backbone"
                / ("fp" * 12)
                / "1"
            )
            cdir.mkdir(parents=True, exist_ok=True)
            for size, owner in pre_claim.items():
                (cdir / f"{size}_{size}_sg0").write_text(owner)
        compile_all_ranges(backend)
        return calls

    calls = _run_claim_rank(pre_claim={256: "rank_0_0", 16: "rank_0_1"})
    # 64 is unclaimed -> rank_0_1 claims and compiles it (wait=0, allowed_dirs=None).
    # 16 was claimed by rank_0_1 -> waits 0 with allowed_dirs={"rank_0_1"}.
    # 256 was claimed by rank_0_0 -> waits up to 42s with allowed_dirs={"rank_0_0"}.
    assert calls == [
        (64, 0.0, None),
        (16, 0.0, {"rank_0_1"}),
        (256, 42.0, {"rank_0_0"}),
    ]


def test_peer_replay_claims_falls_back_to_rotated_without_fingerprint(
    compile_all_ranges, monkeypatch, tmp_path
):
    """When a graph cannot be fingerprinted, claims falls back to the rotated
    order (wait=0) instead of the static deal."""
    monkeypatch.setattr(tpu_compiler, "_graph_fingerprint", lambda graph: None)
    _host(monkeypatch, 8)
    monkeypatch.setenv("LOCAL_RANK", "1")
    calls = []
    cache_dir = str(tmp_path / "hash1" / "rank_0_0" / "backbone")
    vllm_config = SimpleNamespace(additional_config={"peer_replay_scheduler": "claims"})
    backend = SimpleNamespace(
        graph=object(),
        vllm_config=vllm_config,
        range_entries={
            Range(s, s): piecewise_backend.RangeEntry(compile_range=Range(s, s))
            for s in SIZES
        },
        vllm_backend=SimpleNamespace(
            compiler_manager=SimpleNamespace(
                compile=lambda g, a, *r, compile_range, **kw: (
                    calls.append(
                        (
                            compile_range.end,
                            getattr(tpu_compiler._peer_wait, "timeout_s", 0.0),
                        )
                    )
                    or (lambda *x: x)
                ),
                disable_cache=False,
                compiler=SimpleNamespace(cache_dir=cache_dir),
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
    assert all(wait == 0.0 for _, wait in calls)
    assert sorted(end for end, _ in calls) == SIZES


def test_peer_replay_claims_falls_back_on_oserror(
    compile_all_ranges, monkeypatch, tmp_path
):
    """Read-only / unwritable claim directories fall back to rotated order."""
    monkeypatch.setattr(tpu_compiler, "_graph_fingerprint", lambda graph: "fp" * 16)
    monkeypatch.setattr(
        os,
        "makedirs",
        lambda *a, **kw: (_ for _ in ()).throw(PermissionError("read-only")),
    )
    _host(monkeypatch, 8)
    monkeypatch.setenv("LOCAL_RANK", "0")
    calls = []
    cache_dir = str(tmp_path / "hash1" / "rank_0_0" / "backbone")
    vllm_config = SimpleNamespace(additional_config={"peer_replay_scheduler": "claims"})
    backend = SimpleNamespace(
        graph=object(),
        vllm_config=vllm_config,
        range_entries={
            Range(s, s): piecewise_backend.RangeEntry(compile_range=Range(s, s))
            for s in SIZES
        },
        vllm_backend=SimpleNamespace(
            compiler_manager=SimpleNamespace(
                compile=lambda g, a, *r, compile_range, **kw: (
                    calls.append(
                        (
                            compile_range.end,
                            getattr(tpu_compiler._peer_wait, "timeout_s", 0.0),
                        )
                    )
                    or (lambda *x: x)
                ),
                disable_cache=False,
                compiler=SimpleNamespace(cache_dir=cache_dir),
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
    assert all(wait == 0.0 for _, wait in calls)
    assert sorted(end for end, _ in calls) == SIZES


def test_peer_replay_claims_real_compilers_two_ranks_and_deferred_subgraphs(
    compile_all_ranges, monkeypatch, tmp_path
):
    """End-to-end test with real TpuCompilerAdaptor instances:
    1) Two ranks (including across DP engines sharing TORCH_TPU_DP_MASTER_PORT)
       compile each range once, and a waiter waits for the claim owner even
       when another rank already saved an artifact for the same key.
    2) compile_deferred with two subgraphs sharing a range and fingerprint
       claims and compiles both without colliding on one claim file.
    """
    import threading

    import torch

    _host(monkeypatch, 2)
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("TPU_PRECOMPILE_BIGGEST_FIRST", "1")
    monkeypatch.setenv("TPU_PRECOMPILE_PEER_WAIT_S", "5")
    monkeypatch.setenv("TORCH_TPU_DP_MASTER_PORT", "29511")
    monkeypatch.setattr(tpu_compiler, "_PEER_POLL_S", 0.01)

    replayed_calls = []
    gate = threading.Event()
    compiled_by_rank: list[tuple[str, int]] = []

    def _make_adaptor(rank_dir: str):
        adaptor = tpu_compiler.TpuCompilerAdaptor()
        adaptor.initialize_cache(
            cache_dir=str(tmp_path / "hash1" / rank_dir / "backbone")
        )
        return adaptor

    def _make_backend(adaptor, graph, sg_idx: int, ranges: list[int]):
        def cm_compile(g, args, *rest, compile_range, key=None, **kwargs):
            k = key or f"range_{compile_range.start}_{compile_range.end}_sg{sg_idx}"
            fn, _ = adaptor.compile(g, args, {}, compile_range, key=k)
            return fn

        return SimpleNamespace(
            graph=graph,
            _tpu_trace=0,
            vllm_config=SimpleNamespace(
                additional_config={"peer_replay_scheduler": "claims"}
            ),
            range_entries={
                Range(s, s): piecewise_backend.RangeEntry(compile_range=Range(s, s))
                for s in ranges
            },
            vllm_backend=SimpleNamespace(
                compiler_manager=SimpleNamespace(
                    compile=cm_compile,
                    disable_cache=False,
                    compiler=adaptor,
                ),
                inductor_config={},
                is_encoder=False,
            ),
            compilation_config=None,
            piecewise_compile_index=sg_idx,
            total_piecewise_compiles=2,
            _log_compile_start=lambda compile_range: None,
        )

    def tracked_backend(graph, inputs):
        # Identify caller via thread name or call order.
        name = threading.current_thread().name
        if name == "owner-r0":
            gate.wait(timeout=2.0)
        size = (
            int(inputs[0].shape[0]) if hasattr(inputs[0], "shape") else int(inputs[0])
        )
        compiled_by_rank.append((name, size))

        class Entry:
            def __call__(self, *a):
                return f"compiled-{name}"

            def serialize(self):
                return f"entry-{name}"

        return Entry()

    monkeypatch.setattr(tpu_compiler, "_tpu_backend", tracked_backend)
    monkeypatch.setattr(
        tpu_compiler,
        "_deserialize_entry",
        lambda entry: replayed_calls.append(entry) or (lambda *a: entry),
    )

    graph0 = torch.fx.symbolic_trace(lambda x: x + 1)
    graph1 = torch.fx.symbolic_trace(lambda x: x + 1)
    other_graph = torch.fx.symbolic_trace(lambda x: x * 9)

    # Pre-populate rank_2_0 with an artifact for the same key from another graph:
    # waiter rank_0_1 must still wait for claim owner rank_0_0, not fail on rank_2_0.
    r2 = _make_adaptor("rank_2_0")
    r2.compile(
        other_graph, [torch.zeros(64, 4)], {}, Range(64, 64), key="range_64_64_sg0"
    )

    r0 = _make_adaptor("rank_0_0")
    r1 = _make_adaptor("rank_0_1")
    b0 = _make_backend(r0, graph0, 0, [64])
    b1 = _make_backend(r1, graph0, 0, [64])

    # Pre-seed args on range_entries so compile_entry can pass example_inputs.
    b0.range_entries[Range(64, 64)].args = [torch.zeros(64, 4)]
    b1.range_entries[Range(64, 64)].args = [torch.zeros(64, 4)]

    def _compile_entry(b, e):
        re_fn = b.vllm_backend.compiler_manager.compile(
            b.graph, e.args, compile_range=e.compile_range
        )
        e.runnable = re_fn
        e.compiled = True

    def run_r0():
        monkeypatch.setattr(peer_replay, "_claim_call_counter", itertools.count(1))
        peer_replay.compile_claimed(
            [(b0, b0.range_entries[Range(64, 64)])], _compile_entry
        )

    def run_r1():
        monkeypatch.setattr(peer_replay, "_claim_call_counter", itertools.count(1))
        peer_replay.compile_claimed(
            [(b1, b1.range_entries[Range(64, 64)])], _compile_entry
        )

    t0 = threading.Thread(target=run_r0, name="owner-r0")
    t1 = threading.Thread(target=run_r1, name="waiter-r1")
    t0.start()
    time.sleep(0.05)  # Let r0 claim 64 before r1 runs; r0 blocks on gate.
    t1.start()
    time.sleep(0.05)  # r1 is waiting on r0 even though rank_2_0 has an artifact.
    gate.set()
    t0.join(timeout=5.0)
    t1.join(timeout=5.0)

    # Range 64 was compiled only once (by owner-r0) and replayed by waiter-r1.
    assert [c for c in compiled_by_rank if c[0] != "MainThread"] == [("owner-r0", 64)]
    assert "entry-owner-r0" in replayed_calls

    # Part 2: compile_deferred with two subgraphs sharing a range and fingerprint.
    compiled_by_rank.clear()
    monkeypatch.setattr(peer_replay, "_claim_call_counter", itertools.count(1))
    b_sg0 = _make_backend(r0, graph0, 0, [32])
    b_sg1 = _make_backend(r0, graph1, 1, [32])
    e0 = b_sg0.range_entries[Range(32, 32)]
    e1 = b_sg1.range_entries[Range(32, 32)]
    e0.args = [torch.zeros(32, 4)]
    e1.args = [torch.zeros(32, 4)]
    e0._tpu_deferred = "compile"
    e1._tpu_deferred = "compile"
    assert shape_variants._deferred_compiler is not None
    shape_variants._deferred_compiler([b_sg0, b_sg1])
    assert len(compiled_by_rank) == 2
    assert e0.compiled
    assert e1.compiled
