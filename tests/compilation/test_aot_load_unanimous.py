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
"""_patch_vllm_aot_load_unanimous: ranks that share collectives load alike."""

import os
import socket
import time
from datetime import timedelta
from types import CodeType, SimpleNamespace

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker(rank, world, port, case, tmp, out):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    if case.get("force"):
        os.environ["VLLM_FORCE_AOT_LOAD"] = "1"
    # Import before the rendezvous: ranks import at different speeds.
    from vllm.compilation import decorators

    import vllm_torchtpu

    dist.init_process_group(
        "gloo", rank=rank, world_size=world, timeout=timedelta(seconds=60)
    )
    # Only the slow-rank case wants the vote to give up quickly.
    vote_group = (
        dist.new_group(backend="gloo", timeout=timedelta(seconds=10))
        if "slow_rank" in case
        else dist.group.WORLD
    )

    path = os.path.join(tmp, f"rank_{rank}", "model")
    if case["present"][rank]:
        os.makedirs(os.path.dirname(path))
        with open(path, "wb") as f:
            f.write(b"artifact")
    calls = []
    sentinel = object()

    def fake_load(model, aot_path):
        calls.append(aot_path)
        if case.get("slow_rank") == rank:
            time.sleep(30)
        return sentinel if case["loads"][rank] else None

    decorators._try_load_aot_compiled_fn = fake_load
    vllm_torchtpu._aot_load_vote_group = lambda model: SimpleNamespace(
        world_size=world, rank_in_group=rank, cpu_group=vote_group
    )
    vllm_torchtpu._patch_vllm_aot_load_unanimous()
    try:
        got = decorators._try_load_aot_compiled_fn(None, path)
        out[rank] = ("loaded" if got is sentinel else "compile", len(calls))
    except Exception as e:
        out[rank] = (f"{type(e).__name__}: {e}", len(calls))


def _run(case, tmp_path):
    with mp.Manager() as manager:
        out = manager.dict()
        ctx = mp.start_processes(
            _worker,
            args=(2, _free_port(), case, str(tmp_path), out),
            nprocs=2,
            join=False,
            start_method="spawn",
        )
        while not ctx.join(timeout=60):
            pass
        return [out[r] for r in range(2)]


@pytest.mark.parametrize(
    "present, loads, expected",
    [
        # Both have it and load it.
        ((True, True), (True, True), [("loaded", 1), ("loaded", 1)]),
        # A missing file on one rank: nobody loads anything.
        ((True, False), (True, False), [("compile", 0), ("compile", 0)]),
        ((False, True), (False, True), [("compile", 0), ("compile", 0)]),
        ((False, False), (False, False), [("compile", 0), ("compile", 0)]),
        # Present everywhere but unloadable on one rank: both compile.
        ((True, True), (True, False), [("compile", 1), ("compile", 1)]),
    ],
)
def test_ranks_take_the_same_load_path(present, loads, expected, tmp_path):
    case = {"present": present, "loads": loads}
    assert _run(case, tmp_path) == expected


def test_forced_load_raises_on_every_rank_when_one_has_none(tmp_path):
    case = {"present": (True, False), "loads": (True, False), "force": True}
    outcomes = _run(case, tmp_path)
    assert all("VLLM_FORCE_AOT_LOAD is set but ranks [1]" in o for o, _ in outcomes)


def test_forced_load_raises_on_every_rank_when_one_fails_to_load(tmp_path):
    case = {"present": (True, True), "loads": (True, False), "force": True}
    outcomes = _run(case, tmp_path)
    assert all("failed to load on ranks [1]" in o for o, _ in outcomes)


def test_a_vote_from_another_step_is_not_read_as_an_answer(monkeypatch):
    import torch

    import vllm_torchtpu

    def all_gather(reports, mine, group):
        reports[0].copy_(mine)
        reports[1].copy_(torch.tensor([12345, 1]))

    monkeypatch.setattr(dist, "all_gather", all_gather)
    group = SimpleNamespace(world_size=2, rank_in_group=0, cpu_group=None)
    with pytest.raises(RuntimeError, match="reported from different steps"):
        vllm_torchtpu._aot_load_ranks_missing(True, group, "p", "find")


def test_a_rank_that_never_reports_fails_with_the_setting_to_change(tmp_path):
    case = {"present": (True, True), "loads": (True, True), "slow_rank": 1}
    outcome, _ = _run(case, tmp_path)[0]
    assert "AOT artifact load vote: rank 0 stopped waiting" in outcome
    assert "It was voting to load" in outcome
    assert "--cpu-distributed-timeout-seconds" in outcome


def test_single_process_passes_through(monkeypatch):
    from vllm.compilation import decorators

    import vllm_torchtpu

    sentinel = object()
    monkeypatch.setattr(
        decorators, "_try_load_aot_compiled_fn", lambda model, path: sentinel
    )
    monkeypatch.setattr(vllm_torchtpu, "_aot_load_vote_group", lambda model: None)
    vllm_torchtpu._patch_vllm_aot_load_unanimous()
    assert decorators._try_load_aot_compiled_fn(None, "x") is sentinel


@pytest.mark.parametrize(
    "is_encoder, encoder_mode, pp_size, expected",
    [
        (True, "weights", 1, "tp"),
        (True, "data", 1, None),
        (False, "weights", 2, None),
        (False, "weights", 1, "world"),
    ],
)
def test_vote_group_follows_the_collectives(
    monkeypatch, is_encoder, encoder_mode, pp_size, expected
):
    from vllm.distributed import parallel_state

    import vllm_torchtpu

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(parallel_state, "get_tp_group", lambda: "tp")
    monkeypatch.setattr(parallel_state, "get_world_group", lambda: "world")
    model = SimpleNamespace(
        _is_encoder=is_encoder,
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(pipeline_parallel_size=pp_size),
            model_config=SimpleNamespace(
                multimodal_config=SimpleNamespace(mm_encoder_tp_mode=encoder_mode)
            ),
        ),
    )
    assert vllm_torchtpu._aot_load_vote_group(model) == expected


def _code_objects(code: CodeType):
    yield code
    for const in code.co_consts:
        if isinstance(const, CodeType):
            yield from _code_objects(const)


def test_vllm_looks_the_loader_up_at_call_time():
    """The patch replaces a module attribute, which only takes effect if vLLM's
    compiled-model __call__ reads that attribute when it runs."""
    from vllm.compilation import decorators

    callers = [
        code
        for code in _code_objects(decorators._support_torch_compile.__code__)
        if "_try_load_aot_compiled_fn" in code.co_names
    ]
    assert callers, "vLLM no longer calls _try_load_aot_compiled_fn by global name"
