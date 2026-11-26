# SPDX-License-Identifier: Apache-2.0
"""16 ranks run the two-steps-ahead DP agreement over real Gloo collectives."""

import importlib.util
import random
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.request import RequestStatus

# Loaded by path: the spawned ranks need neither TorchTPU nor the platform.
_PATH = Path(__file__).resolve().parents[2] / "src/vllm_torchtpu/distributed/dp_step.py"
_SPEC = importlib.util.spec_from_file_location("dp_step", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
dp_step = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(dp_step)

RANKS = 16
STEPS = 160
DRAFTS = 2  # A decoding request runs 1 + at most DRAFTS tokens a step.
BUDGET = 1024
CHUNK_REQS = 4
LONG_TOKENS = 16  # A chunk holding a longer request takes CHUNK_REQS // 2.
WINDOW = 32  # Draft context window.
MAX_REQS = 64


class _Request:
    """The scheduler's view, with vLLM's next-schedule token formula."""

    def __init__(self, request_id: str, prompt: int, computed: int, image: bool):
        self.request_id = request_id
        self.prompt = self.num_prompt_tokens = prompt
        self.num_computed_tokens = computed
        self.num_output_placeholders = 0
        self.image = image
        self.status = RequestStatus.WAITING
        self.drafts = DRAFTS  # The drafter's proposals for its next step.

    @property
    def num_tokens_with_spec(self) -> int:
        if self.num_computed_tokens < self.prompt:
            return self.prompt
        return self.num_computed_tokens + 1 + self.drafts


class _Scheduler:
    def __init__(self) -> None:
        self._pause_state = PauseState.UNPAUSED
        self.num_spec_tokens = DRAFTS
        self.max_num_active_reqs = MAX_REQS
        self.num_waiting_for_streaming_input = 0
        self.finished_recving_kv_req_ids: set[str] = set()
        self.kv_holding_waiting: list[_Request] = []
        self.waiting: list[_Request] = []
        self.running: list[_Request] = []
        self.finished: set[str] = set()


def _schedule(scheduler: _Scheduler) -> SimpleNamespace:
    """A stand-in for vLLM's schedule: running first, then waiting."""
    output = SimpleNamespace(
        new=[], scheduled={}, computed={}, finished=scheduler.finished
    )
    scheduler.finished = set()
    if scheduler._pause_state == PauseState.PAUSED_ALL:
        return output
    budget = BUDGET

    def run(request: _Request) -> None:
        nonlocal budget
        count = min(request.num_tokens_with_spec - request.num_computed_tokens, budget)
        output.computed[request.request_id] = request.num_computed_tokens
        output.scheduled[request.request_id] = count
        request.num_computed_tokens += count
        budget -= count

    for request in scheduler.running:
        run(request)
    while scheduler._pause_state == PauseState.UNPAUSED and scheduler.waiting:
        request = scheduler.waiting.pop(0)
        request.status = RequestStatus.RUNNING
        scheduler.running.append(request)
        output.new.append(request)
        run(request)
    return output


def _count_chunks(tokens: list[int]) -> tuple[int, int]:
    """The runner's chunk rule in miniature: longer requests shrink a chunk."""
    start = chunks = largest = 0
    while start < len(tokens):
        long = any(count > LONG_TOKENS for count in tokens[start:])
        size = min(CHUNK_REQS // 2 if long else CHUNK_REQS, len(tokens) - start)
        largest = max(largest, size)
        start += size
        chunks += 1
    return chunks, largest


class _Runner:
    """One rank's runner state: cached requests and its last bound."""

    def __init__(self) -> None:
        self.requests: dict[str, list] = {}  # id -> [computed, prompt, image]
        self.bound = None

    def step(self, output: SimpleNamespace) -> "dp_step.StepShape":
        for request_id in output.finished:
            self.requests.pop(request_id, None)
        for request in output.new:
            self.requests[request.request_id] = [0, request.prompt, request.image]
        for request_id, computed in output.computed.items():
            self.requests[request_id][0] = computed
        scheduled = output.scheduled
        ends = {
            request_id: self.requests[request_id][0] + count
            for request_id, count in scheduled.items()
        }
        self.bound = dp_step.next_step_bound(
            getattr(output, dp_step.NEXT_STEP_TOKENS), BUDGET, _count_chunks
        )
        if not scheduled:
            return dp_step.StepShape.local(0, 0, 0)
        tokens = list(scheduled.values())
        chunks, reqs = _count_chunks(tokens)
        prompts = {r: self.requests[r][1] for r in scheduled}
        return dp_step.StepShape.local(
            sum(tokens),
            chunks,
            reqs,
            encoder=any(
                self.requests[r][2] and self.requests[r][0] < prompts[r]
                for r in scheduled
            ),
            proposals=any(ends[r] > prompts[r] - 1 for r in scheduled),
            context=any(ends[r] > prompts[r] - WINDOW for r in scheduled),
        )


def _worker(rank: int, rendezvous: str) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=RANKS,
        timeout=timedelta(seconds=60),
    )
    try:
        _run(rank)
    finally:
        dist.destroy_process_group()


def _run(rank: int) -> None:
    rng = random.Random(rank)
    scheduler, runner = _Scheduler(), _Runner()
    agreement = dp_step.StepAgreement(dist.group.WORLD)
    log, published, arrivals, admitted = [], [], {}, {}
    for step in range(STEPS):
        # Events land in bursts, so quiet stretches must settle to steady.
        burst = step % 10 < 3
        if burst and rng.random() < 0.3:
            prompt = rng.randrange(8, 200)
            loaded = rng.random() < 0.5  # Its KV arrived from a prefill server.
            request = _Request(
                f"{rank}-{step}", prompt, prompt if loaded else 0, rng.random() < 0.3
            )
            scheduler.waiting.append(request)
            bounded = getattr(scheduler, "_tpu_dp_bounded", (False, False))
            arrivals[request.request_id] = (step, bounded)
        decoding = [r for r in scheduler.running if r.num_computed_tokens >= r.prompt]
        if burst and decoding and rng.random() < 0.1:
            # A failed KV load rolls a running request back to a prefill.
            victim = rng.choice(decoding)
            victim.num_computed_tokens = max(0, victim.prompt - 40)
        if scheduler.running and rng.random() < 0.15:
            # Completion is not an event: the row becomes padding.
            done = scheduler.running.pop(rng.randrange(len(scheduler.running)))
            scheduler.finished.add(done.request_id)

        for request in scheduler.running:
            # Draft counts vary step to step, so a bound from two schedules
            # back must cover any count, not the one current then.
            request.drafts = rng.randrange(DRAFTS + 1)
        output = dp_step.gated_schedule(scheduler, _schedule)
        for request in output.new:
            admitted[request.request_id] = step
        local = runner.step(output)
        agreed = agreement.begin(local)
        relayout = agreement.relayout
        # (b) This rank's work fits the step every rank runs.
        assert local.fits(agreed), (step, local, agreed)
        log.append((agreement.synchronous, tuple(agreed.encode()), relayout))
        fields = (rng.randrange(1 << 20), -rng.randrange(1 << 20), -rng.randrange(8))
        published.append(fields)
        agreement.end(runner.bound, fields)

    every = [None] * RANKS
    dist.all_gather_object(
        every, (log, published, len(arrivals)), group=dist.group.WORLD
    )
    # (a) Every rank ran the same collectives at the same shapes.
    assert all(other == log for other, _, _ in every)
    for step, (synchronous, _, relayout) in enumerate(log):
        # Quiet steps launch without a collective before them: the last
        # burst step's arrival enters by step % 10 == 4 and its prompt chunk
        # runs there, so from step % 10 == 7 on every rank's bound holds.
        assert step % 10 < 7 or not synchronous, step
        # The re-layout fields ride the closing all-reduce two steps back.
        expected = (
            (0, 0, 0)
            if step < 2
            else tuple(
                max(fields[step - 2][i] for _, fields, _ in every) for i in range(3)
            )
        )
        assert relayout == expected, step
    # (c) A request enters on the first step that neither of the two schedules
    # before it bounded: its arrival step, one later if only the older one
    # bounded, two later if the newer one did (its arrival schedule publishes
    # no bound); and that step agrees its shape before launch.
    assert len(admitted) == len(arrivals)
    assert sum(count for _, _, count in every) > 5 * RANKS
    for request_id, (step, (older, newer)) in arrivals.items():
        delay = 2 if newer else 1 if older else 0
        assert admitted[request_id] == step + delay, request_id
        assert log[admitted[request_id]][0], request_id


def test_sixteen_ranks_agree_each_step_two_steps_ahead(tmp_path):
    mp.spawn(_worker, args=((tmp_path / "gloo").as_uri(),), nprocs=RANKS)
