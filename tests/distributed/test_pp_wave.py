# SPDX-License-Identifier: Apache-2.0
"""The pipeline wave's launch schedule, driven on CPU with a fake ring.

Every stage runs its own event stream in a thread. The fake ring behaves like
the runtime: a launch completes only when every stage has issued the same
launch number, and it hands stage r-1's payload of that launch to stage r. A
schedule whose launches do not pair therefore deadlocks and fails the join
timeout instead of passing by accident.
"""

import random
import threading
from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu.distributed import pp_wave

HIDDEN = 8
MAX_ROWS = 64
BUCKETS = (8, 16, 32, 64)
SIDE = "topk_indices"
SIDE_WIDTH = 4


class _Ring:
    def __init__(self, size):
        self.size = size
        self.count = [0] * size
        self.slots = {}
        self.cv = threading.Condition()

    def launch(self, rank, carried):
        with self.cv:
            pos = self.count[rank]
            self.slots[(rank, pos)] = tuple(x.clone() for x in carried)
            self.count[rank] += 1
            self.cv.notify_all()
            if not self.cv.wait_for(
                lambda: all(c > pos for c in self.count), timeout=20
            ):
                raise TimeoutError(f"rank {rank} launch {pos} never paired")
            # closed cycle: the first stage receives the last stage's block
            return tuple(x.clone() for x in self.slots[((rank - 1) % self.size, pos)])


def _build_stages(monkeypatch, template, size=8, async_handoff=False):
    """``size`` PPWave instances on CPU sharing one fake ring."""
    ring = _Ring(size)
    current = {}
    monkeypatch.setattr(pp_wave, "stage_chips", lambda: 2 if async_handoff else 1)
    monkeypatch.setattr(pp_wave, "_drain_slice", lambda fence=None: None)
    monkeypatch.setattr(pp_wave, "_fence", lambda: object())
    monkeypatch.setattr(pp_wave, "_flush", lambda tensors: None)
    monkeypatch.setattr(pp_wave, "_wait", lambda tensors: None)
    monkeypatch.setattr(pp_wave, "get_or_create_pp_mesh", lambda: None)
    monkeypatch.setattr(
        pp_wave,
        "pp_permute_op",
        lambda mesh, num_tensors=1: (
            lambda *xs, rank=current["rank"]: ring.launch(rank, xs)
        ),
    )
    monkeypatch.setattr(pp_wave, "_launch_now", lambda op, carried: op(*carried))
    from vllm.distributed import parallel_state

    monkeypatch.setattr(
        parallel_state,
        "get_pp_group",
        lambda: SimpleNamespace(rank_in_group=current["rank"], world_size=size),
    )
    waves = []
    for rank in range(size):
        current["rank"] = rank
        waves.append(pp_wave.PPWave(torch.device("cpu"), MAX_ROWS, template))
    return ring, waves


def _template(side: bool):
    template = {
        "hidden_states": ((HIDDEN,), torch.float32),
        "residual": ((HIDDEN,), torch.float32),
    }
    if side:
        template[SIDE] = ((SIDE_WIDTH,), torch.int32)
    return template


@pytest.fixture(params=[False, True], ids=["activations", "with_side_tensor"])
def stages(request, monkeypatch):
    """Eight stages, with and without a side tensor on each hand-off."""
    ring, waves = _build_stages(monkeypatch, _template(request.param))
    return ring, waves


@pytest.fixture
def async_stages(monkeypatch):
    """Eight stages whose first stage launches from its helper thread."""
    ring, waves = _build_stages(monkeypatch, _template(True), async_handoff=True)
    assert waves[0].async_handoff and not any(w.async_handoff for w in waves[1:])
    return ring, waves


SETTLE = "settle"


def _run_stage(wave, events, results, errors):
    """One stage's event stream: a forward adds 1 to the hidden states it
    received and passes the residual on; the first stage seeds each forward
    with its number as hidden states and a residual of 0.5, which every
    later stage must see as part of the sum. A side tensor is seeded by the
    first stage and relayed unchanged, the way a stage that cannot recompute
    it would. A push is the engine's hand-off that carries nothing; a settle
    closes the burst."""
    side = SIDE in wave.carried
    try:
        wave.warmup()
        g = 0
        for rows in events:
            if rows is None:
                wave.push()
                continue
            if rows == SETTLE:
                wave.settle()
                continue
            received = wave.start_forward(rows)
            if wave.rank == 0:
                h = torch.full((rows, HIDDEN), float(g))
                r = torch.full((rows, HIDDEN), 0.5)
                extra = torch.full((rows, SIDE_WIDTH), g, dtype=torch.int32)
            else:
                assert torch.equal(received["residual"], torch.zeros(rows, HIDDEN))
                h = received["hidden_states"] + 1
                r = received["residual"]
                extra = received[SIDE] if side else None
                assert h.shape == (rows, HIDDEN)
            if wave.rank == wave.last:
                results.append((g, rows, h, r, extra))
                wave.end_forward(None, rows)
            else:
                outgoing = {"hidden_states": h, "residual": r}
                if side:
                    outgoing[SIDE] = extra
                wave.end_forward(outgoing, rows)
            g += 1
    except BaseException as exc:  # reported by the test thread
        errors.append((wave.rank, exc))


def _events(rng, forwards, stages, bursts=3):
    """Forwards with pushes where the engine would send them: after a
    forward, with some probability the engine waits for a step that still
    needs pushes. Each burst ends with the pushes that carry its last
    forward and a settle."""
    events = []
    for _ in range(bursts):
        dispatched = 0
        for _ in range(forwards):
            events.append(rng.choice(BUCKETS))
            dispatched += 1
            if rng.random() < 0.4:
                waited = rng.randint(max(1, dispatched - stages), dispatched)
                need = max(stages - 2 - (dispatched - waited), 0)
                events.extend([None] * need)
                dispatched += need
        events.extend([None] * (stages - 2))
        events.append(SETTLE)
    return events


def _run(waves, events, timeout=60):
    results, errors = [], []
    threads = [
        threading.Thread(
            target=_run_stage, args=(w, events, results, errors), daemon=True
        )
        for w in waves
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=timeout)
    assert not any(t.is_alive() for t in threads), "a stage never finished"
    assert not errors, errors
    return results


def test_random_forwards_and_pushes_deliver_every_forward(stages):
    ring, waves = stages
    rng = random.Random(7)
    events = _events(rng, 12, len(waves))
    results = _run(waves, events)
    forwards = [e for e in events if e is not None and e != SETTLE]
    assert len(results) == len(forwards)
    last = waves[-1].rank
    side = SIDE in waves[0].carried
    for g, rows, h, r, extra in results:
        assert rows == forwards[g]
        assert torch.equal(h, torch.full((rows, HIDDEN), g + last + 0.5))
        assert torch.equal(r, torch.zeros(rows, HIDDEN))
        if side:
            # relayed across every stage, unchanged and still paired with
            # its own forward
            assert torch.equal(
                extra, torch.full((rows, SIDE_WIDTH), g, dtype=torch.int32)
            )
    # Every burst costs its forwards and pushes plus stages - 1 launches;
    # the warmup adds one.
    moves = len(events) - events.count(SETTLE)
    assert len({w.launches for w in waves}) == 1
    assert waves[0].launches == 1 + moves + events.count(SETTLE) * last
    assert all(w.bursts == events.count(SETTLE) for w in waves)


def test_async_first_stage_delivers_every_forward(async_stages):
    ring, waves = async_stages
    rng = random.Random(11)
    events = _events(rng, 12, len(waves))
    results = _run(waves, events)
    forwards = [e for e in events if e is not None and e != SETTLE]
    assert len(results) == len(forwards)
    last = waves[-1].rank
    for g, rows, h, r, extra in results:
        assert rows == forwards[g]
        assert torch.equal(h, torch.full((rows, HIDDEN), g + last + 0.5))
        assert torch.equal(extra, torch.full((rows, SIDE_WIDTH), g, dtype=torch.int32))
    moves = len(events) - events.count(SETTLE)
    assert len({w.launches for w in waves}) == 1
    assert waves[0].launches == 1 + moves + events.count(SETTLE) * last
    assert waves[0]._async_idle()


def test_async_first_stage_reports_a_failed_drain(async_stages, monkeypatch):
    _, waves = async_stages
    w = waves[0]

    def boom(fence=None):
        raise ValueError("ring down")

    monkeypatch.setattr(pp_wave, "_drain_slice", boom)
    w.burst_open = True
    w.burst_launches = 1
    w._launch_async(w.zeros)
    with pytest.raises(RuntimeError, match="hand-off thread failed") as info:
        w._issue_queued()
    assert isinstance(info.value.__cause__, ValueError)
    assert w._issue_queued() is None and w._async_idle()


def _first_stage_alone(monkeypatch, size=2):
    """A two-chip stage 0 PPWave whose drain waits on ``drain_gate`` and whose
    hand-off completion waits on ``done``; ``drains`` records what each drain
    waited on, ``launched`` what each launch carried, ``issuers`` its thread."""
    from vllm.distributed import parallel_state

    drain_gate, done = threading.Event(), threading.Event()
    launched, drains, issuers = [], [], []

    def drain(fence=None):
        drain_gate.wait(10)
        drains.append(fence)

    def launch_now(op, carried):
        issuers.append(threading.current_thread().name)
        launched.append(tuple(x.clone() for x in carried))
        return carried

    monkeypatch.setattr(pp_wave, "stage_chips", lambda: 2)
    monkeypatch.setattr(pp_wave, "_drain_slice", drain)
    monkeypatch.setattr(pp_wave, "_fence", lambda: "fence")
    monkeypatch.setattr(pp_wave, "get_or_create_pp_mesh", lambda: None)
    monkeypatch.setattr(pp_wave, "pp_permute_op", lambda mesh, num_tensors=1: None)
    monkeypatch.setattr(pp_wave, "_launch_now", launch_now)
    monkeypatch.setattr(pp_wave, "_flush", lambda tensors: None)
    monkeypatch.setattr(pp_wave, "_wait", lambda tensors: done.wait(10))
    monkeypatch.setattr(
        parallel_state,
        "get_pp_group",
        lambda: SimpleNamespace(rank_in_group=0, world_size=size),
    )
    wave = pp_wave.PPWave(torch.device("cpu"), MAX_ROWS, _template(False))
    assert wave.async_handoff
    return wave, drain_gate, done, launched, drains, issuers


def _queue_one_forward(wave, launched):
    outgoing = {
        "hidden_states": torch.full((8, HIDDEN), 2.0),
        "residual": torch.full((8, HIDDEN), 0.5),
    }
    wave.end_forward(outgoing, 8)
    # Queued on the helper thread while drain_gate holds the drain.
    assert wave.launches == 2 and len(launched) == 1


def _until(predicate, seconds=2.0):
    for _ in range(int(seconds / 0.01)):
        if predicate():
            return True
        threading.Event().wait(0.01)
    return predicate()


def test_async_helper_issues_the_launch_while_the_worker_thread_is_idle(
    monkeypatch,
):
    wave, drain_gate, done, launched, drains, issuers = _first_stage_alone(monkeypatch)
    drain_gate.set()
    assert wave.start_forward(8) is None
    assert len(launched) == 1 and drains == [None]
    drain_gate.clear()
    _queue_one_forward(wave, launched)
    drain_gate.set()
    # No step is active, so the helper issues the launch once drained.
    assert _until(lambda: len(launched) == 2)
    assert drains[1] == "fence" and issuers[1] == "pp-wave-handoff"
    assert torch.equal(launched[1][0][:8], torch.full((8, HIDDEN), 2.5))
    nxt = threading.Thread(target=wave.start_forward, args=(8,), daemon=True)
    nxt.start()
    nxt.join(0.3)
    assert nxt.is_alive(), "start_forward ran before the hand-off completed"
    done.set()
    nxt.join(5)
    assert not nxt.is_alive()
    assert wave._async_idle()


def test_async_worker_thread_issues_the_launch_inside_a_step(monkeypatch):
    wave, drain_gate, done, launched, drains, issuers = _first_stage_alone(monkeypatch)
    drain_gate.set()
    wave.start_forward(8)
    drain_gate.clear()
    _queue_one_forward(wave, launched)

    def worker():
        with wave.step():
            wave.start_forward(8)

    nxt = threading.Thread(target=worker, name="worker", daemon=True)
    nxt.start()
    nxt.join(0.3)
    assert nxt.is_alive(), "start_forward ran before the drain"
    drain_gate.set()
    # The worker thread is tracing, so the helper leaves the launch to it.
    assert _until(lambda: len(launched) == 2)
    assert drains[1] == "fence" and issuers == [threading.main_thread().name, "worker"]
    nxt.join(0.3)
    assert nxt.is_alive(), "start_forward ran before the hand-off completed"
    done.set()
    nxt.join(5)
    assert not nxt.is_alive()
    assert wave._async_idle()


def test_async_step_waits_for_a_launch_the_helper_is_issuing(monkeypatch):
    wave, drain_gate, done, launched, drains, issuers = _first_stage_alone(monkeypatch)
    drain_gate.set()
    wave.start_forward(8)
    drain_gate.clear()
    _queue_one_forward(wave, launched)
    issue_gate = threading.Event()
    real_launch_now = pp_wave._launch_now

    def slow_launch_now(op, carried):
        issue_gate.wait(10)
        return real_launch_now(op, carried)

    monkeypatch.setattr(pp_wave, "_launch_now", slow_launch_now)
    drain_gate.set()
    assert _until(lambda: wave._async_issuing)
    entered = threading.Event()

    def worker():
        with wave.step():
            entered.set()

    nxt = threading.Thread(target=worker, daemon=True)
    nxt.start()
    assert not entered.wait(0.3), "a step began while the helper was issuing"
    issue_gate.set()
    assert entered.wait(5)
    assert len(launched) == 2 and issuers[1] == "pp-wave-handoff"


def test_async_settle_on_two_stages_reports_a_failed_drain(monkeypatch):
    wave, drain_gate, done, launched, drains, issuers = _first_stage_alone(monkeypatch)
    drain_gate.set()
    wave.start_forward(8)

    def boom(fence=None):
        raise ValueError("ring down")

    monkeypatch.setattr(pp_wave, "_drain_slice", boom)
    wave.end_forward(
        {"hidden_states": torch.zeros(8, HIDDEN), "residual": torch.zeros(8, HIDDEN)},
        8,
    )
    with pytest.raises(RuntimeError, match="hand-off thread failed") as info:
        wave.settle()
    assert isinstance(info.value.__cause__, ValueError)


def test_pushes_alone_carry_a_lone_forward_to_the_last_stage(stages):
    ring, waves = stages
    events = [16] + [None] * (len(waves) - 2) + [SETTLE]
    results = _run(waves, events, timeout=30)
    assert len(results) == 1
    # one forward, six pushes, then the settle's stages - 1 launches
    assert [w.launches for w in waves] == [1 + 7 + 7] * 8


def test_settle_without_an_open_burst_is_a_noop(stages):
    _, waves = stages
    waves[3].settle()
    assert waves[3].launches == 0


def test_a_launch_count_off_by_one_is_caught(stages):
    _, waves = stages
    w = waves[0]
    w.burst_open = True
    w.burst_launches = 3
    w.forwards = 1
    with pytest.raises(RuntimeError, match="issued 3 launches, expected 2"):
        w._check_count()


def test_the_hand_off_always_carries_hidden_states_and_residual(monkeypatch):
    from vllm.distributed import parallel_state

    monkeypatch.setattr(pp_wave, "get_or_create_pp_mesh", lambda: None)
    monkeypatch.setattr(pp_wave, "pp_permute_op", lambda mesh, num_tensors=1: None)
    monkeypatch.setattr(
        parallel_state,
        "get_pp_group",
        lambda: SimpleNamespace(rank_in_group=0, world_size=2),
    )
    with pytest.raises(ValueError, match="hidden_states and residual"):
        pp_wave.PPWave(
            torch.device("cpu"),
            MAX_ROWS,
            {
                "hidden_states": ((HIDDEN,), torch.float32),
                "other": ((HIDDEN,), torch.float32),
            },
        )
    with pytest.raises(ValueError, match="one dtype"):
        pp_wave.PPWave(
            torch.device("cpu"),
            MAX_ROWS,
            {
                "hidden_states": ((HIDDEN,), torch.float32),
                "residual": ((HIDDEN,), torch.bfloat16),
            },
        )
    with pytest.raises(ValueError, match="side tensors"):
        pp_wave.PPWave(
            torch.device("cpu"),
            MAX_ROWS,
            {
                "hidden_states": ((HIDDEN,), torch.float32),
                "residual": ((HIDDEN,), torch.float32),
                SIDE: ((2, SIDE_WIDTH), torch.int32),
            },
        )


def test_a_side_tensor_is_self_tested_on_every_stage(monkeypatch):
    """The warmup checks every tensor, so one the runtime drops is caught
    before a request depends on it."""
    _, waves = _build_stages(monkeypatch, _template(True), size=3)
    _run(waves, [], timeout=30)  # the warmup alone
    assert [w.carried for w in waves] == [("hidden_states", SIDE)] * 3
    assert [w.launches for w in waves] == [1, 1, 1]


def test_a_forward_that_drops_a_side_tensor_is_refused(monkeypatch):
    _, waves = _build_stages(monkeypatch, _template(True), size=3)
    wave = waves[0]
    with pytest.raises(RuntimeError, match=f"no \\['{SIDE}'\\]"):
        wave.end_forward(
            {
                "hidden_states": torch.zeros(8, HIDDEN),
                "residual": torch.zeros(8, HIDDEN),
            },
            8,
        )


def test_summing_before_the_hand_off_keeps_the_residual_and_rounds_the_norm_input_once():  # noqa: E501
    """The next stage's first norm adds hidden states and residual in
    float32, normalizes the sum and keeps it rounded as the residual. The
    hand-off sends the sum rounded to the model dtype with a zero residual,
    so the residual is identical and the norm input differs by at most one
    bfloat16 rounding."""
    # a local generator: the global seed would initialize every accelerator
    # backend in this process, and a process that opened the TPU keeps it
    generator = torch.Generator().manual_seed(0)
    hidden = (torch.randn(256, 512, generator=generator) * 4).to(torch.bfloat16)
    residual = (torch.randn(256, 512, generator=generator) * 4).to(torch.bfloat16)

    # the norm's input and new residual in a single-stage model
    norm_input = hidden.float() + residual.float()
    new_residual = norm_input.to(torch.bfloat16)

    # the same after the hand-off, as PPWave.end_forward and start_forward
    # shape it
    sent = hidden + residual
    received_norm_input = sent.float() + torch.zeros_like(sent).float()
    received_residual = received_norm_input.to(torch.bfloat16)

    assert torch.equal(received_residual, new_residual)
    assert torch.allclose(received_norm_input, norm_input, rtol=2**-8, atol=0)


def _column_wave(monkeypatch, stage, column, received, chips=1):
    """The wave of one device of a slice of three stages of two devices,
    whose hand-off returns ``received`` in every tensor."""
    from vllm.distributed import parallel_state

    from vllm_torchtpu.distributed import pcp

    drains = []
    monkeypatch.setattr(pp_wave, "get_or_create_pp_mesh", lambda: None)
    monkeypatch.setattr(pp_wave, "pp_permute_op", lambda mesh, num_tensors=1: None)
    monkeypatch.setattr(
        pp_wave,
        "_launch_now",
        lambda op, carried: tuple(torch.full_like(x, received) for x in carried),
    )
    monkeypatch.setattr(pp_wave, "stage_chips", lambda: chips)
    monkeypatch.setattr(
        pp_wave, "_drain_slice", lambda fence=None: drains.append(fence)
    )
    monkeypatch.setattr(pp_wave, "_fence", lambda: "fence")
    monkeypatch.setattr(
        parallel_state,
        "get_pp_group",
        lambda: SimpleNamespace(rank_in_group=stage, world_size=3),
    )
    monkeypatch.setattr(
        pcp,
        "get_pcp_group",
        lambda: SimpleNamespace(rank_in_group=column, world_size=2),
    )
    return pp_wave.PPWave(torch.device("cpu"), MAX_ROWS, _template(False)), drains


def test_the_self_test_expects_the_same_column_of_the_previous_stage(monkeypatch):
    # Stage 1 column 1 sends 4 and receives what stage 0 column 1 sent: 2.
    wave, _ = _column_wave(monkeypatch, stage=1, column=1, received=2)
    wave.warmup()
    # 1 is what the other column of the previous stage sent.
    wave, _ = _column_wave(monkeypatch, stage=1, column=1, received=1)
    with pytest.raises(RuntimeError, match="expected 2.0"):
        wave.warmup()


def test_a_stage_of_several_chips_drains_before_every_hand_off(monkeypatch):
    wave, drains = _column_wave(monkeypatch, stage=1, column=0, received=1, chips=2)
    wave.warmup()
    wave.start_forward(16)
    assert drains == [None] * 3 and wave.launches == 3

    wave, drains = _column_wave(monkeypatch, stage=1, column=0, received=1)
    wave.warmup()
    wave.start_forward(16)
    assert drains == [] and wave.launches == 3


def test_only_a_first_stage_of_several_chips_launches_from_the_helper_thread(
    monkeypatch,
):
    wave, _ = _column_wave(monkeypatch, stage=0, column=0, received=1, chips=2)
    assert wave.async_handoff
    wave, _ = _column_wave(monkeypatch, stage=0, column=0, received=1)
    assert not wave.async_handoff
    wave, _ = _column_wave(monkeypatch, stage=1, column=0, received=1, chips=2)
    assert not wave.async_handoff
