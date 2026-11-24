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


def _build_stages(monkeypatch, template, size=8):
    """``size`` PPWave instances on CPU sharing one fake ring."""
    ring = _Ring(size)
    current = {}
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
            if isinstance(rows, tuple):
                wave.flush(force=rows[1])
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


def test_a_forward_larger_than_the_hand_off_buffer_is_refused(monkeypatch):
    _, waves = _build_stages(monkeypatch, _template(False), size=3)
    rows = MAX_ROWS * 2
    with pytest.raises(RuntimeError, match="exceeds the hand-off buffer"):
        waves[0].end_forward(
            {
                "hidden_states": torch.zeros(rows, HIDDEN),
                "residual": torch.zeros(rows, HIDDEN),
            },
            rows,
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
    monkeypatch.setattr(pp_wave, "_drain_slice", lambda: drains.append(1))
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
    assert len(drains) == wave.launches == 3

    wave, drains = _column_wave(monkeypatch, stage=1, column=0, received=1)
    wave.warmup()
    wave.start_forward(16)
    assert drains == [] and wave.launches == 3


def _deferred(monkeypatch, idle):
    """Stage 0 defers its hand-off; ``idle`` answers the device check."""
    monkeypatch.setenv("TPU_PP_DEFER_HANDOFF", "1")
    monkeypatch.delenv("TPU_PP_LAZY_PUSH", raising=False)
    monkeypatch.setattr(pp_wave, "_stage_cpu_group", lambda rank: None)
    monkeypatch.setattr(pp_wave, "_submit", lambda tensors: None)
    monkeypatch.setattr(pp_wave, "_record_event", lambda: "event")
    monkeypatch.setattr(pp_wave, "_device_idle", lambda event: idle())


def _flush_events(rng, forwards, bursts=3):
    """Forwards with the engine's flushes between them: unforced ones as
    results arrive, a forced one when it waits for the newest step, and a
    forced one before each settle."""
    events = []
    for _ in range(bursts):
        for _ in range(forwards):
            events.append(rng.choice(BUCKETS))
            if rng.random() < 0.5:
                events.append(("flush", rng.random() < 0.3))
        events.append(("flush", True))
        events.append(SETTLE)
    return events


@pytest.mark.parametrize("side", [False, True], ids=["activations", "with_side"])
def test_a_deferred_hand_off_sends_the_same_payload_on_every_launch(monkeypatch, side):
    events = _flush_events(random.Random(3), 12)
    ring, waves = _build_stages(monkeypatch, _template(side), size=2)
    plain = _run(waves, events)

    answers = iter(random.Random(5).random() < 0.5 for _ in range(10_000))
    _deferred(monkeypatch, lambda: next(answers))
    deferred_ring, deferred_waves = _build_stages(monkeypatch, _template(side), size=2)
    assert [w.defer for w in deferred_waves] == [True, True]
    deferred = _run(deferred_waves, events)

    assert [(g, rows) for g, rows, *_ in deferred] == [
        (g, rows) for g, rows, *_ in plain
    ]
    for (_, _, h, r, x), (_, _, h2, r2, x2) in zip(plain, deferred):
        assert torch.equal(h, h2) and torch.equal(r, r2)
        assert (x is None and x2 is None) or torch.equal(x, x2)
    # launch p carries the same buffers whenever stage 0 issues it
    assert ring.slots.keys() == deferred_ring.slots.keys()
    for key, sent in ring.slots.items():
        assert all(torch.equal(a, b) for a, b in zip(sent, deferred_ring.slots[key]))
    assert [w.launches for w in deferred_waves] == [w.launches for w in waves]


def test_stage_0_holds_its_output_until_the_next_forward_or_a_flush(monkeypatch):
    idle = [False]
    _deferred(monkeypatch, lambda: idle[0])
    launched = []
    wave, _ = _column_wave(monkeypatch, stage=0, column=0, received=0)
    wave.size, wave.last, wave.defer = 2, 1, True
    monkeypatch.setattr(
        pp_wave, "_launch_now", lambda op, carried: launched.append(carried) or carried
    )
    out = {"hidden_states": torch.ones(8, HIDDEN), "residual": torch.ones(8, HIDDEN)}

    assert wave.start_forward(8) is None  # the prologue only
    wave.end_forward(out, 8)
    assert len(launched) == 1 and wave.pending is not None
    # unforced while the forward still runs: kept
    wave.flush(force=False)
    assert len(launched) == 1
    # the next forward launches it before it runs
    assert wave.start_forward(8) is None
    assert len(launched) == 2
    assert torch.equal(launched[1][0][:8], 2 * torch.ones(8, HIDDEN))
    wave.end_forward(out, 8)
    # unforced once the forward has finished: launched
    idle[0] = True
    wave.flush(force=False)
    assert len(launched) == 3 and wave.pending is None
    # nothing held: a flush does nothing
    wave.flush(force=True)
    assert len(launched) == 3
    assert wave.burst_launches == wave.forwards + 1


def test_deferral_is_two_stages_only_and_not_with_lazy_pushes(monkeypatch):
    monkeypatch.delenv("TPU_PP_DEFER_HANDOFF", raising=False)
    assert not pp_wave.defer_handoff(2)
    monkeypatch.setenv("TPU_PP_DEFER_HANDOFF", "1")
    monkeypatch.delenv("TPU_PP_LAZY_PUSH", raising=False)
    assert pp_wave.defer_handoff(2)
    assert not pp_wave.defer_handoff(3)
    monkeypatch.setenv("TPU_PP_LAZY_PUSH", "1")
    assert not pp_wave.defer_handoff(2)


def test_host_timing_summarizes_the_segments(monkeypatch):
    clock = iter(range(100))
    monkeypatch.setattr(pp_wave.time, "perf_counter", lambda: next(clock) / 1000)
    logged = []
    monkeypatch.setattr(pp_wave.logger, "info", lambda *a: logged.append(a))
    timing = pp_wave._HostTiming("stage 0", every=2)
    for _ in range(2):
        for mark in (
            "ret",
            "entry",
            "prep",
            "drain_start",
            "drain_end",
            "barrier_end",
            "launch_end",
            "dispatched",
        ):
            timing.note(mark)
    assert len(logged) == 1
    line = logged[0][0] % logged[0][1:]
    assert "drain n=2 mean=1.00" in line and "idle_to_dispatch n=2 mean=3.00" in line
    assert "ret_to_entry n=2 mean=1.00" in line and "sample" not in line
