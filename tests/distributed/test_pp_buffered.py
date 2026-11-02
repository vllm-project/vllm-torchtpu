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
"""The buffered pipeline hand-off, point to point, driven on CPU with a fake
transport.

Every device runs its own event stream in a thread, the way the runner calls
the hand-off. The fake transport behaves like the runtime's point-to-point path:
a send waits until the matching receive is posted and then copies its data
into the posted tensor, a receive returns at once, and a device reads a
received tensor only once its data has landed. A schedule whose transfers
do not pair therefore times out instead of passing by accident.
"""

import random
import threading
from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu.distributed import pp_buffered

HIDDEN = 8
MAX_ROWS = 64
BUCKETS = (8, 16, 32, 64)
SIDE = "topk_indices"
SIDE_WIDTH = 4
TIMEOUT = 20


class _Work:
    def wait(self):
        return True


class _Fence:
    def synchronize(self):
        pass


class _Transport:
    """``torch.distributed.isend``/``irecv`` between threads. Channels are
    keyed by (source, destination, tag) global ranks; ``stride`` is the
    distance between a device and the same column of the next stage."""

    def __init__(self, stride=1):
        self.stride = stride
        self.cv = threading.Condition()
        self.posted = {}
        # every receive posted, in the order it was posted
        self.receives = []
        self.landed = {}
        self.sends = []
        self.views_sent = 0
        self.fail_send = None
        self.fail_recv = None

    def deliver(self, tensor):
        return tensor

    def irecv(self, tensor, src, tag):
        if self.fail_recv is not None:
            raise self.fail_recv
        key = (src, src + self.stride, tag)
        with self.cv:
            assert key not in self.posted, f"receive {key} posted twice"
            self.posted[key] = tensor
            self.receives.append(key)
            self.cv.notify_all()
        return _Work()

    def isend(self, tensor, dst, tag):
        if self.fail_send is not None:
            raise self.fail_send
        key = (dst - self.stride, dst, tag)
        with self.cv:
            if not self.cv.wait_for(lambda: key in self.posted, timeout=TIMEOUT):
                raise TimeoutError(f"send {key} never found its receive")
            target = self.posted.pop(key)
            assert (target.shape, target.dtype) == (tensor.shape, tensor.dtype), (
                f"send {key}: {tuple(tensor.shape)} {tensor.dtype} into a "
                f"receive of {tuple(target.shape)} {target.dtype}"
            )
            if tensor._base is not None:
                self.views_sent += 1
            target.copy_(self.deliver(tensor))
            self.landed[id(target)] = target
            self.sends.append((key, int(tensor.shape[0])))
            self.cv.notify_all()
        return _Work()

    def wait_landed(self, tensor):
        """What the device does before a program reads ``tensor``."""
        with self.cv:
            if not self.cv.wait_for(
                lambda: self.landed.get(id(tensor)) is tensor, timeout=TIMEOUT
            ):
                raise TimeoutError("a received tensor never landed")


def _template(side):
    template = {
        "hidden_states": ((HIDDEN,), torch.float32),
        "residual": ((HIDDEN,), torch.float32),
    }
    if side:
        template[SIDE] = ((SIDE_WIDTH,), torch.int32)
    return template


def _build(monkeypatch, stages, columns=1, side=False, depth=2, transport=None):
    """One PPBuffered per device of ``stages`` stages of ``columns`` devices,
    sharing one fake transport; keyed by (stage, column)."""
    from vllm.distributed import parallel_state

    from vllm_torchtpu.distributed import pcp

    transport = transport or _Transport(stride=columns)
    monkeypatch.setattr(torch.distributed, "isend", transport.isend)
    monkeypatch.setattr(torch.distributed, "irecv", transport.irecv)
    monkeypatch.setattr(pp_buffered, "_flush", lambda tensors: None)
    monkeypatch.setattr(pp_buffered, "_fence", _Fence)
    read = pp_buffered._first_and_last

    def first_and_last(x):
        transport.wait_landed(x)
        return read(x)

    monkeypatch.setattr(pp_buffered, "_first_and_last", first_and_last)
    current = {}
    monkeypatch.setattr(
        parallel_state,
        "get_pp_group",
        lambda: SimpleNamespace(
            rank_in_group=current["stage"],
            world_size=stages,
            ranks=[stage * columns + current["column"] for stage in range(stages)],
        ),
    )
    monkeypatch.setattr(
        pcp,
        "get_pcp_group",
        lambda: SimpleNamespace(rank_in_group=current["column"], world_size=columns),
    )
    handoffs = {}
    for stage in range(stages):
        for column in range(columns):
            current.update(stage=stage, column=column)
            handoffs[stage, column] = pp_buffered.PPBuffered(
                torch.device("cpu"), MAX_ROWS, _template(side), depth=depth
            )
    return transport, handoffs


def _forward(handoff, transport, received, rows, g, column):
    """One forward: the first stage seeds forward ``g`` with ``g`` plus 100
    per column as hidden states and a residual of 0.5, which every later
    stage must see as part of the sum; later stages add 1. The side tensor
    is seeded by the first stage as a view of a larger table and relayed
    unchanged. Returns what the last stage sees, None elsewhere."""
    side = SIDE in handoff.carried
    if handoff.rank == 0:
        assert received is None
        h = torch.full((rows, HIDDEN), float(g + 100 * column))
        r = torch.full((rows, HIDDEN), 0.5)
        table = torch.full((MAX_ROWS, SIDE_WIDTH), g, dtype=torch.int32)
        extra = table[:rows]
    else:
        for key in handoff.carried:
            transport.wait_landed(received[key])
        assert torch.equal(received["residual"], torch.zeros(rows, HIDDEN))
        h = received["hidden_states"] + 1
        r = received["residual"]
        extra = received[SIDE] if side else None
    if handoff.rank == handoff.last:
        handoff.end_forward(None, rows)
        return h, r, extra
    outgoing = {"hidden_states": h, "residual": r}
    if side:
        outgoing[SIDE] = extra
    handoff.end_forward(outgoing, rows)
    return None


def _run_device(handoff, transport, steps, results, errors, column):
    """A device's event stream as the runner drives it: each step is the
    padded token count of each of its forwards and whether the runner posts
    its receives first (serving steps) or not (warmup forwards). A step
    without tokens runs nothing."""
    try:
        handoff.warmup()
        g = 0
        for rows_per_forward, ahead in steps:
            if not rows_per_forward:
                continue
            if ahead and handoff.rank > 0:
                handoff.begin_step(rows_per_forward)
            for rows in rows_per_forward:
                received = handoff.start_forward(rows)
                out = _forward(handoff, transport, received, rows, g, column)
                if out is not None:
                    results.append((column, g, rows, *out))
                g += 1
        handoff.close()
    except BaseException as exc:  # reported by the test thread
        errors.append(((handoff.rank, column), exc))


def _run(handoffs, transport, steps, timeout=60):
    results, errors = [], []
    threads = [
        threading.Thread(
            target=_run_device,
            args=(handoff, transport, steps, results, errors, column),
            daemon=True,
        )
        for (_, column), handoff in handoffs.items()
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=timeout)
    assert not any(t.is_alive() for t in threads), "a device never finished"
    assert not errors, errors
    return results


def _steps(rng, count):
    """Serving steps of one to three forwards, steps without tokens, and
    warmup forwards that post no receives ahead."""
    steps = []
    for _ in range(count):
        kind = rng.random()
        if kind < 0.2:
            steps.append(([], True))
        elif kind < 0.3:
            steps.append(([rng.choice(BUCKETS)], False))
        else:
            steps.append(
                ([rng.choice(BUCKETS) for _ in range(rng.randint(1, 3))], True)
            )
    return steps


def _check(results, steps, last, columns=1, side=False):
    forwards = [rows for rows_per_forward, _ in steps for rows in rows_per_forward]
    assert len(results) == len(forwards) * columns
    for column, g, rows, h, r, extra in results:
        assert rows == forwards[g]
        assert torch.equal(h, torch.full((rows, HIDDEN), g + 100 * column + last + 0.5))
        assert torch.equal(r, torch.zeros(rows, HIDDEN))
        if side:
            # relayed across every stage, unchanged and still paired with
            # its own forward
            assert torch.equal(
                extra, torch.full((rows, SIDE_WIDTH), g, dtype=torch.int32)
            )
    return forwards


@pytest.mark.parametrize("side", [False, True], ids=["activations", "with_side"])
@pytest.mark.parametrize("stages", [2, 3, 4])
def test_every_forward_reaches_the_last_stage(monkeypatch, stages, side):
    transport, handoffs = _build(monkeypatch, stages, side=side)
    steps = _steps(random.Random(stages), 24)
    results = _run(handoffs, transport, steps)
    forwards = _check(results, steps, stages - 1, side=side)
    carried = 2 if side else 1
    # The warmup and every forward cross each stage boundary once, with one
    # transfer per carried tensor; nothing is left posted.
    assert len(transport.sends) == (1 + len(forwards)) * (stages - 1) * carried
    assert transport.posted == {}
    assert transport.views_sent == 0
    # Each stage posts transfer by transfer, one receive per carried tensor.
    for stage in range(1, stages):
        tags = [tag for _, dst, tag in transport.receives if dst == stage]
        assert tags == list(range(carried, carried * (2 + len(forwards))))
    for (stage, _), handoff in handoffs.items():
        assert handoff.transfers == 1 + len(forwards)
        assert handoff.forwards == len(forwards)
        assert handoff.sent == (len(forwards) if stage < stages - 1 else 0)
        assert handoff.posted_through == (1 + len(forwards) if stage > 0 else 0)
        assert handoff.landed_through == 1 + len(forwards)


def test_columns_hand_off_to_the_same_column(monkeypatch):
    transport, handoffs = _build(monkeypatch, stages=3, columns=2, side=True)
    steps = _steps(random.Random(5), 12)
    results = _run(handoffs, transport, steps)
    _check(results, steps, last=2, columns=2, side=True)
    # Global ranks are stage * 2 + column: every transfer goes two ranks on.
    assert {dst - src for (src, dst, _), _ in transport.sends} == {2}


def test_steps_without_tokens_transfer_nothing(monkeypatch):
    transport, handoffs = _build(monkeypatch, stages=3)
    results = _run(handoffs, transport, [([], True)] * 5)
    assert results == []
    # the warmup alone
    assert len(transport.sends) == 2
    for handoff in handoffs.values():
        handoff.begin_step([])
        assert handoff.transfers == 1 and not handoff._posted


def _until(predicate, seconds=5.0):
    for _ in range(int(seconds / 0.01)):
        if predicate():
            return True
        threading.Event().wait(0.01)
    return predicate()


@pytest.mark.parametrize("depth", [1, 2])
def test_a_stage_that_runs_ahead_holds_at_most_depth_forwards(monkeypatch, depth):
    transport, handoffs = _build(monkeypatch, stages=2, side=True, depth=depth)
    first, second = handoffs[0, 0], handoffs[1, 0]
    warmups = [threading.Thread(target=r.warmup, daemon=True) for r in (first, second)]
    for t in warmups:
        t.start()
    for t in warmups:
        t.join(TIMEOUT)
    rows = [16, 8, 32, 16, 8]
    queued = []

    def run_first():
        for g, n in enumerate(rows):
            _forward(first, transport, first.start_forward(n), n, g, 0)
            queued.append(g)

    ahead = threading.Thread(target=run_first, daemon=True)
    ahead.start()
    # The second stage has posted nothing: the first sends nothing and stops
    # once ``depth`` forwards wait in its queue.
    assert _until(lambda: len(queued) == depth)
    ahead.join(0.3)
    assert ahead.is_alive() and len(queued) == depth
    assert first.sent == 0 and len(first._sends) == depth

    results = []

    def run_second():
        second.begin_step(rows)
        for g, n in enumerate(rows):
            h, _, extra = _forward(second, transport, second.start_forward(n), n, g, 0)
            results.append((g, h, extra))

    behind = threading.Thread(target=run_second, daemon=True)
    behind.start()
    for t in (ahead, behind):
        t.join(TIMEOUT)
        assert not t.is_alive()
    first.close()
    assert first.sent == len(rows)
    for g, h, extra in results:
        assert torch.equal(h, torch.full((rows[g], HIDDEN), g + 1.5))
        assert torch.equal(
            extra, torch.full((rows[g], SIDE_WIDTH), g, dtype=torch.int32)
        )


def test_the_self_test_catches_data_from_the_wrong_sender(monkeypatch):
    class Misrouting(_Transport):
        def deliver(self, tensor):
            return tensor + 1

    transport, handoffs = _build(monkeypatch, stages=2, transport=Misrouting())
    errors = []

    def warmup(handoff):
        try:
            handoff.warmup()
        except RuntimeError as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=warmup, args=(r,), daemon=True)
        for r in handoffs.values()
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(TIMEOUT)
    (error,) = errors
    assert "self-test tensor 'hidden_states'" in str(error)
    assert "expected 1.0 from global rank 0" in str(error)
    assert "libtpu" in str(error)


def test_a_transport_that_fails_fails_the_self_test_and_names_libtpu(monkeypatch):
    transport, handoffs = _build(monkeypatch, stages=2)
    transport.fail_recv = RuntimeError("cross-host receive unsupported")
    with pytest.raises(RuntimeError, match="self-test failed.*libtpu") as info:
        handoffs[1, 0].warmup()
    assert "unsupported" in str(info.value.__cause__)


def _outgoing(rows):
    return {
        "hidden_states": torch.zeros(rows, HIDDEN),
        "residual": torch.zeros(rows, HIDDEN),
    }


def test_a_send_failure_surfaces_on_the_next_call(monkeypatch):
    transport, handoffs = _build(monkeypatch, stages=2)
    first = handoffs[0, 0]
    transport.fail_send = ValueError("transport down")
    first.start_forward(8)
    first.end_forward(_outgoing(8), 8)
    assert _until(lambda: first._error is not None)
    with pytest.raises(RuntimeError, match="send thread failed") as info:
        first.start_forward(8)
    assert isinstance(info.value.__cause__, ValueError)
    # it stays failed: nothing else could be handed on in order
    with pytest.raises(RuntimeError, match="send thread failed"):
        first.end_forward(_outgoing(8), 8)
    with pytest.raises(RuntimeError, match="send thread failed"):
        first.close()
    assert not first._sends


def test_a_forward_whose_rows_differ_from_its_posted_receive_is_refused(
    monkeypatch,
):
    _, handoffs = _build(monkeypatch, stages=2)
    second = handoffs[1, 0]
    second.begin_step([16])
    with pytest.raises(RuntimeError, match="posted for 16 rows"):
        second.start_forward(8)


def test_receives_posted_for_forwards_that_never_ran_are_reported(monkeypatch):
    _, handoffs = _build(monkeypatch, stages=2)
    second = handoffs[1, 0]
    second.begin_step([16, 16])
    second.start_forward(16)
    second.end_forward(None, 16)
    with pytest.raises(RuntimeError, match="1 receives were posted"):
        second.begin_step([8])


class _Gated(_Transport):
    """A receive blocks until ``gate`` opens, as a real one blocks until its
    landing buffer and receive flag are taken."""

    def __init__(self):
        super().__init__()
        self.gate = threading.Event()
        self.callers = []

    def irecv(self, tensor, src, tag):
        self.callers.append(threading.current_thread())
        if not self.gate.wait(TIMEOUT):
            raise TimeoutError("a receive was never released")
        return super().irecv(tensor, src, tag)


def test_receives_are_posted_off_the_thread_that_runs_the_step(monkeypatch):
    transport, handoffs = _build(monkeypatch, stages=2, transport=_Gated())
    second = handoffs[1, 0]
    second.begin_step([16, 8])
    # begin_step has returned while the first receive is still being posted
    assert _until(lambda: len(transport.callers) == 1)
    assert second.posted_through == 0 and transport.posted == {}
    got = []
    forward = threading.Thread(
        target=lambda: got.append(second.start_forward(16)), daemon=True
    )
    forward.start()
    forward.join(0.3)
    # the forward waits until its receive is posted
    assert forward.is_alive() and got == []
    transport.gate.set()
    forward.join(TIMEOUT)
    assert not forward.is_alive()
    assert got[0]["hidden_states"].shape == (16, HIDDEN)
    assert _until(lambda: second.posted_through == 2)
    # in transfer order, from one thread that is not the step's
    assert list(transport.posted) == [(0, 1, 1), (0, 1, 2)]
    (poster,) = set(transport.callers)
    assert (
        poster is not threading.current_thread() and poster.name == "pp-buffered-post"
    )
    assert second.start_forward(8)["hidden_states"].shape == (8, HIDDEN)
    second.close()
    assert not poster.is_alive()


class _Watched(_Transport):
    """Records, with each receive posted, the last transfer whose forward
    on ``handoff`` had completed by then."""

    def __init__(self):
        super().__init__()
        self.handoff = None
        self.landed_at_post = []

    def irecv(self, tensor, src, tag):
        transfer, index = divmod(tag, len(self.handoff.carried))
        self.landed_at_post.append((transfer, index, self.handoff.landed_through))
        return super().irecv(tensor, src, tag)


class _HeldFence:
    """A device fence that completes when the test sets ``done``, raising
    ``error`` if the test set one."""

    def __init__(self):
        self.done = threading.Event()
        self.error = None

    def synchronize(self):
        if not self.done.wait(TIMEOUT):
            raise TimeoutError("a fence never completed")
        if self.error is not None:
            raise self.error


def _hold_fences(monkeypatch):
    """Makes every fence the hand-off records a _HeldFence; returns them in the
    order they are recorded."""
    fences = []

    def record_fence():
        fences.append(_HeldFence())
        return fences[-1]

    monkeypatch.setattr(pp_buffered, "_fence", record_fence)
    return fences


@pytest.mark.parametrize("side", [False, True], ids=["activations", "with_side"])
def test_a_step_of_many_forwards_keeps_seven_receives_outstanding(monkeypatch, side):
    transport, handoffs = _build(monkeypatch, stages=2, side=side, transport=_Watched())
    fences = _hold_fences(monkeypatch)
    second = transport.handoff = handoffs[1, 0]
    carried = len(second.carried)
    # the transfers whose receive calls fit in seven
    ahead = 7 // carried
    second.begin_step([8] * 9)
    assert _until(lambda: second.posted_through == ahead)
    # forwards that took their receives free nothing until they complete
    for _ in range(ahead):
        second.start_forward(8)
        second.end_forward(None, 8)
    assert not _until(lambda: second.posted_through > ahead, seconds=0.3)
    for completed in range(1, 10 - ahead):
        fences[completed - 1].done.set()
        want = ahead + completed
        assert _until(lambda want=want: second.posted_through == want)
        second.start_forward(8)
        second.end_forward(None, 8)
    for fence in fences:
        fence.done.set()
    second.close()
    assert second.landed_through == 9
    calls = [(transfer, index) for transfer, index, _ in transport.landed_at_post]
    assert calls == [(t, i) for t in range(1, 10) for i in range(carried)]
    for transfer, index, landed in transport.landed_at_post:
        # a transfer beyond the first ``ahead`` only once the forwards before
        # it had completed, never more than seven calls outstanding
        assert landed >= transfer - ahead
        assert (transfer - 1 - landed) * carried + index + 1 <= 7


def test_forwards_that_post_their_own_receives_keep_seven_outstanding(monkeypatch):
    transport, handoffs = _build(monkeypatch, stages=2, transport=_Watched())
    fences = _hold_fences(monkeypatch)
    second = transport.handoff = handoffs[1, 0]
    errors = []

    def warmup():
        # warmup forwards: no begin_step, each posts its own receive
        try:
            for _ in range(9):
                second.start_forward(8)
                second.end_forward(None, 8)
        except BaseException as exc:  # reported by the test thread
            errors.append(exc)

    device = threading.Thread(target=warmup, daemon=True)
    device.start()
    assert _until(lambda: second.posted_through == 7)
    # the eighth forward waits until one before it completes
    assert not _until(lambda: second.posted_through > 7, seconds=0.3)
    for completed in range(1, 3):
        fences[completed - 1].done.set()
        want = 7 + completed
        assert _until(lambda want=want: second.posted_through == want)
    for fence in fences:
        fence.done.set()
    device.join(TIMEOUT)
    assert not device.is_alive() and not errors, errors
    second.close()
    assert second.landed_through == 9
    for transfer, index, landed in transport.landed_at_post:
        assert (transfer - 1 - landed) + index + 1 <= 7


def test_a_send_thread_failure_fails_a_forward_waiting_for_its_receive(monkeypatch):
    _, handoffs = _build(monkeypatch, stages=2)
    fences = _hold_fences(monkeypatch)
    second = handoffs[1, 0]
    second.begin_step([8] * 9)
    assert _until(lambda: second.posted_through == 7)
    for _ in range(7):
        second.start_forward(8)
        second.end_forward(None, 8)
    errors = []

    def eighth_forward():
        try:
            second.start_forward(8)
        except RuntimeError as exc:
            errors.append(exc)

    forward = threading.Thread(target=eighth_forward, daemon=True)
    forward.start()
    forward.join(0.3)
    # it waits for a receive the posting thread holds back
    assert forward.is_alive()
    fences[0].error = RuntimeError("program allocation failed")
    fences[0].done.set()
    forward.join(TIMEOUT)
    assert not forward.is_alive()
    (error,) = errors
    assert "the send thread failed" in str(error)
    assert str(error.__cause__) == "program allocation failed"
    assert _until(lambda: not second._post_thread.is_alive())
    assert not second._to_post


def test_closing_drops_receives_that_wait_for_room(monkeypatch):
    _, handoffs = _build(monkeypatch, stages=2)
    second = handoffs[1, 0]
    second.begin_step([8] * 9)
    assert _until(lambda: second.posted_through == 7)
    poster = second._post_thread
    second.close()
    assert not poster.is_alive() and not second._to_post
    assert [landing.cancelled() for *_, landing in second._posted] == (
        [False] * 7 + [True] * 2
    )


def test_a_post_failure_surfaces_on_the_forward_that_waits_for_it(monkeypatch):
    transport, handoffs = _build(monkeypatch, stages=2)
    second = handoffs[1, 0]
    transport.fail_recv = ValueError("no receive flag")
    second.begin_step([16, 8])
    with pytest.raises(RuntimeError, match="posting thread failed") as info:
        second.start_forward(16)
    assert isinstance(info.value.__cause__, ValueError)
    # it stays failed: no later receive could be posted in order
    with pytest.raises(RuntimeError, match="posting thread failed"):
        second.begin_step([8])
    with pytest.raises(RuntimeError, match="posting thread failed"):
        second.close()
    assert not second._to_post


def test_each_forward_size_reuses_one_zero_residual(monkeypatch):
    _, handoffs = _build(monkeypatch, stages=2)
    second = handoffs[1, 0]
    second.begin_step([16, 8, 16])
    residuals = []
    for rows in (16, 8, 16):
        residuals.append(second.start_forward(rows)["residual"])
        second.end_forward(None, rows)
    sixteen, eight, again = residuals
    assert again is sixteen and eight is not sixteen
    # whole tensors, not views of a larger one
    assert sixteen._base is None and eight._base is None
    assert torch.equal(sixteen, torch.zeros(16, HIDDEN))
    assert torch.equal(eight, torch.zeros(8, HIDDEN))
    assert sorted(second._zero_residuals) == [8, 16]


def test_the_hbm_reserve_covers_every_buffer_the_handoff_holds():
    row = HIDDEN * torch.float32.itemsize
    side_row = SIDE_WIDTH * torch.int32.itemsize
    zero_rows = sum(BUCKETS)
    # depth queued plus one held outgoing forward, seven landing tensors of
    # the widest carried tensor, and a zero residual per bucket
    assert pp_buffered.hbm_reserve_bytes(MAX_ROWS, _template(False), 2, zero_rows) == (
        (2 + 1) * MAX_ROWS * row + 7 * MAX_ROWS * row + zero_rows * row
    )
    assert pp_buffered.hbm_reserve_bytes(MAX_ROWS, _template(True), 3, zero_rows) == (
        (3 + 1) * MAX_ROWS * (row + side_row) + 7 * MAX_ROWS * row + zero_rows * row
    )
    # without a residual in the template there is no zero residual to hold
    hidden_only = {"hidden_states": ((HIDDEN,), torch.float32)}
    assert pp_buffered.hbm_reserve_bytes(MAX_ROWS, hidden_only, 1, zero_rows) == (
        (1 + 1 + 7) * MAX_ROWS * row
    )


@pytest.mark.parametrize(
    ("stages", "depth", "carried", "posted"),
    [
        # the last of two stages holds only the step it runs
        (2, 50, 1, 1),
        (2, 50, 2, 2),
        # no more than the six steps the engine keeps in flight: four stages
        # and two more
        (4, 50, 1, 6),
        # a stage that sends: the step it begins and ``depth`` unsent forwards
        (8, 2, 1, 3),
        (8, 2, 2, 6),
        (8, 6, 1, 7),
        (8, 7, 1, 8),
        (4, 3, 2, 8),
    ],
)
def test_receives_a_device_can_have_posted(stages, depth, carried, posted):
    assert pp_buffered.buffered_receives_posted(stages, depth, carried) == posted


def test_a_handoff_deeper_than_the_receive_flags_is_refused(monkeypatch):
    with pytest.raises(ValueError, match="at most 8 receives can be outstanding"):
        _build(monkeypatch, stages=4, side=True, depth=3)
    _build(monkeypatch, stages=4, side=True, depth=2)
    _build(monkeypatch, stages=2, side=True, depth=10)


def test_rows_beyond_the_hand_off_are_refused(monkeypatch):
    _, handoffs = _build(monkeypatch, stages=2)
    with pytest.raises(RuntimeError, match="1 to 64"):
        handoffs[1, 0].begin_step([MAX_ROWS + 1])


def test_a_forward_that_drops_a_side_tensor_is_refused(monkeypatch):
    _, handoffs = _build(monkeypatch, stages=3, side=True)
    first = handoffs[0, 0]
    first.start_forward(8)
    with pytest.raises(RuntimeError, match=f"no \\['{SIDE}'\\]"):
        first.end_forward(_outgoing(8), 8)


def test_push_settle_and_step_do_nothing(monkeypatch):
    transport, handoffs = _build(monkeypatch, stages=3)
    for handoff in handoffs.values():
        with handoff.step():
            handoff.push()
            handoff.settle()
        handoff.close()
        assert handoff.transfers == 0
        assert handoff.drain_before_handoff is False
    assert transport.sends == [] and transport.posted == {}


def test_the_first_and_last_stage_post_and_send_only_what_they_must(monkeypatch):
    transport, handoffs = _build(monkeypatch, stages=3)
    assert (handoffs[0, 0].prev, handoffs[0, 0].next) == (None, 1)
    assert (handoffs[1, 0].prev, handoffs[1, 0].next) == (0, 2)
    assert (handoffs[2, 0].prev, handoffs[2, 0].next) == (1, None)
    handoffs[0, 0].begin_step([16])
    assert transport.posted == {}


def test_tags_differ_per_transfer_and_tensor():
    handoff = SimpleNamespace(carried=("hidden_states", SIDE))
    tags = {
        pp_buffered.PPBuffered._tag(handoff, transfer, index)
        for transfer in range(1, 100)
        for index in range(2)
    }
    assert len(tags) == 198
    assert pp_buffered.PPBuffered._tag(handoff, pp_buffered._TAG_SPAN, 0) == 0


def test_the_hand_off_always_carries_hidden_states_and_residual(monkeypatch):
    _build(monkeypatch, stages=2)
    with pytest.raises(ValueError, match="hidden_states and residual"):
        pp_buffered.PPBuffered(
            torch.device("cpu"),
            MAX_ROWS,
            {
                "hidden_states": ((HIDDEN,), torch.float32),
                "other": ((HIDDEN,), torch.float32),
            },
        )
    with pytest.raises(ValueError, match="depth of at least 1"):
        pp_buffered.PPBuffered(torch.device("cpu"), MAX_ROWS, _template(False), depth=0)
