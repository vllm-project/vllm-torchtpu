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
"""The pipeline ring's point-to-point hand-off, driven on CPU with a fake
transport.

Every device runs its own event stream in a thread, the way the runner calls
the ring. The fake transport behaves like the runtime's point-to-point path:
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

from vllm_torchtpu.distributed import pp_ring

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
    """One PPRing per device of ``stages`` stages of ``columns`` devices,
    sharing one fake transport; keyed by (stage, column)."""
    from vllm.distributed import parallel_state

    from vllm_torchtpu.distributed import pcp

    transport = transport or _Transport(stride=columns)
    monkeypatch.setattr(torch.distributed, "isend", transport.isend)
    monkeypatch.setattr(torch.distributed, "irecv", transport.irecv)
    monkeypatch.setattr(pp_ring, "_flush", lambda tensors: None)
    monkeypatch.setattr(pp_ring, "_fence", _Fence)
    read = pp_ring._first_and_last

    def first_and_last(x):
        transport.wait_landed(x)
        return read(x)

    monkeypatch.setattr(pp_ring, "_first_and_last", first_and_last)
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
    rings = {}
    for stage in range(stages):
        for column in range(columns):
            current.update(stage=stage, column=column)
            rings[stage, column] = pp_ring.PPRing(
                torch.device("cpu"), MAX_ROWS, _template(side), depth=depth
            )
    return transport, rings


def _forward(ring, transport, received, rows, g, column):
    """One forward: the first stage seeds forward ``g`` with ``g`` plus 100
    per column as hidden states and a residual of 0.5, which every later
    stage must see as part of the sum; later stages add 1. The side tensor
    is seeded by the first stage as a view of a larger table and relayed
    unchanged. Returns what the last stage sees, None elsewhere."""
    side = SIDE in ring.carried
    if ring.rank == 0:
        assert received is None
        h = torch.full((rows, HIDDEN), float(g + 100 * column))
        r = torch.full((rows, HIDDEN), 0.5)
        table = torch.full((MAX_ROWS, SIDE_WIDTH), g, dtype=torch.int32)
        extra = table[:rows]
    else:
        for key in ring.carried:
            transport.wait_landed(received[key])
        assert torch.equal(received["residual"], torch.zeros(rows, HIDDEN))
        h = received["hidden_states"] + 1
        r = received["residual"]
        extra = received[SIDE] if side else None
    if ring.rank == ring.last:
        ring.end_forward(None, rows)
        return h, r, extra
    outgoing = {"hidden_states": h, "residual": r}
    if side:
        outgoing[SIDE] = extra
    ring.end_forward(outgoing, rows)
    return None


def _run_device(ring, transport, steps, results, errors, column):
    """A device's event stream as the runner drives it: each step is the
    padded token count of each of its forwards and whether the runner posts
    its receives first (serving steps) or not (warmup forwards). A step
    without tokens runs nothing."""
    try:
        ring.warmup()
        g = 0
        for rows_per_forward, ahead in steps:
            if not rows_per_forward:
                continue
            if ahead and ring.rank > 0:
                ring.begin_step(rows_per_forward)
            for rows in rows_per_forward:
                received = ring.start_forward(rows)
                out = _forward(ring, transport, received, rows, g, column)
                if out is not None:
                    results.append((column, g, rows, *out))
                g += 1
        ring.close()
    except BaseException as exc:  # reported by the test thread
        errors.append(((ring.rank, column), exc))


def _run(rings, transport, steps, timeout=60):
    results, errors = [], []
    threads = [
        threading.Thread(
            target=_run_device,
            args=(ring, transport, steps, results, errors, column),
            daemon=True,
        )
        for (_, column), ring in rings.items()
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
    transport, rings = _build(monkeypatch, stages, side=side)
    steps = _steps(random.Random(stages), 24)
    results = _run(rings, transport, steps)
    forwards = _check(results, steps, stages - 1, side=side)
    carried = 2 if side else 1
    # The warmup and every forward cross each stage boundary once, with one
    # transfer per carried tensor; nothing is left posted.
    assert len(transport.sends) == (1 + len(forwards)) * (stages - 1) * carried
    assert transport.posted == {}
    assert transport.views_sent == 0
    for (stage, _), ring in rings.items():
        assert ring.transfers == 1 + len(forwards)
        assert ring.forwards == len(forwards)
        assert ring.sent == (len(forwards) if stage < stages - 1 else 0)
        assert ring.posted_through == (1 + len(forwards) if stage > 0 else 0)


def test_columns_hand_off_to_the_same_column(monkeypatch):
    transport, rings = _build(monkeypatch, stages=3, columns=2, side=True)
    steps = _steps(random.Random(5), 12)
    results = _run(rings, transport, steps)
    _check(results, steps, last=2, columns=2, side=True)
    # Global ranks are stage * 2 + column: every transfer goes two ranks on.
    assert {dst - src for (src, dst, _), _ in transport.sends} == {2}


def test_steps_without_tokens_transfer_nothing(monkeypatch):
    transport, rings = _build(monkeypatch, stages=3)
    results = _run(rings, transport, [([], True)] * 5)
    assert results == []
    # the warmup alone
    assert len(transport.sends) == 2
    for ring in rings.values():
        ring.begin_step([])
        assert ring.transfers == 1 and not ring._posted


def _until(predicate, seconds=5.0):
    for _ in range(int(seconds / 0.01)):
        if predicate():
            return True
        threading.Event().wait(0.01)
    return predicate()


@pytest.mark.parametrize("depth", [1, 2])
def test_a_stage_that_runs_ahead_holds_at_most_depth_forwards(monkeypatch, depth):
    transport, rings = _build(monkeypatch, stages=2, side=True, depth=depth)
    first, second = rings[0, 0], rings[1, 0]
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

    transport, rings = _build(monkeypatch, stages=2, transport=Misrouting())
    errors = []

    def warmup(ring):
        try:
            ring.warmup()
        except RuntimeError as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=warmup, args=(r,), daemon=True) for r in rings.values()
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
    transport, rings = _build(monkeypatch, stages=2)
    transport.fail_recv = RuntimeError("cross-host receive unsupported")
    with pytest.raises(RuntimeError, match="self-test failed.*libtpu") as info:
        rings[1, 0].warmup()
    assert "unsupported" in str(info.value.__cause__)


def _outgoing(rows):
    return {
        "hidden_states": torch.zeros(rows, HIDDEN),
        "residual": torch.zeros(rows, HIDDEN),
    }


def test_a_send_failure_surfaces_on_the_next_call(monkeypatch):
    transport, rings = _build(monkeypatch, stages=2)
    first = rings[0, 0]
    transport.fail_send = ValueError("ring down")
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
    _, rings = _build(monkeypatch, stages=2)
    second = rings[1, 0]
    second.begin_step([16])
    with pytest.raises(RuntimeError, match="posted for 16 rows"):
        second.start_forward(8)


def test_receives_posted_for_forwards_that_never_ran_are_reported(monkeypatch):
    _, rings = _build(monkeypatch, stages=2)
    second = rings[1, 0]
    second.begin_step([16, 16])
    second.start_forward(16)
    second.end_forward(None, 16)
    with pytest.raises(RuntimeError, match="1 receives were posted"):
        second.begin_step([8])


def test_rows_beyond_the_hand_off_are_refused(monkeypatch):
    _, rings = _build(monkeypatch, stages=2)
    with pytest.raises(RuntimeError, match="1 to 64"):
        rings[1, 0].begin_step([MAX_ROWS + 1])


def test_a_forward_that_drops_a_side_tensor_is_refused(monkeypatch):
    _, rings = _build(monkeypatch, stages=3, side=True)
    first = rings[0, 0]
    first.start_forward(8)
    with pytest.raises(RuntimeError, match=f"no \\['{SIDE}'\\]"):
        first.end_forward(_outgoing(8), 8)


def test_push_settle_and_step_do_nothing(monkeypatch):
    transport, rings = _build(monkeypatch, stages=3)
    for ring in rings.values():
        with ring.step():
            ring.push()
            ring.settle()
        ring.close()
        assert ring.transfers == 0
        assert ring.drain_before_handoff is False
    assert transport.sends == [] and transport.posted == {}


def test_the_first_and_last_stage_post_and_send_only_what_they_must(monkeypatch):
    transport, rings = _build(monkeypatch, stages=3)
    assert (rings[0, 0].prev, rings[0, 0].next) == (None, 1)
    assert (rings[1, 0].prev, rings[1, 0].next) == (0, 2)
    assert (rings[2, 0].prev, rings[2, 0].next) == (1, None)
    rings[0, 0].begin_step([16])
    assert transport.posted == {}


def test_tags_differ_per_transfer_and_tensor():
    ring = SimpleNamespace(carried=("hidden_states", SIDE))
    tags = {
        pp_ring.PPRing._tag(ring, transfer, index)
        for transfer in range(1, 100)
        for index in range(2)
    }
    assert len(tags) == 198
    assert pp_ring.PPRing._tag(ring, pp_ring._TAG_SPAN, 0) == 0


def test_the_hand_off_always_carries_hidden_states_and_residual(monkeypatch):
    _build(monkeypatch, stages=2)
    with pytest.raises(ValueError, match="hidden_states and residual"):
        pp_ring.PPRing(
            torch.device("cpu"),
            MAX_ROWS,
            {
                "hidden_states": ((HIDDEN,), torch.float32),
                "other": ((HIDDEN,), torch.float32),
            },
        )
    with pytest.raises(ValueError, match="depth of at least 1"):
        pp_ring.PPRing(torch.device("cpu"), MAX_ROWS, _template(False), depth=0)
