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
"""The TCP hand-off between two TPU slices, both ends in one process over
loopback, on CPU.

The device is the CPU, where an upload copies at once; a test that makes
the upload return its host buffer itself shows which buffer each received
tensor reads until its forward has completed.
"""

import socket
import threading
from types import SimpleNamespace

import pytest
import torch

from tests.distributed.store_test_utils import TIMEOUT, FakeStore
from vllm_torchtpu.distributed import pp_dcn_link

# Odd, so the bytes of a row split unevenly across four streams.
HIDDEN = 7
SIDE_WIDTH = 3
SRC, DST = 3, 4
CPU = torch.device("cpu")


@pytest.fixture
def links(monkeypatch):
    """Builds the two ends of links over loopback and closes them after the
    test."""
    monkeypatch.setattr(pp_dcn_link, "get_ip", lambda: "127.0.0.1")
    built = []

    def pair(specs, streams, store=None):
        monkeypatch.setenv("TPU_PP_DCN_STREAMS", str(streams))
        store = FakeStore() if store is None else store
        receiver = pp_dcn_link.DcnLink(store, SRC, DST, DST, specs, CPU)
        sender = pp_dcn_link.DcnLink(store, SRC, DST, SRC, specs, CPU)
        built.extend((sender, receiver))
        return sender, receiver

    yield SimpleNamespace(pair=pair)
    for link in built:
        link.close()


def _in_thread(target, *args):
    """Starts ``target`` in a thread; ``outcome`` gets its result or error."""
    outcome = {}

    def run():
        try:
            outcome["result"] = target(*args)
        except BaseException as exc:  # reported by the test thread
            outcome["error"] = exc

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, outcome


def _finished(thread, outcome):
    thread.join(TIMEOUT)
    assert not thread.is_alive(), "the other end never finished"
    assert "error" not in outcome, outcome
    return outcome.get("result")


def _specs(dtype):
    """Activations of ``dtype`` and an int32 side tensor."""
    return (((HIDDEN,), dtype), ((SIDE_WIDTH,), torch.int32))


def _transfer(generator, rows, dtype):
    return (
        torch.randn(rows, HIDDEN, generator=generator).to(dtype),
        torch.randint(
            -1, 1 << 20, (rows, SIDE_WIDTH), dtype=torch.int32, generator=generator
        ),
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("streams", [1, 4])
def test_transfers_arrive_whole_and_in_order(links, streams, dtype):
    sender, receiver = links.pair(_specs(dtype), streams)
    generator = torch.Generator().manual_seed(streams)
    rows = [8, 64, 8, 1, 3, 64]
    sent = [_transfer(generator, n, dtype) for n in rows]
    thread, outcome = _in_thread(
        lambda: [
            sender.send(t, sender.stage(tensors)) for t, tensors in enumerate(sent, 1)
        ]
    )
    received = [receiver.receive(t, n) for t, n in enumerate(rows, 1)]
    _finished(thread, outcome)
    for got, want in zip(received, sent):
        assert [(x.dtype, x.shape) for x in got] == [(x.dtype, x.shape) for x in want]
        assert all(torch.equal(g, w) for g, w in zip(got, want))
    # Every transfer lands in tensors of its own.
    assert received[0][0].data_ptr() != received[2][0].data_ptr()


def _aliased_uploads(monkeypatch):
    """Makes an upload return its host buffer itself, so a received tensor
    reads the buffer as an upload in progress does."""
    monkeypatch.setattr(pp_dcn_link, "_start_to_device", lambda host, device: host)


def test_receives_ahead_of_their_forwards_hold_their_own_buffers(links, monkeypatch):
    _aliased_uploads(monkeypatch)
    sender, receiver = links.pair(_specs(torch.float32), streams=2)
    generator = torch.Generator().manual_seed(0)
    sent = [_transfer(generator, 8, torch.float32) for _ in range(3)]
    thread, outcome = _in_thread(
        lambda: [
            sender.send(t, sender.stage(tensors)) for t, tensors in enumerate(sent, 1)
        ]
    )
    first = receiver.receive(1, 8)
    second = receiver.receive(2, 8)
    # Two receives before either forward has completed: each fills buffers
    # of its own, and both keep their data.
    assert {x.data_ptr() for x in first}.isdisjoint(x.data_ptr() for x in second)
    for got, want in zip((first, second), sent):
        assert all(torch.equal(g, w) for g, w in zip(got, want))
    # Once transfer 1's forward has completed, the next receive takes its
    # buffers.
    receiver.landed(1)
    third = receiver.receive(3, 8)
    _finished(thread, outcome)
    assert all(x is y for x, y in zip(third, first))
    assert all(torch.equal(g, w) for g, w in zip(third, sent[2]))
    assert all(torch.equal(g, w) for g, w in zip(second, sent[1]))


def test_landed_gives_back_only_the_transfers_it_covers(links, monkeypatch):
    _aliased_uploads(monkeypatch)
    sender, receiver = links.pair(_specs(torch.float32), streams=1)
    # Nothing received yet: nothing to give back.
    receiver.landed(3)
    generator = torch.Generator().manual_seed(0)
    sent = [_transfer(generator, 8, torch.float32) for _ in range(2)]
    thread, outcome = _in_thread(
        lambda: [
            sender.send(t, sender.stage(tensors)) for t, tensors in enumerate(sent, 1)
        ]
    )
    first = receiver.receive(1, 8)
    receiver.landed(0)
    second = receiver.receive(2, 8)
    _finished(thread, outcome)
    assert not any(x is y for x, y in zip(second, first))
    assert all(torch.equal(g, w) for g, w in zip(first, sent[0]))


def test_a_send_returns_only_once_the_receiver_has_read(links):
    width = 1 << 16
    sender, receiver = links.pair((((width,), torch.float32),), streams=2)
    # The first transfer opens the connections; capping their socket buffers
    # then leaves the kernel no room for most of the next one.
    thread, outcome = _in_thread(sender.send, 1, sender.stage((torch.ones(1, width),)))
    receiver.receive(1, 1)
    _finished(thread, outcome)
    for link, option in ((sender, socket.SO_SNDBUF), (receiver, socket.SO_RCVBUF)):
        for connection in link._connections:
            connection.sock.setsockopt(socket.SOL_SOCKET, option, 1 << 16)
    large = torch.arange(128 * width, dtype=torch.float32).reshape(128, width)
    thread, outcome = _in_thread(sender.send, 2, sender.stage((large,)))
    thread.join(0.5)
    assert thread.is_alive(), "a 32 MiB send returned before anything read it"
    (got,) = receiver.receive(2, 128)
    _finished(thread, outcome)
    assert torch.equal(got, large)


@pytest.mark.parametrize(
    ("transfer", "rows", "message"),
    [
        (2, 8, "expects transfer 2, .*sending end sent transfer 1,"),
        (1, 16, "expects transfer 1, tensor 0 of 16 rows.*sent transfer 1, tensor 0 "),
    ],
    ids=["transfer", "rows"],
)
def test_a_receive_the_sender_does_not_match_fails_naming_both(
    links, transfer, rows, message
):
    sender, receiver = links.pair(_specs(torch.float32), streams=4)
    generator = torch.Generator().manual_seed(0)
    staged = sender.stage(_transfer(generator, 8, torch.float32))
    thread, _ = _in_thread(sender.send, 1, staged)
    with pytest.raises(RuntimeError, match=message):
        receiver.receive(transfer, rows)
    # A failed link stays failed: nothing after it would be in step.
    with pytest.raises(RuntimeError, match="has failed"):
        receiver.receive(1, 8)
    thread.join(TIMEOUT)
    assert not thread.is_alive()


def test_a_closed_sender_fails_the_receiver(links):
    sender, receiver = links.pair(_specs(torch.float32), streams=4)
    generator = torch.Generator().manual_seed(0)
    staged = sender.stage(_transfer(generator, 8, torch.float32))
    thread, outcome = _in_thread(sender.send, 1, staged)
    receiver.receive(1, 8)
    _finished(thread, outcome)
    sender.close()
    with pytest.raises(RuntimeError, match="closed the connection"):
        receiver.receive(2, 8)


def test_a_closed_receiver_fails_the_sender(links):
    width = 1 << 16
    sender, receiver = links.pair((((width,), torch.float32),), streams=2)
    thread, outcome = _in_thread(sender.send, 1, sender.stage((torch.ones(1, width),)))
    receiver.receive(1, 1)
    _finished(thread, outcome)
    receiver.close()
    # More than the socket buffers hold, so the writes reach the closed end.
    with pytest.raises(RuntimeError, match="PP DCN link 3 -> 4, stream"):
        sender.send(2, sender.stage((torch.ones(128, width),)))
    with pytest.raises(RuntimeError, match="has failed"):
        sender.stage((torch.ones(1, width),))


def test_staging_a_forward_of_another_shape_is_refused(links):
    sender, receiver = links.pair(_specs(torch.float32), streams=1)
    wrong = (torch.zeros(8, HIDDEN), torch.zeros(8, SIDE_WIDTH))
    with pytest.raises(RuntimeError, match="tensor 1 is .*the link carries"):
        sender.stage(wrong)
    with pytest.raises(RuntimeError, match="hands on 1 tensors; the link carries 2"):
        links.pair(_specs(torch.float32), streams=1)[0].stage(wrong[:1])


def test_staged_transfers_hold_their_own_buffers_until_written(links):
    sender, receiver = links.pair(_specs(torch.float32), streams=2)
    generator = torch.Generator().manual_seed(0)
    first, second = (_transfer(generator, 8, torch.float32) for _ in range(2))
    # Two forwards staged before either is written, as end_forward does when
    # the send thread is behind.
    staged = [sender.stage(first), sender.stage(second)]
    buffers = [{host.data_ptr() for host in hosts} for hosts in staged]
    assert buffers[0].isdisjoint(buffers[1])
    # The copies to the host were started into the staged buffers.
    for hosts, tensors in zip(staged, (first, second)):
        assert all(torch.equal(h, x) for h, x in zip(hosts, tensors))
    # The link holds each forward's tensors while their copies are pending.
    held = {id(x) for x in sender._sources.values()}
    assert held == {id(x) for x in (*first, *second)}
    thread, outcome = _in_thread(
        lambda: [sender.send(t, hosts) for t, hosts in enumerate(staged, 1)]
    )
    received = [receiver.receive(1, 8), receiver.receive(2, 8)]
    _finished(thread, outcome)
    assert not sender._sources
    for got, want in zip(received, (first, second)):
        assert all(torch.equal(g, w) for g, w in zip(got, want))
    # Written buffers are free again, so a later transfer takes one of them.
    third = sender.stage(_transfer(generator, 8, torch.float32))
    assert {host.data_ptr() for host in third} <= buffers[0] | buffers[1]


def test_only_the_receiving_end_publishes_an_address_under_its_rank(links):
    store = FakeStore()
    links.pair(_specs(torch.float32), streams=1, store=store)
    (key,) = store.values
    assert key == f"tpu_pp_dcn_link_{DST}"
    host, port = store.get(key).decode().split(":")
    assert host == "127.0.0.1" and int(port) > 0


def test_a_link_needs_a_stream_and_must_be_one_of_its_ends(monkeypatch):
    monkeypatch.setattr(pp_dcn_link, "get_ip", lambda: "127.0.0.1")
    monkeypatch.setenv("TPU_PP_DCN_STREAMS", "0")
    with pytest.raises(ValueError, match="TPU_PP_DCN_STREAMS must be at least 1"):
        pp_dcn_link.DcnLink(FakeStore(), SRC, DST, DST, _specs(torch.float32), CPU)
    monkeypatch.setenv("TPU_PP_DCN_STREAMS", "1")
    with pytest.raises(ValueError, match="neither end of the link 3 -> 4"):
        pp_dcn_link.DcnLink(FakeStore(), SRC, DST, 5, _specs(torch.float32), CPU)


def _failed(thread, outcome):
    thread.join(TIMEOUT)
    assert not thread.is_alive(), "the receive never returned"
    assert "error" in outcome, outcome


def test_closing_wakes_a_receive_that_waits_for_the_sending_end(links):
    sender, receiver = links.pair(_specs(torch.float32), streams=2)
    thread, outcome = _in_thread(receiver.receive, 1, 8)
    thread.join(0.3)
    # waiting in accept: the sending end has not connected
    assert thread.is_alive()
    receiver.close()
    _failed(thread, outcome)


def test_closing_wakes_a_receive_that_waits_for_data(links):
    sender, receiver = links.pair(_specs(torch.float32), streams=2)
    generator = torch.Generator().manual_seed(0)
    staged = sender.stage(_transfer(generator, 8, torch.float32))
    thread, outcome = _in_thread(sender.send, 1, staged)
    receiver.receive(1, 8)
    _finished(thread, outcome)
    thread, outcome = _in_thread(receiver.receive, 2, 8)
    thread.join(0.3)
    # connected, and nothing has been sent for transfer 2
    assert thread.is_alive()
    receiver.close()
    _failed(thread, outcome)


def test_a_closed_stream_fails_a_job_instead_of_queueing_it():
    ours, theirs = socket.socketpair()
    stream = pp_dcn_link._Stream(ours, "pp-dcn-test")
    stream.close()
    ran = []
    future = stream.submit(lambda: ran.append(True))
    with pytest.raises(ConnectionError, match="closed"):
        future.result(TIMEOUT)
    assert not ran
    stream.close()
    theirs.close()
