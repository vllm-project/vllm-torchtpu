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
"""W7 P1b: empty scheduler steps take no batch-queue slot under the ring."""

import contextlib
import dataclasses
import queue
from collections import deque
from types import SimpleNamespace

import pytest

from vllm_torchtpu.core import pp_ring_step
from vllm_torchtpu.core.pp_ring_step import EmptyStepGrace, install_empty_step_grace


@dataclasses.dataclass
class _Out:
    """The SchedulerOutput fields the step and _carries_state read."""

    total_num_scheduled_tokens: int = 0
    num_scheduled_tokens: dict = dataclasses.field(default_factory=dict)
    scheduled_new_reqs: list = dataclasses.field(default_factory=list)
    scheduled_cached_reqs: object = dataclasses.field(
        default_factory=lambda: SimpleNamespace(req_ids=[])
    )
    finished_req_ids: set = dataclasses.field(default_factory=set)
    num_common_prefix_blocks: list = dataclasses.field(default_factory=list)
    pending_structured_output_tokens: bool = False
    num_spec_tokens_to_schedule: int = 0
    kv_connector_metadata: object = None
    tag: str = ""


def _real(tag):
    return _Out(total_num_scheduled_tokens=8704, num_scheduled_tokens={tag: 8704}, tag=tag)


class _Future:
    def __init__(self, value):
        self.value = value

    def result(self):
        return self.value


class _Scheduler:
    def __init__(self, outputs, requests=1):
        self.outputs = deque(outputs)
        self.requests = {str(i): None for i in range(requests)}
        self.updated = []

    def has_requests(self):
        return bool(self.requests)

    def schedule(self, throttle=False):
        return self.outputs.popleft()

    def get_grammar_bitmask(self, out):
        return None

    def update_from_output(self, out, model_output):
        self.updated.append(out.tag or "empty")
        return {0: out.tag}


class _Executor:
    def __init__(self):
        self.dispatched = []

    def execute_model(self, out, non_block=False):
        self.dispatched.append(out.tag or "empty")
        return _Future("exec:" + (out.tag or "empty"))

    def sample_tokens(self, grammar, non_block=False):
        return _Future("sampled")


class _Clock:
    def __init__(self):
        self.t = 100.0
        self.events = []  # (time, callable)

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s
        for when, fn in list(self.events):
            if self.t >= when:
                fn()
                self.events.remove((when, fn))


def _core(outputs, size=5, requests=1):
    @contextlib.contextmanager
    def _ctx(*a, **k):
        yield None

    return SimpleNamespace(
        batch_queue=deque(maxlen=size),
        batch_queue_size=size,
        scheduler=_Scheduler(outputs, requests),
        model_executor=_Executor(),
        input_queue=queue.Queue(),
        is_mm_encoder_only=False,
        is_pooling_model=False,
        check_for_draft_tokens=False,
        _should_throttle_prefills=lambda: False,
        log_error_detail=_ctx,
        capture_iteration_details=_ctx,
        _process_aborts_queue=lambda: None,
        _attach_iteration_details=lambda *a: None,
    )


def _stepper(core, grace_s=0.025):
    clock = _Clock()
    return EmptyStepGrace(core, grace_s, clock=clock, sleep=clock.sleep), clock


def test_real_steps_are_queued_and_the_full_queue_blocks_like_vllm():
    core = _core([_real(f"s{i}") for i in range(6)])
    st, _ = _stepper(core)
    for i in range(4):
        assert st.step() == (None, True)
    # The fifth fills the queue: vLLM's path blocks on the oldest.
    out, executed = st.step()
    assert executed and out == {0: "s0"}
    assert core.model_executor.dispatched == ["s0", "s1", "s2", "s3", "s4"]
    assert [o.tag for _, o, _ in core.batch_queue] == ["s4", "s3", "s2", "s1"]


def test_empty_step_takes_no_slot_and_blocks_on_the_oldest_without_new_input():
    core = _core([_real("s0"), _Out()])
    st, clock = _stepper(core)
    assert st.step() == (None, True)
    st.last_arrival = clock() - 10.0  # the last request arrived long ago
    out, executed = st.step()
    assert out == {0: "s0"} and not executed
    assert core.model_executor.dispatched == ["s0"]  # the empty step was never sent
    assert core.scheduler.updated == ["s0"] and not core.batch_queue
    assert st.dropped == 1


def test_empty_step_yields_to_input_that_arrives_within_the_grace():
    core = _core([_real("s0"), _Out()])
    st, clock = _stepper(core, grace_s=0.025)
    assert st.step() == (None, True)
    core.scheduler.requests["new"] = None  # a request reached the engine just now
    clock.events.append((clock() + 0.004, lambda: core.input_queue.put("req")))
    assert st.step() == (None, False)
    assert core.scheduler.updated == []  # nothing was waited for
    assert [o.tag for _, o, _ in core.batch_queue] == ["s0"]
    assert core.model_executor.dispatched == ["s0"] and st.yielded == 1
    assert clock() - 100.0 < 0.01


def test_empty_step_waits_out_the_grace_then_blocks_on_the_oldest():
    core = _core([_real("s0"), _Out()])
    st, clock = _stepper(core, grace_s=0.025)
    assert st.step() == (None, True)
    core.scheduler.requests["new"] = None
    out, _ = st.step()
    assert out == {0: "s0"} and core.scheduler.updated == ["s0"]
    assert clock() - 100.0 >= 0.025
    assert core.model_executor.dispatched == ["s0"]


def test_an_empty_step_that_carries_state_is_queued_like_vllm():
    core = _core([_real("s0"), _Out(finished_req_ids={"done"}, tag="fin")])
    st, clock = _stepper(core)
    assert st.step() == (None, True)
    st.last_arrival = clock() - 10.0
    assert st.step() == (None, False)
    assert core.model_executor.dispatched == ["s0", "fin"]
    assert [o.tag for _, o, _ in core.batch_queue] == ["fin", "s0"]
    assert st.kept == 1


def test_the_draft_length_on_every_output_does_not_make_an_empty_step_stateful():
    # vLLM 0.31 sets num_spec_tokens_to_schedule on every output (1 with MTP):
    # l5p1b ran with P1b inert because this field counted as state.
    core = _core([_real("s0"), _Out(num_spec_tokens_to_schedule=1)])
    st, clock = _stepper(core)
    assert st.step() == (None, True)
    st.last_arrival = clock() - 10.0
    out, _ = st.step()
    assert out == {0: "s0"} and core.model_executor.dispatched == ["s0"]
    assert st.dropped == 1 and st.kept == 0


def test_empty_tpu_connector_metadata_leaves_an_empty_step_stateless():
    # Serving caveat: a KV connector sets metadata on every output.
    from vllm_torchtpu.distributed.kv_transfer.connector_metadata import TPUConnectorMetadata

    core = _core([_real("s0"), _Out(kv_connector_metadata=TPUConnectorMetadata())])
    st, clock = _stepper(core)
    assert st.step() == (None, True)
    st.last_arrival = clock() - 10.0
    out, _ = st.step()
    assert out == {0: "s0"} and core.model_executor.dispatched == ["s0"]
    assert st.dropped == 1


def test_tpu_connector_metadata_with_work_keeps_the_empty_step():
    from vllm_torchtpu.distributed.kv_transfer.connector_metadata import TPUConnectorMetadata

    meta = TPUConnectorMetadata(reqs_to_release={"r": 7})
    core = _core([_real("s0"), _Out(kv_connector_metadata=meta)])
    st, clock = _stepper(core)
    assert st.step() == (None, True)
    st.last_arrival = clock() - 10.0
    assert st.step() == (None, False)
    assert core.model_executor.dispatched == ["s0", "empty"] and st.kept == 1


def test_an_empty_step_with_nothing_in_flight_goes_through_vllms_path():
    core = _core([_Out()])
    st, _ = _stepper(core)
    assert st.step() == (None, False)
    assert core.model_executor.dispatched == ["empty"] and len(core.batch_queue) == 1


@pytest.mark.parametrize(
    "value,handoff,queue_present,want",
    [("0", "ring", True, False), ("25", "wave", True, False), ("25", "ring", False, False),
     ("25", "ring", True, True)],
)
def test_install_needs_the_flag_the_ring_and_a_batch_queue(monkeypatch, value, handoff, queue_present, want):
    monkeypatch.setenv("TPU_PP_EMPTY_STEP_GRACE_MS", value)
    # setenv, not setattr on envs: envs resolves through os.environ, and a setattr
    # would leave a module attribute behind that shadows it for later tests.
    monkeypatch.setenv("TPU_PP_HANDOFF", handoff)
    core = _core([])
    if not queue_present:
        core.batch_queue = None
    core.step_fn = "vllm"
    assert install_empty_step_grace(core) is want
    if want:
        assert core.step_fn == core._tpu_empty_step_grace.step
        assert core._tpu_empty_step_grace.grace_s == pytest.approx(0.025)
    else:
        assert core.step_fn == "vllm"


def test_the_reimplemented_vllm_step_is_the_installed_one():
    # A vLLM upgrade that changes step_with_batch_queue must fail here, not
    # silently run an outdated copy (install refuses on a mismatch too).
    assert pp_ring_step.vllm_step_matches()


def test_grace_rejects_garbage(monkeypatch):
    monkeypatch.setenv("TPU_PP_EMPTY_STEP_GRACE_MS", "soon")
    with pytest.raises(ValueError):
        pp_ring_step.grace_ms()
