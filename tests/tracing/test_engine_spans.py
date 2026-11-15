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
"""kv_fetch and xla_compile spans emitted from EngineCore and TPU workers."""

import datetime
import json
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from opentelemetry import trace
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    SupportsHMA,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.metrics.stats import (
    KV_FETCH_COMPLETED_WAITING,
    KV_FETCH_IN_PROGRESS,
    KV_FETCH_WAITING_TO_START,
)
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from vllm_torchtpu.tracing import engine_spans

BLOCK_SIZE = 16
TRACE_ID = 0x90D86CB618C6615E4DD701E734A854A7
PARENT_SPAN_ID = 0x0123456789ABCDEF
TRACEPARENT = f"00-{TRACE_ID:032x}-{PARENT_SPAN_ID:016x}-01"


class _Clock:
    def __init__(self, start_s: float = 1_000.0) -> None:
        self.now_s = start_s

    def time(self) -> float:
        return self.now_s

    def time_ns(self) -> int:
        return int(round(self.now_s * 1e9))


@pytest.fixture
def clock(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(engine_spans.time, "time", clock.time)
    monkeypatch.setattr(engine_spans.time, "time_ns", clock.time_ns)
    return clock


@pytest.fixture
def spans(monkeypatch):
    emitted: list[dict[str, Any]] = []

    def capture(name, start_ns, end_ns, attributes, context):
        emitted.append(
            {
                "name": name,
                "start_ns": start_ns,
                "end_ns": end_ns,
                "attributes": attributes,
                "context": context,
            }
        )

    monkeypatch.setattr(engine_spans, "_emit_span", capture)
    return emitted


def _parent(context) -> tuple[int, int]:
    span_context = trace.get_current_span(context).get_span_context()
    return span_context.trace_id, span_context.span_id


def _stage_attr(stage: str) -> str:
    return f"vllm.kv_fetch.{stage}_s"


# --- KvFetchSpans on a stub scheduler ---------------------------------------


def _stub_scheduler():
    calls: list[tuple[str, str | None]] = []
    scheduler = SimpleNamespace(
        failed_recving_kv_req_ids=set(),
        _set_kv_fetch_stage=lambda req, stage: calls.append((req.request_id, stage)),
    )
    return scheduler, calls


def _stub_request(
    req_id="req-1", *, headers=None, finished=False, num_computed_tokens=64
):
    return SimpleNamespace(
        request_id=req_id,
        trace_headers={"traceparent": TRACEPARENT} if headers is None else headers,
        num_computed_tokens=num_computed_tokens,
        is_finished=lambda: finished,
    )


def test_stage_timeline_becomes_one_span_parented_on_the_request(clock, spans):
    scheduler, calls = _stub_scheduler()
    tracker = engine_spans.KvFetchSpans(scheduler)
    request = _stub_request()

    tracker.set_kv_fetch_stage(request, KV_FETCH_WAITING_TO_START)
    clock.now_s += 0.010
    # Re-marked on a later pass while still waiting: the stage keeps its start.
    tracker.set_kv_fetch_stage(request, KV_FETCH_WAITING_TO_START)
    clock.now_s += 0.015
    tracker.set_kv_fetch_stage(request, KV_FETCH_IN_PROGRESS)
    clock.now_s += 0.200
    tracker.set_kv_fetch_stage(request, KV_FETCH_COMPLETED_WAITING)
    clock.now_s += 0.030
    tracker.set_kv_fetch_stage(request, None)

    # The original method still sees every call.
    assert [stage for _, stage in calls] == [
        KV_FETCH_WAITING_TO_START,
        KV_FETCH_WAITING_TO_START,
        KV_FETCH_IN_PROGRESS,
        KV_FETCH_COMPLETED_WAITING,
        None,
    ]
    (span,) = spans
    assert span["name"] == "kv_fetch"
    assert span["start_ns"] == 1_000_000_000_000
    assert span["end_ns"] - span["start_ns"] == pytest.approx(255_000_000, abs=10)
    attrs = span["attributes"]
    assert attrs[_stage_attr(KV_FETCH_WAITING_TO_START)] == pytest.approx(0.025)
    assert attrs[_stage_attr(KV_FETCH_IN_PROGRESS)] == pytest.approx(0.200)
    assert attrs[_stage_attr(KV_FETCH_COMPLETED_WAITING)] == pytest.approx(0.030)
    assert attrs["vllm.kv_fetch.outcome"] == "scheduled"
    assert attrs["vllm.kv_fetch.load_failed"] is False
    assert attrs["vllm.kv_fetch.num_tokens"] == 64
    assert attrs["vllm.request_id"] == "req-1"
    assert "gen_ai.request.id" not in attrs
    assert _parent(span["context"]) == (TRACE_ID, PARENT_SPAN_ID)
    assert tracker.fetches == {}


def test_no_span_when_the_load_never_started(clock, spans):
    scheduler, _ = _stub_scheduler()
    tracker = engine_spans.KvFetchSpans(scheduler)
    request = _stub_request()

    tracker.set_kv_fetch_stage(request, KV_FETCH_WAITING_TO_START)
    clock.now_s += 0.050
    tracker.set_kv_fetch_stage(request, None)
    # A request without an async load is marked None on admission.
    tracker.set_kv_fetch_stage(_stub_request("req-2"), None)

    assert spans == []
    assert tracker.fetches == {}


@pytest.mark.parametrize("headers", [{}, {"tracestate": "k=v"}])
def test_no_span_without_a_propagated_parent(clock, spans, headers):
    scheduler, _ = _stub_scheduler()
    tracker = engine_spans.KvFetchSpans(scheduler)
    request = _stub_request(headers=headers)

    tracker.set_kv_fetch_stage(request, KV_FETCH_IN_PROGRESS)
    tracker.set_kv_fetch_stage(request, None)

    assert spans == []
    assert tracker.fetches == {}


def test_failed_load_and_abort_are_recorded(clock, spans):
    scheduler, _ = _stub_scheduler()
    tracker = engine_spans.KvFetchSpans(scheduler)
    request = _stub_request(finished=True)

    tracker.set_kv_fetch_stage(request, KV_FETCH_IN_PROGRESS)
    clock.now_s += 1.0
    scheduler.failed_recving_kv_req_ids.add("req-1")
    tracker.set_kv_fetch_stage(request, KV_FETCH_COMPLETED_WAITING)
    tracker.set_kv_fetch_stage(request, None)

    (span,) = spans
    assert span["attributes"]["vllm.kv_fetch.load_failed"] is True
    assert span["attributes"]["vllm.kv_fetch.outcome"] == "finished"


def test_span_errors_never_reach_the_scheduler(clock, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("exporter down")

    monkeypatch.setattr(engine_spans, "_emit_span", boom)
    scheduler, calls = _stub_scheduler()
    tracker = engine_spans.KvFetchSpans(scheduler)
    request = _stub_request()

    tracker.set_kv_fetch_stage(request, KV_FETCH_IN_PROGRESS)
    tracker.set_kv_fetch_stage(request, None)
    tracker.set_kv_fetch_stage(request, KV_FETCH_IN_PROGRESS)

    assert len(calls) == 3


# --- install_kv_fetch_spans gating -------------------------------------------


def _config(endpoint):
    return SimpleNamespace(
        observability_config=SimpleNamespace(otlp_traces_endpoint=endpoint)
    )


def _gated_scheduler():
    scheduler = SimpleNamespace(connector=object())
    scheduler._set_kv_fetch_stage = lambda req, stage: None
    return scheduler


def test_install_is_a_noop_without_an_endpoint_or_connector(monkeypatch):
    monkeypatch.setattr(
        engine_spans,
        "ensure_tracer_provider",
        lambda *a: pytest.fail("tracer must not be initialized"),
    )
    assert (
        engine_spans.install_kv_fetch_spans(_gated_scheduler(), _config(None)) is None
    )

    no_connector = _gated_scheduler()
    no_connector.connector = None
    assert (
        engine_spans.install_kv_fetch_spans(no_connector, _config("http://otel:4317"))
        is None
    )


def test_install_requires_a_working_tracer(monkeypatch):
    monkeypatch.setattr(engine_spans, "ensure_tracer_provider", lambda *a: False)
    scheduler = _gated_scheduler()
    original = scheduler._set_kv_fetch_stage

    assert engine_spans.install_kv_fetch_spans(scheduler, _config("x")) is None
    assert scheduler._set_kv_fetch_stage is original


def test_install_wraps_the_instance(monkeypatch):
    monkeypatch.setattr(engine_spans, "ensure_tracer_provider", lambda *a: True)
    scheduler = _gated_scheduler()

    tracker = engine_spans.install_kv_fetch_spans(scheduler, _config("x"))

    assert tracker is not None
    assert scheduler._set_kv_fetch_stage == tracker.set_kv_fetch_stage
    assert scheduler._tpu_kv_fetch_spans is tracker


def test_patch_is_idempotent(monkeypatch):
    # Patch from a clean class and restore it afterwards (undo deletes the
    # marker again), so later tests build unpatched schedulers.
    monkeypatch.setattr(Scheduler, "__init__", Scheduler.__init__)
    monkeypatch.setattr(Scheduler, "_tpu_kv_fetch_spans_patch", False, raising=False)
    engine_spans._patch_vllm_kv_fetch_spans()
    init = Scheduler.__init__
    engine_spans._patch_vllm_kv_fetch_spans()
    assert Scheduler.__init__ is init
    assert Scheduler.__dict__["_tpu_kv_fetch_spans_patch"] is True


# --- Real vLLM scheduler driving the stage transitions -----------------------


class _AsyncLoadConnector(KVConnectorBase_V1, SupportsHMA):
    """Consumer connector whose first match for a request loads asynchronously."""

    def __init__(self):
        self.consumed: set[str] = set()

    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        if request.request_id in self.consumed:
            return 0, False
        return BLOCK_SIZE, True

    def update_state_after_alloc(
        self, request: Request, blocks: Any, num_external_tokens: int
    ) -> None:
        if num_external_tokens > 0:
            self.consumed.add(request.request_id)

    def get_loaded_kv_cache_group_ids(self, request: Request) -> tuple[int, ...]:
        return (0,)

    def build_connector_meta(self, scheduler_output: Any) -> KVConnectorMetadata:
        return KVConnectorMetadata()

    def start_load_kv(self, forward_context: Any, **kwargs: Any) -> None:
        return None

    def wait_for_layer_load(self, layer_name: str) -> None:
        return None

    def save_kv_layer(
        self, layer_name: str, kv_layer: torch.Tensor, attn_metadata: Any, **kw: Any
    ) -> None:
        return None

    def wait_for_save(self) -> None:
        return None

    def request_finished_all_groups(
        self, request: Request, block_ids: tuple[list[int], ...]
    ) -> tuple[bool, dict[str, Any] | None]:
        return False, None


@pytest.fixture(scope="module")
def shared_model_config(tmp_path_factory):
    model_dir = tmp_path_factory.mktemp("opt_kv_fetch_config")
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["OPTForCausalLM"],
                "model_type": "opt",
                "hidden_size": 64,
                "num_attention_heads": 2,
                "num_hidden_layers": 2,
                "ffn_dim": 128,
                "max_position_embeddings": 2048,
                "vocab_size": 1000,
            }
        )
    )
    return ModelConfig(
        model=str(model_dir),
        tokenizer=str(model_dir),
        skip_tokenizer_init=True,
        dtype="float16",
        seed=42,
        max_model_len=2048,
    )


@pytest.fixture(params=[Scheduler, AsyncScheduler])
def scheduler_cls(request):
    return request.param


def _build_decode_scheduler(model_config, scheduler_cls, num_blocks=16):
    vllm_config = VllmConfig(
        scheduler_config=SchedulerConfig(
            max_num_seqs=8,
            max_num_batched_tokens=256,
            max_model_len=2048,
            enable_chunked_prefill=True,
            is_encoder_decoder=False,
            watermark=0.0,
            async_scheduling=scheduler_cls is AsyncScheduler,
        ),
        model_config=model_config,
        cache_config=CacheConfig(
            block_size=BLOCK_SIZE,
            gpu_memory_utilization=0.9,
            cache_dtype="auto",
            enable_prefix_caching=False,
        ),
        parallel_config=ParallelConfig(),
        device_config=DeviceConfig(device="cpu"),
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer0"],
                FullAttentionSpec(
                    block_size=BLOCK_SIZE,
                    num_kv_heads=1,
                    head_size=32,
                    dtype=torch.float16,
                ),
            )
        ],
    )
    vllm_config.cache_config.num_gpu_blocks = num_blocks
    register_all_kvcache_specs(vllm_config)
    # Build through vLLM's own constructor; the connector is swapped below.
    init = Scheduler.__init__
    while hasattr(init, "__wrapped__"):
        init = init.__wrapped__
    scheduler = scheduler_cls.__new__(scheduler_cls)
    init(
        scheduler,
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        structured_output_manager=StructuredOutputManager(vllm_config),
        block_size=BLOCK_SIZE,
    )
    scheduler.connector = _AsyncLoadConnector()
    tracker = engine_spans.KvFetchSpans(scheduler)
    scheduler._set_kv_fetch_stage = tracker.set_kv_fetch_stage
    init_none_hash(sha256)
    return scheduler, tracker, get_request_block_hasher(BLOCK_SIZE, sha256)


def _add_remote_request(scheduler, hasher, req_id, trace_headers):
    scheduler.add_request(
        Request(
            req_id,
            list(range(1, 20)),
            SamplingParams(max_tokens=4, detokenize=False),
            None,
            block_hasher=hasher,
            trace_headers=trace_headers,
        )
    )


def _no_forward_output(kv_output: KVConnectorOutput) -> ModelRunnerOutput:
    return ModelRunnerOutput(
        req_ids=[],
        req_id_to_index={},
        sampled_token_ids=[],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
        kv_connector_output=kv_output,
    )


def test_real_scheduler_emits_kv_fetch_when_the_pulled_request_is_scheduled(
    shared_model_config, scheduler_cls, clock, spans
):
    scheduler, tracker, hasher = _build_decode_scheduler(
        shared_model_config, scheduler_cls
    )
    _add_remote_request(scheduler, hasher, "d-req", {"traceparent": TRACEPARENT})

    sched_out = scheduler.schedule()
    assert scheduler.requests["d-req"].status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert spans == []

    clock.now_s += 0.250
    scheduler.update_from_output(
        sched_out, _no_forward_output(KVConnectorOutput(finished_recving={"d-req"}))
    )
    clock.now_s += 0.040
    next_out = scheduler.schedule()

    assert [r.req_id for r in next_out.scheduled_new_reqs] == ["d-req"]
    (span,) = spans
    attrs = span["attributes"]
    assert attrs[_stage_attr(KV_FETCH_IN_PROGRESS)] == pytest.approx(0.250)
    assert attrs[_stage_attr(KV_FETCH_COMPLETED_WAITING)] == pytest.approx(0.040)
    # The load starts in the same schedule() pass that admits it.
    assert attrs[_stage_attr(KV_FETCH_WAITING_TO_START)] == pytest.approx(0.0)
    assert attrs["vllm.kv_fetch.outcome"] == "scheduled"
    assert attrs["vllm.kv_fetch.num_tokens"] == BLOCK_SIZE
    assert _parent(span["context"]) == (TRACE_ID, PARENT_SPAN_ID)
    assert tracker.fetches == {}


def test_real_scheduler_emits_kv_fetch_for_a_request_aborted_mid_pull(
    shared_model_config, scheduler_cls, clock, spans
):
    scheduler, tracker, hasher = _build_decode_scheduler(
        shared_model_config, scheduler_cls
    )
    _add_remote_request(scheduler, hasher, "d-req", {"traceparent": TRACEPARENT})
    scheduler.schedule()

    clock.now_s += 0.5
    scheduler.finish_requests("d-req", RequestStatus.FINISHED_ABORTED)

    (span,) = spans
    assert span["attributes"]["vllm.kv_fetch.outcome"] == "finished"
    assert span["attributes"][_stage_attr(KV_FETCH_IN_PROGRESS)] == pytest.approx(0.5)
    assert tracker.fetches == {}


def test_real_scheduler_without_trace_headers_emits_nothing(
    shared_model_config, scheduler_cls, clock, spans
):
    scheduler, tracker, hasher = _build_decode_scheduler(
        shared_model_config, scheduler_cls
    )
    _add_remote_request(scheduler, hasher, "d-req", None)
    sched_out = scheduler.schedule()
    scheduler.update_from_output(
        sched_out, _no_forward_output(KVConnectorOutput(finished_recving={"d-req"}))
    )
    scheduler.schedule()

    assert spans == []
    assert tracker.fetches == {}


# --- XlaCompileSpans ---------------------------------------------------------


class _Entry:
    def __init__(self, compile_s: float, last_read_s: float, read_count: int = 1):
        self.compilation_duration = datetime.timedelta(seconds=compile_s)
        # TorchTPU returns a naive local datetime (pybind11 chrono).
        self.last_read = datetime.datetime.fromtimestamp(last_read_s)
        self.read_count = read_count


class _FakeTpu:
    def __init__(self, entries=None, misses=0, with_misses=True):
        self.entries = list(entries or [])
        self.misses = misses
        self.stats_calls = 0
        if with_misses:
            self._get_cache_misses = lambda: self.misses

    def _get_cache_stats(self):
        self.stats_calls += 1
        return SimpleNamespace(per_entry_stats=list(self.entries))


def test_startup_compiles_are_not_reported(clock, spans):
    tpu = _FakeTpu([_Entry(30.0, clock.now_s - 5)], misses=1)
    emitter = engine_spans.XlaCompileSpans(tpu, {"vllm.worker.rank": 0})

    clock.now_s += 1
    emitter.poll()

    assert spans == []


def test_unchanged_miss_count_skips_the_cache_scan(clock, spans):
    tpu = _FakeTpu([_Entry(30.0, clock.now_s - 5)], misses=1)
    emitter = engine_spans.XlaCompileSpans(tpu, {})
    calls_after_baseline = tpu.stats_calls

    for _ in range(5):
        clock.now_s += 0.01
        emitter.poll()

    assert tpu.stats_calls == calls_after_baseline
    assert spans == []


def test_serving_compile_becomes_a_root_span_ending_at_first_read(clock, spans):
    tpu = _FakeTpu([_Entry(30.0, clock.now_s - 5)], misses=1)
    emitter = engine_spans.XlaCompileSpans(tpu, {"vllm.worker.rank": 0})
    polled_at = clock.now_s

    clock.now_s += 10.0
    compile_end = clock.now_s - 1.0
    tpu.entries.append(_Entry(8.718, compile_end))
    tpu.misses += 1
    emitter.poll()

    (span,) = spans
    assert span["name"] == "xla_compile"
    assert span["end_ns"] == pytest.approx(compile_end * 1e9, abs=1e3)
    assert span["start_ns"] == pytest.approx((compile_end - 8.718) * 1e9, abs=1e3)
    assert span["start_ns"] >= polled_at * 1e9
    attrs = span["attributes"]
    assert attrs["vllm.xla_compile.compile_s"] == pytest.approx(8.718)
    assert attrs["vllm.xla_compile.num_graphs"] == 1
    assert attrs["vllm.worker.rank"] == 0
    # A root span: no parent in the passed context.
    assert not trace.get_current_span(span["context"]).get_span_context().is_valid


def test_reread_and_parallel_compiles_are_clamped_to_the_poll_interval(clock, spans):
    tpu = _FakeTpu(misses=0)
    emitter = engine_spans.XlaCompileSpans(tpu, {})
    polled_at = clock.now_s

    clock.now_s += 2.0
    # Already read again (eager graphs reused across layers), so the first
    # read is unknown; together they took longer than the interval.
    tpu.entries += [_Entry(1.5, clock.now_s, read_count=4), _Entry(1.5, clock.now_s)]
    tpu.entries[-1].last_read = datetime.datetime.fromtimestamp(polled_at - 1)
    tpu.misses = 2
    emitter.poll()

    (span,) = spans
    assert span["end_ns"] == pytest.approx(clock.now_s * 1e9, abs=1e3)
    assert span["start_ns"] == pytest.approx(polled_at * 1e9, abs=1e3)
    assert span["attributes"]["vllm.xla_compile.compile_s"] == pytest.approx(3.0)
    assert span["attributes"]["vllm.xla_compile.num_graphs"] == 2


def test_cache_hits_from_disk_and_evictions_emit_nothing(clock, spans):
    tpu = _FakeTpu([_Entry(4.0, clock.now_s - 1)], misses=1)
    emitter = engine_spans.XlaCompileSpans(tpu, {})

    clock.now_s += 1
    tpu.entries.append(_Entry(0.0, clock.now_s))  # loaded from a persistent cache
    tpu.misses += 1
    emitter.poll()
    clock.now_s += 1
    tpu.entries = [tpu.entries[-1]]  # the compiled entry was evicted
    tpu.misses += 1
    emitter.poll()
    clock.now_s += 1
    tpu.entries.append(_Entry(2.0, clock.now_s))
    tpu.misses += 1
    emitter.poll()

    (span,) = spans
    assert span["attributes"]["vllm.xla_compile.compile_s"] == pytest.approx(2.0)


def test_compile_still_running_at_the_poll_is_reported_when_it_finishes(clock, spans):
    # TorchTPU counts the miss and inserts the entry when the graph is looked
    # up, but records the duration only when the compile pool finishes, which
    # the worker thread does not wait for unless it reads a result.
    tpu = _FakeTpu(misses=0)
    emitter = engine_spans.XlaCompileSpans(tpu, {})
    before_miss = clock.now_s

    clock.now_s += 0.01
    running = _Entry(0.0, clock.now_s)
    tpu.entries.append(running)
    tpu.misses = 1
    emitter.poll()
    assert spans == []
    calls = tpu.stats_calls

    clock.now_s += 0.01  # before the first rescan is due
    emitter.poll()
    assert tpu.stats_calls == calls

    clock.now_s += 3.0
    compile_end = clock.now_s - 0.5
    running.compilation_duration = datetime.timedelta(seconds=2.0)
    running.last_read = datetime.datetime.fromtimestamp(compile_end)
    emitter.poll()

    (span,) = spans
    assert span["end_ns"] == pytest.approx(compile_end * 1e9, abs=1e3)
    assert span["start_ns"] == pytest.approx((compile_end - 2.0) * 1e9, abs=1e3)
    assert span["start_ns"] >= before_miss * 1e9
    assert span["attributes"]["vllm.xla_compile.num_graphs"] == 1

    # Resolved: an unchanged miss count no longer scans.
    calls = tpu.stats_calls
    for _ in range(3):
        clock.now_s += 5.0
        emitter.poll()
    assert tpu.stats_calls == calls
    assert len(spans) == 1


def test_carried_compile_window_starts_at_the_poll_before_the_miss(clock, spans):
    tpu = _FakeTpu(misses=0)
    emitter = engine_spans.XlaCompileSpans(tpu, {})
    before_miss = clock.now_s

    clock.now_s += 0.5
    running = _Entry(0.0, clock.now_s)
    tpu.entries.append(running)
    tpu.misses = 1
    emitter.poll()

    # Several polls while the step waits; then a compile longer than the
    # last poll interval finishes.
    for _ in range(3):
        clock.now_s += 0.5
        emitter.poll()
    running.compilation_duration = datetime.timedelta(seconds=10.0)
    running.last_read = datetime.datetime.fromtimestamp(clock.now_s)
    clock.now_s += 1.0
    emitter.poll()

    (span,) = spans
    assert span["start_ns"] == pytest.approx(before_miss * 1e9, abs=1e3)


def test_carried_compile_does_not_end_at_an_earlier_reported_compile(clock, spans):
    tpu = _FakeTpu(misses=0)
    emitter = engine_spans.XlaCompileSpans(tpu, {})

    # Two misses: a one-off program that is never read again finishes before
    # the poll, the other is still compiling.
    clock.now_s += 1.0
    one_off_end = clock.now_s - 0.2
    tpu.entries.append(_Entry(0.5, one_off_end))
    running = _Entry(0.0, clock.now_s)
    tpu.entries.append(running)
    tpu.misses = 2
    emitter.poll()
    assert len(spans) == 1

    # The carried compile finishes and is read again before the next scan, so
    # its first read is lost; it must not reuse the one-off program's.
    clock.now_s += 5.0
    running.compilation_duration = datetime.timedelta(seconds=3.0)
    running.last_read = datetime.datetime.fromtimestamp(clock.now_s - 0.1)
    running.read_count = 7
    emitter.poll()

    assert len(spans) == 2
    span = spans[1]
    assert span["end_ns"] == pytest.approx(clock.now_s * 1e9, abs=1e3)
    assert span["start_ns"] == pytest.approx((clock.now_s - 3.0) * 1e9, abs=1e3)
    assert span["attributes"]["vllm.xla_compile.compile_s"] == pytest.approx(3.0)


def test_persistent_cache_loads_stop_rescanning_after_a_timeout(clock, spans):
    tpu = _FakeTpu(misses=0)
    emitter = engine_spans.XlaCompileSpans(tpu, {})

    clock.now_s += 1.0
    tpu.entries.append(_Entry(0.0, clock.now_s))  # never reports a duration
    tpu.misses = 1
    emitter.poll()
    calls = tpu.stats_calls

    # Rescans back off instead of running on every step.
    for _ in range(200):
        clock.now_s += 0.01
        emitter.poll()
    assert 0 < tpu.stats_calls - calls < 10

    clock.now_s += engine_spans.XlaCompileSpans._PENDING_TIMEOUT_S
    emitter.poll()
    calls = tpu.stats_calls
    for _ in range(10):
        clock.now_s += 2.0
        emitter.poll()
    assert tpu.stats_calls == calls
    assert spans == []


def test_without_a_miss_counter_every_poll_scans(clock, spans):
    tpu = _FakeTpu(with_misses=False)
    emitter = engine_spans.XlaCompileSpans(tpu, {})

    clock.now_s += 1
    tpu.entries.append(_Entry(0.5, clock.now_s))
    emitter.poll()

    assert len(spans) == 1


def test_maybe_create_is_gated_on_the_endpoint(monkeypatch):
    monkeypatch.setattr(torch, "tpu", _FakeTpu(), raising=False)
    monkeypatch.setattr(engine_spans, "ensure_tracer_provider", lambda *a: True)
    config = SimpleNamespace(
        observability_config=SimpleNamespace(otlp_traces_endpoint=None),
        parallel_config=SimpleNamespace(data_parallel_rank=1),
    )
    assert engine_spans.maybe_create_xla_compile_spans(config, rank=0) is None

    config.observability_config.otlp_traces_endpoint = "http://otel:4317"
    emitter = engine_spans.maybe_create_xla_compile_spans(config, rank=2)
    assert emitter is not None
    assert emitter._attributes == {"vllm.worker.rank": 2, "vllm.dp_rank": 1}


def test_maybe_create_skips_non_zero_tp_ranks(monkeypatch):
    monkeypatch.setattr(torch, "tpu", _FakeTpu(), raising=False)
    monkeypatch.setattr(engine_spans, "ensure_tracer_provider", lambda *a: True)
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_tensor_model_parallel_rank", lambda: 1
    )
    config = SimpleNamespace(
        observability_config=SimpleNamespace(otlp_traces_endpoint="x"),
        parallel_config=SimpleNamespace(data_parallel_rank=0),
    )
    assert engine_spans.maybe_create_xla_compile_spans(config, rank=1) is None
