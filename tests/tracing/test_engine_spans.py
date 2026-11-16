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
"""kv_fetch spans emitted from EngineCore."""

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

    def time_ns(self) -> int:
        return int(round(self.now_s * 1e9))


@pytest.fixture
def clock(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(engine_spans.time, "time_ns", clock.time_ns)
    return clock


@pytest.fixture
def exporter(monkeypatch):
    """Exports spans through vLLM's `instrument_manual` into memory."""
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, "get_tracer", provider.get_tracer)
    return exporter


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
    tracker = engine_spans.KvFetchSpans(scheduler._set_kv_fetch_stage)
    scheduler._set_kv_fetch_stage = tracker.set_kv_fetch_stage
    init_none_hash(sha256)
    return scheduler, tracker, get_request_block_hasher(BLOCK_SIZE, sha256)


def _add_remote_request(scheduler, hasher, trace_headers):
    scheduler.add_request(
        Request(
            "d-req",
            list(range(1, 20)),
            SamplingParams(max_tokens=4, detokenize=False),
            None,
            block_hasher=hasher,
            trace_headers=trace_headers,
        )
    )


def _finish_pull(scheduler, sched_out):
    scheduler.update_from_output(
        sched_out,
        ModelRunnerOutput(
            req_ids=[],
            req_id_to_index={},
            sampled_token_ids=[],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
            kv_connector_output=KVConnectorOutput(finished_recving={"d-req"}),
        ),
    )


def test_span_covers_the_pull_and_the_wait_to_be_scheduled(
    shared_model_config, scheduler_cls, clock, exporter
):
    scheduler, tracker, hasher = _build_decode_scheduler(
        shared_model_config, scheduler_cls
    )
    _add_remote_request(scheduler, hasher, {"traceparent": TRACEPARENT})
    clock.now_s += 0.010
    start_ns = clock.time_ns()
    sched_out = scheduler.schedule()
    assert scheduler.requests["d-req"].status == RequestStatus.WAITING_FOR_REMOTE_KVS

    clock.now_s += 0.250
    _finish_pull(scheduler, sched_out)
    assert exporter.get_finished_spans() == ()
    clock.now_s += 0.040
    next_out = scheduler.schedule()

    assert [r.req_id for r in next_out.scheduled_new_reqs] == ["d-req"]
    (span,) = exporter.get_finished_spans()
    assert span.name == "kv_fetch"
    assert span.kind is trace.SpanKind.INTERNAL
    assert span.context.trace_id == TRACE_ID
    assert span.parent is not None
    assert span.parent.span_id == PARENT_SPAN_ID
    # Starts at the load, not at admission: the 10 ms before it is queueing.
    assert span.start_time == start_ns
    assert span.end_time == clock.time_ns()
    assert dict(span.attributes) == pytest.approx(
        {
            "vllm.kv_fetch.in_progress_s": 0.250,
            "vllm.kv_fetch.completed_waiting_s": 0.040,
        }
    )
    assert tracker.fetches == {}


def test_span_for_a_request_aborted_mid_pull(
    shared_model_config, scheduler_cls, clock, exporter
):
    scheduler, tracker, hasher = _build_decode_scheduler(
        shared_model_config, scheduler_cls
    )
    _add_remote_request(scheduler, hasher, {"traceparent": TRACEPARENT})
    scheduler.schedule()

    clock.now_s += 0.5
    scheduler.finish_requests("d-req", RequestStatus.FINISHED_ABORTED)

    (span,) = exporter.get_finished_spans()
    assert dict(span.attributes) == pytest.approx({"vllm.kv_fetch.in_progress_s": 0.5})
    assert tracker.fetches == {}


@pytest.mark.parametrize("headers", [None, {}, {"tracestate": "k=v"}])
def test_no_span_without_a_propagated_parent(
    shared_model_config, clock, exporter, headers
):
    scheduler, tracker, hasher = _build_decode_scheduler(shared_model_config, Scheduler)
    _add_remote_request(scheduler, hasher, headers)
    sched_out = scheduler.schedule()
    _finish_pull(scheduler, sched_out)
    scheduler.schedule()

    assert exporter.get_finished_spans() == ()
    assert tracker.fetches == {}


def test_no_span_when_the_load_never_started(shared_model_config, clock, exporter):
    scheduler, tracker, hasher = _build_decode_scheduler(shared_model_config, Scheduler)
    _add_remote_request(scheduler, hasher, {"traceparent": TRACEPARENT})
    scheduler.finish_requests("d-req", RequestStatus.FINISHED_ABORTED)

    assert exporter.get_finished_spans() == ()
    assert tracker.fetches == {}


def _config(endpoint):
    return SimpleNamespace(
        observability_config=SimpleNamespace(otlp_traces_endpoint=endpoint)
    )


def _gated_scheduler(connector=True):
    return SimpleNamespace(
        connector=object() if connector else None,
        _set_kv_fetch_stage=lambda req, stage: None,
    )


@pytest.mark.parametrize(
    ("endpoint", "connector", "tracer"),
    [(None, True, True), ("x", False, True), ("x", True, False)],
)
def test_install_is_a_noop_without_something_to_trace(
    monkeypatch, endpoint, connector, tracer
):
    monkeypatch.setattr(engine_spans, "_ensure_tracer_provider", lambda _: tracer)
    scheduler = _gated_scheduler(connector)
    original = scheduler._set_kv_fetch_stage

    assert engine_spans.install_kv_fetch_spans(scheduler, _config(endpoint)) is None
    assert scheduler._set_kv_fetch_stage is original


def test_install_wraps_the_instance(monkeypatch):
    monkeypatch.setattr(engine_spans, "_ensure_tracer_provider", lambda _: True)
    scheduler = _gated_scheduler()

    tracker = engine_spans.install_kv_fetch_spans(scheduler, _config("x"))

    assert tracker is not None
    assert scheduler._set_kv_fetch_stage == tracker.set_kv_fetch_stage


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
