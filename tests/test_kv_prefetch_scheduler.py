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
"""TPU_SCHED_KV_PREFETCH_WAITING on the real vLLM scheduler loop.

These tests build a real PrefixAwareScheduler (stock Scheduler.schedule(),
KVCacheManager and block pool) as a PD consumer whose connector loads every
remote request asynchronously, and drive schedule() / update_from_output()
step by step. A remote request's pull starts when the connector's
update_state_after_alloc runs; its KV "arrives" when a step's output reports
it in finished_recving.
"""

import contextlib
import json
import random

import pytest
import torch
from vllm import SamplingParams
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    KVTransferConfig,
    ModelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    SupportsHMA,
)
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from vllm_torchtpu.core.prefix_aware_scheduler import PrefixAwareScheduler

BLOCK = 16
PROMPT = 64  # 4 blocks
CAP = 4


class _PrefetchTestConnector(KVConnectorBase_V1, SupportsHMA):
    """Consumer connector: a request with kv_transfer_params loads its whole
    uncached prompt asynchronously; any other request computes locally."""

    def __init__(self, vllm_config, role, kv_cache_config):
        super().__init__(vllm_config, role, kv_cache_config)
        self.loads: list[str] = []

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        params = request.kv_transfer_params
        if not params or params.get("_started"):
            return 0, False
        return request.num_tokens - num_computed_tokens, True

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        if num_external_tokens > 0:
            request.kv_transfer_params["_started"] = True
            self.loads.append(request.request_id)

    def build_connector_meta(self, scheduler_output):
        return KVConnectorMetadata()

    def request_finished_all_groups(self, request, block_ids):
        return False, None

    def start_load_kv(self, kv_caches, finished_req_ids):
        pass

    def wait_for_layer_load(self, layer_name):
        pass

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        pass

    def wait_for_save(self):
        pass


_CONNECTOR = "KvPrefetchTestConnector"
# A second import of this module finds the name registered.
with contextlib.suppress(ValueError):
    KVConnectorFactory.register_connector(
        _CONNECTOR, __name__, _PrefetchTestConnector.__name__
    )


@pytest.fixture(autouse=True)
def _decode_env(monkeypatch):
    monkeypatch.setenv("TPU_SCHED_KV_PREFETCH_RESERVE_BLOCKS", "8")
    monkeypatch.delenv("TPU_SCHED_KV_PREFETCH_WAITING", raising=False)


def _model_dir(tmp_path):
    d = tmp_path / "tiny-opt"
    d.mkdir(exist_ok=True)
    (d / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["OPTForCausalLM"],
                "model_type": "opt",
                "hidden_size": 64,
                "num_attention_heads": 4,
                "num_hidden_layers": 2,
                "ffn_dim": 128,
                "max_position_embeddings": 2048,
                "vocab_size": 50272,
                "torch_dtype": "float16",
                "word_embed_proj_dim": 64,
                "do_layer_norm_before": True,
            }
        )
    )
    return str(d)


def _make_sched(tmp_path, monkeypatch, prefetch=0, num_blocks=2000, role="kv_consumer"):
    if prefetch:
        monkeypatch.setenv("TPU_SCHED_KV_PREFETCH_WAITING", str(prefetch))
    vllm_config = VllmConfig(
        scheduler_config=SchedulerConfig(
            max_num_seqs=CAP,
            max_num_batched_tokens=256,
            max_model_len=2048,
            enable_chunked_prefill=True,
            is_encoder_decoder=False,
        ),
        model_config=ModelConfig(
            model=_model_dir(tmp_path),
            dtype="float16",
            seed=42,
            skip_tokenizer_init=True,
        ),
        cache_config=CacheConfig(
            block_size=BLOCK, gpu_memory_utilization=0.9, enable_prefix_caching=True
        ),
        # The TPU platform validates the connector name against a fixed set;
        # pass with an allowed stock connector, then swap in the test one
        # before the scheduler constructs it.
        kv_transfer_config=KVTransferConfig(
            kv_connector="DecodeBenchConnector", kv_role=role
        ),
        device_config=DeviceConfig("cpu"),
    )
    vllm_config.kv_transfer_config.kv_connector = _CONNECTOR
    vllm_config.cache_config.num_gpu_blocks = num_blocks
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=BLOCK, num_kv_heads=1, head_size=1, dtype=torch.float32
                ),
            )
        ],
    )
    return PrefixAwareScheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        structured_output_manager=StructuredOutputManager(vllm_config),
        block_size=BLOCK,
        log_stats=True,
    )


_none_hash_ready = False


def _req(rid: str, remote: bool = True, max_tokens: int = 1000) -> Request:
    global _none_hash_ready
    if not _none_hash_ready:
        init_none_hash(sha256)
        _none_hash_ready = True
    seed = sum(ord(c) for c in rid) * 7919
    sp = SamplingParams(max_tokens=max_tokens, ignore_eos=True)
    r = Request(
        request_id=rid,
        prompt_token_ids=[(seed + i) % 50000 + 1 for i in range(PROMPT)],
        sampling_params=sp,
        pooling_params=None,
        mm_features=None,
        block_hasher=get_request_block_hasher(BLOCK, sha256),
    )
    r.kv_transfer_params = {"remote": True} if remote else None
    return r


class _Engine:
    """Alternates schedule() and update_from_output(), checking the cap."""

    def __init__(self, sched: PrefixAwareScheduler):
        self.s = sched
        self.admitted: list[str] = []
        self.max_running = 0

    @property
    def loads(self) -> list[str]:
        return self.s.connector.loads

    def step(self, recv=()):
        out = self.s.schedule()
        assert len(self.s.running) <= CAP
        assert len(out.num_scheduled_tokens) <= CAP
        self.max_running = max(self.max_running, len(self.s.running))
        self.admitted += [r.req_id for r in out.scheduled_new_reqs]
        ids = list(out.num_scheduled_tokens)
        recv = set(recv)
        mro = ModelRunnerOutput(
            req_ids=ids,
            req_id_to_index={rid: i for i, rid in enumerate(ids)},
            sampled_token_ids=[[0] for _ in ids],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=None,
            kv_connector_output=(
                KVConnectorOutput(finished_recving=recv) if recv else None
            ),
        )
        self.s.update_from_output(out, mro)
        return out

    def status(self, rid):
        return self.s.requests[rid].status

    def held(self) -> int:
        return self.s._kv_prefetch_num_held()

    def blocks(self, rid) -> int:
        mgr = self.s.kv_cache_manager.coordinator.single_type_managers[0]
        return len(mgr.req_to_blocks.get(rid, ()))

    def free_blocks(self) -> int:
        return self.s.kv_cache_manager.block_pool.get_num_free_blocks()

    def fill_running(self, n=CAP, prefix="run"):
        ids = [f"{prefix}{i}" for i in range(n)]
        for rid in ids:
            self.s.add_request(_req(rid))
        self.step()  # pulls start (a slot is free)
        self.step(recv=ids)  # KV arrives
        for _ in range(3):
            if all(self.status(r) == RequestStatus.RUNNING for r in ids):
                break
            self.step()
        assert all(self.status(r) == RequestStatus.RUNNING for r in ids)
        assert len(self.s.running) == n
        return ids


def test_stock_loop_defers_pull_while_running_is_full(tmp_path, monkeypatch):
    """Gate off: at the cap no waiting request's pull starts (the deferral)."""
    eng = _Engine(_make_sched(tmp_path, monkeypatch, prefetch=0))
    assert eng.s._kv_prefetch_n == 0
    eng.fill_running()
    n_loads = len(eng.loads)
    for rid in ("w0", "w1", "w2"):
        eng.s.add_request(_req(rid))
    for _ in range(3):
        eng.step()
    assert len(eng.loads) == n_loads
    assert all(eng.status(r) == RequestStatus.WAITING for r in ("w0", "w1", "w2"))
    assert all(eng.blocks(r) == 0 for r in ("w0", "w1", "w2"))


def test_prefetch_pulls_head_of_waiting_without_exceeding_cap(tmp_path, monkeypatch):
    eng = _Engine(_make_sched(tmp_path, monkeypatch, prefetch=2))
    assert eng.s._kv_prefetch_n == 2
    running = eng.fill_running()
    n_loads = len(eng.loads)
    for rid in ("w0", "w1", "w2"):
        eng.s.add_request(_req(rid))

    eng.step()
    # The two oldest waiting requests start their pulls; the third waits.
    assert eng.loads[n_loads:] == ["w0", "w1"]
    assert eng.status("w0") == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert eng.status("w1") == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert eng.status("w2") == RequestStatus.WAITING
    assert eng.blocks("w0") == eng.blocks("w1") == PROMPT // BLOCK
    assert eng.blocks("w2") == 0
    assert [r.request_id for r in eng.s.running] == running
    assert eng.s._kv_prefetch_started == 2

    # KV arrives while the running set is still full: held, not admitted, and
    # no third pull (two already held).
    eng.step(recv=["w0", "w1"])
    eng.step()
    eng.step()
    assert [r.request_id for r in eng.s.running] == running
    assert "w0" not in eng.admitted and "w1" not in eng.admitted
    assert eng.held() == 2
    assert eng.loads[n_loads:] == ["w0", "w1"]

    # A slot frees: the oldest KV-ready request takes it on the very next
    # step, and the pull for w2 starts in the same step.
    eng.s.finish_requests(running[0], RequestStatus.FINISHED_ABORTED)
    out = eng.step()
    assert [r.req_id for r in out.scheduled_new_reqs] == ["w0"]
    assert len(eng.s.running) == CAP
    assert eng.loads[n_loads:] == ["w0", "w1", "w2"]
    assert eng.status("w1") != RequestStatus.RUNNING
    assert eng.max_running == CAP


@pytest.mark.parametrize("prefetch", [0, 3])
def test_freed_slot_refill_latency(tmp_path, monkeypatch, prefetch):
    """Steps from a slot freeing to a waiting request filling it, when KV
    takes 2 steps to arrive after the pull starts."""
    eng = _Engine(_make_sched(tmp_path, monkeypatch, prefetch=prefetch))
    running = eng.fill_running()
    n_loads = len(eng.loads)
    eng.s.add_request(_req("w0"))
    started_at = {}
    for step in range(4):  # running is full throughout
        eng.step(recv=[r for r, t in started_at.items() if step - t == 2])
        for rid in eng.loads[n_loads:]:
            started_at.setdefault(rid, step)
    eng.s.finish_requests(running[0], RequestStatus.FINISHED_ABORTED)
    idle_steps = 0
    for step in range(4, 12):
        out = eng.step(recv=[r for r, t in started_at.items() if step - t == 2])
        for rid in eng.loads[n_loads:]:
            started_at.setdefault(rid, step)
        if "w0" in [r.req_id for r in out.scheduled_new_reqs]:
            break
        idle_steps += 1
    else:
        pytest.fail("w0 never admitted")
    if prefetch:
        assert idle_steps == 0  # KV already held: the slot is refilled at once
    else:
        assert idle_steps >= 2  # the slot idles while the pull runs


def test_abort_of_prefetched_requests_frees_blocks(tmp_path, monkeypatch):
    eng = _Engine(_make_sched(tmp_path, monkeypatch, prefetch=2))
    running = eng.fill_running()
    eng.s.add_request(_req("w0"))
    eng.s.add_request(_req("w1"))
    eng.step()
    per_req = PROMPT // BLOCK
    assert eng.status("w0") == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert eng.status("w1") == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert eng.blocks("w0") == eng.blocks("w1") == per_req

    # w1's KV arrives (it stays held at the cap); w0's pull is still running.
    eng.step(recv=["w1"])
    eng.step()
    assert eng.status("w1") == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert "w1" in eng.s.finished_recving_kv_req_ids

    # Abort the KV-ready one: its blocks return immediately.
    free = eng.free_blocks()
    eng.s.finish_requests("w1", RequestStatus.FINISHED_ABORTED)
    assert eng.free_blocks() == free + per_req
    assert eng.blocks("w1") == 0 and "w1" not in eng.s.requests
    # Abort the one still loading: its blocks stay until the worker reports
    # the load done (it may still be writing them), then return.
    free = eng.free_blocks()
    eng.s.finish_requests("w0", RequestStatus.FINISHED_ABORTED)
    assert eng.free_blocks() == free
    assert eng.blocks("w0") == per_req and "w0" in eng.s.requests
    eng.step(recv=["w0"])
    assert eng.blocks("w0") == 0 and "w0" not in eng.s.requests
    assert eng.held() == 0
    assert not eng.s._inflight_prefills

    # Drain: nothing leaks.
    eng.s.finish_requests(running, RequestStatus.FINISHED_ABORTED)
    eng.step()
    assert not eng.s.requests and not eng.s.waiting and not eng.s.kv_holding_waiting
    assert not eng.s.deferred_waiting and not eng.s.running
    pool = eng.s.kv_cache_manager.block_pool
    assert pool.get_num_free_blocks() == pool.num_gpu_blocks - 1
    assert all(b.ref_cnt == 0 for b in pool.blocks)


def test_prefetch_respects_free_blocks_and_reserve(tmp_path, monkeypatch):
    # 4 running x 5 blocks (4 prompt + 1 decode) = 20 used. The stock
    # async-load reservation is 0 once every running prefill has finished.
    # With 50 blocks (49 usable): 29 free - prefetch reserve must fit 4 blocks.
    monkeypatch.setenv("TPU_SCHED_KV_PREFETCH_RESERVE_BLOCKS", "64")
    eng = _Engine(_make_sched(tmp_path, monkeypatch, prefetch=2, num_blocks=50))
    running = eng.fill_running()
    free_before = eng.free_blocks()
    eng.s.add_request(_req("w0"))
    for _ in range(3):
        eng.step()
    assert eng.status("w0") == RequestStatus.WAITING
    assert eng.blocks("w0") == 0
    assert [r.request_id for r in eng.s.running] == running
    assert eng.free_blocks() >= free_before - CAP  # only running growth

    monkeypatch.setenv("TPU_SCHED_KV_PREFETCH_RESERVE_BLOCKS", "0")
    eng2 = _Engine(_make_sched(tmp_path, monkeypatch, prefetch=2, num_blocks=50))
    eng2.fill_running()
    eng2.s.add_request(_req("w0"))
    eng2.step()
    assert eng2.status("w0") == RequestStatus.WAITING_FOR_REMOTE_KVS


def test_request_without_async_load_stops_prefetch_and_never_passes_cap(
    tmp_path, monkeypatch
):
    eng = _Engine(_make_sched(tmp_path, monkeypatch, prefetch=3))
    running = eng.fill_running()
    n_loads = len(eng.loads)
    eng.s.add_request(_req("local0", remote=False))  # would compute locally
    eng.s.add_request(_req("w1"))
    for _ in range(3):
        eng.step()
    # FCFS: nothing is pulled ahead of the local request, and it does not run.
    assert len(eng.loads) == n_loads
    assert eng.status("local0") == RequestStatus.WAITING
    assert [r.request_id for r in eng.s.running] == running

    eng.s.finish_requests(running[0], RequestStatus.FINISHED_ABORTED)
    out = eng.step()
    assert [r.req_id for r in out.scheduled_new_reqs] == ["local0"]
    assert eng.loads[n_loads:] == ["w1"]  # prefetched once local0 has the slot
    assert len(eng.s.running) == CAP


def test_gate_ignored_on_kv_producer(tmp_path, monkeypatch):
    s = _make_sched(tmp_path, monkeypatch, prefetch=4, role="kv_producer")
    assert s._kv_prefetch_n == 0


@pytest.mark.parametrize("prefetch", [0, 1, 4])
def test_randomized_cap_bound_and_no_leaks(tmp_path, monkeypatch, prefetch):
    """Random arrivals, finishes, aborts (running and held) and KV arrival
    delays. Every step: the running set never passes the cap (_Engine.step),
    and a step that starts with the running set full starts at most
    max(0, prefetch - held) pulls (none with the gate off). At the end
    everything drains with every block free."""
    rng = random.Random(1234 + prefetch)
    eng = _Engine(_make_sched(tmp_path, monkeypatch, prefetch=prefetch, num_blocks=200))
    started_at: dict[str, int] = {}
    delay: dict[str, int] = {}
    n = 0
    at_cap_steps = 0
    for step in range(400):
        for _ in range(rng.choice((0, 0, 1, 2))):
            rid = f"r{n}"
            n += 1
            eng.s.add_request(_req(rid, remote=rng.random() > 0.05))
            delay[rid] = rng.choice((1, 2, 3, 5))
        live = [r.request_id for r in eng.s.running]
        if live and rng.random() < 0.3:
            eng.s.finish_requests(rng.choice(live), RequestStatus.FINISHED_ABORTED)
        held = [
            r.request_id
            for r in eng.s._inflight_prefills
            if r.status != RequestStatus.RUNNING
        ]
        if held and rng.random() < 0.05:
            eng.s.finish_requests(rng.choice(held), RequestStatus.FINISHED_ABORTED)
        # A load is reported done once, even if its request was aborted
        # meanwhile (the scheduler keeps such a request until then).
        recv = [
            rid
            for rid, t in started_at.items()
            if step - t == delay[rid] and rid in eng.s.requests
        ]
        at_cap = len(eng.s.running) >= CAP
        held_start = eng.held()
        loads_start = len(eng.loads)
        eng.step(recv=recv)
        new_loads = len(eng.loads) - loads_start
        if at_cap:
            at_cap_steps += 1
            assert new_loads <= max(0, prefetch - held_start), (step, held_start)
        for rid in eng.loads:
            started_at.setdefault(rid, step)
    assert eng.max_running == CAP
    assert at_cap_steps > 50
    if prefetch:
        assert eng.s._kv_prefetch_started > 0
    else:
        assert eng.s._kv_prefetch_started == 0
    # Drain: abort everything, report every outstanding load, step.
    eng.s.finish_requests(list(eng.s.requests), RequestStatus.FINISHED_ABORTED)
    outstanding = [r for r in started_at if r in eng.s.requests]
    eng.step(recv=outstanding)
    eng.step()
    assert not eng.s.requests and not eng.s.running and not eng.s._inflight_prefills
    pool = eng.s.kv_cache_manager.block_pool
    assert pool.get_num_free_blocks() == pool.num_gpu_blocks - 1
    assert all(b.ref_cnt == 0 for b in pool.blocks)
