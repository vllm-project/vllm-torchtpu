# SPDX-License-Identifier: Apache-2.0
"""A schedule pass starts at most N remote KV loads, in queue order.

See `vllm_torchtpu.core.kv_load_pacing`. A real vLLM ``Scheduler`` runs a
KV-consumer engine whose stand-in connector loads every request that carries
``kv_transfer_params``; the tests land loads and free running slots by hand
and check, pass by pass, which loads start and which requests run.
"""

import inspect

import pytest
import torch
from vllm import SamplingParams
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    KVTransferConfig,
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
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from vllm_torchtpu.core import kv_load_pacing

BLOCK_SIZE = 16
NUM_BLOCKS = 512
PROMPT_LEN = 40


class PacingTestConnector(KVConnectorBase_V1, SupportsHMA):
    """Scheduler half of a P/D consumer: a request with unprocessed
    ``kv_transfer_params`` loads everything past its local hit, async."""

    def __init__(self, vllm_config, role, kv_cache_config=None):
        super().__init__(vllm_config, role, kv_cache_config)
        self.started: list[str] = []

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        params = request.kv_transfer_params
        if not params or params.get("_remote_kv_processed"):
            return 0, False
        count = request.num_tokens - num_computed_tokens
        return count, count > 0

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        if num_external_tokens > 0:
            self.started.append(request.request_id)
            request.kv_transfer_params["_remote_kv_processed"] = True

    def build_connector_meta(self, scheduler_output):
        return KVConnectorMetadata()

    def request_finished_all_groups(self, request, block_ids):
        return False, None

    def start_load_kv(self, forward_context, **kwargs):
        pass

    def wait_for_layer_load(self, layer_name):
        pass

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        pass

    def wait_for_save(self):
        pass


@pytest.fixture
def pacing(monkeypatch):
    """Installs the patch with a given cap; the class is restored after."""
    for name in ("schedule", "_handle_blocked_waiting_request"):
        monkeypatch.setattr(Scheduler, name, getattr(Scheduler, name))

    def install(cap: int) -> None:
        monkeypatch.setenv("VLLM_TPU_KV_LOAD_STARTS_PER_STEP", str(cap))
        kv_load_pacing.patch_kv_load_pacing()

    return install


def _scheduler(
    monkeypatch, *, max_num_seqs: int, role: str = "kv_consumer", cls=Scheduler
):
    monkeypatch.setenv("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
    vllm_config = VllmConfig(
        scheduler_config=SchedulerConfig(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=2048,
            max_model_len=2048,
            enable_chunked_prefill=True,
            is_encoder_decoder=False,
        ),
        model_config=ModelConfig(
            model="facebook/opt-125m", dtype="float16", seed=42, max_model_len=2048
        ),
        cache_config=CacheConfig(
            block_size=BLOCK_SIZE,
            gpu_memory_utilization=0.9,
            cache_dtype="auto",
            enable_prefix_caching=True,
        ),
        parallel_config=ParallelConfig(),
        device_config=DeviceConfig(device="cpu"),
    )
    # Set after validation: the TPU platform admits only its own connectors.
    vllm_config.kv_transfer_config = KVTransferConfig(
        kv_connector="PacingTestConnector",
        kv_connector_module_path=__name__,
        kv_role=role,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=NUM_BLOCKS,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=BLOCK_SIZE,
                    num_kv_heads=1,
                    head_size=64,
                    dtype=torch.float16,
                ),
            )
        ],
    )
    vllm_config.cache_config.num_gpu_blocks = NUM_BLOCKS
    init_none_hash(sha256)
    return cls(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        structured_output_manager=StructuredOutputManager(vllm_config),
        block_size=BLOCK_SIZE,
        hash_block_size=BLOCK_SIZE,
        log_stats=True,
    )


def _add(scheduler, index: int, *, remote: bool = True) -> str:
    request_id = f"r{index:02d}"
    params = SamplingParams(max_tokens=8, ignore_eos=True)
    if remote:
        params.extra_args = {"kv_transfer_params": {"num_tokens": PROMPT_LEN}}
    # Distinct prompts: no request hits another's prefix.
    prompt = [1000 * (index + 1) + t for t in range(PROMPT_LEN)]
    scheduler.add_request(
        Request(
            request_id,
            prompt,
            params,
            None,
            block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
        )
    )
    return request_id


class Run:
    """Drives one scheduler pass by pass and records each pass."""

    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.connector = scheduler.connector
        self.starts: list[list[str]] = []
        self.admitted: list[list[str]] = []

    def step(self) -> list[str]:
        before = len(self.connector.started)
        out = self.scheduler.schedule()
        started = self.connector.started[before:]
        self.starts.append(started)
        self.admitted.append([r.req_id for r in out.scheduled_new_reqs])
        return started

    def land(self, request_ids) -> None:
        self.scheduler._update_from_kv_xfer_finished(
            KVConnectorOutput(finished_recving=set(request_ids))
        )

    def finish(self, request_ids) -> None:
        self.scheduler.finish_requests(request_ids, RequestStatus.FINISHED_STOPPED)

    def queue(self) -> list[str]:
        return [r.request_id for r in self.scheduler.kv_holding_waiting] + [
            r.request_id for r in self.scheduler.waiting
        ]


def test_zero_cap_leaves_the_scheduler_unpatched(pacing):
    schedule = Scheduler.schedule
    handle = Scheduler._handle_blocked_waiting_request
    pacing(0)
    assert Scheduler.schedule is schedule
    assert Scheduler._handle_blocked_waiting_request is handle


def test_patch_is_idempotent(pacing):
    pacing(4)
    schedule = Scheduler.schedule
    handle = Scheduler._handle_blocked_waiting_request
    pacing(4)
    assert Scheduler.schedule is schedule
    assert Scheduler._handle_blocked_waiting_request is handle
    assert inspect.unwrap(handle) is not handle


def test_unpatched_pass_starts_every_waiting_load(monkeypatch):
    run = Run(_scheduler(monkeypatch, max_num_seqs=4))
    ids = [_add(run.scheduler, i) for i in range(10)]
    assert run.step() == ids
    assert run.step() == []


@pytest.mark.parametrize(
    "cap,cls",
    [
        (1, Scheduler),
        (2, Scheduler),
        (4, Scheduler),
        (8, Scheduler),
        (4, AsyncScheduler),
    ],
)
def test_each_pass_starts_at_most_the_cap_in_arrival_order(
    monkeypatch, pacing, cap, cls
):
    pacing(cap)
    run = Run(_scheduler(monkeypatch, max_num_seqs=4, cls=cls))
    ids = [_add(run.scheduler, i) for i in range(10)]
    while len(run.connector.started) < len(ids):
        started = run.step()
        assert 0 < len(started) <= cap
        # Started loads lead the queue, the rest follow in arrival order.
        assert run.queue() == ids
        assert len(run.starts) <= len(ids)
    assert [r for s in run.starts for r in s] == ids
    assert [len(s) for s in run.starts] == [
        min(cap, len(ids) - k) for k in range(0, len(ids), cap)
    ]
    assert run.step() == []
    # Requests whose load landed are admitted as before, oldest first.
    run.land(ids)
    run.step()
    assert run.admitted[-1] == ids[:4]


def test_a_cap_that_never_binds_matches_the_unpatched_scheduler(monkeypatch, pacing):
    def scenario() -> tuple[list[list[str]], list[list[str]]]:
        run = Run(_scheduler(monkeypatch, max_num_seqs=3))
        ids = [_add(run.scheduler, i) for i in range(5)]
        run.step()
        run.land(ids[:2])
        ids += [_add(run.scheduler, i) for i in range(5, 8)]
        run.step()
        run.finish([ids[0]])
        run.land(ids[2:6])
        run.step()
        run.finish(ids[1:3])
        run.step()
        run.land(ids[6:])
        run.step()
        return run.starts, run.admitted

    unpatched = scenario()
    pacing(16)
    assert scenario() == unpatched


def test_every_waiting_request_starts_and_runs_in_order(monkeypatch, pacing):
    """Loads land one pass after they start; a running request finishes
    every other pass. Requests without remote KV (r03, r07) are not paced."""
    cap, max_num_seqs = 2, 3
    pacing(cap)
    run = Run(_scheduler(monkeypatch, max_num_seqs=max_num_seqs))
    local = {3, 7}
    ids = [_add(run.scheduler, i, remote=i not in local) for i in range(12)]
    remote_ids = [r for i, r in enumerate(ids) if i not in local]
    running: list[str] = []
    for index in range(200):
        started = run.step()
        assert len(started) <= cap
        running += run.admitted[-1]
        if index:
            run.land(run.starts[-2])
        if index % 2 and running:
            run.finish([running.pop(0)])
        if len(run.connector.started) == len(remote_ids) and not run.queue():
            break
    assert [r for s in run.starts for r in s] == remote_ids
    admitted = [r for a in run.admitted for r in a]
    assert sorted(admitted) == ids
    # Paced loads and local prefills both go to running oldest first.
    assert [r for r in admitted if r in remote_ids] == remote_ids


def test_aborting_waiting_requests(monkeypatch, pacing):
    pacing(2)
    run = Run(_scheduler(monkeypatch, max_num_seqs=4))
    free_blocks = run.scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
    ids = [_add(run.scheduler, i) for i in range(7)]
    assert run.step() == ids[:2]
    # r02 waits unstarted behind the cap; r00 is loading.
    run.finish([ids[2], ids[0]])
    assert ids[2] not in run.scheduler.requests
    assert ids[2] not in {r.request_id for r in run.scheduler.deferred_waiting}
    assert run.step() == ids[3:5]
    assert run.queue() == [ids[1]] + ids[3:]
    run.land([ids[1], ids[3], ids[4]])
    assert run.step() == ids[5:7]
    assert run.admitted[-1] == [ids[1], ids[3], ids[4]]
    run.land(ids[5:7])
    run.finish([ids[0]])  # already finished: ignored
    assert run.step() == []
    assert run.admitted[-1] == ids[5:6]
    run.finish(ids[1:2] + ids[3:7])
    run.land([ids[0]])  # the aborted load reports in late
    assert not run.scheduler.requests
    # Every scheduled step's output processed: the blocks freed behind the
    # in-flight step fence go back too, and none is missing.
    run.scheduler.processed_step_seq = run.scheduler.sched_step_seq
    run.scheduler._drain_deferred_frees()
    assert (
        run.scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == free_blocks
    )


@pytest.mark.parametrize("role", ["kv_producer", "kv_both"])
def test_engines_that_are_not_pure_consumers_are_not_paced(monkeypatch, pacing, role):
    pacing(2)
    run = Run(_scheduler(monkeypatch, max_num_seqs=4, role=role))
    ids = [_add(run.scheduler, i) for i in range(6)]
    assert run.step() == ids
