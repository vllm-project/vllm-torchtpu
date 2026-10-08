# ==============================================================================
# 3-TIER EXECUTION PROVENANCE BANNER (MANDATORY RED-TEAM AUDIT CONTRACT)
# ==============================================================================
# ANALYTIC / MICROBENCHMARK PROJECTION (NOT E2E PHYSICAL VLLM RUN)
# - Tier 1 (Compiled Host C-ABI Shared Library): NOT INVOKED IN THIS TEST
# - Tier 2 (Compiled Host C++20 LloRegionBuilder IR): NOT INVOKED IN THIS TEST
# - Tier 3 (Execution Provenance): CPU NUMERICAL PARITY & HOST CONTROL-PLANE
#   UNIT TEST ONLY (NOT PHYSICAL TPU EXECUTION).
#   Validates Stage 3 M3 (2x2x4 16-Chip + TPURaidenOffloadingConnector +
#   DSpark K=5 + EP=32) control-plane invariants (RC35-A, RC35-B, RC35-C),
#   Gate 7B partitioned EP=32/TP=2 vs unpartitioned EP=1 numerical parity
#   (TRAP24), and 3-Way Vectorized NumPy baseline fairness (TRAP25).
# ==============================================================================
"""Stage 3 M3 (2x2x4 + TPURaidenOffloadingConnector + DSpark K=5 + EP=32) tests."""

from __future__ import annotations

import dataclasses
import os
import sys
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

for _cand_path in (
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
):
    if os.path.isdir(_cand_path) and _cand_path not in sys.path:
        sys.path.insert(0, _cand_path)

import torch  # noqa: E402


def _install_vllm_test_stubs():
    import contextlib
    import dataclasses
    import importlib.abc
    import importlib.machinery
    import logging
    import types
    from unittest.mock import MagicMock

    try:
        import torch_tpu  # noqa: F401
        import vllm  # noqa: F401

        return
    except ImportError:
        pass

    class _DummyMeta(type):
        def __getattr__(cls, attr: str):
            if attr.startswith("__") and attr.endswith("__"):
                raise AttributeError(attr)
            val = MagicMock(name=f"{cls.__name__}.{attr}")
            setattr(cls, attr, val)
            return val

    class _DummyBase(metaclass=_DummyMeta):
        def __init__(self, *args, **kwargs):
            pass

    class _DummyMod(types.ModuleType):
        def __init__(self, name: str):
            super().__init__(name)
            self.__path__ = []

        def __getattr__(self, attr: str):
            if attr.startswith("__") and attr.endswith("__"):
                raise AttributeError(attr)
            if attr == "_offload_stubs_installed":
                raise AttributeError(attr)
            sub_name = f"{self.__name__}.{attr}"
            if sub_name in sys.modules:
                val = sys.modules[sub_name]
                setattr(self, attr, val)
                return val
            if self.__name__ == "vllm.envs" and attr.isupper():
                setattr(self, attr, None)
                return None
            if attr[0].isupper():
                val = _DummyMeta(attr, (_DummyBase,), {})
            else:
                val = MagicMock(name=sub_name)
            setattr(self, attr, val)
            return val

    class _AutoStubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        PREFIXES = (
            "vllm",
            "torch_tpu",
            "einops",
            "compressed_tensors",
            "transformers",
            "safetensors",
            "portpicker",
            "requests",
        )

        def find_spec(self, fullname, path, target=None):
            if any(
                fullname == p or fullname.startswith(p + ".") for p in self.PREFIXES
            ):
                return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
            return None

        def create_module(self, spec):
            if spec.name in sys.modules:
                return sys.modules[spec.name]
            mod = _DummyMod(spec.name)
            sys.modules[spec.name] = mod
            if "." in spec.name:
                parent_name, child_attr = spec.name.rsplit(".", 1)
                if parent_name in sys.modules:
                    setattr(sys.modules[parent_name], child_attr, mod)
            return mod

        def exec_module(self, module):
            pass

    sys.meta_path.insert(0, _AutoStubFinder())
    stub_modules = [
        "torch_tpu",
        "vllm",
        "vllm.envs",
        "vllm.logger",
        "vllm.config",
        "vllm.compilation",
        "vllm.compilation.backends",
        "vllm.distributed",
        "vllm.distributed.parallel_state",
        "vllm.forward_context",
        "vllm.v1",
        "vllm.v1.outputs",
        "vllm.v1.core",
        "vllm.v1.core.sched",
        "vllm.v1.core.sched.async_scheduler",
        "vllm.v1.core.sched.interface",
        "vllm.v1.core.sched.output",
        "vllm.v1.core.sched.request_queue",
        "vllm.v1.worker",
        "vllm.v1.worker.gpu_model_runner",
        "vllm.v1.kv_offload",
        "vllm.v1.kv_offload.base",
        "vllm.distributed.kv_transfer.kv_connector.v1.base",
        "vllm.distributed.kv_transfer.kv_connector.v1.offloading",
        "vllm.distributed.kv_transfer.kv_connector.v1.offloading.common",
        "vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler",
        "vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker",
        "vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector",
        "vllm.v1.core.sched.scheduler",
        "vllm.v1.kv_cache_interface",
        "vllm.v1.request",
    ]
    for mod_name in stub_modules:
        parts = mod_name.split(".")
        for i in range(1, len(parts) + 1):
            sub = ".".join(parts[:i])
            if sub not in sys.modules:
                m = _DummyMod(sub)
                sys.modules[sub] = m
                if i > 1:
                    setattr(sys.modules[".".join(parts[: i - 1])], parts[i - 1], m)

    @contextlib.contextmanager
    def _noop_cm(*args, **kwargs):
        yield

    sys.modules["vllm.compilation.backends"].set_model_tag = _noop_cm
    sys.modules["vllm.config"].set_current_vllm_config = _noop_cm

    @dataclasses.dataclass
    class LogprobsTensors:
        logprob_token_ids: object = None
        logprobs: object = None
        selected_token_ranks: object = None

    @dataclasses.dataclass
    class LogprobsLists:
        logprob_token_ids: object = None
        logprobs: object = None
        sampled_token_ranks: object = None
        cu_num_generated_tokens: object = None

        def slice_request(self, req_idx: int, num_tokens: int) -> LogprobsLists:
            start = self.cu_num_generated_tokens[req_idx]
            end = start + num_tokens
            return LogprobsLists(
                logprob_token_ids=self.logprob_token_ids[start:end],
                logprobs=self.logprobs[start:end],
                sampled_token_ranks=self.sampled_token_ranks[start:end],
                cu_num_generated_tokens=[0, num_tokens],
            )

    sys.modules["vllm.v1.outputs"].LogprobsTensors = LogprobsTensors
    sys.modules["vllm.v1.outputs"].LogprobsLists = LogprobsLists

    @dataclasses.dataclass
    class ModelRunnerOutput:
        req_ids: list = dataclasses.field(default_factory=list)
        req_id_to_index: dict = dataclasses.field(default_factory=dict)
        sampled_token_ids: list = dataclasses.field(default_factory=list)
        logprobs: object = None
        prompt_logprobs_dict: dict = dataclasses.field(default_factory=dict)
        pooler_output: list = dataclasses.field(default_factory=list)
        kv_connector_output: object = None
        ec_connector_output: object = None
        num_nans_in_logits: dict = dataclasses.field(default_factory=dict)
        finish_sending: object = None
        finish_recving: object = None

    sys.modules["vllm.v1.outputs"].ModelRunnerOutput = ModelRunnerOutput
    sys.modules["vllm.v1.outputs"].EMPTY_MODEL_RUNNER_OUTPUT = ModelRunnerOutput()

    try:
        import torch._dynamo

        if "tpu" not in torch._dynamo.list_backends():

            @torch._dynamo.register_backend(name="tpu")
            def _dummy_tpu_backend(model, inputs):
                return model
    except Exception as e:
        logging.debug("torch._dynamo backend setup skipped: %s", e)

    class _GPUModelRunner:
        def __init__(self, *args, **kwargs):
            pass

    sys.modules["vllm.v1.worker.gpu_model_runner"].GPUModelRunner = _GPUModelRunner

    class _AsyncScheduler:
        def __init__(self, *args, **kwargs):
            pass

        def schedule(self, *args, **kwargs):
            return None

    sys.modules["vllm.v1.core.sched.async_scheduler"].AsyncScheduler = _AsyncScheduler
    sys.modules["vllm.logger"].init_logger = lambda name: logging.getLogger(name)


_install_vllm_test_stubs()


def _ensure_vllm_offload_stubs() -> None:
    """Ensure lightweight vllm offloading base classes exist for CPU unit testing."""
    vllm_mod = sys.modules.get("vllm")
    if (
        vllm_mod is not None
        and vllm_mod.__dict__.get("_offload_stubs_installed") is True
    ):
        return
    try:
        import vllm  # noqa: F401
    except ImportError:
        _install_vllm_test_stubs()
    sys.modules["vllm"]._offload_stubs_installed = True

    import enum
    from itertools import chain
    from typing import NewType

    import vllm.distributed.kv_transfer.kv_connector.v1.base as v1_base
    import vllm.distributed.kv_transfer.kv_connector.v1.offloading.common as off_common
    import vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler as off_sched  # noqa: E501
    import vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker as off_worker
    import vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector as off_conn
    import vllm.v1.core.sched.scheduler as v1_sched_mod
    import vllm.v1.kv_cache_interface as kv_iface
    import vllm.v1.kv_offload.base as kv_base
    import vllm.v1.outputs as v1_outputs
    import vllm.v1.request as v1_req_mod

    class AttentionSpec:
        pass

    class FullAttentionSpec(AttentionSpec):
        pass

    class MambaSpec:
        pass

    kv_iface.AttentionSpec = AttentionSpec
    kv_iface.FullAttentionSpec = FullAttentionSpec
    kv_iface.MambaSpec = MambaSpec

    @dataclasses.dataclass
    class CanonicalKVCacheTensor:
        tensor: torch.Tensor
        page_size_bytes: int = 0
        num_pages: int = 0

    @dataclasses.dataclass
    class CanonicalKVCaches:
        tensors: list[CanonicalKVCacheTensor] = dataclasses.field(default_factory=list)

    kv_base.CanonicalKVCacheTensor = CanonicalKVCacheTensor
    kv_base.CanonicalKVCaches = CanonicalKVCaches

    class RequestStatus(enum.Enum):
        WAITING = 0
        RUNNING = 1
        FINISHED_STOPPED = 2
        FINISHED_ABORTED = 3

    v1_req_mod.RequestStatus = RequestStatus

    if not isinstance(getattr(v1_base, "KVConnectorMetadata", None), type):

        class KVConnectorMetadata:
            pass

        v1_base.KVConnectorMetadata = KVConnectorMetadata

    class KVConnectorBase_V1:
        def __init__(self, vllm_config=None, role=None, kv_cache_config=None):
            self.vllm_config = vllm_config
            self.role = role
            self.kv_cache_config = kv_cache_config
            self._connector_metadata = None

        def handle_preemptions(self, connector_metadata) -> None:
            pass

        def bind_connector_metadata(self, connector_metadata) -> None:
            self._connector_metadata = connector_metadata

        def clear_connector_metadata(self) -> None:
            self._connector_metadata = None

        def _get_connector_metadata(self):
            return getattr(self, "_connector_metadata", None)

        def start_load_kv(self, forward_context, **kwargs) -> None:
            worker = getattr(self, "connector_worker", None)
            if worker is not None and hasattr(worker, "start_load_kv"):
                worker.start_load_kv(self._connector_metadata, **kwargs)

        def get_block_ids_with_load_errors_group_index(self) -> int | None:
            return None

        def get_kv_connector_stats(self):
            return None

    v1_base.KVConnectorBase_V1 = KVConnectorBase_V1

    @dataclasses.dataclass
    class TransferJob:
        req_id: str
        src_spec: object
        dst_spec: object

    @dataclasses.dataclass
    class OffloadingConnectorMetadata(v1_base.KVConnectorMetadata):
        load_jobs: dict = dataclasses.field(default_factory=dict)
        store_jobs: dict = dataclasses.field(default_factory=dict)
        jobs_to_flush: set = dataclasses.field(default_factory=set)
        reqs_to_load: dict = dataclasses.field(default_factory=dict)

    class _FakeTransferStats:
        def aggregate(self, other):
            return self

    @dataclasses.dataclass
    class OffloadingWorkerMetadata:
        completed_jobs: dict = dataclasses.field(default_factory=dict)
        transfer_stats: object = dataclasses.field(default_factory=_FakeTransferStats)

    _FakeTransferStats.__module__ = off_common.__name__
    _FakeTransferStats.__qualname__ = "_FakeTransferStats"
    OffloadingWorkerMetadata.__module__ = off_common.__name__
    OffloadingWorkerMetadata.__qualname__ = "OffloadingWorkerMetadata"
    off_common._FakeTransferStats = _FakeTransferStats
    off_common.TransferJob = TransferJob
    off_common.OffloadingConnectorMetadata = OffloadingConnectorMetadata
    off_common.OffloadingWorkerMetadata = OffloadingWorkerMetadata

    OffloadKey = NewType("OffloadKey", bytes)

    def make_offload_key(block_hash: bytes, group_idx: int) -> OffloadKey:
        return OffloadKey(block_hash + group_idx.to_bytes(4, "big", signed=False))

    def get_offload_block_hash(key: OffloadKey) -> bytes:
        return key[:-4]

    def get_offload_group_idx(key: OffloadKey) -> int:
        return int.from_bytes(key[-4:], "big", signed=False)

    class Medium(enum.Enum):
        CPU = "CPU"
        STORAGE = "STORAGE"

    class Locality(enum.Enum):
        LOCAL = "LOCAL"
        REMOTE = "REMOTE"

    class LookupResult(enum.Enum):
        MISS = 0
        HIT = 1
        HIT_PENDING = 2
        RETRY = 3

    class TransferResult(enum.Enum):
        COMPLETED = 0
        FAILED = 1

    @dataclasses.dataclass
    class ReqContext:
        req_id: str
        kv_transfer_params: dict | None = None

    class RequestOffloadingContext:
        pass

    @dataclasses.dataclass(frozen=True)
    class ScheduleEndContext:
        new_req_ids: tuple = ()
        preempted_req_ids: tuple = ()

    @dataclasses.dataclass(frozen=True)
    class OffloadingEvent:
        keys: list[OffloadKey]
        medium: Medium
        removed: bool
        locality: Locality = Locality.LOCAL

    class LoadStoreSpec:
        pass

    class GPULoadStoreSpec(LoadStoreSpec):
        def __init__(self, block_ids, group_sizes=None, block_indices=None):
            self.block_ids = np.asarray(block_ids, dtype=np.int64)
            self.group_sizes = group_sizes
            self.block_indices = block_indices

    @dataclasses.dataclass
    class PrepareStoreOutput:
        keys_to_store: list[OffloadKey]
        store_spec: LoadStoreSpec
        evicted_keys: list[OffloadKey] = dataclasses.field(default_factory=list)

    @dataclasses.dataclass(frozen=True)
    class CanonicalKVCacheTensor:
        tensor: torch.Tensor
        page_size_bytes: int

    @dataclasses.dataclass(frozen=True)
    class CanonicalKVCacheRef:
        tensor_idx: int
        page_size_bytes: int

    @dataclasses.dataclass(frozen=True)
    class CanonicalKVCaches:
        tensors: list[CanonicalKVCacheTensor]
        group_data_refs: list[list[CanonicalKVCacheRef]]

    class OffloadingManager:
        def complete_load(self, keys, req_context) -> None:
            pass

    class OffloadingSpec:
        def __init__(self, vllm_config=None, kv_cache_config=None):
            self.vllm_config = vllm_config
            self.kv_cache_config = kv_cache_config
            extra = getattr(vllm_config, "extra_config", None)
            if extra is None and vllm_config is not None:
                ktc = getattr(vllm_config, "kv_transfer_config", None)
                extra = getattr(ktc, "kv_connector_extra_config", None)
            self.extra_config = dict(extra) if extra else {}

    class OffloadingWorker:
        pass

    kv_base.OffloadKey = OffloadKey
    kv_base.make_offload_key = make_offload_key
    kv_base.get_offload_block_hash = get_offload_block_hash
    kv_base.get_offload_group_idx = get_offload_group_idx
    kv_base.Medium = Medium
    kv_base.Locality = Locality
    kv_base.LookupResult = LookupResult
    kv_base.TransferResult = TransferResult
    kv_base.ReqContext = ReqContext
    kv_base.RequestOffloadingContext = RequestOffloadingContext
    kv_base.ScheduleEndContext = ScheduleEndContext
    kv_base.OffloadingEvent = OffloadingEvent
    kv_base.PrepareStoreOutput = PrepareStoreOutput
    kv_base.CanonicalKVCacheTensor = CanonicalKVCacheTensor
    kv_base.CanonicalKVCacheRef = CanonicalKVCacheRef
    kv_base.CanonicalKVCaches = CanonicalKVCaches
    kv_base.OffloadingManager = OffloadingManager
    kv_base.OffloadingSpec = OffloadingSpec
    kv_base.OffloadingWorker = OffloadingWorker
    kv_base.LoadStoreSpec = LoadStoreSpec
    kv_base.GPULoadStoreSpec = GPULoadStoreSpec

    @dataclasses.dataclass
    class TransferJobStatus:
        req_id: str
        pending_count: int
        keys: set[OffloadKey]
        is_store: bool
        deferred_fence_block_ids: list[int] | None = None
        fenced_block_ids: list[int] | None = None

    @dataclasses.dataclass(frozen=True)
    class GroupOffloadConfig:
        group_idx: int
        tokens_per_block: int
        tokens_per_chunk: int
        hashes_per_chunk: int
        kv_event_group_spec: object = None
        sliding_window_size_in_chunks: int | None = None
        requires_cow_source: bool = False
        alignment_chunk_count: int | None = None
        is_eagle_group: bool = False

    @dataclasses.dataclass(frozen=True)
    class SchedulerOffloadConfig:
        kv_group_configs: tuple[GroupOffloadConfig, ...]
        blocks_per_chunk: int = 1
        tokens_per_hash: int = 1024
        num_workers: int = 1
        offload_prompt_only: bool = False
        supports_partial_tail: bool = False

    def is_store_reachable_swa_chunk(
        absolute_chunk_index: int,
        storable_chunk_count: int,
        alignment_chunk_count: int | None,
        sliding_window_chunks: int | None,
        is_eagle_group: bool,
    ) -> bool:
        if alignment_chunk_count is None:
            return True
        assert sliding_window_chunks is not None
        position_in_segment = absolute_chunk_index % alignment_chunk_count
        segment_start = absolute_chunk_index - position_in_segment
        actual_segment_length = min(
            alignment_chunk_count, storable_chunk_count - segment_start
        )
        reachable_tail = sliding_window_chunks + int(is_eagle_group)
        return position_in_segment >= actual_segment_length - reachable_tail

    @dataclasses.dataclass
    class RequestGroupState:
        offload_keys: list[OffloadKey] = dataclasses.field(default_factory=list)
        block_ids: list[int] = dataclasses.field(default_factory=list)
        next_stored_chunk_idx: int = 0
        num_hit_chunks: int = 0

    @dataclasses.dataclass
    class RequestOffloadState:
        config: SchedulerOffloadConfig
        req: object
        req_context: ReqContext
        offloading_context: RequestOffloadingContext
        group_states: tuple[RequestGroupState, ...] = dataclasses.field(init=False)
        max_offload_tokens: int | None = None
        num_locally_computed_tokens: int = 0
        transfer_jobs: set[int] = dataclasses.field(default_factory=set)

        def __post_init__(self) -> None:
            self.group_states = tuple(
                RequestGroupState() for _ in self.config.kv_group_configs
            )

        def storable_chunks(
            self,
            group_config: GroupOffloadConfig,
            group_state: RequestGroupState,
            num_offloadable_tokens: int,
        ) -> int:
            num_chunks = num_offloadable_tokens // group_config.tokens_per_chunk
            is_decoding = num_offloadable_tokens > self.req.num_prompt_tokens
            if group_config.is_eagle_group and is_decoding:
                num_chunks = max(0, num_chunks - 1)
            num_allocated_chunks = (
                len(group_state.block_ids) // self.config.blocks_per_chunk
            )
            return min(num_chunks, num_allocated_chunks)

        def advance_stored_idx(self, num_offloadable_tokens: int) -> None:
            for group_config, group_state in zip(
                self.config.kv_group_configs, self.group_states
            ):
                group_state.next_stored_chunk_idx = max(
                    group_state.next_stored_chunk_idx,
                    self.storable_chunks(
                        group_config, group_state, num_offloadable_tokens
                    ),
                )

    class OffloadingConnectorScheduler:
        def __init__(self, spec=None, vllm_config=None, kv_cache_config=None):
            self.spec = spec
            self.config = getattr(spec, "config", SimpleNamespace(num_workers=1))
            self.block_size = 64
            self._group_ids = (0,)
            self._pending_store_jobs = {}
            self._pending_load_reqs = {}
            self._reqs_being_loaded = set()
            self._job_counter = 0

        def has_pending_push_work(self) -> bool:
            return bool(self._pending_store_jobs)

        def reset_cache(self) -> None:
            self._pending_store_jobs.clear()

        def _generate_job_id(self) -> int:
            jid = self._job_counter
            self._job_counter += 1
            return jid

        def _calc_num_offloadable_tokens(
            self, req_status: RequestOffloadState, num_computed_tokens: int
        ) -> int:
            num = min(num_computed_tokens, req_status.req.num_tokens)
            if req_status.max_offload_tokens is not None:
                num = min(num, req_status.max_offload_tokens)
            if self.config.offload_prompt_only:
                num = min(num, req_status.req.num_prompt_tokens)
            return num

        def _touch(self, req_status: RequestOffloadState) -> None:
            for group_config, group_state in zip(
                self.config.kv_group_configs, req_status.group_states
            ):
                if group_config.sliding_window_size_in_chunks is None:
                    self.manager.touch(group_state.offload_keys, req_status.req_context)
                else:
                    chunks_to_skip = max(
                        0,
                        group_state.num_hit_chunks
                        - group_config.sliding_window_size_in_chunks,
                    )
                    self.manager.touch(
                        group_state.offload_keys[chunks_to_skip:],
                        req_status.req_context,
                    )

        def _build_store_jobs(self, scheduler_output) -> dict[int, TransferJob]:
            blocks_per_chunk = self.config.blocks_per_chunk
            store_jobs: dict[int, TransferJob] = {}
            for req_id in chain(
                scheduler_output.num_scheduled_tokens,
                scheduler_output.finished_req_ids or (),
            ):
                req_status = self._req_status.get(req_id)
                if req_status is None:
                    continue
                req = req_status.req
                if req.status is RequestStatus.FINISHED_ABORTED:
                    num_tokens_after_batch = req.num_computed_tokens
                elif req.is_finished():
                    num_tokens_after_batch = req.num_tokens
                else:
                    num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
                    num_tokens_after_batch = (
                        req.num_computed_tokens + num_scheduled_tokens
                    )
                num_offloadable_tokens = self._calc_num_offloadable_tokens(
                    req_status, num_tokens_after_batch
                )
                new_offload_keys: list[OffloadKey] = []
                for group_config, group_state in zip(
                    self.config.kv_group_configs, req_status.group_states
                ):
                    num_chunks = req_status.storable_chunks(
                        group_config, group_state, num_offloadable_tokens
                    )
                    start_chunk_idx = group_state.next_stored_chunk_idx
                    if num_chunks <= start_chunk_idx:
                        continue
                    offload_keys = group_state.offload_keys[start_chunk_idx:num_chunks]
                    offload_block_ids = group_state.block_ids[
                        start_chunk_idx * blocks_per_chunk
                        + blocks_per_chunk
                        - 1 : num_chunks * blocks_per_chunk : blocks_per_chunk
                    ]
                    for key_idx, (offload_key, block_id) in enumerate(
                        zip(offload_keys, offload_block_ids)
                    ):
                        if block_id == 0:
                            continue
                        abs_chunk_idx = start_chunk_idx + key_idx
                        if not is_store_reachable_swa_chunk(
                            abs_chunk_idx,
                            num_chunks,
                            group_config.alignment_chunk_count,
                            group_config.sliding_window_size_in_chunks,
                            group_config.is_eagle_group,
                        ):
                            continue
                        new_offload_keys.append(offload_key)
                if not new_offload_keys:
                    req_status.advance_stored_idx(num_offloadable_tokens)
                    continue
                store_output = self.manager.prepare_store(
                    new_offload_keys, req_status.req_context
                )
                if store_output is None or not store_output.keys_to_store:
                    if store_output is not None:
                        req_status.advance_stored_idx(num_offloadable_tokens)
                    continue
                self._touch(req_status)
                keys_to_store = set(store_output.keys_to_store)
                group_sizes: list[int] = []
                block_indices: list[int] = []
                src_block_ids: list[int] = []
                fenced_block_ids: list[int] = []
                deferred_fence_block_ids: list[int] = []
                for group_config, group_state in zip(
                    self.config.kv_group_configs, req_status.group_states
                ):
                    is_sliding_window = (
                        group_config.sliding_window_size_in_chunks is not None
                    )
                    num_chunks = req_status.storable_chunks(
                        group_config, group_state, num_offloadable_tokens
                    )
                    start_chunk_idx = group_state.next_stored_chunk_idx
                    block_ids = group_state.block_ids
                    num_group_blocks = 0
                    start_gpu_block_idx: int | None = None
                    for idx, offload_key in enumerate(
                        group_state.offload_keys[start_chunk_idx:num_chunks]
                    ):
                        if offload_key not in keys_to_store:
                            continue
                        chunk_idx = start_chunk_idx + idx
                        gpu_block_idx = chunk_idx * blocks_per_chunk
                        for i in range(blocks_per_chunk):
                            block_id = block_ids[gpu_block_idx + i]
                            if block_id == 0:
                                continue
                            if start_gpu_block_idx is None:
                                start_gpu_block_idx = gpu_block_idx + i
                            src_block_ids.append(block_id)
                            num_group_blocks += 1
                            if is_sliding_window:
                                fenced_block_ids.append(block_id)
                            else:
                                deferred_fence_block_ids.append(block_id)
                    group_sizes.append(num_group_blocks)
                    block_indices.append(start_gpu_block_idx or 0)
                    group_state.next_stored_chunk_idx = max(
                        group_state.next_stored_chunk_idx, num_chunks
                    )
                src_spec = GPULoadStoreSpec(
                    src_block_ids, group_sizes=group_sizes, block_indices=block_indices
                )
                dst_spec = store_output.store_spec
                job_id = self._generate_job_id()
                req_status.transfer_jobs.add(job_id)
                for bid in fenced_block_ids:
                    self._block_id_to_pending_jobs.setdefault(bid, set()).add(job_id)
                self._jobs[job_id] = TransferJobStatus(
                    req_id=req_id,
                    pending_count=self.config.num_workers,
                    keys=set(keys_to_store),
                    is_store=True,
                    deferred_fence_block_ids=deferred_fence_block_ids,
                    fenced_block_ids=fenced_block_ids or None,
                )
                store_jobs[job_id] = TransferJob(
                    req_id=req_id, src_spec=src_spec, dst_spec=dst_spec
                )
            return store_jobs

        def update_connector_output(self, connector_output) -> None:
            pass

        def build_connector_meta(self, scheduler_output):
            return OffloadingConnectorMetadata()

    off_sched.TransferJobStatus = TransferJobStatus
    off_sched.GroupOffloadConfig = GroupOffloadConfig
    off_sched.SchedulerOffloadConfig = SchedulerOffloadConfig
    off_sched.RequestGroupState = RequestGroupState
    off_sched.RequestOffloadState = RequestOffloadState
    off_sched.is_store_reachable_swa_chunk = is_store_reachable_swa_chunk
    off_sched.OffloadingConnectorScheduler = OffloadingConnectorScheduler
    _sched_pkg = "vllm.distributed.kv_transfer.kv_connector.v1.offloading"
    sys.modules[f"{_sched_pkg}.scheduler"] = off_sched
    sys.modules[_sched_pkg].scheduler = off_sched

    class OffloadingConnectorWorker:
        def __init__(self, spec=None, vllm_config=None, kv_cache_config=None):
            self.spec = spec
            self.vllm_config = vllm_config
            self.kv_cache_config = kv_cache_config

        def _init_worker(self, canonical_kv_caches) -> None:
            self.canonical_kv_caches = canonical_kv_caches

    off_worker.OffloadingConnectorWorker = OffloadingConnectorWorker

    class OffloadingConnector(v1_base.KVConnectorBase_V1):
        pass

    off_conn.OffloadingConnector = OffloadingConnector

    if not dataclasses.is_dataclass(
        getattr(v1_outputs, "KVConnectorOutput", None)
    ) or not hasattr(getattr(v1_outputs, "KVConnectorOutput", None), "is_empty"):

        @dataclasses.dataclass
        class KVConnectorOutput:
            finished_sending: set[str] | None = None
            finished_recving: set[str] | None = None
            kv_connector_stats: object | None = None
            kv_connector_worker_meta: object | None = None
            invalid_block_ids: set[int] = dataclasses.field(default_factory=set)
            invalid_block_group_index: int | None = None

            def is_empty(self) -> bool:
                return (
                    not self.finished_sending
                    and not self.finished_recving
                    and not self.kv_connector_stats
                    and not self.kv_connector_worker_meta
                    and not self.invalid_block_ids
                )

        KVConnectorOutput.__module__ = "vllm.v1.outputs"
        KVConnectorOutput.__qualname__ = "KVConnectorOutput"
        v1_outputs.KVConnectorOutput = KVConnectorOutput
        import vllm.v1.worker.kv_connector_model_runner_mixin as kv_mixin_mod

        kv_mixin_mod.KVConnectorOutput = KVConnectorOutput
        if "vllm_torchtpu.runner.tpu_runner" in sys.modules:
            sys.modules[
                "vllm_torchtpu.runner.tpu_runner"
            ].KVConnectorOutput = KVConnectorOutput
            sys.modules[
                "vllm_torchtpu.runner.tpu_runner"
            ]._KV_CONNECTOR_OUTPUT_SUPPORTS_INVALID_BLOCK_GROUP = True

    if hasattr(v1_outputs, "ModelRunnerOutput"):
        v1_outputs.ModelRunnerOutput.__module__ = "vllm.v1.outputs"
        v1_outputs.ModelRunnerOutput.__qualname__ = "ModelRunnerOutput"

    if not isinstance(getattr(v1_sched_mod, "Scheduler", None), type):

        class Scheduler:
            def __init__(self, *args, **kwargs):
                pass

        v1_sched_mod.Scheduler = Scheduler


_ensure_vllm_offload_stubs()

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (  # noqa: E402
    GroupOffloadConfig,
    OffloadingConnectorScheduler,
    SchedulerOffloadConfig,
    is_store_reachable_swa_chunk,
)
from vllm.v1.kv_offload.base import OffloadKey, make_offload_key  # noqa: E402
from vllm.v1.outputs import KVConnectorOutput  # noqa: E402

from vllm_torchtpu.core.prefix_aware_scheduler import PrefixAwareScheduler  # noqa: E402
from vllm_torchtpu.offload.raiden_connector import (  # noqa: E402
    RaidenOffloadingConnectorMetadata,
    RaidenOffloadingWorkerMetadata,
    TPURaidenOffloadingConnector,
    TPURaidenOffloadingScheduler,
    _ParkedStore,
)

PROVENANCE_BANNER = (
    "ANALYTIC / MICROBENCHMARK PROJECTION (NOT E2E PHYSICAL VLLM RUN) | "
    "Tier 1: Host C-ABI Shared Library (NOT INVOKED) | "
    "Tier 2: Host C++20 LloRegionBuilder IR (NOT INVOKED) | "
    "Tier 3: CPU Numerical Parity & Host Control-Plane Unit Test (NOT Physical TPU)"
)


# ------------------------------------------------------------------------------
# Check (a): DSpark (layers.61..63) + Target (layers.0..60) KV Registration
# ------------------------------------------------------------------------------


def test_dspark_and_target_kv_cache_registration_in_raiden_connector():
    """Verify 61 Target + 3 DSpark Draft layers register cleanly with HCA pool first."""
    num_blocks = 32
    device = torch.device("cpu")

    # Simulate DeepSeek-V4 + DSpark heterogeneous backing pools:
    # 1. CSA pool (36 KiB / block)
    csa_storage = torch.zeros(
        num_blocks * 36864, dtype=torch.uint8, device=device
    ).untyped_storage()
    # 2. HCA pool (128 KiB / block - largest page, must sort to index 0)
    hca_storage = torch.zeros(
        num_blocks * 131072, dtype=torch.uint8, device=device
    ).untyped_storage()
    # 3. DSpark Draft + SWA overlay pool (8 KiB / block)
    dspark_storage = torch.zeros(
        num_blocks * 8192, dtype=torch.uint8, device=device
    ).untyped_storage()
    # 4. Indexer k/s pool (4 KiB / block)
    indexer_storage = torch.zeros(
        num_blocks * 4096, dtype=torch.uint8, device=device
    ).untyped_storage()

    kv_caches: dict[str, list[torch.Tensor]] = {}
    for layer_id in range(61):
        pool = hca_storage if layer_id in (3, 19, 35, 51) else csa_storage
        page_bytes = 131072 if layer_id in (3, 19, 35, 51) else 36864
        t_main = (
            torch.empty(0, dtype=torch.uint8, device=device)
            .set_(pool)
            .view(num_blocks, page_bytes)
        )
        t_idx = (
            torch.empty(0, dtype=torch.uint8, device=device)
            .set_(indexer_storage)
            .view(num_blocks, 4096)
        )
        kv_caches[f"model.layers.{layer_id}.self_attn.attn"] = [t_main, t_idx]

    # 3 DSpark draft layers (layers.61, 62, 63)
    for draft_layer_id in (61, 62, 63):
        t_draft = (
            torch.empty(0, dtype=torch.uint8, device=device)
            .set_(dspark_storage)
            .view(num_blocks, 8192)
        )
        kv_caches[f"mtp.layers.{draft_layer_id}.self_attn.attn"] = [t_draft]

    connector = TPURaidenOffloadingConnector.__new__(TPURaidenOffloadingConnector)
    connector._fenced_jobs = {}
    connector._completed_jobs = {}
    connector._load_error_block_ids = set()
    connector._pool_sync_tensors = []
    connector._pending_fences = []

    mock_worker = MagicMock()
    mock_worker.kv_cache_config.num_blocks = num_blocks
    # 2 KV cache groups (target + dspark draft)
    mock_worker.kv_cache_config.kv_cache_groups = [MagicMock(), MagicMock()]
    connector.connector_worker = mock_worker

    connector.register_kv_caches(kv_caches)

    # All 4 unique backing storages (CSA, HCA, DSpark, Indexer) are
    # deduplicated and registered
    assert len(connector._pool_sync_tensors) == 4
    mock_worker._init_worker.assert_called_once()
    canonical_kv = mock_worker._init_worker.call_args[0][0]
    page_sizes = [pt.page_size_bytes for pt in canonical_kv.tensors]
    # Sorted descending so HCA 131,072 B is at index 0
    assert page_sizes == [131072, 36864, 8192, 4096]
    assert len(canonical_kv.group_data_refs) == 2
    assert len(canonical_kv.group_data_refs[0]) == 4
    assert len(canonical_kv.group_data_refs[1]) == 4


# ------------------------------------------------------------------------------
# Check (b1): RC35-A — Idle DP Rank KV-Connector Setup & Fence Ordering
# ------------------------------------------------------------------------------


def test_rc35a_idle_dp_rank_records_fence_before_dummy_forward():
    """Verify RC35-A records save fence before idle pairing and retires in
    the same step."""
    connector = TPURaidenOffloadingConnector.__new__(TPURaidenOffloadingConnector)
    connector.connector_worker = MagicMock()
    connector._fenced_jobs = {}
    connector._completed_jobs = {}
    connector._load_error_block_ids = set()
    connector._pool_sync_tensors = [torch.zeros((4, 64), dtype=torch.int8)]
    connector._pending_fences = []

    meta = RaidenOffloadingConnectorMetadata(
        reqs_to_load={},
        fence_job_ids={101, 102},
        finished_store_job_ids=set(),
        finished_load_jobs={},
        failed_load_device_block_ids=set(),
    )
    connector._get_connector_metadata = MagicMock(return_value=meta)

    event_log: list[str] = []
    dummy_forward_completed = {"done": False}

    class FakeEvent:
        def __init__(self):
            # Recorded before dummy forward; retires once dummy forward runs
            self.recorded_before_dummy = not dummy_forward_completed["done"]

    with (
        patch(
            "vllm_torchtpu.offload.raiden_connector._enqueue_pool_writes",
            side_effect=lambda tensors: event_log.append("enqueue_pool_writes"),
        ),
        patch(
            "vllm_torchtpu.offload.raiden_connector._record_fence_event",
            side_effect=lambda: (event_log.append("record_fence_event"), FakeEvent())[
                1
            ],
        ),
        patch(
            "vllm_torchtpu.offload.raiden_connector._event_is_retired",
            side_effect=lambda ev: ev.recorded_before_dummy
            and dummy_forward_completed["done"],
        ),
    ):
        # Step 1: Call record_pending_fences() BEFORE _run_dp_idle_pairing
        connector.record_pending_fences()
        assert meta.fence_job_ids == set(), (
            "fence_job_ids must be cleared after recording"
        )
        assert len(connector._pending_fences) == 1

        # Step 2: Simulate _run_dp_idle_pairing (Target + DSpark dummy forwards)
        event_log.append("run_dp_idle_pairing")
        dummy_forward_completed["done"] = True

        # Step 3: Call get_finished() AFTER _run_dp_idle_pairing
        sending, recving = connector.get_finished(set())
        assert sending == set()
        assert recving == set()

        # Verify ordering: enqueue_pool_writes -> record_fence_event ->
        # run_dp_idle_pairing
        assert event_log == [
            "enqueue_pool_writes",
            "record_fence_event",
            "run_dp_idle_pairing",
        ]
        # Verify fence retired in the SAME step without double-recording!
        assert connector._fenced_jobs == {101: 1, 102: 1}
        assert connector._pending_fences == []


def test_rc35a_negative_control_unretired_fence_stays_pending():
    """Negative control: if device fence has not retired yet, fenced_jobs
    stays empty."""
    connector = TPURaidenOffloadingConnector.__new__(TPURaidenOffloadingConnector)
    connector.connector_worker = MagicMock()
    connector._fenced_jobs = {}
    connector._completed_jobs = {}
    connector._load_error_block_ids = set()
    connector._pool_sync_tensors = [torch.zeros((4, 64), dtype=torch.int8)]
    connector._pending_fences = []

    meta = RaidenOffloadingConnectorMetadata(
        reqs_to_load={},
        fence_job_ids={202},
        finished_store_job_ids=set(),
        finished_load_jobs={},
        failed_load_device_block_ids=set(),
    )
    connector._get_connector_metadata = MagicMock(return_value=meta)

    with (
        patch("vllm_torchtpu.offload.raiden_connector._enqueue_pool_writes"),
        patch(
            "vllm_torchtpu.offload.raiden_connector._record_fence_event",
            return_value=object(),
        ),
        patch(
            "vllm_torchtpu.offload.raiden_connector._event_is_retired",
            return_value=False,
        ),
    ):
        connector.record_pending_fences()
        connector.get_finished(set())
        assert connector._fenced_jobs == {}
        assert len(connector._pending_fences) == 1
        assert connector._pending_fences[0][0] == {202}


def test_rc41_record_pending_fences_handles_none_metadata_safely():
    """RC41: On idle DP ranks, _connector_metadata is None; record_pending_fences
    must not assert."""
    connector = TPURaidenOffloadingConnector.__new__(TPURaidenOffloadingConnector)
    connector.connector_worker = SimpleNamespace()
    connector._pending_fences = []
    connector._pool_sync_tensors = []
    connector._connector_metadata = None

    # Must return cleanly without raising AssertionError or AttributeError
    connector.record_pending_fences()
    assert connector._pending_fences == []

    # Case 2: _connector_metadata not set at all
    delattr(connector, "_connector_metadata")
    connector.record_pending_fences()
    assert connector._pending_fences == []


def test_rc41_record_pending_fences_negative_control_pre_rc41_assert():
    """RC41 Negative control: pre-RC41 self._get_connector_metadata() asserted
    not None."""
    connector = TPURaidenOffloadingConnector.__new__(TPURaidenOffloadingConnector)
    connector._connector_metadata = None

    def legacy_get_connector_metadata():
        assert connector._connector_metadata is not None, (
            "AssertionError on None metadata"
        )
        return connector._connector_metadata

    with pytest.raises(AssertionError):
        legacy_get_connector_metadata()


def test_rc41_tpu_worker_execute_dummy_batch_record_fences_exception_safe():
    """RC41: Even if record_pending_fences raises an unexpected exception,
    execute_dummy_batch catches it."""
    mock_runner = MagicMock()
    mock_runner._dp_step_executed_in_empty_execute_model = False
    mock_runner._dp_coordinated_step.return_value = (SimpleNamespace(), 1)

    mock_kv_connector = MagicMock()
    mock_kv_connector.record_pending_fences.side_effect = RuntimeError(
        "Simulated crash"
    )

    from vllm_torchtpu.worker.tpu_worker import TPUWorker

    worker = object.__new__(TPUWorker)
    worker.model_runner = mock_runner

    with (
        patch(
            "vllm_torchtpu.worker.tpu_worker.has_kv_transfer_group", return_value=True
        ),
        patch(
            "vllm_torchtpu.worker.tpu_worker.get_kv_transfer_group",
            return_value=mock_kv_connector,
        ),
    ):
        # Must not raise RuntimeError
        worker.execute_dummy_batch()

    # Verify dummy pairing forward still ran
    mock_runner._run_dp_idle_pairing.assert_called_once()


def test_rc42_get_finished_safe_none_metadata_and_negative_control():
    """RC42: TPURaidenOffloadingConnector.get_finished must handle None
    metadata safely and not assert."""
    connector = TPURaidenOffloadingConnector.__new__(TPURaidenOffloadingConnector)
    connector.connector_worker = SimpleNamespace()
    connector._pending_fences = []
    connector._pool_sync_tensors = []
    connector._fenced_jobs = {}
    connector._completed_jobs = {}
    connector._load_error_block_ids = set()
    connector._connector_metadata = None

    # Case 1: _connector_metadata is None (idle DP step or after
    # clear_connector_metadata)
    sent, recvd = connector.get_finished(set())
    assert sent == set()
    assert recvd == set()

    # Case 2: _connector_metadata deleted
    delattr(connector, "_connector_metadata")
    sent, recvd = connector.get_finished(set())
    assert sent == set()
    assert recvd == set()

    # Case 3: connector_worker is None
    connector.connector_worker = None
    sent, recvd = connector.get_finished(set())
    assert sent == set()
    assert recvd == set()

    # Negative control: pre-RC42 logic asserted meta was
    # RaidenOffloadingConnectorMetadata
    connector._connector_metadata = None

    def legacy_get_finished():
        assert connector._connector_metadata is not None
        assert isinstance(
            connector._connector_metadata, RaidenOffloadingConnectorMetadata
        )

    with pytest.raises(AssertionError):
        legacy_get_finished()


def test_rc42_tpu_runner_maybe_setup_kv_connector_none_metadata_safe():
    """RC42: maybe_setup_kv_connector must return early without asserting
    when metadata is None."""
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    runner = TPUModelRunner.__new__(TPUModelRunner)
    mock_sched_out = SimpleNamespace(kv_connector_metadata=None)

    with (
        patch(
            "vllm_torchtpu.runner.tpu_runner.has_kv_transfer_group", return_value=True
        ),
        patch(
            "vllm_torchtpu.runner.tpu_runner.get_kv_transfer_group"
        ) as mock_get_group,
    ):
        runner.maybe_setup_kv_connector(mock_sched_out)
        mock_get_group.assert_not_called()


def test_rc42_hybrid_tp2_dp8_ep16_invariants_and_gate7b_parity():
    """RC42: Verify hybrid TP=2, DP=8, EP=16 topology invariants and Gate 7B
    (TRAP24) precision parity."""
    num_target_experts = 384
    num_dspark_experts = 160
    ep_size = 16
    tp_size = 2
    dp_size = 8
    total_chips = tp_size * dp_size
    assert total_chips == 16, "TP=2 * DP=8 must equal 16 TPU chips"

    assert num_target_experts % ep_size == 0
    assert num_dspark_experts % ep_size == 0
    g_local_target = num_target_experts // ep_size
    g_local_dspark = num_dspark_experts // ep_size
    assert g_local_target == 24, "384 / 16 must be 24 local target experts per rank"
    assert g_local_dspark == 10, "160 / 16 must be 10 local DSpark experts per rank"

    # Gate 7B precision parity (TRAP24):
    torch.manual_seed(20261006)
    num_tokens = 64
    hidden_size = 256
    expert_outputs_bf16 = torch.randn(
        (ep_size, num_tokens, hidden_size), dtype=torch.bfloat16
    )

    # Golden unpartitioned reference (float32 accumulation across all 16 EP shards)
    golden_ref = torch.sum(expert_outputs_bf16.to(torch.float32), dim=0).to(
        torch.bfloat16
    )

    # Hybrid TP=2, DP=8, EP=16: TP=2 split in float32 + EP=16 reduction
    tp0_half = (
        expert_outputs_bf16[:, :, : hidden_size // 2].to(torch.float32).sum(dim=0)
    )
    tp1_half = (
        expert_outputs_bf16[:, :, hidden_size // 2 :].to(torch.float32).sum(dim=0)
    )
    hybrid_merged = torch.cat([tp0_half, tp1_half], dim=-1).to(torch.bfloat16)

    cossim, sqnr = _compute_cossim_and_sqnr(golden_ref, hybrid_merged)
    assert cossim >= 0.99999, f"TP=2 DP=8 EP=16 CosSim {cossim} < 0.99999"
    assert sqnr >= 45.0, f"TP=2 DP=8 EP=16 SQNR {sqnr} dB < 45.0 dB"


def test_rc43_get_connector_metadata_safe_override_and_negative_control():
    """RC43: TPURaidenOffloadingConnector._get_connector_metadata must safely
    return getattr(self, '_connector_metadata', None) without asserting."""
    connector = TPURaidenOffloadingConnector.__new__(TPURaidenOffloadingConnector)

    # Path 1: metadata is None (idle DP step) -> must return None without
    # raising AssertionError
    connector._connector_metadata = None
    assert connector._get_connector_metadata() is None

    # Path 2: metadata is unset (deleted attribute) -> must return None
    # without raising AttributeError or AssertionError
    delattr(connector, "_connector_metadata")
    assert connector._get_connector_metadata() is None

    # Path 3: metadata is set to valid object -> must return the exact object
    mock_meta = SimpleNamespace(tag="valid_metadata")
    connector._connector_metadata = mock_meta
    assert connector._get_connector_metadata() is mock_meta

    # Negative control: legacy un-overridden base class method asserted not None
    class LegacyOffloadingConnector:
        _connector_metadata = None

        def _get_connector_metadata(self):
            assert self._connector_metadata is not None, "Legacy assertion failed"
            return self._connector_metadata

    legacy = LegacyOffloadingConnector()
    with pytest.raises(AssertionError, match="Legacy assertion failed"):
        legacy._get_connector_metadata()


def _make_real_raiden_connector_for_idle_dp(
    call_trace: list[str] | None = None,
) -> TPURaidenOffloadingConnector:
    connector = TPURaidenOffloadingConnector.__new__(TPURaidenOffloadingConnector)
    connector._connector_metadata = None
    connector._fenced_jobs = {}
    connector._completed_jobs = {}
    connector._load_error_block_ids = set()
    connector._pool_sync_tensors = [torch.zeros((4, 64), dtype=torch.int8)]
    connector._pending_fences = []

    class _WorkerStub:
        def start_load_kv(self, meta, **kwargs):
            if call_trace is not None:
                call_trace.append(
                    f"start_load_kv:wait={kwargs.get('wait_for_completion', False)}:"
                    f"report={kwargs.get('report_completion', True)}"
                )

    connector.connector_worker = _WorkerStub()  # type: ignore[assignment]
    return connector


def test_rc35a_tpu_runner_execute_model_idle_dp_rank_with_kv_connector():
    """Verify execute_model idle DP rank calls REAL maybe_setup_kv_connector,
    record_pending_fences before idle pairing, REAL get_finished,
    and module-level _build_kv_connector_output without mutating
    EMPTY_MODEL_RUNNER_OUTPUT.
    """
    from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT

    import vllm_torchtpu.runner.tpu_runner as tpu_runner_mod
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    tpu_runner_mod.KVConnectorOutput = KVConnectorOutput
    tpu_runner_mod._KV_CONNECTOR_OUTPUT_SUPPORTS_INVALID_BLOCK_GROUP = True

    runner = TPUModelRunner.__new__(TPUModelRunner)
    runner.execute_model_state = None
    runner.scheduler_config = SimpleNamespace(async_scheduling=False)
    runner.vllm_config = SimpleNamespace()
    runner._stage_timing = SimpleNamespace(take_samples=lambda: ({}, {}))
    runner._update_states = MagicMock()
    runner._dp_lockstep_enabled = MagicMock(return_value=True)
    runner._dp_step_has_active_decode = True

    def _coord_step(*args, **kwargs):
        runner._dp_step_has_active_decode = True
        return (256, 4), 2

    runner._dp_coordinated_step = MagicMock(side_effect=_coord_step)

    call_trace: list[str] = []
    dummy_forward_done = {"done": False}
    connector = _make_real_raiden_connector_for_idle_dp(call_trace)

    orig_handle_preemptions = connector.handle_preemptions
    orig_bind = connector.bind_connector_metadata
    orig_clear = connector.clear_connector_metadata

    def _traced_handle_preemptions(meta):
        call_trace.append("handle_preemptions")
        orig_handle_preemptions(meta)

    def _traced_bind(meta):
        call_trace.append("bind_connector_metadata")
        orig_bind(meta)

    def _traced_clear():
        call_trace.append("clear_connector_metadata")
        orig_clear()

    connector.handle_preemptions = _traced_handle_preemptions  # type: ignore[method-assign]
    connector.bind_connector_metadata = _traced_bind  # type: ignore[method-assign]
    connector.clear_connector_metadata = _traced_clear  # type: ignore[method-assign]

    def _idle_pairing(bucket, chunks):
        call_trace.append(f"run_dp_idle_pairing:{bucket}:{chunks}")
        dummy_forward_done["done"] = True

    runner._run_dp_idle_pairing = _idle_pairing  # type: ignore[method-assign]

    meta = RaidenOffloadingConnectorMetadata(
        reqs_to_load={},
        fence_job_ids={101},
        finished_store_job_ids={303},
        finished_load_jobs={},
        failed_load_device_block_ids=set(),
    )
    sched_out = SimpleNamespace(
        total_num_scheduled_tokens=0,
        scheduled_encoder_inputs={},
        kv_connector_metadata=meta,
        finished_req_ids=set(),
    )

    class _FenceEvent:
        def __init__(self):
            self.recorded_before_dummy = not dummy_forward_done["done"]

    with (
        patch(
            "vllm_torchtpu.runner.tpu_runner.has_kv_transfer_group", return_value=True
        ),
        patch(
            "vllm_torchtpu.runner.tpu_runner.get_kv_transfer_group",
            return_value=connector,
        ),
        patch(
            "vllm_torchtpu.runner.tpu_runner.dist_utils.get_raiden_inline_load",
            return_value=False,
        ),
        patch(
            "vllm_torchtpu.offload.raiden_connector._enqueue_pool_writes",
            side_effect=lambda tensors: call_trace.append("enqueue_pool_writes"),
        ),
        patch(
            "vllm_torchtpu.offload.raiden_connector._record_fence_event",
            side_effect=lambda: (
                call_trace.append("record_fence_event"),
                _FenceEvent(),
            )[1],
        ),
        patch(
            "vllm_torchtpu.offload.raiden_connector._event_is_retired",
            side_effect=lambda ev: ev.recorded_before_dummy
            and dummy_forward_done["done"],
        ),
    ):
        out = runner.execute_model(sched_out)

    assert call_trace == [
        "handle_preemptions",
        "bind_connector_metadata",
        "start_load_kv:wait=False:report=True",
        "enqueue_pool_writes",
        "record_fence_event",
        "run_dp_idle_pairing:(256, 4):2",
        "clear_connector_metadata",
    ]
    assert out is not EMPTY_MODEL_RUNNER_OUTPUT
    assert out.kv_connector_output is not None
    assert isinstance(
        out.kv_connector_output.kv_connector_worker_meta,
        RaidenOffloadingWorkerMetadata,
    )
    assert out.kv_connector_output.kv_connector_worker_meta.completed_jobs == {303: 1}
    assert out.kv_connector_output.kv_connector_worker_meta.fenced_jobs == {101: 1}
    assert out.dp_step_has_active_decode is True
    assert not hasattr(EMPTY_MODEL_RUNNER_OUTPUT, "dp_step_has_active_decode")
    assert EMPTY_MODEL_RUNNER_OUTPUT.kv_connector_output is None
    assert connector._get_connector_metadata() is None


def test_rc38a_idle_dp_rank_real_get_finished_kv_transfers_6tuple_and_worker_meta():
    """Verify RC38-A: Idle DP rank executes real 6-tuple get_finished_kv_transfers and
    module-level _build_kv_connector_output for both non-empty and empty
    KVConnectorOutput.
    """
    from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT

    import vllm_torchtpu.runner.tpu_runner as tpu_runner_mod
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    tpu_runner_mod.KVConnectorOutput = KVConnectorOutput
    tpu_runner_mod._KV_CONNECTOR_OUTPUT_SUPPORTS_INVALID_BLOCK_GROUP = True

    runner = TPUModelRunner.__new__(TPUModelRunner)
    runner.execute_model_state = None
    runner.scheduler_config = SimpleNamespace(async_scheduling=False)
    runner.vllm_config = SimpleNamespace()
    runner._stage_timing = SimpleNamespace(take_samples=lambda: ({}, {}))
    runner._update_states = MagicMock()
    runner._dp_lockstep_enabled = MagicMock(return_value=True)
    runner._run_dp_idle_pairing = MagicMock()

    connector = _make_real_raiden_connector_for_idle_dp()

    # Case 1: Non-empty KVConnectorOutput (completed store 303, completed
    # load 404 -> req_loaded_1, retired fence 101, load error block 19, inline
    # mode=True) with active peer decode=True.
    def _coord_active(*args, **kwargs):
        runner._dp_step_has_active_decode = True
        return 256, 1

    runner._dp_coordinated_step = MagicMock(side_effect=_coord_active)
    meta_non_empty = RaidenOffloadingConnectorMetadata(
        reqs_to_load={},
        fence_job_ids={101},
        finished_store_job_ids={303},
        finished_load_jobs={404: "req_loaded_1"},
        failed_load_device_block_ids={19},
    )
    sched_non_empty = SimpleNamespace(
        total_num_scheduled_tokens=0,
        scheduled_encoder_inputs={},
        kv_connector_metadata=meta_non_empty,
        finished_req_ids=set(),
    )

    with (
        patch(
            "vllm_torchtpu.runner.tpu_runner.has_kv_transfer_group", return_value=True
        ),
        patch(
            "vllm_torchtpu.runner.tpu_runner.get_kv_transfer_group",
            return_value=connector,
        ),
        patch(
            "vllm_torchtpu.runner.tpu_runner.dist_utils.get_raiden_inline_load",
            return_value=True,
        ),
        patch("vllm_torchtpu.offload.raiden_connector._enqueue_pool_writes"),
        patch(
            "vllm_torchtpu.offload.raiden_connector._record_fence_event",
            return_value=object(),
        ),
        patch(
            "vllm_torchtpu.offload.raiden_connector._event_is_retired",
            return_value=True,
        ),
    ):
        out_non_empty = runner.execute_model(sched_non_empty)

    assert out_non_empty is not EMPTY_MODEL_RUNNER_OUTPUT
    assert out_non_empty.dp_step_has_active_decode is True
    kv_out = out_non_empty.kv_connector_output
    assert kv_out is not None
    assert not kv_out.is_empty()
    assert kv_out.finished_sending == set()
    assert kv_out.finished_recving == {"req_loaded_1"}
    assert kv_out.invalid_block_ids == {19}
    assert kv_out.kv_connector_worker_meta is not None
    assert kv_out.kv_connector_worker_meta.completed_jobs == {303: 1, 404: 1}
    assert kv_out.kv_connector_worker_meta.fenced_jobs == {101: 1}
    assert connector._get_connector_metadata() is None

    # Case 2: Empty KVConnectorOutput on idle DP rank (no fences, no completed
    # jobs, no errors)
    # and peer decode=False -> kv_connector_output.is_empty() is True, returns copy of
    # EMPTY_MODEL_RUNNER_OUTPUT with dp_step_has_active_decode=False and
    # kv_connector_output=None.
    def _coord_idle(*args, **kwargs):
        runner._dp_step_has_active_decode = False
        return 256, 1

    runner._dp_coordinated_step = MagicMock(side_effect=_coord_idle)
    meta_empty = RaidenOffloadingConnectorMetadata(
        reqs_to_load={},
        fence_job_ids=set(),
        finished_store_job_ids=set(),
        finished_load_jobs={},
        failed_load_device_block_ids=set(),
    )
    sched_empty = SimpleNamespace(
        total_num_scheduled_tokens=0,
        scheduled_encoder_inputs={},
        kv_connector_metadata=meta_empty,
        finished_req_ids=set(),
    )

    with (
        patch(
            "vllm_torchtpu.runner.tpu_runner.has_kv_transfer_group", return_value=True
        ),
        patch(
            "vllm_torchtpu.runner.tpu_runner.get_kv_transfer_group",
            return_value=connector,
        ),
        patch(
            "vllm_torchtpu.runner.tpu_runner.dist_utils.get_raiden_inline_load",
            return_value=False,
        ),
    ):
        out_empty = runner.execute_model(sched_empty)

    assert out_empty is not EMPTY_MODEL_RUNNER_OUTPUT
    assert out_empty.kv_connector_output is None
    assert out_empty.dp_step_has_active_decode is False
    assert not hasattr(EMPTY_MODEL_RUNNER_OUTPUT, "dp_step_has_active_decode")
    assert EMPTY_MODEL_RUNNER_OUTPUT.kv_connector_output is None


# ------------------------------------------------------------------------------
# Check (b2): RC35-B — _fence_pending in has_pending_push_work & reset_cache
# ------------------------------------------------------------------------------


def _make_mock_raiden_scheduler(num_workers: int = 2) -> TPURaidenOffloadingScheduler:
    sched = TPURaidenOffloadingScheduler.__new__(TPURaidenOffloadingScheduler)
    sched.config = SimpleNamespace(num_workers=num_workers)
    sched.block_size = 64
    sched._group_ids = (0,)
    sched._raiden_manager = MagicMock()
    sched._pending_store_jobs = {}
    sched._pending_load_reqs = {}
    sched._reqs_being_loaded = set()
    sched._NextJobId = 1
    sched._fence_pending = {}
    return sched


def test_rc35b_has_pending_push_work_and_reset_cache_three_paths():
    """Verify RC35-B across (1) normal ack, (2) reset/cancel crash path,
    (3) repeat call."""
    sched = _make_mock_raiden_scheduler(num_workers=2)

    # Initially no pending push work
    assert not sched.has_pending_push_work()

    # Park job 10 in _fence_pending (popped from _pending_store_jobs by
    # build_connector_meta)
    key_a = make_offload_key(b"hash_a_012345678", 0)
    sched._fence_pending[10] = _ParkedStore(
        keys=[key_a],
        device_block_ids=[5],
        acks=0,
    )

    # Path 1: While parked in _fence_pending (even with _pending_store_jobs empty),
    # has_pending_push_work() MUST return True
    assert sched.has_pending_push_work() is True

    # 1st worker ack arrives -> still 1 ack short of num_workers=2
    out_w0 = KVConnectorOutput(
        kv_connector_worker_meta=RaidenOffloadingWorkerMetadata(
            completed_jobs={}, fenced_jobs={10: 1}
        )
    )
    with patch.object(OffloadingConnectorScheduler, "update_connector_output"):
        sched.update_connector_output(out_w0)
    assert sched.has_pending_push_work() is True
    sched._raiden_manager.submit_store_job.assert_not_called()

    # 2nd worker ack arrives -> job 10 transitions out of _fence_pending to
    # submit_store_job
    out_w1 = KVConnectorOutput(
        kv_connector_worker_meta=RaidenOffloadingWorkerMetadata(
            completed_jobs={}, fenced_jobs={10: 1}
        )
    )
    with patch.object(OffloadingConnectorScheduler, "update_connector_output"):
        sched.update_connector_output(out_w1)
    assert 10 not in sched._fence_pending
    sched._raiden_manager.submit_store_job.assert_called_once_with(10, [key_a], [5])

    # Path 2 (Crash / Reset path): Park job 11 in _fence_pending, then call
    # reset_cache()
    key_b = make_offload_key(b"hash_b_012345678", 0)
    sched._fence_pending[11] = _ParkedStore(
        keys=[key_b],
        device_block_ids=[9],
        acks=1,
    )
    assert sched.has_pending_push_work() is True

    with patch.object(OffloadingConnectorScheduler, "reset_cache"):
        sched.reset_cache()

    sched._raiden_manager.on_store_job_cancelled.assert_called_once_with([key_b])
    assert sched._fence_pending == {}

    # Path 3 (Repeat / Late-Ack path): Late worker ack for cancelled job 11
    # arrives after reset_cache()
    late_ack = KVConnectorOutput(
        kv_connector_worker_meta=RaidenOffloadingWorkerMetadata(
            completed_jobs={}, fenced_jobs={11: 1}
        )
    )
    with patch.object(OffloadingConnectorScheduler, "update_connector_output"):
        sched.update_connector_output(late_ack)
    # Must be ignored cleanly without submitting cancelled job 11
    assert sched._raiden_manager.submit_store_job.call_count == 1


# ------------------------------------------------------------------------------
# Check (c): 2x2x4 EP=32 (g_local=12) & Gate 7B Partitioned vs Unpartitioned Parity
# ------------------------------------------------------------------------------


def _compute_cossim_and_sqnr(
    ref: torch.Tensor, cand: torch.Tensor
) -> tuple[float, float]:
    ref_f64 = ref.to(torch.float64).flatten()
    cand_f64 = cand.to(torch.float64).flatten()
    cossim = torch.nn.functional.cosine_similarity(
        ref_f64.unsqueeze(0), cand_f64.unsqueeze(0)
    ).item()
    noise = ref_f64 - cand_f64
    sig_power = torch.sum(ref_f64 * ref_f64).item()
    noise_power = torch.sum(noise * noise).item()
    if noise_power == 0.0:
        return cossim, float("inf")
    sqnr_db = 10.0 * np.log10(sig_power / noise_power)
    return cossim, float(sqnr_db)


def test_ep32_topology_invariants_and_gate7b_sharded_merge_parity():
    """Verify 2x2x4 EP=32 (g_local=12) alignment invariants and Gate 7B
    (TRAP24) parity."""
    num_target_experts = 384
    num_dspark_experts = 160
    ep_size = 32
    rowblk = 2
    alignment_slot_field = 64

    assert num_target_experts % ep_size == 0
    assert num_dspark_experts % ep_size == 0
    g_local_target = num_target_experts // ep_size
    g_local_dspark = num_dspark_experts // ep_size
    assert g_local_target == 12
    assert g_local_dspark == 5
    # Verify worst-case alignment pad overhead <= ALIGNMENT_SLOT_FIELD (64)
    assert (rowblk - 1) * (ep_size - 1) == 31 <= alignment_slot_field
    assert (rowblk - 1) * (g_local_target - 1) == 11 <= alignment_slot_field

    # Gate 7B (TRAP24_SHARDED_STATE_MERGE_PRECISION_DRIFT):
    # Compare Unpartitioned (EP=1, TP=1) vs:
    #   Arm A: DP=32, EP=32, TP=1 (32 expert shards of g_local=12 experts)
    #   Arm B: TP=2, DP=16, EP=32 (32 expert shards + TP=2 intermediate split)
    torch.manual_seed(20261006)
    num_tokens = 64
    hidden_size = 256

    expert_outputs_bf16 = torch.randn(
        (ep_size, num_tokens, hidden_size), dtype=torch.bfloat16
    )

    # Golden Unpartitioned Reference (float32 accumulation across all 384 experts)
    golden_unpartitioned = torch.sum(expert_outputs_bf16.to(torch.float32), dim=0).to(
        torch.bfloat16
    )

    # Candidate Arm A (DP=32, EP=32, TP=1): 2-stage tree reduction in float32
    # (TRAP24 compliant)
    stage1_shards = expert_outputs_bf16.view(4, 8, num_tokens, hidden_size).to(
        torch.float32
    )
    arm_a_merged = torch.sum(torch.sum(stage1_shards, dim=1), dim=0).to(torch.bfloat16)

    # Candidate Arm B (TP=2, DP=16, EP=32): TP=2 half-width accumulation in
    # float32 + AllReduce
    tp0_part = (
        expert_outputs_bf16[:, :, : hidden_size // 2].to(torch.float32).sum(dim=0)
    )
    tp1_part = (
        expert_outputs_bf16[:, :, hidden_size // 2 :].to(torch.float32).sum(dim=0)
    )
    arm_b_merged = torch.cat([tp0_part, tp1_part], dim=-1).to(torch.bfloat16)

    cossim_a, sqnr_a = _compute_cossim_and_sqnr(golden_unpartitioned, arm_a_merged)
    cossim_b, sqnr_b = _compute_cossim_and_sqnr(golden_unpartitioned, arm_b_merged)

    assert cossim_a >= 0.99999, f"Arm A CosSim {cossim_a} < 0.99999"
    assert sqnr_a >= 45.0, f"Arm A SQNR {sqnr_a} dB < 45.0 dB"
    assert cossim_b >= 0.99999, f"Arm B CosSim {cossim_b} < 0.99999"
    assert sqnr_b >= 45.0, f"Arm B SQNR {sqnr_b} dB < 45.0 dB"

    # Negative Control: Naive sequential bf16 accumulation across 32 EP shards drifts!
    naive_bf16_accum = torch.zeros((num_tokens, hidden_size), dtype=torch.bfloat16)
    for shard_idx in range(ep_size):
        naive_bf16_accum = (naive_bf16_accum + expert_outputs_bf16[shard_idx]).to(
            torch.bfloat16
        )
    _, sqnr_naive_bf16 = _compute_cossim_and_sqnr(
        golden_unpartitioned, naive_bf16_accum
    )
    assert sqnr_naive_bf16 < 45.0, (
        f"Negative control failed: naive bf16 merge achieved {sqnr_naive_bf16:.2f} dB"
    )


# ------------------------------------------------------------------------------
# Check (d): RC35-C — PrefixAwareScheduler + Non-Pinning Raiden Warmth + 3-Way NumPy
# ------------------------------------------------------------------------------


def test_rc35c_probe_key_resident_non_pinning_and_prefix_aware_scheduler():
    """Verify RC35-C.1/C.2/C.3 against FakeKVCacheStore tuple API and
    DeepSeek-V4 hybrid MLA+SWA."""
    from tests.offload.test_raiden_store_offload import (
        FakeBlockStatus,
        FakeKVCacheStore,
        _Entry,
        make_global_manager,
        seed_global_entries,
        seed_host_entries,
    )
    from tests.offload.test_raiden_store_offload import (
        sub_hashes as expand_test_sub_hashes,
    )

    store = FakeKVCacheStore()
    mgr = make_global_manager(store, device_block_size_factor=4, num_blocks=32)

    # 1. Verify RC35-C.1 on real FakeKVCacheStore returning
    # list[tuple[bytes, SimpleNamespace(status=...)]]:
    #    - HOST resident key -> True, zero pins taken, _remote_slices untouched
    #    - REMOTE resident key -> True, zero pins taken, _remote_slices untouched
    #    - Negative control A: HBM status WITHOUT active admission -> False
    #      (stale residue)
    #    - Positive control B: HBM status WITH active admission -> True (save in flight)
    #    - Negative control C: Partial sub-hash miss (3 of 4 sub-hashes) -> False
    hashes = [f"dsv4_blk_{i:04d}".encode() for i in range(8)]
    key_host = make_offload_key(hashes[0], 0)
    key_remote = make_offload_key(hashes[1], 0)
    key_stale_hbm = make_offload_key(hashes[2], 0)
    key_inflight_hbm = make_offload_key(hashes[3], 0)
    key_partial_miss = make_offload_key(hashes[4], 0)

    seed_host_entries(store, key_host, device_block_size_factor=4)
    seed_global_entries(store, key_remote, device_block_size_factor=4)

    for sh in expand_test_sub_hashes(key_stale_hbm, device_block_size_factor=4):
        store.entries[sh] = _Entry(status=FakeBlockStatus.HBM)

    for sh in expand_test_sub_hashes(key_inflight_hbm, device_block_size_factor=4):
        store.entries[sh] = _Entry(status=FakeBlockStatus.HBM)
        mgr._hash_to_admission[sh] = object()  # type: ignore[assignment]

    for sh in expand_test_sub_hashes(key_partial_miss, device_block_size_factor=4)[:3]:
        store.entries[sh] = _Entry(status=FakeBlockStatus.HOST)

    assert mgr.probe_key_resident(key_host) is True
    assert mgr.probe_key_resident(key_remote) is True
    assert mgr.probe_key_resident(key_stale_hbm) is False
    assert mgr.probe_key_resident(key_inflight_hbm) is True
    assert mgr.probe_key_resident(key_partial_miss) is False
    assert store.pinned_hashes() == {}, "probe_key_resident must NOT pin any entries"
    assert mgr._remote_slices == {}, "probe_key_resident must NOT mutate _remote_slices"

    # Clean up test entries before DeepSeek-V4 hybrid MLA + SWA scenario
    store.entries.clear()
    store.global_entries.clear()
    mgr._hash_to_admission.clear()

    # 2. Verify RC35-C.2 & RC35-C.3: DeepSeek-V4 Hybrid MLA (full-attn) +
    #    SWA (sliding window)
    #    - Group 0 (MLA full-attn): tokens_per_chunk=4096, hashes_per_chunk=4,
    #      sliding_window=None
    #    - Group 1 (SWA sliding-window): tokens_per_chunk=1024, hashes_per_chunk=1,
    #      sliding_window_size_in_chunks=1, alignment_chunk_count=4
    #    In upstream scheduler.py, is_store_reachable_swa_chunk skips storing
    #    early SWA chunks
    #    0, 1, 2 (segment 0) and 4, 5, 6 (segment 1), storing ONLY trailing
    #    SWA chunks 3 and 7!
    mla_group = GroupOffloadConfig(
        group_idx=0,
        tokens_per_block=1024,
        tokens_per_chunk=4096,
        hashes_per_chunk=4,
        sliding_window_size_in_chunks=None,
        alignment_chunk_count=None,
    )
    swa_group = GroupOffloadConfig(
        group_idx=1,
        tokens_per_block=1024,
        tokens_per_chunk=1024,
        hashes_per_chunk=1,
        sliding_window_size_in_chunks=1,
        alignment_chunk_count=4,
    )
    dsv4_offload_cfg = SchedulerOffloadConfig(
        kv_group_configs=(mla_group, swa_group),
        blocks_per_chunk=1,
        tokens_per_hash=1024,
        num_workers=1,
    )

    # Seed MLA chunks 0 (hashes[3]) and 1 (hashes[7])
    seed_host_entries(store, make_offload_key(hashes[3], 0), device_block_size_factor=4)
    seed_host_entries(store, make_offload_key(hashes[7], 0), device_block_size_factor=4)

    # Seed SWA chunks ONLY where is_store_reachable_swa_chunk is True
    # (chunks 3 and 7; 0..2 and 4..6 skipped!)
    reachable_swa_chunks = []
    for swa_chunk_idx in range(8):
        if is_store_reachable_swa_chunk(
            swa_chunk_idx,
            storable_chunk_count=8,
            alignment_chunk_count=swa_group.alignment_chunk_count,
            sliding_window_chunks=swa_group.sliding_window_size_in_chunks,
            is_eagle_group=False,
        ):
            reachable_swa_chunks.append(swa_chunk_idx)
            seed_host_entries(
                store,
                make_offload_key(hashes[swa_chunk_idx], 1),
                device_block_size_factor=4,
            )
    assert reachable_swa_chunks == [3, 7]

    raiden_sched = _make_mock_raiden_scheduler(num_workers=1)
    raiden_sched.config = dsv4_offload_cfg
    raiden_sched._raiden_manager = mgr

    cold_hashes = [f"cold_blk_{i:04d}".encode() for i in range(8)]
    req_warm_raiden = SimpleNamespace(
        num_tokens=8193,
        num_computed_tokens=0,
        num_preemptions=0,
        skip_reading_prefix_cache=False,
        block_hashes=hashes,
        arrival_time=10.0,
    )
    req_cold = SimpleNamespace(
        num_tokens=8193,
        num_computed_tokens=0,
        num_preemptions=0,
        skip_reading_prefix_cache=False,
        block_hashes=cold_hashes,
        arrival_time=9.0,
    )

    # Both MLA chunks (0, 1) and trailing SWA chunks (3, 7) are resident
    # -> full 8192 tokens!
    ext_warmth = raiden_sched.estimate_external_prefix_warmth(req_warm_raiden, 0)
    assert ext_warmth == 8192
    assert store.pinned_hashes() == {}

    # Step-down verification: if trailing SWA chunk 7 is evicted, warmth
    # steps down to 4096 (segment 0)
    for sh in expand_test_sub_hashes(
        make_offload_key(hashes[7], 1), device_block_size_factor=4
    ):
        del store.entries[sh]
    assert raiden_sched.estimate_external_prefix_warmth(req_warm_raiden, 0) == 4096

    # Negative control: if SWA chunk 3 is ALSO evicted, no aligned segment
    # has SWA coverage -> 0
    for sh in expand_test_sub_hashes(
        make_offload_key(hashes[3], 1), device_block_size_factor=4
    ):
        del store.entries[sh]
    assert raiden_sched.estimate_external_prefix_warmth(req_warm_raiden, 0) == 0

    # Restore SWA chunks 3 and 7 and verify PrefixAwareScheduler._rank_and_warmth
    seed_host_entries(store, make_offload_key(hashes[3], 1), device_block_size_factor=4)
    seed_host_entries(store, make_offload_key(hashes[7], 1), device_block_size_factor=4)

    paw = PrefixAwareScheduler.__new__(PrefixAwareScheduler)
    paw._paw_scoring_broken = False
    paw._paw_ext_scoring_broken = False
    paw.kv_cache_manager = SimpleNamespace(
        enable_caching=True,
        coordinator=SimpleNamespace(
            find_longest_cache_hit=lambda bh, max_len: (None, 0)
        ),
    )
    paw.connector = SimpleNamespace(
        estimate_external_prefix_warmth=raiden_sched.estimate_external_prefix_warmth
    )

    def _prefix_warmth_with_ext(self, request):
        kvm = self.kv_cache_manager
        if not kvm.enable_caching or getattr(
            request, "skip_reading_prefix_cache", False
        ):
            return 0
        if getattr(request, "num_computed_tokens", 0) > 0:
            return request.num_computed_tokens
        max_hit_len = max(getattr(request, "num_tokens", 1) - 1, 0)
        try:
            result = kvm.coordinator.find_longest_cache_hit(
                request.block_hashes, max_hit_len
            )
            local_warmth = int(result[1])
        except Exception:
            return 0
        connector = getattr(self, "connector", None)
        if connector is not None and local_warmth < max_hit_len:
            estimate_ext = getattr(connector, "estimate_external_prefix_warmth", None)
            if callable(estimate_ext):
                try:
                    ext_warmth = int(estimate_ext(request, local_warmth))
                except Exception:
                    if not getattr(self, "_paw_ext_scoring_broken", False):
                        self._paw_ext_scoring_broken = True
                    ext_warmth = 0
                if ext_warmth > 0:
                    local_warmth = min(max_hit_len, local_warmth + ext_warmth)
        return local_warmth

    import types

    paw._prefix_warmth = types.MethodType(_prefix_warmth_with_ext, paw)

    key_raiden, warmth_raiden = paw._rank_and_warmth(req_warm_raiden)
    key_cold, warmth_cold = paw._rank_and_warmth(req_cold)

    assert warmth_raiden == 8192
    assert warmth_cold == 0
    assert key_raiden < key_cold

    # Fault-isolation check: if external connector raises an unexpected error,
    # PrefixAwareScheduler preserves local HBM warmth (2048) instead of returning 0!
    paw_fault = PrefixAwareScheduler.__new__(PrefixAwareScheduler)
    paw_fault._paw_scoring_broken = False
    paw_fault._paw_ext_scoring_broken = False
    paw_fault.kv_cache_manager = SimpleNamespace(
        enable_caching=True,
        coordinator=SimpleNamespace(
            find_longest_cache_hit=lambda bh, max_len: (None, 2048)
        ),
    )

    def _boom_ext(req, local_w):
        raise RuntimeError("simulated external connector error")

    paw_fault.connector = SimpleNamespace(estimate_external_prefix_warmth=_boom_ext)
    paw_fault._prefix_warmth = types.MethodType(_prefix_warmth_with_ext, paw_fault)
    assert paw_fault._prefix_warmth(req_warm_raiden) == 2048
    assert paw_fault._paw_ext_scoring_broken is True
    assert paw_fault._paw_scoring_broken is False


def benchmark_3way_prefix_warmth_ranking(iters: int = 500) -> dict[str, float]:
    """Run 3-Way benchmark for _promote_warmest selection: Scalar Python
    vs Vectorized NumPy vs O(W) Candidate."""
    window = 64
    num_blocks_per_req = 64
    block_size = 64
    rng = np.random.default_rng(42)

    hit_matrix = rng.random((window, num_blocks_per_req)) < 0.75
    hit_matrix = np.cumprod(hit_matrix, axis=1).astype(bool)
    num_tokens = np.full(window, num_blocks_per_req * block_size, dtype=np.int64)
    preempted = (rng.random(window) < 0.1).astype(np.int64)
    arrival_times = np.arange(window, dtype=np.float64)

    # 1. Scalar Python for-loop baseline (evaluates window & sorts tuples)
    def scalar_python_best():
        scored = []
        for i in range(window):
            contig = 0
            for b in range(num_blocks_per_req):
                if bool(hit_matrix[i, b]):
                    contig += 1
                else:
                    break
            warmth = contig * block_size
            cold = float(int(num_tokens[i]) - warmth)
            tier = 0 if int(preempted[i]) > 0 else 1
            scored.append(((tier, cold, float(arrival_times[i])), i))
        scored.sort(key=lambda x: x[0])
        return scored[0][1]

    # 2. Pre-allocated, loop-free Vectorized NumPy baseline (sum + O(W log W) lexsort)
    contig_buf = np.empty((window,), dtype=np.int64)
    warmth_buf = np.empty((window,), dtype=np.int64)
    cold_buf = np.empty((window,), dtype=np.float64)
    tier_buf = np.where(preempted > 0, 0, 1).astype(np.int64)

    def vectorized_numpy_best():
        np.sum(hit_matrix, axis=1, out=contig_buf)
        np.multiply(contig_buf, block_size, out=warmth_buf)
        np.subtract(num_tokens, warmth_buf, out=cold_buf)
        order = np.lexsort((arrival_times, cold_buf, tier_buf))
        return int(order[0])

    # 3. Candidate O(W) composite int64 argmin (pre-packed static request
    # base key - shifted contig)
    composite_key = np.empty((window,), dtype=np.int64)
    arrival_rank = np.arange(window, dtype=np.int64)
    static_base_key = (
        np.left_shift(tier_buf, 48) | np.left_shift(num_tokens, 16) | arrival_rank
    )

    def candidate_composite_best():
        np.sum(hit_matrix, axis=1, out=contig_buf)
        np.left_shift(contig_buf, 22, out=contig_buf)
        np.subtract(static_base_key, contig_buf, out=composite_key)
        return int(np.argmin(composite_key))

    ref_best = scalar_python_best()
    vec_best = vectorized_numpy_best()
    cand_best = candidate_composite_best()
    assert ref_best == vec_best == cand_best

    # Warmup
    for _ in range(100):
        scalar_python_best()
        vectorized_numpy_best()
        candidate_composite_best()

    t0 = time.perf_counter()
    for _ in range(iters):
        scalar_python_best()
    scalar_us = (time.perf_counter() - t0) * 1e6 / iters

    t1 = time.perf_counter()
    for _ in range(iters):
        vectorized_numpy_best()
    vec_us = (time.perf_counter() - t1) * 1e6 / iters

    t2 = time.perf_counter()
    for _ in range(iters):
        candidate_composite_best()
    cand_us = (time.perf_counter() - t2) * 1e6 / iters

    return {
        "scalar_python_us": scalar_us,
        "vectorized_numpy_us": vec_us,
        "candidate_composite_us": cand_us,
    }


def test_check5_three_way_vectorized_numpy_baseline_prefix_warmth_ranking():
    """Check 5: Verify 3-Way fairness (Candidate < Vectorized NumPy < Scalar Python)."""
    res = benchmark_3way_prefix_warmth_ranking(iters=500)
    assert res["vectorized_numpy_us"] < res["scalar_python_us"]
    assert res["candidate_composite_us"] < res["vectorized_numpy_us"]
    # Executes in single-digit microseconds (<< 0.1% of a ~45 ms decode step,
    # TRAP25 <= 5.0%)
    assert res["candidate_composite_us"] < 50.0


if __name__ == "__main__":

    class _CpuDevicePlugin:
        def pytest_configure(self, config):
            config.option.use_tpu = False

    print(PROVENANCE_BANNER)
    perf = benchmark_3way_prefix_warmth_ranking(iters=1000)
    print(
        "[Stage 3 3-Way Prefix Warmth Ranking Summary] "
        f"Scalar Python={perf['scalar_python_us']:.2f} us | "
        f"Vectorized NumPy={perf['vectorized_numpy_us']:.2f} us | "
        f"Candidate Composite={perf['candidate_composite_us']:.2f} us"
    )
    raise SystemExit(
        pytest.main(
            [__file__, "-v", "-p", "no:cacheprovider"],
            plugins=[_CpuDevicePlugin()],
        )
    )
