# SPDX-License-Identifier: Apache-2.0
"""Standalone pure-Python unit tests for lossless KV-cache & scheduler optimizations.

Runs directly with stdlib ``python3 -m unittest`` (no torch/jax/vllm/tpu_sync
required) and also under ``pytest``. Exercises:
- ``envs.py`` registration and default-OFF guarantees
- ``pp_chunk_scheduler.py`` ``TPU_PP_WARM_FIRST_WINDOW``
- ``offload/lookup_log.py`` per-key gap-safe probing + ``tier_runs``
- ``offload/slru.py`` + ``RaidenOffloadingManager`` SLRU scan resistance & pin invariants
- ``offload/victim.py`` ``SupersededCheckpointTracker`` (branch safety, bounds)
- ``offload/raiden_connector.py`` + ``tpu_multi_connector.py`` ``VLLM_TPU_OFFLOAD_DIVERGENT_HITS``
- ``core/arrival_touch.py`` ``VLLM_TPU_ARRIVAL_TOUCH`` (ref_cnt, order, victim stamps, rollback)
"""

from __future__ import annotations

import ast
import enum
import importlib.util
import json
import os
import sys
import types
import unittest
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _load_module_direct(mod_name: str, rel_path: str):
    """Load a pure-Python module directly by file path without importing ``vllm_torchtpu.__init__``."""
    spec = importlib.util.spec_from_file_location(mod_name, SRC_ROOT / rel_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


class _FakeMambaSpec:
    prefix_cacheable = True


class _FakeAttnSpec:
    prefix_cacheable = True


class _FakeLookupResult(enum.Enum):
    MISS = 0
    HIT = 1
    HIT_PENDING = 2


class _FakeMedium(enum.Enum):
    CPU = 0


class _FakeLocality(enum.Enum):
    LOCAL = 0


class _FakeOffloadingManager:
    pass


class _FakeOffloadingSpec:
    pass


class _FakeOffloadingWorker:
    pass


class _FakeLoadStoreSpec:
    pass


class _FakeGPULoadStoreSpec(_FakeLoadStoreSpec):
    def __init__(self, block_ids):
        self.block_ids = block_ids


class _FakeReqContext:
    def __init__(self, req_id: str = "req"):
        self.req_id = req_id


class _FakeOffloadingEvent:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FakePrepareStoreOutput:
    def __init__(self, keys_to_store, store_spec, evicted_keys):
        self.keys_to_store = keys_to_store
        self.store_spec = store_spec
        self.evicted_keys = evicted_keys


class _FakeSchedulingPolicy(enum.Enum):
    FCFS = 0
    PRIORITY = 1


class _FakeFCFSQueue(list):
    def add_request(self, req):
        self.append(req)

    def pop_request(self):
        return self.pop(0)

    def prepend_requests(self, reqs):
        # Match vLLM FCFSRequestQueue.prepend_requests (reverses its input).
        for req in reqs:
            self.insert(0, req)


def _fake_make_offload_key(block_hash: bytes, group_idx: int) -> bytes:
    if isinstance(block_hash, str):
        block_hash = block_hash.encode()
    return block_hash + group_idx.to_bytes(4, "big")


def _fake_get_offload_group_idx(key: bytes) -> int:
    return int.from_bytes(key[-4:], "big", signed=False)


def _new_mamba_spec():
    from vllm.v1.kv_cache_interface import MambaSpec

    if MambaSpec is _FakeMambaSpec:
        return _FakeMambaSpec()
    spec = object.__new__(MambaSpec)
    spec.__dict__["prefix_cacheable"] = True
    return spec


def _new_attn_spec():
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    if FullAttentionSpec is _FakeAttnSpec:
        return _FakeAttnSpec()
    spec = object.__new__(FullAttentionSpec)
    spec.__dict__["prefix_cacheable"] = True
    return spec


def _make_offload_key(block_hash: bytes, group_idx: int) -> bytes:
    from vllm.v1.kv_offload.base import make_offload_key

    return make_offload_key(block_hash, group_idx)


def _new_req_ctx(req_id: str = "req"):
    from vllm.v1.kv_offload.base import ReqContext

    return ReqContext(req_id=req_id)


def _new_fcfs_queue():
    from vllm.v1.core.sched.request_queue import FCFSRequestQueue

    return FCFSRequestQueue()


def _fcfs_policy():
    from vllm.v1.core.sched.request_queue import SchedulingPolicy

    return SchedulingPolicy.FCFS


class _StubEnv:
    """Temporarily provide minimal stubs when torch/vllm/numpy are unavailable."""

    def __enter__(self):
        self._saved: dict[str, object] = {}
        self._pre_keys = set(sys.modules)

        def set_mod(name: str, mod: types.ModuleType) -> types.ModuleType:
            if name not in self._saved:
                self._saved[name] = sys.modules.get(name)
            sys.modules[name] = mod
            if "." in name:
                parent_name, attr = name.rsplit(".", 1)
                if parent_name in sys.modules:
                    setattr(sys.modules[parent_name], attr, mod)
            return mod

        def pkg(name: str) -> types.ModuleType:
            if name in sys.modules:
                return sys.modules[name]
            parts = name.split(".")
            for i in range(1, len(parts) + 1):
                cur = ".".join(parts[:i])
                if cur not in sys.modules:
                    m = types.ModuleType(cur)
                    m.__path__ = []  # type: ignore[attr-defined]
                    set_mod(cur, m)
            return sys.modules[name]

        has_vllm = importlib.util.find_spec("vllm") is not None
        has_torch = importlib.util.find_spec("torch") is not None
        has_numpy = importlib.util.find_spec("numpy") is not None

        if not has_vllm and "vllm_torchtpu" not in sys.modules:
            vt = types.ModuleType("vllm_torchtpu")
            vt.__path__ = [str(SRC_ROOT / "vllm_torchtpu")]  # type: ignore[attr-defined]
            set_mod("vllm_torchtpu", vt)

        if not has_numpy and "numpy" not in sys.modules:
            np_mod = pkg("numpy")
            np_mod.ndarray = object
            np_mod.float32 = float
            np_mod.float64 = float
            np_mod.int64 = int
        if not has_torch and "torch" not in sys.modules:
            torch_mod = pkg("torch")
            torch_mod.Tensor = object
            torch_mod.Event = object
            torch_mod.UntypedStorage = object
            torch_mod.device = object

        if not has_vllm and "vllm" not in sys.modules:
            import logging

            for mod_name in (
                "vllm",
                "vllm.logger",
                "vllm.config",
                "vllm.distributed",
                "vllm.distributed.kv_transfer",
                "vllm.distributed.kv_transfer.kv_connector",
                "vllm.distributed.kv_transfer.kv_connector.v1",
                "vllm.distributed.kv_transfer.kv_connector.v1.base",
                "vllm.distributed.kv_transfer.kv_connector.v1.multi_connector",
                "vllm.distributed.kv_transfer.kv_connector.v1.offloading",
                "vllm.distributed.kv_transfer.kv_connector.v1.offloading.config",
                "vllm.distributed.kv_transfer.kv_connector.v1.offloading.common",
                "vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler",
                "vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker",
                "vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector",
                "vllm.v1",
                "vllm.v1.core",
                "vllm.v1.core.sched",
                "vllm.v1.core.sched.output",
                "vllm.v1.core.sched.request_queue",
                "vllm.v1.core.sched.scheduler",
                "vllm.v1.core.sched.async_scheduler",
                "vllm.v1.kv_cache_interface",
                "vllm.v1.kv_offload",
                "vllm.v1.kv_offload.base",
                "vllm.v1.kv_offload.config",
                "vllm.v1.outputs",
                "vllm.v1.request",
            ):
                pkg(mod_name)

            sys.modules["vllm.logger"].init_logger = logging.getLogger
            sys.modules["vllm.config"].VllmConfig = object
            sys.modules["vllm.v1.outputs"].KVConnectorOutput = object
            sys.modules["vllm.v1.request"].Request = object
            sys.modules["vllm.v1.core.sched.output"].SchedulerOutput = object

            rq = sys.modules["vllm.v1.core.sched.request_queue"]
            rq.SchedulingPolicy = _FakeSchedulingPolicy
            rq.FCFSRequestQueue = _FakeFCFSQueue
            rq.RequestQueue = _FakeFCFSQueue

            class _FakeScheduler:
                def schedule(self, throttle_prefills: bool = False):
                    return None

                def add_request(self, request):
                    self.requests[request.request_id] = request
                    self.waiting.append(request)

            sys.modules["vllm.v1.core.sched.scheduler"].Scheduler = _FakeScheduler
            sys.modules["vllm.v1.core.sched.async_scheduler"].AsyncScheduler = _FakeScheduler

            kvi = sys.modules["vllm.v1.kv_cache_interface"]
            kvi.MambaSpec = _FakeMambaSpec
            kvi.FullAttentionSpec = _FakeAttnSpec
            kvi.AttentionSpec = _FakeAttnSpec
            kvi.KVCacheConfig = object

            kvo = sys.modules["vllm.v1.kv_offload.base"]
            kvo.CanonicalKVCacheRef = SimpleNamespace
            kvo.CanonicalKVCaches = SimpleNamespace
            kvo.CanonicalKVCacheTensor = SimpleNamespace
            kvo.GPULoadStoreSpec = _FakeGPULoadStoreSpec
            kvo.LoadStoreSpec = _FakeLoadStoreSpec
            kvo.Locality = _FakeLocality
            kvo.LookupResult = _FakeLookupResult
            kvo.Medium = _FakeMedium
            kvo.OffloadingEvent = _FakeOffloadingEvent
            kvo.OffloadingManager = _FakeOffloadingManager
            kvo.OffloadingSpec = _FakeOffloadingSpec
            kvo.OffloadingWorker = _FakeOffloadingWorker
            kvo.OffloadKey = bytes
            kvo.PrepareStoreOutput = _FakePrepareStoreOutput
            kvo.ReqContext = _FakeReqContext
            kvo.RequestOffloadingContext = SimpleNamespace
            kvo.ScheduleEndContext = SimpleNamespace
            kvo.TransferResult = SimpleNamespace
            kvo.get_offload_group_idx = _fake_get_offload_group_idx
            kvo.make_offload_key = _fake_make_offload_key
            sys.modules["vllm.v1.kv_offload.config"].OffloadingConfig = object

            sys.modules[
                "vllm.distributed.kv_transfer.kv_connector.v1"
            ].KVConnectorRole = SimpleNamespace(SCHEDULER="sched", WORKER="worker")
            base_m = sys.modules[
                "vllm.distributed.kv_transfer.kv_connector.v1.base"
            ]
            base_m.KVConnectorBase_V1 = object
            base_m.KVConnectorMetadata = object

            class _FakeMultiConnector:
                def __init__(self, connectors):
                    self._connectors = list(connectors)

                @property
                def supports_divergent_local_hybrid_hits(self) -> bool:
                    return all(
                        c.supports_divergent_local_hybrid_hits for c in self._connectors
                    )

            sys.modules[
                "vllm.distributed.kv_transfer.kv_connector.v1.multi_connector"
            ].MultiConnector = _FakeMultiConnector

            @dataclass
            class _FakeOffloadMeta:
                load_jobs: dict = None  # type: ignore[assignment]
                store_jobs: dict = None  # type: ignore[assignment]
                jobs_to_flush: set | None = None

            @dataclass
            class _FakeWorkerMeta:
                completed_jobs: dict = None  # type: ignore[assignment]
                transfer_stats: object = None

            common_m = sys.modules[
                "vllm.distributed.kv_transfer.kv_connector.v1.offloading.common"
            ]
            common_m.OffloadingConnectorMetadata = _FakeOffloadMeta
            common_m.OffloadingWorkerMetadata = _FakeWorkerMeta

            class _FakeOffloadSched:
                def get_num_new_matched_tokens(self, request, num_computed_tokens, *a, **kw):
                    return (0, False)

                def request_finished(self, request, *a, **kw):
                    return (False, None)

            sys.modules[
                "vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler"
            ].OffloadingConnectorScheduler = _FakeOffloadSched
            sys.modules[
                "vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker"
            ].OffloadingConnectorWorker = object
            sys.modules[
                "vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector"
            ].OffloadingConnector = object

        # Only install direct envs/logger stubs when running standalone without vllm.
        if not has_vllm:
            if "vllm_torchtpu.envs" not in sys.modules:
                mod = _load_module_direct("vllm_torchtpu.envs", "vllm_torchtpu/envs.py")
                self._saved["vllm_torchtpu.envs"] = None
                setattr(sys.modules["vllm_torchtpu"], "envs", mod)
            if "vllm_torchtpu.logger" not in sys.modules:
                import logging

                lm = types.ModuleType("vllm_torchtpu.logger")
                lm.init_logger = logging.getLogger
                set_mod("vllm_torchtpu.logger", lm)
        self._stubbed = not has_vllm
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._stubbed:
            for name in list(sys.modules):
                if name not in self._pre_keys and name not in self._saved:
                    sys.modules.pop(name, None)
        for name in list(self._saved):
            orig = self._saved[name]
            if orig is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = orig  # type: ignore[assignment]


class _FakeBlockStatus(enum.Enum):
    INIT = 0
    REMOTE = 1
    HOST = 2
    HBM = 3
    HOST_AND_HBM = 4
    SHARED_STORAGE = 5


class _FakeRaidenBlockId:
    def __init__(self, raiden_id=None, host_block_id=-1, device_block_id=-1, status=None):
        self.raiden_id = raiden_id
        self.host_block_id = host_block_id
        self.device_block_id = device_block_id
        self.status = status


class _LRUStoreEntry:
    def __init__(self, status: _FakeBlockStatus, device_block_id: int = -1, pin_count: int = 0):
        self.status = status
        self.device_block_id = device_block_id
        self.pin_count = pin_count


class CapacityLRUStore:
    """In-memory capacity-bounded mock matching stock C++ ``KVCacheStore`` LRU + pins."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.entries: OrderedDict[bytes, _LRUStoreEntry] = OrderedDict()
        self.raiden_id = object()
        self.eviction_callback = None
        self.storage_entries: set[bytes] = set()
        self.lookup_calls = 0
        self.release_calls = 0
        self.evicted_history: list[bytes] = []
        self._saving: list[bytes] = []
        self._save_done: list[bytes] = []
        self._loading: list[bytes] = []
        self._load_done: list[bytes] = []

    def set_eviction_callback(self, cb) -> None:
        self.eviction_callback = cb

    def lookup(self, block_hashes, enable_global=False, pin_found=True):
        self.lookup_calls += 1
        out = []
        block_hashes = list(block_hashes)
        for h in block_hashes:
            entry = self.entries.get(h)
            if entry is None:
                break
            if pin_found:
                entry.pin_count += 1
            out.append(
                (
                    h,
                    SimpleNamespace(
                        status=entry.status,
                        device_block_id=entry.device_block_id,
                        host_block_id=-1,
                    ),
                )
            )
        for h in block_hashes[len(out) :]:
            if h not in self.storage_entries:
                break
            out.append(
                (
                    h,
                    SimpleNamespace(
                        status=_FakeBlockStatus.SHARED_STORAGE,
                        device_block_id=-1,
                        host_block_id=-1,
                    ),
                )
            )
        return out

    def release(self, block_hashes) -> None:
        self.release_calls += 1
        for h in block_hashes:
            entry = self.entries.get(h)
            assert entry is not None, f"release() called on absent hash {h!r}"
            assert entry.pin_count > 0, f"double release() on hash {h!r}"
            entry.pin_count -= 1
            if entry.pin_count == 0:
                self.entries.move_to_end(h)

    def insert(self, block_hashes, slices, on_host=False) -> bool:
        new_hashes = [h for h in block_hashes if h not in self.entries]
        need = len(self.entries) + len(new_hashes) - self.capacity
        evict_victims: list[bytes] = []
        if need > 0:
            for h, entry in self.entries.items():
                if entry.pin_count == 0:
                    evict_victims.append(h)
                    if len(evict_victims) == need:
                        break
            if len(evict_victims) < need:
                return False
            for h in evict_victims:
                del self.entries[h]
            self.evicted_history.extend(evict_victims)
            # Fire callback synchronously on the SAME thread (tests deadlock freedom!).
            if self.eviction_callback is not None:
                self.eviction_callback(evict_victims)
        for h, s in zip(block_hashes, slices):
            existing = self.entries.get(h)
            if existing is None:
                self.entries[h] = _LRUStoreEntry(
                    status=s.status, device_block_id=s.device_block_id, pin_count=1
                )
            else:
                existing.pin_count += 1
        return True

    def save(self, block_hashes) -> bool:
        for h in block_hashes:
            entry = self.entries.get(h)
            if entry is None or entry.pin_count <= 0:
                return False
        self._saving.extend(block_hashes)
        return True

    def complete_saves(self) -> None:
        for h in self._saving:
            entry = self.entries[h]
            entry.status = _FakeBlockStatus.HOST_AND_HBM
            entry.pin_count -= 1
            if entry.pin_count == 0:
                self.entries.move_to_end(h)
            self._save_done.append(h)
        self._saving.clear()

    def poll_save_status(self):
        done, self._save_done = self._save_done, []
        return done, [], list(self._saving), [], []

    def load(self, block_hashes, device_block_ids) -> bool:
        for h in block_hashes:
            entry = self.entries.get(h)
            if entry is None or entry.pin_count <= 0:
                return False
        self._loading.extend(block_hashes)
        return True

    def complete_loads(self) -> None:
        for h in self._loading:
            entry = self.entries[h]
            entry.pin_count -= 1
            if entry.pin_count == 0:
                self.entries.move_to_end(h)
            self._load_done.append(h)
        self._loading.clear()

    def poll_load_status(self):
        done, self._load_done = self._load_done, []
        return done, [], list(self._loading)

    def pinned_counts(self) -> dict[bytes, int]:
        return {h: e.pin_count for h, e in self.entries.items() if e.pin_count != 0}


def _block_key(n: int, group: int = 4) -> bytes:
    return (f"blk{n:06d}".encode() + b"\x00" * 24)[:32] + group.to_bytes(4, "big")


class TestEnvsAndSyntax(unittest.TestCase):
    def test_new_flags_registered_and_default_off(self):
        envs_path = SRC_ROOT / "vllm_torchtpu" / "envs.py"
        src = envs_path.read_text()
        tree = ast.parse(src)

        typed_names: set[str] = set()
        for node in tree.body:
            if isinstance(node, ast.If) and getattr(node.test, "id", None) == "TYPE_CHECKING":
                for stmt in node.body:
                    if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                        typed_names.add(stmt.target.id)

        expected = {
            "TPU_PP_WARM_FIRST_WINDOW": 0,
            "VLLM_TPU_OFFLOAD_DIVERGENT_HITS": False,
            "VLLM_TPU_OFFLOAD_TIER_EVICT_POLICY": "",
            "VLLM_TPU_OFFLOAD_TIER_SLRU_PROBATION_FRAC": 0.25,
            "VLLM_TPU_OFFLOAD_TIER_SKIP_SUPERSEDED": False,
            "VLLM_TPU_ARRIVAL_TOUCH": False,
        }
        mod = _load_module_direct("_test_envs", "vllm_torchtpu/envs.py")
        try:
            for name, default_val in expected.items():
                self.assertIn(name, typed_names, f"{name} missing from TYPE_CHECKING")
                self.assertIn(name, mod.environment_variables, f"{name} missing from dict")
                with patch.dict(os.environ, {}, clear=False):
                    os.environ.pop(name, None)
                    self.assertEqual(getattr(mod, name), default_val, name)

            with patch.dict(os.environ, {"VLLM_TPU_OFFLOAD_TIER_EVICT_POLICY": "bogus"}):
                with self.assertRaises(ValueError):
                    _ = mod.VLLM_TPU_OFFLOAD_TIER_EVICT_POLICY

            for good in ("", "lru", "LRU", "slru", "SLRU", "slru_all"):
                with patch.dict(os.environ, {"VLLM_TPU_OFFLOAD_TIER_EVICT_POLICY": good}):
                    self.assertEqual(mod.VLLM_TPU_OFFLOAD_TIER_EVICT_POLICY.lower(), good.lower())
        finally:
            sys.modules.pop("_test_envs", None)

    def test_no_ttft_shield_names_introduced(self):
        rel_files = (
            "vllm_torchtpu/core/pp_chunk_scheduler.py",
            "vllm_torchtpu/core/arrival_touch.py",
            "vllm_torchtpu/offload/lookup_log.py",
            "vllm_torchtpu/offload/slru.py",
            "vllm_torchtpu/offload/victim.py",
        )
        for rel in rel_files:
            text = (SRC_ROOT / rel).read_text()
            for banned in ("protect=", "hold=", "deadline=", "maxmix="):
                self.assertNotIn(banned, text, f"{banned} found in {rel}")

    def test_compile_and_annotation_names_valid(self):
        files = (
            "vllm_torchtpu/envs.py",
            "vllm_torchtpu/patch_registry.py",
            "vllm_torchtpu/core/pp_chunk_scheduler.py",
            "vllm_torchtpu/core/arrival_touch.py",
            "vllm_torchtpu/distributed/kv_transfer/tpu_multi_connector.py",
            "vllm_torchtpu/offload/lookup_log.py",
            "vllm_torchtpu/offload/slru.py",
            "vllm_torchtpu/offload/victim.py",
            "vllm_torchtpu/offload/raiden_store.py",
            "vllm_torchtpu/offload/raiden_connector.py",
            "vllm_torchtpu/offload/tier_classes.py",
        )
        for rel in files:
            path = SRC_ROOT / rel
            source = path.read_text()
            compile(source, str(path), "exec")
            tree = ast.parse(source)
            has_future_ann = any(
                isinstance(n, ast.ImportFrom)
                and n.module == "__future__"
                and any(a.name == "annotations" for a in n.names)
                for n in tree.body
            )
            if not has_future_ann:
                # Verify module-level imported/defined names cover any top-level annotations.
                top_names = set(dir(__builtins__)) if isinstance(__builtins__, dict) else set(dir(__builtins__))
                for n in tree.body:
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        top_names.add(n.name)
                    elif isinstance(n, ast.Import):
                        for a in n.names:
                            top_names.add(a.asname or a.name.split(".")[0])
                    elif isinstance(n, ast.ImportFrom):
                        for a in n.names:
                            top_names.add(a.asname or a.name)
                    elif isinstance(n, ast.Assign):
                        for t in n.targets:
                            if isinstance(t, ast.Name):
                                top_names.add(t.id)
                for n in tree.body:
                    if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
                        for sub in ast.walk(n):
                            if isinstance(sub, ast.FunctionDef):
                                ann_nodes = [sub.returns] + [a.annotation for a in sub.args.args + sub.args.kwonlyargs]
                                for an in ann_nodes:
                                    if an is None:
                                        continue
                                    for leaf in ast.walk(an):
                                        if isinstance(leaf, ast.Name):
                                            self.assertIn(
                                                leaf.id,
                                                top_names | {"int", "float", "str", "bool", "bytes", "object", "list", "dict", "set", "tuple", "None"},
                                                f"Unresolved runtime annotation {leaf.id!r} in {rel}",
                                            )


class TestWarmFirstWindow(unittest.TestCase):
    def _make_sched(self, reqs, window: int = 0, warmth=None, aging: float = 0.0):
        pp = sys.modules["vllm_torchtpu.core.pp_chunk_scheduler"]
        cls = pp.TpuPipelineChunkScheduler
        s = cls.__new__(cls)
        s.connector = None
        s._init_warm_first(True, aging, False, window)
        s.policy = _fcfs_policy()
        q = _new_fcfs_queue()
        for r in reqs:
            q.add_request(r)
        s.waiting = q
        warmth_map = warmth or {}
        s.kv_cache_manager = SimpleNamespace(
            enable_caching=True,
            coordinator=SimpleNamespace(
                find_longest_cache_hit=lambda bh, ml: ((), min(warmth_map.get(bh, 0), ml), 0)
            ),
        )
        return s

    @staticmethod
    def _req(rid: str, num_tokens: int, arrival: float, computed: int = 0):
        return SimpleNamespace(
            request_id=rid,
            num_tokens=num_tokens,
            num_prompt_tokens=num_tokens,
            num_computed_tokens=computed,
            num_preemptions=0,
            arrival_time=arrival,
            skip_reading_prefix_cache=False,
            block_hashes=rid,
        )

    def test_window_default_positive_unbounded_and_validation(self):
        with _StubEnv():
            import vllm_torchtpu.core.pp_chunk_scheduler as pp  # noqa: F401

            # Default 3-arg call keeps window=128
            reqs = [self._req(f"c{i}", 100_000, float(i)) for i in range(150)]
            reqs.append(self._req("warm_tail", 100_000, 151.0))
            warmth = {"warm_tail": 98_000}

            s_def = self._make_sched(reqs, window=0, warmth=warmth)
            self.assertEqual(s_def._warm_first_window, 128)
            s_def._order_waiting(now=200.0)
            self.assertEqual(s_def.waiting[-1].request_id, "warm_tail")

            # Larger window reaches warm_tail at index 150
            s_256 = self._make_sched(list(reqs), window=256, warmth=warmth)
            s_256._order_waiting(now=200.0)
            self.assertEqual(s_256.waiting[0].request_id, "warm_tail")

            # Unbounded (-1) scans entire waiting queue
            big = [self._req(f"c{i}", 100_000, float(i)) for i in range(300)]
            big.append(self._req("late_warm", 100_000, 301.0))
            s_all = self._make_sched(big, window=-1, warmth={"late_warm": 99_000})
            self.assertIsNone(s_all._warm_first_window)
            s_all._order_waiting(now=310.0)
            self.assertEqual(s_all.waiting[0].request_id, "late_warm")
            self.assertEqual(len(s_all.waiting), 301)

            # Started requests stay pinned ahead of fresh warm ones
            mixed = [
                self._req("c0", 100_000, 0.0),
                self._req("started", 100_000, 1.0, computed=3200),
                self._req("warm", 100_000, 2.0),
            ]
            s_m = self._make_sched(mixed, window=-1, warmth={"warm": 99_000})
            s_m._order_waiting(now=5.0)
            self.assertEqual([r.request_id for r in s_m.waiting], ["started", "warm", "c0"])

            # Validation rejects window < -1
            sched_cls = pp.TpuPipelineChunkScheduler
            bad = sched_cls.__new__(sched_cls)
            bad.connector = None
            with self.assertRaises(ValueError):
                bad._init_warm_first(True, 0.0, False, -2)

            # Existing 2-arg and 3-arg call signatures remain valid
            compat = sched_cls.__new__(sched_cls)
            compat.connector = None
            compat._init_warm_first(True, 0.0)
            self.assertEqual(compat._warm_first_window, 128)


class TestSlruTracker(unittest.TestCase):
    def setUp(self):
        self.slru_mod = _load_module_direct("_test_slru", "vllm_torchtpu/offload/slru.py")

    def tearDown(self):
        sys.modules.pop("_test_slru", None)

    def test_policy_and_fraction_validation(self):
        norm = self.slru_mod.normalize_evict_policy
        self.assertEqual(norm(""), "lru")
        self.assertEqual(norm(None), "lru")
        self.assertEqual(norm("  LRU "), "lru")
        self.assertEqual(norm("SLRU"), "slru")
        self.assertEqual(norm("slru_all"), "slru_all")
        with self.assertRaises(ValueError):
            norm("fifo")

        val = self.slru_mod.validate_probation_frac
        self.assertEqual(val(0.25), 0.25)
        for bad in (0.0, 1.0, -0.2, 1.5):
            with self.assertRaises(ValueError):
                val(bad)

    def test_promotion_demotion_and_recency_preservation(self):
        t = self.slru_mod.SlruTracker(capacity=4, probation_frac=0.25)  # protected_cap=3
        self.assertEqual(t.protected_cap, 3)
        for k in ("a", "b", "c", "d"):
            t.record_insert(k)
        self.assertTrue(all(t.is_probation(k) for k in ("a", "b", "c", "d")))
        self.assertTrue(t.record_hit("a"))
        self.assertTrue(t.record_hit("b"))
        self.assertTrue(t.record_hit("c"))
        self.assertEqual(t.stats().protected, 3)
        # Rescuing "a" does NOT mark it newer inside _protected.
        t.mark_rescued("a")
        # Promoting a 4th key "d" must demote "a" (the oldest real hit), not "b".
        self.assertTrue(t.record_hit("d"))
        self.assertTrue(t.is_probation("a"))
        self.assertTrue(t.is_protected("b"))
        self.assertTrue(t.is_protected("c"))
        self.assertTrue(t.is_protected("d"))
        self.assertEqual(t.stats().demotes, 1)

    def test_checkpoint_promote_hits_two(self):
        t = self.slru_mod.SlruTracker(capacity=8, probation_frac=0.25, promote_hits=2)
        t.record_insert("ck1")
        # 1st load (single-line continuation): stays in probation.
        self.assertFalse(t.record_hit("ck1"))
        self.assertTrue(t.is_probation("ck1"))
        # 2nd load (shared branch junction!): promoted to protected.
        self.assertTrue(t.record_hit("ck1"))
        self.assertTrue(t.is_protected("ck1"))
        # Explicit demote pushes it back to probation.
        self.assertTrue(t.demote("ck1"))
        self.assertTrue(t.is_probation("ck1"))
        self.assertFalse(t.demote("ck1"))

    def test_rescue_handles_max_rescues_cap_and_all_protected_edge(self):
        t = self.slru_mod.SlruTracker(capacity=8, probation_frac=0.25)  # protected_cap=6
        for k in range(6):
            t.record_insert(f"p{k}")
            t.record_hit(f"p{k}")
        t.record_insert("c0")
        t.record_insert("c1")
        # When max_rescues=2 caps the scan before hitting "c0", still returns 2 keys
        # because unpinned probation keys exist.
        rescued = t.rescue_candidates(1, max_rescues=2)
        self.assertEqual(rescued, ["p0", "p1"])
        # If every probation key is excluded (pinned), returns [] (no spinning!).
        self.assertEqual(t.rescue_candidates(1, exclude={"c0", "c1"}), [])


class TestSlruWithRaidenManager(unittest.TestCase):
    """End-to-end tests of real ``RaidenOffloadingManager`` on ``CapacityLRUStore``."""

    def _store_batch(self, mgr, store: CapacityLRUStore, job_id: int, keys: list[bytes]):
        out = mgr.prepare_store(keys, _new_req_ctx("s"))
        assert out is not None
        to_store = list(out.keys_to_store)
        if not to_store:
            return
        mgr.submit_store_job(job_id, to_store, list(range(len(to_store))))
        store.complete_saves()
        done = mgr.poll_finished_jobs()
        self.assertTrue(all(f.success for f in done))

    def _load_batch(self, mgr, store: CapacityLRUStore, job_id: int, keys: list[bytes]):
        spec = mgr.prepare_load(keys, _new_req_ctx("l"))
        self.assertTrue(spec.pinned)
        mgr.submit_load_job(job_id, keys, list(range(len(keys))), req_id="l", pinned=True)
        store.complete_loads()
        done = mgr.poll_finished_jobs()
        self.assertTrue(all(f.success for f in done))

    def test_slru_survives_cold_scan_while_plain_lru_evicts_hot_keys(self):
        with _StubEnv():
            rs = sys.modules.get("vllm_torchtpu.offload.raiden_store")
            if rs is None:
                import vllm_torchtpu.offload.raiden_store as rs

            store_types = SimpleNamespace(
                BlockStatus=_FakeBlockStatus, RaidenBlockId=_FakeRaidenBlockId
            )
            hot = [_block_key(i) for i in range(4)]
            cold = [_block_key(100 + i) for i in range(20)]

            # Plain LRU (default OFF)
            lru_store = CapacityLRUStore(capacity=8)
            lru_mgr = rs.RaidenOffloadingManager(
                kernel_physical_blocks_capacity=8,
                offload_logical_blocks_capacity=8,
                device_block_size_factor=1,
                world_size=1,
                raiden_controller_port=0,
                raiden_job_name="lru",
                track_held=True,
                store=lru_store,
                store_types=store_types,
            )
            self._store_batch(lru_mgr, lru_store, 1, hot + cold[:4])
            self._load_batch(lru_mgr, lru_store, 2, hot)
            for j, ck in enumerate(cold[4:], start=10):
                self._store_batch(lru_mgr, lru_store, j, [ck])
            lru_survivors = sum(
                1 for k in hot if lru_mgr.host_resident_prefix([k]) == 1
            )
            self.assertEqual(lru_survivors, 0)
            self.assertEqual(lru_store.pinned_counts(), {})

            # SLRU enabled (probation_frac=0.25 -> protected_cap=6)
            slru_store = CapacityLRUStore(capacity=8)
            slru_mgr = rs.RaidenOffloadingManager(
                kernel_physical_blocks_capacity=8,
                offload_logical_blocks_capacity=8,
                device_block_size_factor=1,
                world_size=1,
                raiden_controller_port=0,
                raiden_job_name="slru",
                track_held=True,
                slru_probation_frac=0.25,
                store=slru_store,
                store_types=store_types,
            )
            self._store_batch(slru_mgr, slru_store, 1, hot + cold[:4])
            # Re-load hot keys -> promoted to protected.
            self._load_batch(slru_mgr, slru_store, 2, hot)
            # Stream 16 one-shot cold keys (2x full cache capacity!).
            for j, ck in enumerate(cold[4:], start=10):
                self._store_batch(slru_mgr, slru_store, j, [ck])
            slru_survivors = sum(
                1 for k in hot if slru_mgr.host_resident_prefix([k]) == 1
            )
            self.assertEqual(slru_survivors, 4)
            # Exact pin balance invariant: ZERO leaked pins!
            self.assertEqual(slru_store.pinned_counts(), {})
            self.assertGreater(slru_mgr.slru_stats().rescues, 0)
            self.assertEqual(slru_mgr.slru_stats().protected, 4)

    def test_slru_rescue_missing_and_shared_storage_pin_balance_factor2(self):
        with _StubEnv():
            import vllm_torchtpu.offload.raiden_store as rs

            store_types = SimpleNamespace(
                BlockStatus=_FakeBlockStatus, RaidenBlockId=_FakeRaidenBlockId
            )
            store = CapacityLRUStore(capacity=8)
            mgr = rs.RaidenOffloadingManager(
                kernel_physical_blocks_capacity=8,
                offload_logical_blocks_capacity=4,
                device_block_size_factor=2,
                world_size=1,
                raiden_controller_port=0,
                raiden_job_name="slru-f2",
                slru_probation_frac=0.25,
                store=store,
                store_types=store_types,
            )
            k0, k1, k2, k3 = (_block_key(i) for i in range(4))
            self._store_batch(mgr, store, 1, [k0, k1, k2])
            self._load_batch(mgr, store, 2, [k0, k1, k2])
            # Admit probation k3 AFTER k0..k2 were promoted so LRU order is [k0, k1, k2, k3].
            self._store_batch(mgr, store, 3, [k3])
            # Quietly erase one sub-hash of k1 and place it in storage tier to
            # force a mid-batch SHARED_STORAGE break during rescue!
            missing_sub = mgr._sub_hashes(k1)[1]
            del store.entries[missing_sub]
            store.storage_entries.add(missing_sub)
            for s in mgr._sub_hashes(k2):
                store.storage_entries.add(s)

            # Trigger rescue via storing a new key k4.
            k4 = _block_key(99)
            self._store_batch(mgr, store, 4, [k4])
            # No pin may leak and no SHARED_STORAGE hash may be released!
            self.assertEqual(store.pinned_counts(), {})
            self.assertFalse(mgr._slru.contains(k1))
            self.assertGreaterEqual(mgr.slru_stats().pruned, 1)

    def test_slru_victim_rehit_touch_unwind_and_tiered_telemetry(self):
        with _StubEnv():
            import vllm_torchtpu.offload.raiden_store as rs
            from vllm_torchtpu.offload.tier_classes import (
                TierClass,
                TieredOffloadingManager,
                TierPlan,
            )

            store_types = SimpleNamespace(
                BlockStatus=_FakeBlockStatus, RaidenBlockId=_FakeRaidenBlockId
            )
            store_m = CapacityLRUStore(capacity=8)
            store_c = CapacityLRUStore(capacity=4)
            main_mgr = rs.RaidenOffloadingManager(
                kernel_physical_blocks_capacity=8,
                offload_logical_blocks_capacity=8,
                device_block_size_factor=1,
                world_size=1,
                raiden_controller_port=0,
                raiden_job_name="main",
                track_held=True,
                slru_probation_frac=0.25,
                store=store_m,
                store_types=store_types,
            )
            ckpt_mgr_lru = rs.RaidenOffloadingManager(
                kernel_physical_blocks_capacity=4,
                offload_logical_blocks_capacity=4,
                device_block_size_factor=1,
                world_size=1,
                raiden_controller_port=0,
                raiden_job_name="gdn-c",
                track_held=True,
                slru_probation_frac=None,
                store=store_c,
                store_types=store_types,
            )
            c_main = TierClass("main", 0, "main", (4,), ((0,),), row_bytes=1024)
            c_ckpt = TierClass("gdn-c", 1, "gdn", (2,), ((0,),), row_bytes=1024)
            plan = TierPlan((c_main, c_ckpt), budget=12 * 1024, ratio=2.0, token_blocks=8, checkpoints=4)
            tiered = TieredOffloadingManager(plan, [main_mgr, ckpt_mgr_lru])

            # Store one key in main, one checkpoint key in gdn-c.
            km0, km1 = _block_key(0, group=4), _block_key(1, group=4)
            kc0 = _block_key(0, group=2)
            self._store_batch(main_mgr, store_m, 1, [km0, km1])
            self._store_batch(ckpt_mgr_lru, store_c, 2, [kc0])
            self.assertTrue(main_mgr._slru.is_probation(km0))
            self.assertTrue(main_mgr._slru.is_probation(km1))

            # Unwind of prepared load must NOT promote a probation key to protected.
            spec = main_mgr.prepare_load([km1], _new_req_ctx("u"))
            main_mgr.unwind_prepared_load(spec)
            self.assertTrue(main_mgr._slru.is_probation(km1))
            self.assertEqual(store_m.pinned_counts(), {})

            # touch() promotes km0 to protected with zero leaked pins.
            tiered.touch([km0], _new_req_ctx("t"))
            self.assertTrue(main_mgr._slru.is_protected(km0))
            self.assertEqual(store_m.pinned_counts(), {})

            # Re-eviction of resident victim km1 in victim mode promotes it cleanly.
            main_mgr.set_victim_mode(True)
            needed = main_mgr.prepare_victim_store([km1])
            self.assertEqual(needed, [])
            self.assertTrue(main_mgr._slru.is_protected(km1))
            self.assertEqual(store_m.pinned_counts(), {})

            # demote_keys moves km0 back to probation; gdn-c (plain LRU) returns 0.
            self.assertEqual(tiered.demote_keys([km0, kc0]), 1)
            self.assertTrue(main_mgr._slru.is_probation(km0))

            # Telemetry line contains slru= for main ONLY (not for plain-LRU gdn-c!).
            line = tiered.telemetry_line(now=100.0)
            self.assertIn("tier[main]", line)
            self.assertIn("slru=1/1(", line)
            ckpt_part = line.split("tier[gdn-c]")[1]
            self.assertNotIn("slru=", ckpt_part)

    def test_lookup_log_factor2_partial_subhash_is_miss(self):
        with _StubEnv():
            from vllm_torchtpu.offload import lookup_log

            mgr = SimpleNamespace(
                _sub_hashes=lambda k: [k + b"0", k + b"1"],
                _block_status=SimpleNamespace(HOST="H", HOST_AND_HBM="HH"),
                _probe=lambda hs: [(h, SimpleNamespace(status="H")) for h in hs if h == b"k0"],
            )
            self.assertFalse(lookup_log.is_key_host_resident(mgr, b"k"))


class TestLookupLog(unittest.TestCase):
    def test_per_key_probe_survives_gaps_and_zero_pins(self):
        with _StubEnv():
            from vllm_torchtpu.offload import lookup_log

            pins_seen: list[bool] = []
            # Real KVCacheStore prefix-stopping lookup!
            class _PrefixStoppingStore:
                def __init__(self, resident: set[str]):
                    self.resident = resident

                def probe(self, hs):
                    pins_seen.append(False)
                    out = []
                    for h in hs:
                        if h not in self.resident:
                            break
                        out.append((h, SimpleNamespace(status="H")))
                    return out

            in_tier = {"a4", "a5", "s5"}
            store = _PrefixStoppingStore(in_tier)
            lines: list[str] = []
            orig_logger = lookup_log.logger
            lookup_log.logger = SimpleNamespace(
                info=lambda fmt, *a: lines.append(fmt % a)
            )
            try:
                mgr = SimpleNamespace(
                    _sub_hashes=lambda k: [k],
                    _block_status=SimpleNamespace(HOST="H", HOST_AND_HBM="HH"),
                    _probe=store.probe,
                )
                configs = [
                    SimpleNamespace(tokens_per_chunk=16, kv_cache_spec=_new_mamba_spec()),
                    SimpleNamespace(tokens_per_chunk=16, kv_cache_spec=_new_attn_spec()),
                ]
                sched = SimpleNamespace(
                    config=SimpleNamespace(kv_group_configs=configs),
                    _raiden_manager=mgr,
                    _req_status={
                        "r": SimpleNamespace(
                            group_states=(
                                SimpleNamespace(offload_keys=[f"s{i}" for i in range(6)]),
                                SimpleNamespace(offload_keys=[f"a{i}" for i in range(6)]),
                            )
                        )
                    },
                )
                req = SimpleNamespace(
                    request_id="r",
                    num_tokens=6 * 16 + 1,
                    shared_prefix_boundary=4 * 16,
                    _tpu_hbm_lookup=([True, True, True, True, False, False], [False] * 6),
                )
                lookup_log.log_lookup(sched, req, local_tokens=4 * 16, tier_tokens=2 * 16)
            finally:
                lookup_log.logger = orig_logger

            self.assertEqual(len(lines), 1)
            data = json.loads(lines[0].split(" ", 1)[1])
            self.assertEqual(data["tier_attn"], 6)
            self.assertEqual(data["tier_attn_any"], 2)
            self.assertEqual(data["tier_ckpts"], [6])
            self.assertEqual(data["union_attn"], 6)
            self.assertEqual(data["union_ckpt"], 6)

    def test_tiered_manager_class_runs_and_presence(self):
        with _StubEnv():
            from vllm_torchtpu.offload import lookup_log

            class _ClassMgr:
                def __init__(self, held: set[bytes]):
                    self.held = held

                def host_resident_prefix(self, keys):
                    n = 0
                    for k in keys:
                        if k not in self.held:
                            break
                        n += 1
                    return n

            main_held = {b"m2", b"m3", b"m5"}
            ckpt_held = {b"c3", b"c5"}
            c_main = SimpleNamespace(name="main", groups=(4,), per_checkpoint=False)
            c_ckpt = SimpleNamespace(name="gdn-c", groups=(2,), per_checkpoint=True)
            mgr = SimpleNamespace(
                classes=(c_main, c_ckpt),
                managers=[_ClassMgr(main_held), _ClassMgr(ckpt_held)],
                class_of=lambda k: 0 if k.startswith(b"m") else 1,
            )
            self.assertFalse(lookup_log.is_key_host_resident(mgr, b"m0"))
            self.assertTrue(lookup_log.is_key_host_resident(mgr, b"m3"))
            self.assertTrue(lookup_log.is_key_host_resident(mgr, b"c5"))

            keys_of = {
                4: [f"m{i}".encode() for i in range(6)],
                2: [f"c{i}".encode() for i in range(6)],
            }
            self.assertEqual(lookup_log.class_presence(mgr, keys_of, 0), {"main": 0, "gdn-c": [4, 6]})
            self.assertEqual(lookup_log.class_runs(mgr, keys_of), {"main": [[2, 4], [5, 6]]})


class TestSupersededCheckpointTracker(unittest.TestCase):
    def setUp(self):
        self.victim = _load_module_direct("_test_victim", "vllm_torchtpu/offload/victim.py")

    def tearDown(self):
        sys.modules.pop("_test_victim", None)

    def test_single_line_skip_and_branch_protection(self):
        tr = self.victim.SupersededCheckpointTracker(max_entries=16)
        k_t0 = (b"g0_t0", b"g1_t0")
        tr.record_resume("req1", 3200, k_t0)
        # Repeat lookup for SAME req_id must not falsely mark as shared branch!
        tr.record_resume("req1", 3200, k_t0)
        # Request completes deeper prefill at boundary 6400 -> supersedes t0.
        newly = tr.on_request_finished("req1", 6400, completed_prompt=True)
        self.assertEqual(newly, list(k_t0))
        self.assertTrue(tr.is_superseded(b"g0_t0"))

        # Victim eviction drops superseded checkpoints while keeping attention blocks.
        kept, skipped = tr.filter_victims([(b"g0_t0", 1), (b"attn_0", 2), (b"g1_t0", 3)])
        self.assertEqual(kept, [(b"attn_0", 2)])
        self.assertEqual(skipped, 2)
        self.assertFalse(tr.is_superseded(b"g0_t0"))

        # Shared branch (sequential!): req2 supersedes k_br, then req3 ALSO resumes k_br
        # before eviction -> k_br must be removed from superseded & kept!
        tr.record_resume("req2", 3200, [b"k_br"])
        tr.on_request_finished("req2", 6400, completed_prompt=True)
        self.assertTrue(tr.is_superseded(b"k_br"))
        tr.record_resume("req3", 3200, [b"k_br"])
        self.assertFalse(tr.is_superseded(b"k_br"))
        tr.on_request_finished("req3", 9600, completed_prompt=True)
        self.assertFalse(tr.is_superseded(b"k_br"))
        kept2, skipped2 = tr.filter_victims([(b"k_br", 9)])
        self.assertEqual((kept2, skipped2), ([(b"k_br", 9)], 0))

    def test_aborted_and_same_block_do_not_supersede(self):
        tr = self.victim.SupersededCheckpointTracker()
        tr.record_resume("abort", 3200, [b"ka"])
        self.assertEqual(tr.on_request_finished("abort", 9600, completed_prompt=False), [])
        self.assertFalse(tr.is_superseded(b"ka"))

        tr.record_resume("same", 3200, [b"ks"])
        self.assertEqual(tr.on_request_finished("same", 3200, completed_prompt=True), [])
        self.assertFalse(tr.is_superseded(b"ks"))

    def test_divergent_fallback_replaces_owner_and_bounded_memory(self):
        tr = self.victim.SupersededCheckpointTracker(max_entries=4)
        # Tentative divergent hit at 6400 replaced by convergent hit at 3200.
        tr.record_resume("r", 6400, [b"div"])
        tr.record_resume("r", 3200, [b"conv"])
        self.assertNotIn(b"div", tr._first_owner)
        self.assertEqual(tr.on_request_finished("r", 6400, completed_prompt=True), [b"conv"])

        # Bounds hold under continuous load.
        for i in range(50):
            k = f"k{i}".encode()
            tr.record_resume(f"req{i}", 3200, [k])
            tr.on_request_finished(f"req{i}", 6400, completed_prompt=True)
        self.assertLessEqual(len(tr._superseded), 4)
        self.assertLessEqual(len(tr._first_owner), 4)
        self.assertLessEqual(len(tr._pending), 4)

    def test_scheduler_wiring_skips_and_demotes_superseded_checkpoints(self):
        with _StubEnv():
            from vllm_torchtpu.offload import raiden_connector, victim as victim_mod

            demoted_calls: list[list] = []
            sched = raiden_connector.TPURaidenOffloadingScheduler.__new__(
                raiden_connector.TPURaidenOffloadingScheduler
            )
            state_key = _make_offload_key(b"blk0" + b"\x00" * 28, 0)
            attn_key = _make_offload_key(b"blk0" + b"\x00" * 28, 4)
            sched.config = SimpleNamespace(
                kv_group_configs=(
                    SimpleNamespace(group_idx=0, tokens_per_chunk=3200, kv_cache_spec=_new_mamba_spec()),
                    SimpleNamespace(group_idx=4, tokens_per_chunk=3200, kv_cache_spec=_new_attn_spec()),
                )
            )
            sched._req_status = {
                "r1": SimpleNamespace(
                    group_states=(
                        SimpleNamespace(offload_keys=[state_key]),
                        SimpleNamespace(offload_keys=[attn_key]),
                    )
                )
            }
            sched._raiden_manager = SimpleNamespace(
                demote_keys=lambda keys: demoted_calls.append(list(keys)) or len(keys)
            )
            sched._superseded_tracker = victim_mod.SupersededCheckpointTracker()
            req = SimpleNamespace(request_id="r1", num_prompt_tokens=6401, num_computed_tokens=6401)
            sched.get_num_new_matched_tokens(req, 3200)
            sched.request_finished(req)
            self.assertEqual(demoted_calls, [[state_key]])
            self.assertEqual(sched.telemetry["ck_demote_superseded"], 1)
            kept, skipped = sched._superseded_tracker.filter_victims([(state_key, 1), (attn_key, 2)])
            self.assertEqual(kept, [(attn_key, 2)])
            self.assertEqual(skipped, 1)


class TestDivergentHybridHits(unittest.TestCase):
    def test_flags_and_connector_voting(self):
        with _StubEnv():
            from vllm_torchtpu.distributed.kv_transfer.tpu_multi_connector import (
                TPUMultiConnector,
            )
            from vllm_torchtpu.offload.raiden_connector import (
                TPURaidenOffloadingConnector,
                divergent_hits_restore_state,
            )

            cfg_hybrid = SimpleNamespace(
                kv_cache_groups=[
                    SimpleNamespace(kv_cache_spec=_new_mamba_spec()),
                    SimpleNamespace(kv_cache_spec=_new_mamba_spec()),
                    SimpleNamespace(kv_cache_spec=_new_attn_spec()),
                ]
            )
            self.assertTrue(divergent_hits_restore_state(cfg_hybrid, [0, 1, 2]))
            self.assertFalse(divergent_hits_restore_state(cfg_hybrid, [0, 2]))
            cfg_attn_only = SimpleNamespace(
                kv_cache_groups=[SimpleNamespace(kv_cache_spec=_new_attn_spec())]
            )
            self.assertFalse(divergent_hits_restore_state(cfg_attn_only, [0]))

            conn = TPURaidenOffloadingConnector.__new__(TPURaidenOffloadingConnector)
            conn.connector_scheduler = SimpleNamespace(restores_state_with_every_load=True)
            with patch.dict(os.environ, {"VLLM_TPU_OFFLOAD_DIVERGENT_HITS": "0"}):
                self.assertFalse(conn.supports_divergent_local_hybrid_hits)
            with patch.dict(os.environ, {"VLLM_TPU_OFFLOAD_DIVERGENT_HITS": "1"}):
                self.assertTrue(conn.supports_divergent_local_hybrid_hits)

            producer = SimpleNamespace(
                _kv_transfer_config=SimpleNamespace(kv_role="kv_producer"),
                supports_divergent_local_hybrid_hits=False,
            )
            loader_ok = SimpleNamespace(
                _kv_transfer_config=SimpleNamespace(kv_role="kv_both"),
                supports_divergent_local_hybrid_hits=True,
            )
            multi = TPUMultiConnector.__new__(TPUMultiConnector)
            multi._connectors = [producer, loader_ok]
            with patch.dict(os.environ, {"VLLM_TPU_OFFLOAD_DIVERGENT_HITS": "0"}):
                self.assertFalse(multi.supports_divergent_local_hybrid_hits)
            with patch.dict(os.environ, {"VLLM_TPU_OFFLOAD_DIVERGENT_HITS": "1"}):
                self.assertTrue(multi.supports_divergent_local_hybrid_hits)

            prod_only = TPUMultiConnector.__new__(TPUMultiConnector)
            prod_only._connectors = [producer]
            with patch.dict(os.environ, {"VLLM_TPU_OFFLOAD_DIVERGENT_HITS": "1"}):
                self.assertFalse(prod_only.supports_divergent_local_hybrid_hits)


class _FakeFreeBlock:
    def __init__(self, block_id: int, block_hash=b"h", ref_cnt: int = 0, is_null: bool = False, linked: bool = True):
        self.block_id = block_id
        self.block_hash = block_hash
        self.ref_cnt = ref_cnt
        self.is_null = is_null
        self.prev_free_block = object() if linked else None
        self.next_free_block = object() if linked else None


class _FakeFreeQueue:
    def __init__(self, blocks, tap=None, fail_on_id: int | None = None):
        self.items: list[_FakeFreeBlock] = list(blocks)
        self.tap = tap
        self.fail_on_id = fail_on_id

    def remove(self, block):
        if self.fail_on_id is not None and block.block_id == self.fail_on_id:
            raise RuntimeError("injected queue error")
        self.items.remove(block)

    def append_n(self, blocks):
        for b in blocks:
            self.items.append(b)
            if self.tap is not None and not b.is_null:
                self.tap.on_freed([b.block_id])


class TestArrivalTouch(unittest.TestCase):
    def test_hbm_touch_order_refcnt_and_victim_stamps(self):
        with _StubEnv():
            from vllm_torchtpu.core import arrival_touch
            from vllm_torchtpu.offload import victim

            # Create blocks: attn0..2 (group 1), state0 (older ckpt), state2 (newest ckpt),
            # plus an unrelated block "other", a pinned block (ref_cnt=1), and an unlinked block.
            a0 = _FakeFreeBlock(10)
            a1 = _FakeFreeBlock(11)
            a2 = _FakeFreeBlock(12)
            pinned_a = _FakeFreeBlock(13, ref_cnt=1)
            unlinked = _FakeFreeBlock(14, linked=False)
            old_ckpt = _FakeFreeBlock(20)
            new_ckpt = _FakeFreeBlock(22)
            other = _FakeFreeBlock(99)

            tap = victim.VictimTap(step=50)
            tap.freed_at[10] = 3
            tap.freed_at[11] = 4
            tap.freed_at[22] = 5
            # Intentionally leave block 12 unstamped to verify it stays unstamped!
            victim.activate(tap)
            try:
                fq = _FakeFreeQueue([a0, a1, a2, old_ckpt, new_ckpt, other], tap=tap)
                cached = {
                    (b"h0", (1,)): [a0],
                    (b"h1", (1,)): [a1],
                    (b"h2", (1,)): [a2],
                    (b"h0", (2,)): [pinned_a],
                    (b"h1", (2,)): [unlinked],
                    (b"h0", (0,)): [old_ckpt],
                    (b"h2", (0,)): [new_ckpt],
                }
                pool = SimpleNamespace(
                    hash_block_size=16,
                    free_block_queue=fq,
                    get_cached_block=lambda h, gids: cached.get((h, tuple(gids))),
                )
                moved, conv_tok, attn_tok = arrival_touch._touch_hbm_blocks(
                    pool,
                    [b"h0", b"h1", b"h2"],
                    num_tokens=3 * 16 + 1,
                    attn_gids=[1, 2],
                    state_gids=[0],
                )
                # min_attn_hit is 2 (group 2 stopped at index 2); so rightmost ckpt in [0,2) is h0 (old_ckpt)!
                self.assertEqual(attn_tok, 2 * 16)
                self.assertEqual(conv_tok, 1 * 16)
                # Moved blocks: a2, a1, a0 (tail-first), old_ckpt (last -> MRU). pinned & unlinked untouched!
                self.assertEqual(moved, 4)
                self.assertEqual(
                    [b.block_id for b in fq.items],
                    [22, 99, 12, 11, 10, 20],
                )
                # ref_cnt NEVER changed!
                for b in (a0, a1, a2, old_ckpt, new_ckpt, other):
                    self.assertEqual(b.ref_cnt, 0)
                self.assertEqual(pinned_a.ref_cnt, 1)
                # Victim stamps MUST be preserved (not reset to tap.step=50!).
                self.assertEqual(tap.freed_at[10], 3)
                self.assertEqual(tap.freed_at[11], 4)
                self.assertEqual(tap.freed_at[22], 5)
                self.assertNotIn(12, tap.freed_at)
            finally:
                victim.deactivate(tap)

    def test_rollback_restores_removed_blocks_on_exception(self):
        with _StubEnv():
            from vllm_torchtpu.core import arrival_touch

            b0, b1, b2 = _FakeFreeBlock(1), _FakeFreeBlock(2), _FakeFreeBlock(3)
            fq = _FakeFreeQueue([b0, b1, b2], fail_on_id=2)
            with self.assertRaises(RuntimeError):
                arrival_touch._requeue_tail_preserving_stamps(fq, [b0, b1])
            # b0 was removed before b1 raised, and MUST have been put back!
            self.assertEqual(sorted(b.block_id for b in fq.items), [1, 2, 3])

    def test_host_tier_touch_calls_manager_touch(self):
        with _StubEnv():
            from vllm_torchtpu.core import arrival_touch

            touched: list[bytes] = []
            manager = SimpleNamespace(
                touch=lambda keys, ctx: touched.extend(keys),
            )
            attn_cfg = SimpleNamespace(
                group_idx=4, tokens_per_chunk=16, hashes_per_chunk=1,
                sliding_window_size_in_chunks=None, is_eagle_group=False,
            )
            state_cfg = SimpleNamespace(
                group_idx=0, tokens_per_chunk=16, hashes_per_chunk=1,
                sliding_window_size_in_chunks=1, is_eagle_group=False,
            )
            offload_sched = SimpleNamespace(
                _raiden_manager=manager,
                config=SimpleNamespace(kv_group_configs=(attn_cfg, state_cfg)),
                host_resident_tokens=lambda req, local: 32 if local == 16 else 0,
            )
            child = SimpleNamespace(
                host_tier_hit_tokens=lambda r, l: 32,
                connector_scheduler=offload_sched,
            )
            multi = SimpleNamespace(sub_connectors=[SimpleNamespace(), child])
            req = SimpleNamespace(
                request_id="r1",
                num_tokens=50,
                block_hashes=[b"b0", b"b1", b"b2"],
            )
            n = arrival_touch.touch_host_tier_prefix(multi, req, convergent_tokens=16)
            self.assertEqual(n, 3)
            # Attention keys for blocks 1, 2 and state window key at boundary block 2.
            self.assertEqual(
                touched,
                [
                    _make_offload_key(b"b1", 4),
                    _make_offload_key(b"b2", 4),
                    _make_offload_key(b"b2", 0),
                ],
            )

    def test_patch_inert_when_default_off(self):
        with _StubEnv():
            from vllm.v1.core.sched.scheduler import Scheduler
            from vllm_torchtpu.core import arrival_touch

            orig = Scheduler.add_request
            with patch.dict(os.environ, {"VLLM_TPU_ARRIVAL_TOUCH": "0"}):
                arrival_touch.patch_arrival_touch()
            self.assertIs(Scheduler.add_request, orig)

    def test_patch_add_request_skips_duplicate_and_skip_cache(self):
        with _StubEnv():
            from vllm.v1.core.sched.scheduler import Scheduler
            from vllm_torchtpu.core import arrival_touch

            orig = Scheduler.add_request
            calls: list[str] = []
            orig_touch = arrival_touch.touch_on_arrival
            try:
                arrival_touch.touch_on_arrival = lambda s, r: (calls.append(r.request_id) or (2, 1))  # type: ignore[assignment]
                sched = Scheduler.__new__(Scheduler)
                sched.requests = {}
                sched.waiting = []
                # Stub out the real Scheduler.add_request body cleanly for unit check.
                def _stub_add(self, r):
                    self.requests[r.request_id] = r
                    self.waiting.append(r)

                Scheduler.add_request = _stub_add  # type: ignore[assignment]
                with patch.dict(os.environ, {"VLLM_TPU_ARRIVAL_TOUCH": "1"}):
                    arrival_touch.patch_arrival_touch()
                r1 = SimpleNamespace(request_id="r1")
                sched.add_request(r1)
                sched.add_request(r1)  # duplicate/streaming chunk -> must NOT re-touch!
                self.assertEqual(calls, ["r1"])
                self.assertEqual(sched._tpu_arrival_touch_stats["reqs"], 1)
                self.assertEqual(sched._tpu_arrival_touch_stats["hbm_blocks"], 2)
                self.assertEqual(sched._tpu_arrival_touch_stats["tier_keys"], 1)
            finally:
                Scheduler.add_request = orig
                arrival_touch.touch_on_arrival = orig_touch

    def test_divergent_host_tier_fallback(self):
        with _StubEnv():
            from vllm_torchtpu.core import arrival_touch

            probed: list[int] = []
            touched: list[bytes] = []
            attn_cfg = SimpleNamespace(
                group_idx=4, tokens_per_chunk=16, hashes_per_chunk=1,
                sliding_window_size_in_chunks=None, is_eagle_group=False,
            )

            def fake_resident(req, local):
                probed.append(local)
                # Under divergent boundary 32, tier hits 16 tokens (block 2).
                return 16 if local == 32 else 0

            offload_sched = SimpleNamespace(
                _raiden_manager=SimpleNamespace(touch=lambda ks, ctx: touched.extend(ks)),
                config=SimpleNamespace(kv_group_configs=(attn_cfg,)),
                host_resident_tokens=fake_resident,
            )
            conn = SimpleNamespace(
                host_tier_hit_tokens=lambda r, l: 16,
                connector_scheduler=offload_sched,
            )
            req = SimpleNamespace(request_id="d1", num_tokens=50, block_hashes=[b"b0", b"b1", b"b2"])
            with patch.dict(os.environ, {"VLLM_TPU_OFFLOAD_DIVERGENT_HITS": "1"}):
                n = arrival_touch.touch_host_tier_prefix(conn, req, convergent_tokens=0, attn_tokens=32)
            self.assertEqual(probed, [32])
            self.assertEqual(n, 1)
            self.assertEqual(touched, [_make_offload_key(b"b2", 4)])


if __name__ == "__main__":
    unittest.main()
