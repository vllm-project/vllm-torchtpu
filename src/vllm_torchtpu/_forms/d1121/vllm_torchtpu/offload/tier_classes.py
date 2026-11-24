# SPDX-License-Identifier: Apache-2.0
"""Host tier in size classes: one Raiden store for each set of pool regions.

Turned on by VLLM_TPU_OFFLOAD_TIER_CLASSES=1. Without the flag one store holds every
offloaded group, and an entry moves one row of every pool region on every stage,
whatever its group uses. With it, the offloaded groups are split into classes by what
they hold (main KV, compressed keys, GDN state, PLE state;
``qwen4_exp.offload.tier_group_kind``) and by the pool regions their layers occupy on
each worker, as the model's own group-to-layer plan places them
(``kv_cache_materializer.layer_to_pool_index`` over each worker's KV cache config). Each
class is its own ``KVCacheStore`` with its own controller port (31515 up by default,
below the ephemeral port range), LRU, directory and host rows, and a worker registers
with a class only the regions that class's groups use on its stage. On the served
prefill (PP8 TP1) the classes are ``main``, ``keys``, ``gdn-ab`` (GDN groups A and B),
``gdn-c`` and ``ple``, with 12, 12, 12, 12 and 1 regions over 8, 8, 8, 8 and 1 workers;
the startup check holds the derived plan to that table.

With ``VLLM_TPU_OFFLOAD_TIER_KEY_ROWS=1`` the ``keys`` class keeps only the keys'
leading bytes of each page: its regions are registered as views of ``page / live`` times
as many rows, and block ``b``'s keys are device row ``b * page / live``. That
translation lives in the class manager's call into Raiden
(``RaidenOffloadingManager._expand_device_blocks``); keys, host slots and the IDs this
module reports are the scheduler's.

Scheduler side, ``TieredOffloadingManager`` routes each key to its class by group. A
store or load job is split into one child job a class, under the parent's job ID in each
class manager. The rules of a split job:

- Each class manager keeps its own admissions and pins and finalizes its child exactly
  once. The parent releases nothing.
- A prepare that fails part way gives back only the pins it took itself: those of the
  classes prepared before the one that failed, which released its own.
- The parent is terminal only when every child it launched is, and succeeds only if all
  did. A child whose launch is refused finalizes at once; the parent still waits for the
  others.
- A failed parent load reports every scheduler block ID of the job, the destinations of
  children that succeeded included.
- Host entries a successful child wrote stay; "not loaded" is the scheduler's completion
  state for the job.

Capacities are fixed at startup from the one byte budget (``cpu_bytes_to_use``) and one
ratio of token-blocks to checkpoints (``VLLM_TPU_OFFLOAD_TIER_CLASS_RATIO``). With M
token-blocks and C checkpoints, an attention class holds M entries a group and a state
class C entries a group, and M x (bytes of a token-block's entries) + C x (bytes of a
checkpoint's entries) fits the budget.

Scope: the local host DRAM tier. The flag is refused with a global registry, a peer
store group, a storage tier or a shared-memory host pool: the classes register different
region counts by stage, and Raiden's transfer spec for those paths wants equal worker
geometry.

TIER-X2 (``VLLM_TPU_OFFLOAD_TIER_X2=1``) lifts the first two refusals: ``shard_plan`` cuts
each class into shards of one per-worker region count, each shard store joins the global
registry under its own pool group and key namespace, Raiden's evict sweep demotes its cold
blocks to that group's store nodes on another host, and a miss reads them back with
``read_remote``. A group then belongs to several classes (the shards of one) and every key
goes to each.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from vllm.v1.kv_offload.base import (
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    LoadStoreSpec,
    LookupResult,
    OffloadingEvent,
    OffloadingManager,
    OffloadingWorker,
    OffloadKey,
    PrepareStoreOutput,
    ReqContext,
    RequestOffloadingContext,
    ScheduleEndContext,
    TransferResult,
    get_offload_group_idx,
)

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.offload import raiden_store
from vllm_torchtpu.offload.fp4 import SlotPool
from vllm_torchtpu.offload.raiden_store import (
    AdmissionOp,
    FinishedJob,
    RaidenLoadStoreSpec,
    RaidenOffloadingManager,
)

logger = init_logger(__name__)

# Order of the classes (and of their controller ports): by kind, then by first group.
KIND_ORDER = ("main", "keys", "gdn", "ple")
STATE_KINDS = ("gdn", "ple")

# The served prefill's classes (Qwen3.8-Flash, PP8 TP1): regions summed over the
# workers, and workers registered.
SERVED_CLASSES = {
    "main": (12, 8),
    "keys": (12, 8),
    "gdn-ab": (12, 8),
    "gdn-c": (12, 8),
    "ple": (1, 1),
}

# TIER-X2 (VLLM_TPU_OFFLOAD_TIER_X2): the served classes split into shards of one
# per-worker geometry (``shard_plan``), regions and workers as above.
SERVED_SHARDS = {
    "main.r1": (4, 4),
    "main.r2": (8, 4),
    "keys.r1": (4, 4),
    "keys.r2": (8, 4),
    "gdn-ab.r1": (4, 4),
    "gdn-ab.r2": (8, 4),
    "gdn-c.r1": (4, 4),
    "gdn-c.r2": (8, 4),
    "ple": (1, 1),
}


@dataclass(frozen=True)
class TierClass:
    """One host-tier class: its groups and the pool regions they occupy by worker."""

    name: str
    index: int
    kind: str
    groups: tuple[int, ...]
    # Per worker, in the order of the recorded per-worker configs: the pool region
    # indices this class's groups use.
    worker_regions: tuple[tuple[int, ...], ...]
    # Host bytes one region contributes to an entry.
    row_bytes: int
    # Device rows a scheduler page spans in this class's registered views (1: whole-page
    # rows).
    rows_per_page: int = 1
    # TIER-X2: the class this shard was cut from (``shard_plan``); None for a whole
    # class.
    shard_of: str | None = None

    @property
    def per_checkpoint(self) -> bool:
        return self.kind in STATE_KINDS

    @property
    def workers(self) -> tuple[int, ...]:
        """The workers (indices into ``worker_regions``) that register with this class."""
        return tuple(i for i, regions in enumerate(self.worker_regions) if regions)

    def node_id(self, worker: int) -> int:
        """The worker's dense node id in this class's store: its position among the
        workers that register. Raiden's transfer spec and a store node pair workers by
        node id over the range [0, workers)."""
        return self.workers.index(worker)

    @property
    def num_workers(self) -> int:
        return sum(1 for regions in self.worker_regions if regions)

    @property
    def num_regions(self) -> int:
        return sum(len(regions) for regions in self.worker_regions)

    @property
    def bytes_per_entry(self) -> int:
        return self.num_regions * self.row_bytes


@dataclass(frozen=True)
class TierPlan:
    """The classes and their fixed capacities (entries) for one byte budget."""

    classes: tuple[TierClass, ...]
    budget: int
    ratio: float
    token_blocks: int
    checkpoints: int

    def capacity(self, tier_class: TierClass) -> int:
        unit = self.checkpoints if tier_class.per_checkpoint else self.token_blocks
        return len(tier_class.groups) * unit

    def describe(self) -> str:
        return "; ".join(
            f"{c.name}: groups {list(c.groups)}, regions by worker "
            f"{[len(r) for r in c.worker_regions]} "
            f"({c.num_regions} over {c.num_workers} workers), "
            f"{c.bytes_per_entry} B an entry, {self.capacity(c)} entries"
            for c in self.classes
        )


def worker_group_regions(kv_cache_config: Any, group_id: int) -> tuple[int, ...]:
    """Pool regions a group's layers occupy on one worker, by that worker's plan."""
    from vllm_torchtpu.kv_cache_materializer import layer_to_pool_index

    index = layer_to_pool_index(kv_cache_config)
    return tuple(
        sorted(
            {
                index[name]
                for name in kv_cache_config.kv_cache_groups[group_id].layer_names
            }
        )
    )


def build_tier_plan(
    worker_configs: Sequence[Any],
    group_ids: Iterable[int],
    group_kind: Callable[[Any, list[str]], str],
    budget: int,
    ratio: float,
    key_row_bytes: int | None = None,
    main_row_bytes: int | None = None,
    gdn_row_bytes: int | None = None,
) -> TierPlan:
    """Splits the offloaded groups into classes and sizes them.

    Classes are by kind and by per-worker region set.

    ``group_kind(spec, layer_names)`` names what a group holds. ``key_row_bytes``, when
    set, is the live bytes of a compressed-key page: the ``keys`` class then keeps that
    many bytes a region instead of the page.
    ``main_row_bytes``, when set (VLLM_TPU_OFFLOAD_FP4), is the host bytes of one main-KV
    page in MXFP4: the ``main`` class keeps that many bytes a region (registered as rows
    of the workers' staging buffers, not of the pool). ``gdn_row_bytes``
    (VLLM_TPU_OFFLOAD_FP4_GDN) does the same for the ``gdn`` classes.
    """
    if not worker_configs:
        raise ValueError("host tier classes: no per-worker KV cache configs recorded")
    if ratio <= 0:
        raise ValueError(
            "host tier classes: the token-block to checkpoint ratio must be "
            f"positive, got {ratio}"
        )
    pages = {
        group.kv_cache_spec.page_size_bytes
        for cfg in worker_configs
        for group in cfg.kv_cache_groups
    }
    if len(pages) != 1:
        raise ValueError(
            f"host tier classes need one uniform pool page, got {sorted(pages)}"
        )
    (page,) = pages
    buckets: dict[tuple[str, tuple[tuple[int, ...], ...]], list[int]] = {}
    kinds: dict[int, str] = {}
    for gid in sorted(set(group_ids)):
        names = [
            name
            for cfg in worker_configs
            for name in cfg.kv_cache_groups[gid].layer_names
        ]
        kind = group_kind(worker_configs[0].kv_cache_groups[gid].kv_cache_spec, names)
        if kind not in KIND_ORDER:
            raise ValueError(
                f"host tier classes: group {gid} has unknown kind {kind!r}"
            )
        kinds[gid] = kind
        signature = tuple(worker_group_regions(cfg, gid) for cfg in worker_configs)
        buckets.setdefault((kind, signature), []).append(gid)
    ordered = sorted(
        buckets.items(), key=lambda item: (KIND_ORDER.index(item[0][0]), min(item[1]))
    )
    classes: list[TierClass] = []
    for index, ((kind, signature), groups) in enumerate(ordered):
        name = kind
        if sum(1 for (other, _) in buckets if other == kind) > 1:
            of_kind = sorted(g for g, k in kinds.items() if k == kind)
            name = f"{kind}-" + "".join(
                chr(ord("a") + of_kind.index(g)) for g in groups
            )
        row_bytes, rows_per_page = page, 1
        if kind == "keys" and key_row_bytes is not None:
            if key_row_bytes <= 0 or page % key_row_bytes:
                raise ValueError(
                    f"host tier classes: compressed-key rows of {key_row_bytes} B "
                    f"do not tile a {page} B page"
                )
            row_bytes, rows_per_page = key_row_bytes, page // key_row_bytes
        if kind == "main" and main_row_bytes is not None:
            if main_row_bytes <= 0 or main_row_bytes >= page:
                raise ValueError(
                    f"host tier classes: FP4 main-KV rows of {main_row_bytes} B for a "
                    f"{page} B page"
                )
            row_bytes, rows_per_page = main_row_bytes, 1
        if kind == "gdn" and gdn_row_bytes is not None:
            # VLLM_TPU_OFFLOAD_FP4_GDN: MXFP4 of the state's FP32 words, registered
            # as rows of the workers' GDN staging buffers.
            if gdn_row_bytes <= 0 or gdn_row_bytes >= page:
                raise ValueError(
                    f"host tier classes: FP4 GDN rows of {gdn_row_bytes} B for a "
                    f"{page} B page"
                )
            row_bytes, rows_per_page = gdn_row_bytes, 1
        classes.append(
            TierClass(
                name=name,
                index=index,
                kind=kind,
                groups=tuple(groups),
                worker_regions=signature,
                row_bytes=row_bytes,
                rows_per_page=rows_per_page,
            )
        )
    token = sum(
        len(c.groups) * c.bytes_per_entry for c in classes if not c.per_checkpoint
    )
    ckpt = sum(len(c.groups) * c.bytes_per_entry for c in classes if c.per_checkpoint)
    if ckpt == 0:
        checkpoints, token_blocks = 0, int(budget // token)
    elif token == 0:
        checkpoints, token_blocks = int(budget // ckpt), 0
    else:
        checkpoints = int(budget // (ratio * token + ckpt))
        token_blocks = int(ratio * checkpoints)
    plan = TierPlan(
        classes=tuple(classes),
        budget=budget,
        ratio=ratio,
        token_blocks=token_blocks,
        checkpoints=checkpoints,
    )
    empty = [c.name for c in classes if plan.capacity(c) <= 0 or c.num_workers == 0]
    if empty:
        raise ValueError(
            f"host tier classes: budget {budget} B leaves no entry (or no worker) "
            f"for {empty}: {plan.describe()}"
        )
    return plan


def shard_plan(plan: TierPlan) -> TierPlan:
    """TIER-X2: each class cut into shards whose workers all register the same number
    of regions, renumbered in order.

    Raiden's transfer spec (the one a store node mirrors to receive a pool group's
    demoted blocks) describes every worker of a store with one block geometry, and a
    class's workers register 1 or 2 regions by stage. A shard keeps the class's kind,
    groups, row and capacity, and only its own workers' regions; a class whose workers
    already agree stays whole under its own name. Shards of one class hold the same
    keys, so every key goes to each of them (``TieredOffloadingManager``), and the
    shard sizes sum to the class's.
    """
    shards: list[TierClass] = []
    for tier_class in plan.classes:
        by_count: dict[int, list[int]] = {}
        for worker in tier_class.workers:
            by_count.setdefault(len(tier_class.worker_regions[worker]), []).append(worker)
        for count, workers in sorted(by_count.items()):
            keep = set(workers)
            shards.append(
                TierClass(
                    name=(
                        f"{tier_class.name}.r{count}"
                        if len(by_count) > 1
                        else tier_class.name
                    ),
                    index=len(shards),
                    kind=tier_class.kind,
                    groups=tier_class.groups,
                    worker_regions=tuple(
                        regions if i in keep else ()
                        for i, regions in enumerate(tier_class.worker_regions)
                    ),
                    row_bytes=tier_class.row_bytes,
                    rows_per_page=tier_class.rows_per_page,
                    shard_of=tier_class.name,
                )
            )
    return TierPlan(
        classes=tuple(shards),
        budget=plan.budget,
        ratio=plan.ratio,
        token_blocks=plan.token_blocks,
        checkpoints=plan.checkpoints,
    )


def check_served_shards(plan: TierPlan) -> None:
    """Holds a sharded plan to the reviewed served table."""
    got = {c.name: (c.num_regions, c.num_workers) for c in plan.classes}
    if got != SERVED_SHARDS:
        raise ValueError(
            "VLLM_TPU_OFFLOAD_TIER_X2: the model's plan gives shards (regions, "
            f"workers) {got}, not the served prefill's {SERVED_SHARDS}. The shards are "
            "sized and validated for that geometry only."
        )


def check_served_plan(plan: TierPlan) -> None:
    """Holds the derived classes to the reviewed served table."""
    got = {c.name: (c.num_regions, c.num_workers) for c in plan.classes}
    if got != SERVED_CLASSES:
        raise ValueError(
            "VLLM_TPU_OFFLOAD_TIER_CLASSES: the model's plan gives classes "
            f"(regions, workers) {got}, not the served prefill's {SERVED_CLASSES}. "
            "The classes are sized and validated for that "
            "geometry only."
        )


@dataclass
class _Parent:
    """A split job: the scheduler's view and the classes still running a child."""

    op: AdmissionOp
    req_id: str | None
    keys: list[OffloadKey]
    device_block_ids: list[int]
    waiting: set[int]
    failed: bool = False


class TieredOffloadingManager(OffloadingManager):
    """Scheduler-side manager over one ``RaidenOffloadingManager`` per host-tier class.

    Presents the single manager's surface to the connector: every key goes to its class
    by group, and every job is split into one child a class under the same job ID (rules
    in the module docstring).
    """

    def __init__(
        self,
        plan: TierPlan,
        managers: Sequence[RaidenOffloadingManager],
        check: bool = False,
        fp4_slots: int | dict[str, tuple[int, int]] | None = None,
    ):
        assert len(managers) == len(plan.classes)
        # VLLM_TPU_OFFLOAD_FP4 (offload/fp4.py): an FP4 class's device rows are the
        # workers' staging slots, not pool blocks. A store or load job binds a slot
        # to each of its FP4-class keys; store slots are freed when the child ends,
        # load slots once every worker has dequantized them. ``fp4_slots``: kind ->
        # (first global slot id, slots) (fp4.slot_layout); an int is the main kind
        # alone. Kind "main" is the main class, kind "gdn" every gdn class
        # (VLLM_TPU_OFFLOAD_FP4_GDN), which share one pool.
        if isinstance(fp4_slots, int):
            fp4_slots = {"main": (0, fp4_slots)} if fp4_slots else None
        self._fp4_kind: dict[int, str] = {}
        self._fp4_pools: dict[str, SlotPool] = {}
        self._fp4_first: dict[str, int] = {}
        for kind, (first, count) in (fp4_slots or {}).items():
            members = [c.index for c in plan.classes if c.kind == kind]
            if kind == "main":
                assert len(members) == 1, [c.name for c in plan.classes]
            assert members, (kind, [c.name for c in plan.classes])
            for c in members:
                self._fp4_kind[c] = kind
            self._fp4_pools[kind] = SlotPool(count)
            self._fp4_first[kind] = first
        self._fp4_bound: dict[int, dict[OffloadKey, int]] = {}
        self._fp4_dequant: list[tuple[int, int]] = []
        self._fp4_release: list[int] = []
        names = ("store_keys", "load_keys", "store_no_slot", "load_no_slot", "dequant_keys")
        self.fp4_counts: dict[str, int] = {
            n + ("" if kind == "main" else f"_{kind}"): 0
            for kind in self._fp4_pools
            for n in names
        }
        self._plan = plan
        self._classes = plan.classes
        self._managers = list(managers)
        # Group -> the classes holding it, in class order: one class, or under TIER-X2
        # the shards of one class (every key goes to each).
        classes_of_group: dict[int, list[int]] = {}
        for c in plan.classes:
            for g in c.groups:
                classes_of_group.setdefault(g, []).append(c.index)
        self._classes_of_group = {g: tuple(cs) for g, cs in classes_of_group.items()}
        self._class_of_group = {g: cs[0] for g, cs in self._classes_of_group.items()}
        self._parents: dict[int, _Parent] = {}
        self._finished: list[FinishedJob] = []
        self._load_keys = [0] * len(managers)
        # Read by lookup_log and tests the way they read the single manager.
        self._block_status = managers[0]._block_status
        self._key_namespace = managers[0]._key_namespace
        self._telemetry_last: tuple[float, list[tuple[int, int]]] | None = None
        # Check mode (VLLM_TPU_OFFLOAD_TIER_CHECK): (class, key, block) of
        # every entry a class saved and every key a load restored, for the
        # workers to hash; None when off.
        self._check: tuple[list, list] | None = ([], []) if check else None

    @property
    def classes(self) -> tuple[TierClass, ...]:
        return self._classes

    @property
    def managers(self) -> list[RaidenOffloadingManager]:
        return self._managers

    # -- FP4 main class (VLLM_TPU_OFFLOAD_FP4) ------------------------------------

    @property
    def fp4_class(self) -> int | None:
        """The main class when it keeps FP4 (None: FP4 off)."""
        return next((c for c, k in self._fp4_kind.items() if k == "main"), None)

    @property
    def fp4_kinds(self) -> dict[int, str]:
        """FP4 class index -> its kind."""
        return dict(self._fp4_kind)

    def fp4_kind_of(self, key: OffloadKey) -> str | None:
        return self._fp4_kind.get(self.class_of(key))

    @property
    def fp4_free_slots(self) -> int:
        pool = self._fp4_pools.get("main")
        return 0 if pool is None else pool.free

    def fp4_free_by_kind(self) -> dict[str, int]:
        return {kind: pool.free for kind, pool in self._fp4_pools.items()}

    def _fp4_count(self, name: str, kind: str, n: int) -> None:
        self.fp4_counts[name + ("" if kind == "main" else f"_{kind}")] += n

    def _fp4_take(self, kind: str, n: int) -> list[int] | None:
        local = self._fp4_pools[kind].take(n)
        if local is None:
            return None
        first = self._fp4_first[kind]
        return [first + s for s in local]

    def _fp4_local(self, kind: str, slots: Sequence[int]) -> list[int]:
        first = self._fp4_first[kind]
        return [s - first for s in slots]

    def fp4_bind_store(self, job_id: int, keys: Sequence[OffloadKey]) -> list[int] | None:
        """Global slots for a store job's keys, all of one FP4 kind (None: not enough
        free)."""
        kinds = {self.fp4_kind_of(k) for k in keys}
        assert len(kinds) == 1 and None not in kinds, kinds
        (kind,) = kinds
        slots = self._fp4_take(kind, len(keys))
        if slots is None:
            self._fp4_count("store_no_slot", kind, len(keys))
            return None
        self._fp4_bound.setdefault(job_id, {}).update(zip(keys, slots))
        self._fp4_count("store_keys", kind, len(keys))
        return slots

    def fp4_unbind(self, job_id: int) -> None:
        """Frees a job's slots (a store that will never be submitted)."""
        bound = self._fp4_bound.pop(job_id, None)
        if bound:
            self.fp4_give(list(bound.values()))

    def _fp4_unbind_keys(self, job_id: int, keys: Iterable[OffloadKey]) -> None:
        """Frees the slots of ``keys`` alone (one child of a job that spans classes)."""
        bound = self._fp4_bound.get(job_id)
        if not bound:
            return
        slots = [bound.pop(k) for k in keys if k in bound]
        if not bound:
            del self._fp4_bound[job_id]
        self.fp4_give(slots)

    def fp4_give(self, slots: Sequence[int]) -> None:
        """Returns global slot ids to their kinds' pools."""
        for kind, pool in self._fp4_pools.items():
            first = self._fp4_first[kind]
            mine = [s - first for s in slots if first <= s < first + pool.total]
            if mine:
                pool.give(mine)

    def take_fp4_dequant(self) -> tuple[list[tuple[int, int]], list[int]]:
        """``(global slot, pool block)`` pairs of the loads that finished since the last
        call, and their slots, to free once every worker has dequantized them."""
        pairs, slots = self._fp4_dequant, self._fp4_release
        self._fp4_dequant, self._fp4_release = [], []
        for kind, pool in self._fp4_pools.items():
            first = self._fp4_first[kind]
            n = sum(1 for s, _ in pairs if first <= s < first + pool.total)
            self._fp4_count("dequant_keys", kind, n)
        return pairs, slots

    # -- routing ------------------------------------------------------------

    def classes_of(self, key: OffloadKey) -> tuple[int, ...]:
        """The classes holding the key's group: one, or a class's shards (TIER-X2)."""
        group = get_offload_group_idx(key)
        try:
            return self._classes_of_group[group]
        except KeyError:
            raise KeyError(
                f"host tier classes: group {group} belongs to no class"
            ) from None

    def class_of(self, key: OffloadKey) -> int:
        """The key's class (the first shard of it under TIER-X2)."""
        return self.classes_of(key)[0]

    def _split(
        self, keys: Iterable[OffloadKey], device_block_ids: Sequence[int] | None = None
    ) -> dict[int, tuple[list[OffloadKey], list[int]]]:
        """Keys (and their block IDs) by class, each class keeping the job's order. A
        sharded class's key goes to every shard."""
        parts: dict[int, tuple[list[OffloadKey], list[int]]] = {}
        for i, key in enumerate(keys):
            for c in self.classes_of(key):
                ks, ids = parts.setdefault(c, ([], []))
                ks.append(key)
                if device_block_ids is not None:
                    ids.append(device_block_ids[i])
        return dict(sorted(parts.items()))

    def _runs(
        self, keys: Sequence[OffloadKey]
    ) -> list[tuple[tuple[int, ...], list[OffloadKey]]]:
        """Maximal runs of consecutive keys held by the same classes, in order."""
        runs: list[tuple[tuple[int, ...], list[OffloadKey]]] = []
        for key in keys:
            cs = self.classes_of(key)
            if runs and runs[-1][0] == cs:
                runs[-1][1].append(key)
            else:
                runs.append((cs, [key]))
        return runs

    # -- OffloadingManager contract -------------------------------------------

    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        return RequestOffloadingContext()

    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        classes = self.classes_of(key)
        if len(classes) == 1:
            return self._managers[classes[0]].lookup(key, req_context)
        # A sharded class (TIER-X2) holds the key only if every shard does; the first
        # shard that misses ends the walk, so the rest are not asked.
        pending = False
        for c in classes:
            result = self._managers[c].lookup(key, req_context)
            if result == LookupResult.MISS:
                return LookupResult.MISS
            if result == LookupResult.HIT_PENDING:
                pending = True
        return LookupResult.HIT_PENDING if pending else LookupResult.HIT

    def host_resident_prefix(self, keys: list[OffloadKey]) -> int:
        resident = 0
        for cs, run in self._runs(list(keys)):
            found = min(self._managers[c].host_resident_prefix(run) for c in cs)
            resident += found
            if found < len(run):
                break
        return resident

    def touch(self, keys, req_context: ReqContext) -> None:
        for c, (ks, _) in self._split(keys).items():
            self._managers[c].touch(ks, req_context)

    def prepare_load(self, keys, req_context: ReqContext) -> LoadStoreSpec:
        keys = list(keys)
        taken: list[tuple[int, RaidenLoadStoreSpec]] = []
        for c, (ks, _) in self._split(keys).items():
            spec = self._managers[c].prepare_load(ks, req_context)
            assert isinstance(spec, RaidenLoadStoreSpec)
            if not spec.pinned:
                # This class gave back its own partial pins; give back the
                # classes prepared before it, nothing else.
                for done, done_spec in taken:
                    self._managers[done].unwind_prepared_load(done_spec)
                return RaidenLoadStoreSpec(keys, pinned=False)
            taken.append((c, spec))
        return RaidenLoadStoreSpec(keys)

    def prepare_store(self, keys, req_context: ReqContext) -> PrepareStoreOutput | None:
        keys = list(keys)
        outputs: list[tuple[int, PrepareStoreOutput]] = []
        for c, (ks, _) in self._split(keys).items():
            output = self._managers[c].prepare_store(ks, req_context)
            if output is None:
                for done, done_output in outputs:
                    self._managers[done].on_store_job_cancelled(
                        list(done_output.keys_to_store)
                    )
                return None
            outputs.append((c, output))
        chosen = {k for _, output in outputs for k in output.keys_to_store}
        keys_to_store = [k for k in keys if k in chosen]
        return PrepareStoreOutput(
            keys_to_store=keys_to_store,
            store_spec=RaidenLoadStoreSpec(keys_to_store),
            evicted_keys=[],
        )

    def set_victim_mode(self, on: bool) -> None:
        for manager in self._managers:
            manager.set_victim_mode(on)

    def set_exclusive_restore(self, on: bool) -> None:
        """Each class evicts its part of a load only once the whole parent
        succeeded: a failed parent recomputes every block, so no class may have
        dropped its host copy (``RaidenOffloadingManager.excl_resolve``)."""
        for manager in self._managers:
            manager.set_exclusive_restore(on, defer=True)
        self.__dict__["_excl"] = bool(on)

    def excl_stats(self) -> list[dict[str, int]]:
        return [manager.excl_stats() for manager in self._managers]

    def x2_stats(self) -> dict[str, int] | None:
        """TIER-X2 counters summed over the shard stores; None when TIER-X2 is off."""
        on = [m for m in self._managers if m.__dict__.get("_x2", False)]
        if not on:
            return None
        total = dict.fromkeys(raiden_store.X2_COUNTERS, 0)
        for manager in on:
            for name, value in manager.x2_stats().items():
                total[name] += value
        return total

    def prepare_victim_store(self, keys) -> list[OffloadKey]:
        keys = list(keys)
        chosen: set[OffloadKey] = set()
        for c, (ks, _) in self._split(keys).items():
            chosen.update(self._managers[c].prepare_victim_store(ks))
        return [k for k in keys if k in chosen]

    def take_events(self) -> list[OffloadingEvent]:
        return [event for manager in self._managers for event in manager.take_events()]

    def on_schedule_end(self, context: ScheduleEndContext) -> None:
        for manager in self._managers:
            manager.on_schedule_end(context)

    def reset_cache(self) -> None:
        raise NotImplementedError(
            "TPURaidenOffloadingConnector: external cache reset is not supported "
            "by the Raiden offload store."
        )

    def has_pending_work(self) -> bool:
        return bool(self._parents) or any(m.has_pending_work() for m in self._managers)

    def shutdown(self) -> None:
        for manager in self._managers:
            manager.shutdown()

    # -- the lookup log's probe surface ----------------------------------------

    def _sub_hashes(self, key: OffloadKey) -> list[bytes]:
        return self._managers[self.class_of(key)]._sub_hashes(key)

    def _classes_of_sub_hash(self, sub_hash: bytes) -> tuple[int, ...]:
        group = int.from_bytes(sub_hash[-8:-4], "big", signed=False)
        return self._classes_of_group[group]

    def _class_of_sub_hash(self, sub_hash: bytes) -> int:
        return self._classes_of_sub_hash(sub_hash)[0]

    def _shard_probe(self, classes: tuple[int, ...], run: list[bytes]) -> list:
        """A run of the first shard's sub-hashes, found only where every shard holds
        it (each shard keys its entries under its own namespace)."""
        first = self._managers[classes[0]]
        found = first._probe(run)
        for c in classes[1:]:
            if not found:
                break
            manager = self._managers[c]
            theirs = []
            for sub_hash in run[: len(found)]:
                parsed = first._sub_hash_to_offload_key(sub_hash)
                assert parsed is not None, sub_hash
                key, sub_idx = parsed
                theirs.append(manager._sub_hashes(key)[sub_idx])
            found = found[: len(manager._probe(theirs))]
        return found

    def _probe(
        self, sub_hashes: list[bytes], *, enable_global: bool = False
    ) -> list[tuple[bytes, Any]]:
        """Probes like the single store: class runs in order, up to the first miss."""
        out: list[tuple[bytes, Any]] = []
        run: list[bytes] = []
        for i, sub_hash in enumerate(sub_hashes):
            run.append(sub_hash)
            last = i + 1 == len(sub_hashes)
            if last or self._classes_of_sub_hash(
                sub_hashes[i + 1]
            ) != self._classes_of_sub_hash(sub_hash):
                classes = self._classes_of_sub_hash(sub_hash)
                if len(classes) == 1:
                    found = self._managers[classes[0]]._probe(run)
                else:
                    found = self._shard_probe(classes, run)
                out.extend(found)
                if len(found) < len(run):
                    break
                run = []
        return out

    # -- job submission bridge --------------------------------------------------

    def _open_parent(self, job_id: int, parent: _Parent) -> None:
        assert job_id not in self._parents, f"job {job_id} is already open"
        self._parents[job_id] = parent

    def submit_store_job(
        self, job_id: int, keys: list[OffloadKey], device_block_ids: list[int]
    ) -> None:
        assert len(keys) == len(device_block_ids), (len(keys), len(device_block_ids))
        parts = self._split(keys, device_block_ids)
        if not parts:
            self._finished.append(
                FinishedJob(job_id=job_id, op=AdmissionOp.STORE, success=True)
            )
            return
        self._open_parent(
            job_id,
            _Parent(
                AdmissionOp.STORE, None, list(keys), list(device_block_ids), set(parts)
            ),
        )
        for c, (ks, ids) in parts.items():
            kind = self._fp4_kind.get(c)
            if kind is not None:
                # An FP4 class saves the staging slots its keys were quantized into
                # (its kind's local slot ids are its device rows).
                bound = self._fp4_bound[job_id]
                ids = self._fp4_local(kind, [bound[k] for k in ks])
            self._managers[c].submit_store_job(job_id, ks, ids)

    def submit_load_job(
        self,
        job_id: int,
        keys: list[OffloadKey],
        device_block_ids: list[int],
        req_id: str,
        pinned: bool = True,
    ) -> None:
        assert len(keys) == len(device_block_ids), (len(keys), len(device_block_ids))
        if not pinned:
            # prepare_load gave back every pin it took: nothing to launch.
            self._finished.append(
                FinishedJob(
                    job_id=job_id,
                    op=AdmissionOp.LOAD,
                    success=False,
                    req_id=req_id,
                    failed_device_block_ids=list(device_block_ids),
                )
            )
            return
        parts = self._split(keys, device_block_ids)
        if not parts:
            self._finished.append(
                FinishedJob(
                    job_id=job_id, op=AdmissionOp.LOAD, success=True, req_id=req_id
                )
            )
            return
        self._open_parent(
            job_id,
            _Parent(
                AdmissionOp.LOAD, req_id, list(keys), list(device_block_ids), set(parts)
            ),
        )
        # Every child is launched even after a sibling's launch is refused:
        # each owns the pins its prepare took and finalizes them itself.
        for c, (ks, ids) in parts.items():
            self._load_keys[c] += len(ks)
            kind = self._fp4_kind.get(c)
            if kind is not None:
                # An FP4 class lands in staging slots; dequantized into ``ids`` once
                # the whole job has finished (take_fp4_dequant).
                slots = self._fp4_take(kind, len(ks))
                if slots is None:
                    self._fp4_count("load_no_slot", kind, len(ks))
                    self._managers[c].unwind_prepared_load(RaidenLoadStoreSpec(ks))
                    self._child_finished(
                        c,
                        FinishedJob(
                            job_id=job_id,
                            op=AdmissionOp.LOAD,
                            success=False,
                            req_id=req_id,
                            failed_device_block_ids=list(ids),
                        ),
                    )
                    continue
                # One load spans several FP4 classes (main, gdn-*): merge.
                self._fp4_bound.setdefault(job_id, {}).update(zip(ks, slots))
                self._fp4_count("load_keys", kind, len(ks))
                ids = self._fp4_local(kind, slots)
            self._managers[c].submit_load_job(job_id, ks, ids, req_id, pinned=True)

    def on_store_job_cancelled(self, keys: list[OffloadKey]) -> None:
        for c, (ks, _) in self._split(keys).items():
            self._managers[c].on_store_job_cancelled(ks)

    def is_job_launched(self, job_id: int) -> bool:
        return job_id in self._parents

    # -- completion ---------------------------------------------------------------

    def _child_finished(self, c: int, finished: FinishedJob) -> None:
        parent = self._parents.get(finished.job_id)
        if parent is None or c not in parent.waiting:
            raise AssertionError(
                f"host tier class {self._classes[c].name}: job {finished.job_id} "
                "finished with no parent "
                "waiting for it"
            )
        parent.waiting.discard(c)
        if not finished.success:
            parent.failed = True
        if c in self._fp4_kind and (
            parent.op == AdmissionOp.STORE or not finished.success
        ):
            # A store child has read its slots (or failed); a failed load child
            # left nothing to dequantize. Only this child's slots: a sibling FP4
            # class of the same job may still be landing in its own.
            self._fp4_unbind_keys(
                finished.job_id, [k for k in parent.keys if self.class_of(k) == c]
            )
        if self._check is not None and finished.stored_keys:
            # Saved by this child, whatever its siblings do: a later load
            # can restore them.
            block_of = dict(zip(parent.keys, parent.device_block_ids))
            name = self._classes[c].name
            self._check[0].extend(
                (name, bytes(k), block_of[k]) for k in finished.stored_keys
            )
        if parent.waiting:
            return
        del self._parents[finished.job_id]
        bound = self._fp4_bound.pop(finished.job_id, None)
        if bound:
            if parent.op == AdmissionOp.LOAD and not parent.failed:
                block_of = dict(zip(parent.keys, parent.device_block_ids))
                self._fp4_dequant.extend((s, block_of[k]) for k, s in bound.items())
                self._fp4_release.extend(bound.values())
            else:
                self.fp4_give(list(bound.values()))
        if parent.op == AdmissionOp.LOAD and self.__dict__.get("_excl", False):
            for manager in self._managers:
                manager.excl_resolve(finished.job_id, not parent.failed)
        if (
            self._check is not None
            and parent.op == AdmissionOp.LOAD
            and not parent.failed
        ):
            self._check[1].extend(
                (self._classes[c].name, bytes(k), b)
                for k, b in zip(parent.keys, parent.device_block_ids)
                for c in self.classes_of(k)
            )
        self._finished.append(
            FinishedJob(
                job_id=finished.job_id,
                op=parent.op,
                success=not parent.failed,
                req_id=parent.req_id,
                # The scheduler's own IDs, every destination of the job.
                failed_device_block_ids=(
                    list(parent.device_block_ids)
                    if parent.failed and parent.op == AdmissionOp.LOAD
                    else []
                ),
            )
        )

    def poll_finished_jobs(self) -> list[FinishedJob]:
        for c, manager in enumerate(self._managers):
            for finished in manager.poll_finished_jobs():
                self._child_finished(c, finished)
        finished, self._finished = self._finished, []
        return finished

    def drain_jobs(
        self, job_ids: set[int], deadline: float | None = None
    ) -> list[FinishedJob]:
        """Drains the children of ``job_ids`` against one deadline.

        Returns the parents that ended.

        A parent missing from the result has a child still in flight at the deadline
        (that child is poisoned as the single manager does).
        """
        if deadline is None:
            deadline = time.perf_counter() + raiden_store._DRAIN_TIMEOUT_S
        for c, manager in enumerate(self._managers):
            ids = {
                j
                for j in job_ids
                if (p := self._parents.get(j)) is not None and c in p.waiting
            }
            if ids:
                for finished in manager.drain_jobs(ids, deadline=deadline):
                    self._child_finished(c, finished)
        drained = [f for f in self._finished if f.job_id in job_ids]
        self._finished = [f for f in self._finished if f.job_id not in job_ids]
        return drained

    def take_check_requests(self) -> tuple[list, list]:
        """Takes the (class, key, block) entries to hash and to compare.

        Empty when check mode is off.
        """
        if self._check is None:
            return [], []
        stores, loads = self._check
        self._check = ([], [])
        return stores, loads

    # -- telemetry --------------------------------------------------------------

    def bytes_of_keys(self, keys: Sequence[OffloadKey]) -> int:
        """Host bytes behind ``keys``, each at its class's entry size."""
        return sum(
            self._classes[c].bytes_per_entry for k in keys for c in self.classes_of(k)
        )

    def telemetry_line(self, now: float | None = None) -> str:
        """One line of per-class tier state.

        Per class: held of capacity, stores and evictions (total and a minute since the
        last line), store time (``time.time()``) and age of the oldest held entry, load
        keys.
        """
        now = time.time() if now is None else now
        stats = [m.held_stats() for m in self._managers]
        last = self._telemetry_last
        parts = []
        for c, (held, stored, evicted, oldest) in zip(self._classes, stats):
            rate = ""
            if last is not None and now > last[0]:
                prev_stored, prev_evicted = last[1][c.index]
                per_min = 60.0 / (now - last[0])
                rate = (
                    f" stores_per_min={(stored - prev_stored) * per_min:.0f}"
                    f" evictions_per_min={(evicted - prev_evicted) * per_min:.0f}"
                )
            age = "n/a" if oldest is None else f"{now - oldest:.0f}"
            stamp = "n/a" if oldest is None else f"{oldest:.1f}"
            parts.append(
                f"tier[{c.name}] held={held}/{self._plan.capacity(c)} "
                f"stored={stored} evicted={evicted}{rate} "
                f"oldest_stored_at={stamp} oldest_age_s={age} "
                f"load_keys={self._load_keys[c.index]}"
                + (
                    f" discarded={self._managers[c.index].excl_stats()['discarded']}"
                    if self.__dict__.get("_excl", False)
                    else ""
                )
            )
        self._telemetry_last = (now, [(s[1], s[2]) for s in stats])
        return " ".join(parts)


class TieredStoreOffloadingWorker(OffloadingWorker):
    """Worker-side endpoint: one ``RaidenStoreOffloadingWorker`` a class."""

    def __init__(self, workers: dict[str, Any]):
        self.workers = workers

    def submit_store(self, job_id, src_spec, dst_spec) -> bool:
        raise AssertionError(
            "TPURaidenOffloadingConnector: submit_store reached a worker; "
            "transfers are controller-driven"
        )

    def submit_load(self, job_id, src_spec, dst_spec) -> bool:
        raise AssertionError(
            "TPURaidenOffloadingConnector: submit_load reached a worker; "
            "transfers are controller-driven"
        )

    def get_finished(self) -> list[TransferResult]:
        return []

    def wait(self, job_ids: set[int]) -> None:
        return

    def shutdown(self) -> None:
        for worker in self.workers.values():
            worker.shutdown()


def class_kv_caches(
    kv_caches: CanonicalKVCaches, tensor_indices: Sequence[int]
) -> CanonicalKVCaches:
    """The canonical tensors at ``tensor_indices``, group refs renumbered."""
    position = {idx: pos for pos, idx in enumerate(tensor_indices)}
    return CanonicalKVCaches(
        tensors=[kv_caches.tensors[i] for i in tensor_indices],
        group_data_refs=[
            [
                CanonicalKVCacheRef(
                    tensor_idx=position[ref.tensor_idx],
                    page_size_bytes=ref.page_size_bytes,
                )
                for ref in refs
                if ref.tensor_idx in position
            ]
            for refs in kv_caches.group_data_refs
        ],
    )
