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
"""A pipeline that spans several TPU slices.

The devices of a slice are joined by ICI; two slices are joined only by the
host network. A sliced pipeline is cut into runs of consecutive stages, one
per slice, and the slices may differ in size. One engine drives every
slice's workers, one ``vllm serve`` per host as for any engine of several
hosts, with the hosts numbered slice-major. Every host runs the same number
of workers and every stage has the same number of workers, and engine ranks
keep their stage-major order, so a slice holds a run of consecutive engine
ranks: the P_s stages they make and the H_s hosts they cover, which come
after those of the slices before it.

``TPU_PP_SLICES`` sets the cut: a number of equal slices, or the stages of
each slice in pipeline order.

Each slice is a torch.distributed world of its own, brought up as a
stand-alone deployment of that slice would be: ranks from 0, the slice's
workers alone, a pipeline group of its P_s stages. torch_tpu never sees a
rank or a device of another slice. What spans the engine is layered on top:

  * ``SlicedPipelineGroup`` stands in for vLLM's pipeline group and reports
    a worker's place in the whole pipeline, so model layers, the runner and
    the KV connectors see every stage; the group of vLLM's own, the
    worker's column over its slice's stages, stays reachable as
    ``in_slice`` for code that works inside the torch world.
  * ``EngineQueues`` stands in for vLLM's inner data-parallel world group,
    through which a worker joins the engine's message queues: the queues'
    handles go through the start-up rendezvous store instead of a world
    that spans every worker.
  * Stages inside a slice hand activations on over ICI; the hop from the
    last stage of one slice to the first of the next goes over TCP
    (``pp_buffered``, ``pp_dcn_link``).
"""

import base64
import bisect
import copy
import itertools
import pickle
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from torch.distributed import Store
from torch.distributed.constants import default_pg_timeout

from vllm_torchtpu import envs

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import GroupCoordinator

# Prefix of the rendezvous-store keys the engine's message queues meet at.
_QUEUE_KEY = "tpu_pp_slices_queue"

# Prefix of the rendezvous-store keys of the objects a column broadcasts.
_BROADCAST_KEY = "tpu_pp_slices_broadcast"


def _set_object(store: Store, key: str, obj: Any) -> None:
    """Sets ``key`` to ``obj``, pickled. The store's values are text, so the
    pickle goes in as base64."""
    store.set(key, base64.b64encode(pickle.dumps(obj)).decode("ascii"))


def _get_object(store: Store, key: str) -> Any:
    """The object ``_set_object`` put under ``key``, which must be set."""
    return pickle.loads(base64.b64decode(store.get(key)))


@dataclass(frozen=True)
class PipelineSlices:
    """A pipeline of ``stages`` stages on ``nnodes`` hosts, cut into slices of
    ``ranks[s]`` consecutive engine ranks each, in pipeline order. Every
    stage has world_size // stages workers and every host world_size //
    nnodes, so a slice's stages and hosts follow from its ranks."""

    ranks: tuple[int, ...]
    stages: int
    nnodes: int

    def __post_init__(self) -> None:
        if (
            not self.ranks
            or min(self.ranks) < 1
            or self.stages < 1
            or self.nnodes < 1
            or self.world_size % self.stages
            or self.world_size % self.nnodes
            or any(
                ranks % self.stage_width or ranks % self.devices_per_host
                for ranks in self.ranks
            )
        ):
            raise ValueError(
                f"{self} is not a pipeline cut into slices of whole stages and "
                "hosts: every slice holds at least one rank, the world size is "
                "a multiple of the stages and of the hosts, and every slice's "
                "ranks make whole stages and cover whole hosts"
            )

    @property
    def slices(self) -> int:
        return len(self.ranks)

    @property
    def world_size(self) -> int:
        return sum(self.ranks)

    @property
    def stage_width(self) -> int:
        """The workers of one stage."""
        return self.world_size // self.stages

    @property
    def devices_per_host(self) -> int:
        return self.world_size // self.nnodes

    def first_rank_of(self, slice_: int) -> int:
        """The first engine rank of slice ``slice_``."""
        return sum(self.ranks[:slice_])

    def world_size_of(self, slice_: int) -> int:
        """The workers, and devices, of slice ``slice_``."""
        return self.ranks[slice_]

    def stages_of(self, slice_: int) -> int:
        return self.ranks[slice_] // self.stage_width

    def first_stage_of(self, slice_: int) -> int:
        """The first stage of slice ``slice_`` in the whole pipeline."""
        return self.first_rank_of(slice_) // self.stage_width

    def hosts_of(self, slice_: int) -> int:
        return self.ranks[slice_] // self.devices_per_host

    def first_node_of(self, slice_: int) -> int:
        """The first node rank of slice ``slice_``."""
        return self.first_rank_of(slice_) // self.devices_per_host

    def slice_of_rank(self, rank: int) -> int:
        return bisect.bisect_right(list(itertools.accumulate(self.ranks)), rank)

    def slice_of_node(self, node_rank: int) -> int:
        return self.slice_of_rank(node_rank * self.devices_per_host)

    def slice_start(self, rank: int) -> int:
        """The first engine rank of ``rank``'s slice."""
        return self.first_rank_of(self.slice_of_rank(rank))


def pipeline_slices(parallel_config: Any) -> PipelineSlices | None:
    """How ``TPU_PP_SLICES`` cuts this engine's pipeline, or None for a
    pipeline on one slice. The setting is a number of equal slices, or a list
    with the stages of each slice in pipeline order, such as ``4,2``.

    Raises:
      ValueError: TPU_PP_SLICES is neither form, a number below 1, or a list
        of one slice, of a slice without stages, of stages that do not add up
        to the pipeline's, or of a slice that does not cover whole hosts; or
        equal slices would not each hold the same number of stages and hosts,
        as the one stage of an engine without a pipeline cannot.
    """
    # Set but empty means unset, as for a number.
    text = envs.TPU_PP_SLICES or "1"
    if "," in text:
        return _listed_slices(text, parallel_config)
    try:
        slices = int(text)
    except ValueError:
        raise ValueError(_malformed(text)) from None
    if slices < 1:
        raise ValueError(f"TPU_PP_SLICES must be at least 1, got {slices}.")
    if slices == 1:
        return None
    stages = parallel_config.pipeline_parallel_size
    nnodes = parallel_config.nnodes
    if stages % slices:
        raise ValueError(
            f"TPU_PP_SLICES={slices} must divide --pipeline-parallel-size "
            f"({stages}): every slice holds the same number of stages."
        )
    if nnodes % slices:
        raise ValueError(
            f"TPU_PP_SLICES={slices} must divide --nnodes ({nnodes}): every "
            "slice spans the same number of hosts, and every host runs "
            "vllm serve with its --node-rank."
        )
    return PipelineSlices(
        ranks=(parallel_config.world_size // slices,) * slices,
        stages=stages,
        nnodes=nnodes,
    )


def _malformed(text: str) -> str:
    return (
        f"TPU_PP_SLICES={text!r} is neither a number of equal slices, such as "
        "2, nor a list with the stages of each slice in pipeline order, such "
        "as 4,2."
    )


def _listed_slices(text: str, parallel_config: Any) -> PipelineSlices:
    """The slices of ``TPU_PP_SLICES`` given as a list, such as ``4,2``; a
    trailing comma ends the list."""
    entries = text.split(",")
    if not entries[-1].strip():
        entries.pop()
    try:
        counts = tuple(int(entry) for entry in entries)
    except ValueError:
        raise ValueError(_malformed(text)) from None
    if len(counts) == 1:
        raise ValueError(
            f"TPU_PP_SLICES={text} lists one slice, which is a pipeline on one "
            "slice: unset TPU_PP_SLICES."
        )
    ranks = _ranks_of_listed_stages(text, counts, parallel_config)
    nnodes = parallel_config.nnodes
    per_host = parallel_config.world_size // nnodes
    for slice_, count in enumerate(ranks):
        if count % per_host:
            raise ValueError(
                f"TPU_PP_SLICES={text}: slice {slice_} holds {count} engine "
                f"ranks, which do not cover whole hosts; every host runs "
                f"{per_host} workers (the world size over --nnodes {nnodes}), so "
                "every slice must cover whole hosts."
            )
    return PipelineSlices(
        ranks=ranks, stages=parallel_config.pipeline_parallel_size, nnodes=nnodes
    )


def _ranks_of_listed_stages(
    text: str, counts: tuple[int, ...], parallel_config: Any
) -> tuple[int, ...]:
    """The engine ranks of each slice of a ``TPU_PP_SLICES`` list, whose
    entries count the stages of each slice: every stage has world_size //
    stages workers."""
    stages = parallel_config.pipeline_parallel_size
    if min(counts) < 1:
        raise ValueError(
            f"TPU_PP_SLICES={text}: every slice holds at least 1 stage, as in 4,2."
        )
    if sum(counts) != stages:
        raise ValueError(
            f"TPU_PP_SLICES={text} lists {sum(counts)} stages; the stages of "
            f"the slices must add up to --pipeline-parallel-size ({stages})."
        )
    width = parallel_config.world_size // stages
    return tuple(count * width for count in counts)


class SlicedPipelineGroup:
    """Stands in for vLLM's pipeline group on a worker of a sliced pipeline.

    ``in_slice`` is vLLM's group: this worker's column over its slice's
    stages, in ranks of the slice's torch world. This one reports the
    worker's place in the whole pipeline, in engine ranks. Beyond the stage
    layout it defines one collective, ``broadcast_object``, which vLLM runs
    over the pipeline group when a KV connector starts; the whole pipeline
    has no torch world for any other to run in, so code that asks for one
    fails.

    ``store`` is the start-up rendezvous store, which every host of the
    engine reaches; the broadcast and the hand-off between slices meet
    through it.
    """

    def __init__(
        self, in_slice: Any, layout: PipelineSlices, engine_rank: int, store: Store
    ):
        self.in_slice = in_slice
        self.store = store
        # Broadcasts this worker has taken part in; every worker of a column
        # makes the same ones in the same order, so the count names each.
        self._broadcasts = 0
        self.rank = engine_rank
        self.world_size = layout.stages
        self.rank_in_group = layout.first_stage_of(
            layout.slice_of_rank(engine_rank)
        ) + int(in_slice.rank_in_group)
        column = engine_rank % layout.stage_width
        self.ranks = [
            stage * layout.stage_width + column for stage in range(layout.stages)
        ]
        if (
            not 0 <= self.rank_in_group < layout.stages
            or self.ranks[self.rank_in_group] != engine_rank
        ):
            raise RuntimeError(
                f"engine rank {engine_rank} is stage {in_slice.rank_in_group} of "
                f"slice {layout.slice_of_rank(engine_rank)} in its torch world, "
                f"but stage-major engine ranks put it at stage "
                f"{engine_rank // layout.stage_width} of {layout}"
            )

    @property
    def first_rank(self) -> int:
        return self.ranks[0]

    @property
    def last_rank(self) -> int:
        return self.ranks[-1]

    @property
    def is_first_rank(self) -> bool:
        return self.rank == self.first_rank

    @property
    def is_last_rank(self) -> bool:
        return self.rank == self.last_rank

    def broadcast_object(self, obj: Any | None = None, src: int = 0) -> Any:
        """Returns, on every worker of this column, the object the worker at
        stage ``src`` passes; the others wait for it."""
        if not 0 <= src < self.world_size:
            raise ValueError(
                f"stage {src} is not one of the pipeline's {self.world_size}"
            )
        key = f"{_BROADCAST_KEY}_{self.first_rank}_{self._broadcasts}"
        self._broadcasts += 1
        if self.rank_in_group == src:
            _set_object(self.store, key, obj)
            return obj
        self.store.wait([key], default_pg_timeout)
        return _get_object(self.store, key)

    def destroy(self) -> None:
        self.in_slice.destroy()


def install_sliced_pipeline_group(
    layout: PipelineSlices, engine_rank: int, store: Store
) -> SlicedPipelineGroup:
    """Puts a SlicedPipelineGroup in place of the pipeline group vLLM built
    over this worker's slice."""
    from vllm.distributed import parallel_state

    group = SlicedPipelineGroup(
        parallel_state.get_pp_group(), layout, engine_rank, store
    )
    # vLLM types the slot as its own group class; the stand-in defines what
    # readers of the pipeline group use.
    parallel_state._PP = cast("GroupCoordinator", group)
    return group


def in_slice_pipeline_group() -> Any:
    """The pipeline group inside this worker's torch world: its slice's
    stages under a sliced pipeline, the whole pipeline otherwise."""
    from vllm.distributed import parallel_state

    group = parallel_state.get_pp_group()
    if isinstance(group, SlicedPipelineGroup):
        return group.in_slice
    return group


def slice_view(vllm_config: Any, layout: PipelineSlices, slice_master_addr: str) -> Any:
    """A shallow copy of ``vllm_config`` that describes this worker's slice
    as a deployment of its own: the slice's stages, workers and hosts, with
    the first host of the slice as the master of a slice of several hosts.

    vLLM's ``init_distributed_environment`` derives a worker's torch world
    from the current config whenever it spans several hosts: rank, world
    size and the rendezvous address. It runs under this view.
    """
    parallel_config = copy.copy(vllm_config.parallel_config)
    slice_ = layout.slice_of_node(parallel_config.node_rank)
    parallel_config.pipeline_parallel_size = layout.stages_of(slice_)
    parallel_config.world_size = layout.world_size_of(slice_)
    parallel_config.nnodes = layout.hosts_of(slice_)
    parallel_config.node_rank -= layout.first_node_of(slice_)
    if layout.hosts_of(slice_) > 1:
        parallel_config.master_addr = slice_master_addr
    view = copy.copy(vllm_config)
    view.parallel_config = parallel_config
    return view


class EngineQueues:
    """Stands in for vLLM's inner data-parallel world group
    (``parallel_state._INNER_DP_WORLD``) on a worker of a sliced pipeline.

    A worker of an engine on several hosts joins the engine's message queues
    through that group (``WorkerProc._init_message_queues``), which hands
    the queues' handles around with collectives over one world of every
    worker. A sliced pipeline has a world per slice, so the handles go
    through the start-up rendezvous store. Ranks are engine ranks. Workers
    get here once their models have loaded, minutes apart, so a wait is
    bounded like the collectives it replaces.
    """

    def __init__(self, store: Store, rank: int, world_size: int, local_world_size: int):
        self.store = store
        self.rank = rank
        self.world_size = world_size
        # The workers of a host; the engine numbers ranks host by host.
        self.local_world_size = local_world_size

    def create_mq_broadcaster(
        self,
        writer_rank: int = 0,
        external_writer_handle: Any = None,
        blocking: bool = True,
    ) -> Any:
        """Joins the queue every worker reads: ``writer_rank`` holds its
        handle, from the process that writes it, and publishes it."""
        from vllm.distributed.device_communicators.shm_broadcast import MessageQueue

        key = f"{_QUEUE_KEY}_broadcast_{writer_rank}"
        if self.rank == writer_rank:
            if external_writer_handle is None:
                raise NotImplementedError(
                    "a sliced pipeline joins the engine's broadcast queue only; "
                    "its writer rank needs the queue's handle"
                )
            queue = MessageQueue.create_from_handle(external_writer_handle, self.rank)
            _set_object(self.store, key, queue.export_handle())
        else:
            self.store.wait([key], default_pg_timeout)
            handle = _get_object(self.store, key)
            queue = MessageQueue.create_from_handle(handle, self.rank)
        if blocking:
            queue.wait_until_ready()
        return queue

    def create_single_reader_mq_broadcasters(
        self, reader_rank_in_group: int = 0, blocking: bool = False
    ) -> tuple[Any, list[Any]]:
        """Creates this worker's reply queue, which ``reader_rank_in_group``
        reads; that rank gets every worker's handle, in rank order, and the
        others get none."""
        from vllm.distributed.device_communicators.shm_broadcast import MessageQueue

        reader = reader_rank_in_group
        same_node = (
            self.rank // self.local_world_size == reader // self.local_world_size
        )
        # Sizes from vLLM's GroupCoordinator.create_single_reader_mq_broadcasters.
        queue = MessageQueue(
            n_reader=1,
            n_local_reader=1 if same_node else 0,
            max_chunk_bytes=1 << 22,
            max_chunks=6,
        )
        prefix = f"{_QUEUE_KEY}_reply_{reader}"
        _set_object(self.store, f"{prefix}_{self.rank}", queue.export_handle())
        handles = []
        if self.rank == reader:
            keys = [f"{prefix}_{rank}" for rank in range(self.world_size)]
            self.store.wait(keys, default_pg_timeout)
            handles = [_get_object(self.store, key) for key in keys]
        if blocking:
            queue.wait_until_ready()
        return queue, handles


def install_engine_queues(
    store: Store, rank: int, world_size: int, local_world_size: int
) -> EngineQueues:
    """Puts an EngineQueues in place of vLLM's inner data-parallel world
    group on this worker."""
    from vllm.distributed import parallel_state

    queues = EngineQueues(store, rank, world_size, local_world_size)
    # vLLM types the slot as its own group class; the stand-in defines the
    # two calls a worker makes on it to join the engine's message queues.
    parallel_state._INNER_DP_WORLD = cast("GroupCoordinator", queues)
    return queues
