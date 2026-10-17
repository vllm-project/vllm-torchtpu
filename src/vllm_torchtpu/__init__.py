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
# The environment variables override should be imported before any other
# modules to ensure that the environment variables are set before any
# other modules are imported.
import vllm_torchtpu.env_override  # noqa: F401
from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


def _patch_vllm_compact_mamba_streaming_input() -> None:
    """Reject input sessions whose retained state can exceed the compact pool."""
    import inspect
    from functools import wraps

    from vllm.v1.engine.input_processor import InputProcessor

    from vllm_torchtpu.platforms.tpu_block_size_utils import unified_kv_layout_enabled

    original = InputProcessor.process_inputs
    marker = "_tpu_compact_mamba_input_patch"
    if original.__dict__.get(marker, False):
        return
    signature = inspect.signature(original)

    @wraps(original)
    def process_inputs(self, *args, **kwargs):
        config = self.vllm_config
        transfer = config.kv_transfer_config
        uses_slots = transfer is None or transfer.kv_connector == "DecodeBenchConnector"
        if (
            uses_slots
            and config.model_config.is_hybrid
            and not unified_kv_layout_enabled(config)
            and signature.bind(self, *args, **kwargs).arguments.get("resumable", False)
        ):
            raise ValueError(
                "resumable input is unsupported with the compact Mamba state pool: "
                "waiting sessions retain state beyond the running-request limit. "
                "Use the unified KV pool on a supported architecture, or submit "
                "complete inputs. Streaming output remains supported."
            )
        return original(self, *args, **kwargs)

    process_inputs.__dict__[marker] = True
    InputProcessor.process_inputs = process_inputs


def _patch_vllm_hybrid_kv_load_failure_recovery() -> None:
    """Backport hybrid KV cache load failure recovery from vLLM #50388.

    In vLLM <= 0.29.0 (prior to vLLM #50388),
    ``Scheduler._update_requests_with_invalid_blocks`` unpacks
    ``(req_block_ids,) = self.kv_cache_manager.get_block_ids(req_id)``, which
    raises ``ValueError: too many values to unpack (expected 1)`` whenever a
    hybrid model (multiple KV cache groups) encounters a failed KV load.
    """
    from functools import wraps

    from vllm.v1.core.sched.scheduler import Scheduler

    if Scheduler.__dict__.get("_tpu_hybrid_kv_load_failure_patch", False):
        return

    original_update = getattr(Scheduler, "_update_requests_with_invalid_blocks", None)
    if original_update is None:
        return

    @wraps(original_update)
    def _hybrid_update_requests_with_invalid_blocks(
        self,
        requests,
        invalid_block_ids: set[int],
        num_scheduled_tokens: dict[str, int],
        evict_blocks: bool = True,
    ) -> tuple[set[str], int, set[int]]:
        if len(self.kv_cache_config.kv_cache_groups) <= 1:
            return original_update(
                self,
                requests,
                invalid_block_ids,
                num_scheduled_tokens,
                evict_blocks=evict_blocks,
            )

        affected_req_ids: set[str] = set()
        total_affected_tokens = 0
        blocks_to_evict: set[int] = set()
        marked_invalid_block_ids: set[int] = set()
        null_block_id = self.kv_cache_manager.block_pool.null_block.block_id
        for request in requests:
            req_id = request.request_id
            req_block_ids_per_group = self.kv_cache_manager.get_block_ids(req_id)
            req_num_computed_tokens = (
                request.num_computed_tokens - num_scheduled_tokens.get(req_id, 0)
            )
            request_invalid_block_ids = {
                block_id
                for req_block_ids in req_block_ids_per_group
                for block_id in req_block_ids
                if block_id != null_block_id and block_id in invalid_block_ids
            }
            if request_invalid_block_ids:
                marked_invalid_block_ids |= request_invalid_block_ids
                total_affected_tokens += req_num_computed_tokens
                request.num_computed_tokens = 0
                if evict_blocks:
                    for req_block_ids in req_block_ids_per_group:
                        blocks_to_evict.update(
                            block_id
                            for block_id in req_block_ids
                            if block_id != null_block_id
                        )
                affected_req_ids.add(req_id)
        return affected_req_ids, total_affected_tokens, blocks_to_evict

    Scheduler._update_requests_with_invalid_blocks = (
        _hybrid_update_requests_with_invalid_blocks
    )
    Scheduler._tpu_hybrid_kv_load_failure_patch = True
    logger.info(
        "Applied TPU patch: hybrid KV cache load failure recovery (vLLM #50388)."
    )


def _patch_vllm_mamba_split_scheduler_block_size() -> None:
    """Align Mamba prefill splits to the scheduler block size.

    vLLM 0.27.0 uses the rank-local physical ``CacheConfig.block_size`` in
    ``Scheduler._mamba_block_aligned_split``.  Under hybrid PCP, prefix hashes
    and reusable Mamba states instead advance at ``Scheduler.block_size``, the
    logical block spanning all PCP ranks.  Calling the upstream method with a
    scheduler-local cache-config view backports the corresponding one-line fix
    without mutating the shared ``VllmConfig.cache_config`` or copying the
    scheduling algorithm into this plugin.
    """
    import copy
    from functools import wraps

    from vllm.v1.core.sched.scheduler import Scheduler

    original_split = Scheduler._mamba_block_aligned_split
    if getattr(original_split, "_tpu_scheduler_block_size_patch", False):
        return

    @wraps(original_split)
    def split_with_scheduler_block_size(self, *args, **kwargs):
        physical_cache_config = self.cache_config
        if physical_cache_config.block_size == self.block_size:
            return original_split(self, *args, **kwargs)

        scheduler_cache_config = copy.copy(physical_cache_config)
        scheduler_cache_config.block_size = self.block_size
        # schedule() calls this method synchronously on the EngineCore thread,
        # and the upstream implementation contains no callback or await point.
        self.cache_config = scheduler_cache_config
        try:
            return original_split(self, *args, **kwargs)
        finally:
            self.cache_config = physical_cache_config

    split_with_scheduler_block_size._tpu_scheduler_block_size_patch = True
    Scheduler._mamba_block_aligned_split = split_with_scheduler_block_size
    logger.info("Applied TPU patch: align Mamba prefill splits to scheduler blocks.")


def _patch_vllm_aot_compile_cache_key() -> None:
    """Include the TPU compiler hash, and the trace ordinal, in vLLM's outer
    AOT cache key."""
    from vllm.compilation import caching

    original = caching.aot_compile_hash_factors
    if getattr(original, "_tpu_compiler_hash_patch", False):
        return

    def tpu_aot_compile_hash_factors(vllm_config):
        from vllm_torchtpu.compilation import shape_variants
        from vllm_torchtpu.compilation.tpu_compiler import compute_tpu_compilation_hash

        return [
            *original(vllm_config),
            compute_tpu_compilation_hash(vllm_config),
            *filter(None, [shape_variants.aot_cache_tag()]),
        ]

    tpu_aot_compile_hash_factors._tpu_compiler_hash_patch = True
    caching.aot_compile_hash_factors = tpu_aot_compile_hash_factors
    logger.info("Applied TPU patch: include compiler hash in AOT cache key.")


def _aot_load_vote_group(model):
    """The ranks that must take the same AOT load path as this one: those the
    model's compiled graphs run collectives with, or None for no vote.

    An encoder only talks within its tensor-parallel group, and not at all in
    data-parallel encoder mode, where a rank without images skips it. Under
    pipeline parallelism there is no vote: speculative decoding, the one
    known trigger of a split, is rejected there (pp_validation).
    """
    import torch.distributed as dist
    from vllm.distributed import parallel_state

    if not dist.is_initialized():
        return None
    if model._is_encoder:
        mm_config = model.vllm_config.model_config.multimodal_config
        if mm_config is not None and mm_config.mm_encoder_tp_mode == "data":
            return None
        return parallel_state.get_tp_group()
    if model.vllm_config.parallel_config.pipeline_parallel_size > 1:
        return None
    return parallel_state.get_world_group()


def _aot_load_ranks_missing(ok, group, path, step):
    """Gather `ok` from every rank of `group`; return the ranks where it is false.

    `step` ("find" or "load") is the same on every rank, unlike `path`, which
    names this rank's own artifact; it tags the gather, so a rank at the other
    vote fails loudly instead of being read as an answer.
    """
    from vllm_torchtpu.distributed.rank_agreement import gather_step_reports

    logger.debug("Waiting for %d ranks to %s %s.", group.world_size, step, path)
    reports = gather_step_reports(
        group,
        f"AOT artifact {step} vote",
        (int(ok),),
        on_timeout=f"It was voting to {step} {path}. "
        "Clear the compile cache, or raise --cpu-distributed-timeout-seconds if "
        "a rank is legitimately slow to load.",
    )
    return [r for r, (rank_ok,) in enumerate(reports) if not rank_ok]


def _patch_vllm_aot_load_unanimous() -> None:
    """Use a rank's AOT artifact only if every rank that shares its collectives
    loaded its own.

    ``_try_load_aot_compiled_fn`` decides per rank: a rank whose
    ``rank_{r}_{dp}/model`` is missing, truncated, or stale falls back to a
    Dynamo compile while its peers run the loaded artifact. The two paths are
    not collective-equivalent. The spec-decode warmup's first attempt fails
    only on a Dynamo-compiling rank (see ``_warmup_one_spec_request``), so
    that rank alone unwinds and retries while its peers are already inside
    the draft step's collectives, which can halt the device.

    The ranks agree twice. First on whether each one has a non-empty artifact,
    before anything is loaded, so the common split (a missing file) leaves no
    loaded state behind. Then on whether the load itself succeeded, which
    catches an artifact that exists but cannot be loaded.
    """
    from vllm import envs as vllm_envs
    from vllm.compilation import decorators

    original = decorators._try_load_aot_compiled_fn
    if getattr(original, "_tpu_unanimous_patch", False):
        return

    def tpu_try_load_aot_compiled_fn(model, aot_compilation_path):
        import os

        group = _aot_load_vote_group(model)
        if group is None or group.world_size == 1:
            return original(model, aot_compilation_path)

        try:
            present = os.path.getsize(aot_compilation_path) > 0
        except OSError:
            present = False
        missing = _aot_load_ranks_missing(present, group, aot_compilation_path, "find")
        if missing:
            if vllm_envs.VLLM_FORCE_AOT_LOAD:
                raise RuntimeError(
                    f"VLLM_FORCE_AOT_LOAD is set but ranks {missing} have no AOT "
                    f"artifact {aot_compilation_path}."
                )
            if present:
                logger.warning(
                    "Not loading AOT artifact %s: ranks %s of %d have none, so "
                    "every rank compiles.",
                    aot_compilation_path,
                    missing,
                    group.world_size,
                )
            return None

        error = None
        try:
            loaded_fn = original(model, aot_compilation_path)
        except Exception as e:  # VLLM_FORCE_AOT_LOAD re-raises
            loaded_fn, error = None, e
        failed = _aot_load_ranks_missing(
            loaded_fn is not None, group, aot_compilation_path, "load"
        )
        if error is not None:
            raise error
        if not failed:
            return loaded_fn
        if vllm_envs.VLLM_FORCE_AOT_LOAD:
            raise RuntimeError(
                f"VLLM_FORCE_AOT_LOAD is set but the AOT artifact "
                f"{aot_compilation_path} failed to load on ranks {failed}."
            )
        if loaded_fn is not None:
            logger.warning(
                "Discarding AOT artifact %s: it failed to load on ranks %s of %d, "
                "so every rank compiles.",
                aot_compilation_path,
                failed,
                group.world_size,
            )
        return None

    tpu_try_load_aot_compiled_fn.__dict__["_tpu_unanimous_patch"] = True
    decorators._try_load_aot_compiled_fn = tpu_try_load_aot_compiled_fn
    logger.info("Applied TPU patch: load AOT artifacts only when all ranks can.")


def _patch_vllm_config_hash_ignore_diagnostics() -> None:
    """Keep diagnostics-only ``additional_config`` keys out of cache keys.

    ``VllmConfig.compute_hash()`` unconditionally folds
    ``json.dumps(additional_config)`` into its result, and that hash feeds both
    the AOT compile cache key (``caching.aot_compile_hash_factors``) and the
    piecewise cache dir (``backends.py``). The phased profiler is configured
    through ``additional_config``, so a per-run trace directory would force a
    full recompile on every server start. Upstream takes the same care in
    ``ProfilerConfig.compute_hash()`` ("this config will not affect the
    computation graph"); this is the equivalent carve-out for the plugin-owned
    keys, mirroring ``_TPU_COMPILE_ENV_IGNORED`` for env vars.
    """
    import copy

    from vllm.config import VllmConfig

    from vllm_torchtpu.runner.utils import HASH_IGNORED_ADDITIONAL_CONFIG_KEYS

    if getattr(VllmConfig, "_tpu_additional_config_hash_patch", False):
        return

    original = VllmConfig.compute_hash

    def compute_hash(self, *args, **kwargs) -> str:
        additional_config = self.additional_config
        if not isinstance(additional_config, dict):
            return original(self, *args, **kwargs)
        filtered = {
            key: value
            for key, value in additional_config.items()
            if key not in HASH_IGNORED_ADDITIONAL_CONFIG_KEYS
        }
        if filtered == additional_config:
            return original(self, *args, **kwargs)
        # Hash a shallow copy with the filtered dict rather than mutating the
        # live config: compute_hash() may run while other code holds the same
        # VllmConfig, and a shallow copy shares every sub-config by reference
        # (cheap, no re-validation -- __post_init__ is not re-run) while giving
        # compute_hash its own additional_config to read.
        proxy = copy.copy(self)
        proxy.additional_config = filtered
        return original(proxy, *args, **kwargs)

    VllmConfig._tpu_upstream_compute_hash = original
    VllmConfig.compute_hash = compute_hash
    VllmConfig._tpu_additional_config_hash_patch = True
    logger.info(
        "Applied TPU patch: exclude diagnostics-only additional_config keys "
        "%s from the compile cache key.",
        sorted(HASH_IGNORED_ADDITIONAL_CONFIG_KEYS),
    )


def _patch_moe_explicit_pcp_collectives() -> None:
    """Keep PCP dispatch/combine explicit unless MoE chunk pipelining owns it.

    vLLM treats PCP+EP as an all-to-all-kernel configuration and therefore
    skips the explicit PCP all-gather/reduce-scatter in ``MoERunner``.  The
    TorchTPU monolithic MoE kernels only compute local expert contributions;
    they only implement internal PCP dispatch/combine when MoE collective
    chunking is enabled.  Otherwise keep dispatch/combine explicit.

    Chunk pipelining is a process-wide configuration, so it can select the
    contract through ``FusedMoEParallelConfig.use_all2all_kernels``.  The fused
    EP kernel is armed per layer instead; ``VllmMoERunner`` handles that case.
    """
    from vllm.model_executor.layers.fused_moe import FusedMoEParallelConfig

    if not getattr(
        FusedMoEParallelConfig, "_tpu_explicit_pcp_collectives_patch", False
    ):
        upstream_property = FusedMoEParallelConfig.use_all2all_kernels
        upstream_getter = upstream_property.fget
        assert upstream_getter is not None

        def use_all2all_kernels(self):
            if (
                self.use_ep
                and self.pcp_size > 1
                and self.dp_size == 1
                and not self.is_sequence_parallel
                and envs.TPU_MOE_COLLECTION_CHUNK_SIZE <= 0
            ):
                return False
            return upstream_getter(self)

        FusedMoEParallelConfig.use_all2all_kernels = property(
            use_all2all_kernels, doc=upstream_property.__doc__
        )
        FusedMoEParallelConfig._tpu_explicit_pcp_collectives_patch = True

    logger.info(
        "Applied TPU patch: use explicit PCP collectives unless MoE "
        "chunk pipelining owns them."
    )


def _patch_expert_map_host_lookup() -> None:
    """Build the expert map on the host and serve host readers from a copy.

    ``ExpertMapManager`` builds its maps under the model-init device context,
    so every op of ``determine_expert_map`` is a device op, and host readers
    then read the map back:

    - ``map_global_to_local`` reads it with ``.item()`` once per (expert,
      shard) tensor during MoE weight loading (~200k per worker on
      Qwen3.5-397B).
    - ``get_compressed_map_string`` calls ``.item()`` on every element for
      the per-layer EP log line; the ``info_once`` argument is evaluated for
      every MoE layer, and each element index compiled its own slice module
      (128 compiles, ~5.4 min serialized, at EP8).

    On TPU every read is a blocking sync that flushes the deferred graph.
    Build the maps on CPU, keep the host copy, move the tensors to the
    device once, and index the host copy in Python. The cache follows
    reassignment and in-place rebalance via the tensor's version counter.
    """
    import torch
    from vllm.model_executor.layers.fused_moe import expert_map_manager as _em

    cls = _em.ExpertMapManager
    if getattr(cls, "_tpu_host_expert_map_patch", False):
        return

    def _calculate_expert_maps(self) -> None:
        device = torch.get_default_device()
        with torch.device("cpu"):
            local_num_experts, expert_map, expert_mask = _em.determine_expert_map(
                ep_size=self.ep_size,
                ep_rank=self.ep_rank,
                global_num_experts=self.global_num_experts,
                expert_placement_strategy=self._placement_strategy,
                num_fused_shared_experts=self.num_fused_shared_experts,
                return_expert_mask=self.rocm_aiter_enabled,
            )
        self._local_num_experts = local_num_experts + self.num_fused_shared_experts
        self._expert_map = None if expert_map is None else expert_map.to(device)
        self._expert_mask = None if expert_mask is None else expert_mask.to(device)
        # The device map as built; a fresh tensor starts at version 0.
        self._tpu_expert_map_source = self._expert_map
        self._tpu_expert_map_version = (
            None if self._expert_map is None else self._expert_map._version
        )
        self._tpu_expert_map_host = None if expert_map is None else expert_map.tolist()

    def _host_expert_map(self) -> list[int]:
        expert_map = self._expert_map
        if (
            self._tpu_expert_map_source is not expert_map
            or self._tpu_expert_map_version != expert_map._version
        ):
            self._tpu_expert_map_source = expert_map
            self._tpu_expert_map_version = expert_map._version
            self._tpu_expert_map_host = expert_map.tolist()
        return self._tpu_expert_map_host

    def map_global_to_local(self, global_id: int) -> int:
        if self._expert_map is None:
            return global_id
        return _host_expert_map(self)[global_id]

    def get_compressed_map_string(self) -> str:
        if self._expert_map is None:
            return f"[0..{self.global_num_experts - 1}]"
        return ", ".join(
            f"{local_index}->{global_index}"
            for global_index, local_index in enumerate(_host_expert_map(self))
            if local_index != -1
        )

    cls._calculate_expert_maps = _calculate_expert_maps
    cls.map_global_to_local = map_global_to_local
    cls.get_compressed_map_string = get_compressed_map_string
    cls._tpu_expert_map_source = None
    cls._tpu_expert_map_version = None
    cls._tpu_expert_map_host = None
    cls._tpu_host_expert_map_patch = True
    logger.info("Applied TPU patch: host-built expert map with host-side reads.")


def _patch_vllm_disable_compile_ranges() -> None:
    """Force TPU to compile only fixed bucketed sizes, never dynamic ranges.

    ``VllmConfig._set_compile_ranges()`` always appends
    ``scheduler_config.max_num_batched_tokens`` to ``compile_ranges_endpoints``,
    so ``PiecewiseBackend`` builds a dynamic ``Range(1, max)`` and compiles it
    with backed ``SymInt`` shapes. The torch_tpu backend rejects SymInts ("does
    not support dynamic shape" / "No shape env"). The TPU runner pads every shape
    to an exact ``compile_size``, so dynamic ranges are never needed.

    Patch the consumption point: ``CompilationConfig.get_compile_ranges() -> []``.
    Clearing the field (e.g. in the worker) does not stick --
    ``_set_compile_ranges`` rewrites it on every config reconstruction
    (Qwen3-VL ``with_hf_config`` -> ``replace``), and the compile-time
    compilation_config is not the instance a platform/worker hook can reach --
    so a class-level override is the robust fix.
    """
    from vllm.config.compilation import CompilationConfig

    def get_compile_ranges(self):
        return []

    CompilationConfig.get_compile_ranges = get_compile_ranges
    logger.info(
        "Applied TPU patch: disable dynamic compile_ranges (static compile_sizes only)."
    )


def _patch_vllm_compile_all_ranges() -> None:
    """Patch vllm compile all ranges so the ranks of a host split the ranges.

    With TPU_PRECOMPILE_PEER_REPLAY=1, each range is compiled by one rank of
    the host, its owner; every other rank replays the artifact the owner saves
    to vLLM's compile cache instead of tracing and compiling the range again
    (see TpuCompilerAdaptor.compile). Startup then takes about as long as the
    slowest single range instead of the sum over ranges.

    Off, the default, each rank compiles every range itself, starting on a
    different one, and the ranks share only through the Tier-2 cache.
    """
    import os
    from collections import deque

    from vllm.compilation.piecewise_backend import (
        PiecewiseBackend,
        create_concrete_args,
        get_fake_args_from_graph,
    )

    def ranks_per_host() -> int | None:
        """How many ranks share this host, or None if that is unknown."""
        try:
            from vllm.distributed.parallel_state import (
                get_node_count,
                get_world_group,
            )

            world_size = get_world_group().world_size
            nodes = get_node_count()
        except Exception:
            return None
        if nodes <= 0 or world_size % nodes:
            return None
        return world_size // nodes

    def compile_entry(self, range_entry) -> None:
        self._log_compile_start(range_entry.compile_range)

        if range_entry.compile_range.is_single_size():
            args_list = create_concrete_args(
                self.graph, range_entry.compile_range.start
            )
        else:
            args_list = get_fake_args_from_graph(self.graph)

        range_entry.runnable = self.vllm_backend.compiler_manager.compile(
            self.graph,
            args_list,
            self.vllm_backend.inductor_config,
            self.compilation_config,
            compile_range=range_entry.compile_range,
            graph_index=self.piecewise_compile_index,
            num_graphs=self.total_piecewise_compiles,
            is_encoder=self.vllm_backend.is_encoder,
        )

        range_entry.compiled = True

    def compile_rotated(self, local_rank: int) -> None:
        """Compile every range here, each local rank starting on a different
        one: the order without peer replay."""
        range_entry_list = list(self.range_entries.values())
        if envs.TPU_PRECOMPILE_BIGGEST_FIRST:
            # Biggest first: local rank 0 starts on the biggest range, which
            # takes far longer than the others and so bounds startup, instead
            # of reaching it only after the smallest one.
            range_entry_list.sort(key=lambda e: e.compile_range.end, reverse=True)

        range_entry_len = len(range_entry_list)

        expanded_indexes = list(range((range_entry_len + 7) // 8 * 8))
        d = deque(sorted(expanded_indexes, key=lambda x: (x % 8, x)))
        d.rotate(-local_rank * (len(d) // 8))
        indexes = list(d)

        for i in indexes:
            if i >= len(range_entry_list):
                continue

            range_entry = range_entry_list[i]

            if range_entry.compiled:
                continue

            compile_entry(self, range_entry)

    def compile_all_ranges(self) -> None:
        """Compile all range entries for this piecewise subgraph up front."""
        assert self.graph is not None, (
            "Cannot compile without a graph. "
            "When loading from cache/AOT artifacts, "
            "compile_all_ranges should not be called."
        )
        local_rank = int(os.environ["LOCAL_RANK"])
        if not envs.TPU_PRECOMPILE_PEER_REPLAY:
            compile_rotated(self, local_rank)
            return

        from vllm_torchtpu.compilation.tpu_compiler import (
            PEER_WAIT_S,
            wait_for_peer_artifact,
        )

        ranks = ranks_per_host()
        # Unknown: assume a full host, but do not wait on owners that may not
        # exist; a range nobody saved yet is then compiled here.
        wait_s = PEER_WAIT_S if ranks is not None and local_rank < ranks else 0.0
        ranks = ranks or 8

        # Biggest first, dealt out in a snake: local rank 0 starts on the
        # biggest range, which takes far longer than the others and so bounds
        # startup, and a rank whose first range is small gets the next one.
        # TPU_PRECOMPILE_BIGGEST_FIRST=0 deals them in vLLM's order instead.
        entries = [e for e in self.range_entries.values() if not e.compiled]
        if envs.TPU_PRECOMPILE_BIGGEST_FIRST:
            entries.sort(key=lambda e: e.compile_range.end, reverse=True)
        mine, others = [], []
        for i, range_entry in enumerate(entries):
            turn, slot = divmod(i, ranks)
            owner = slot if turn % 2 == 0 else ranks - 1 - slot
            if owner == local_rank:
                mine.append(range_entry)
            else:
                # In the order the owners finish them: by turn, small first.
                others.append((turn, range_entry.compile_range.end, range_entry))

        for range_entry in mine:
            compile_entry(self, range_entry)
        with wait_for_peer_artifact(wait_s):
            for _, _, range_entry in sorted(others, key=lambda o: o[:2]):
                compile_entry(self, range_entry)

    if envs.TPU_PARALLEL_PRECOMPILE:
        PiecewiseBackend.compile_all_ranges = compile_all_ranges
        logger.info("Applied TPU patch: compile_all_ranges.")


def _patch_sequence_parallel_moe_admission() -> None:
    """Gate vLLM's sequence-parallel MoE on TPU.

    Off by default. With TPU_MOE_SEQUENCE_PARALLEL=1, admit it at DP1
    when EP is on and TP > 1. The GPU all2all-backend and DP > 1 conjuncts do
    not apply on TPU. EP is required: without it the row-parallel expert sum
    is never reduced.
    """
    from vllm.config.parallel import ParallelConfig

    if getattr(ParallelConfig, "_tpu_no_sp_moe_patch", False):
        return

    if envs.TPU_MOE_SEQUENCE_PARALLEL:
        ParallelConfig.use_sequence_parallel_moe = property(
            lambda self: self.enable_expert_parallel
            and self.tensor_parallel_size > 1
            and self.data_parallel_size == 1
        )
    else:
        ParallelConfig.use_sequence_parallel_moe = property(lambda self: False)
    ParallelConfig._tpu_no_sp_moe_patch = True
    logger.info(
        "Applied TPU patch: sequence-parallel MoE %s.",
        "admitted" if envs.TPU_MOE_SEQUENCE_PARALLEL else "disabled",
    )


def _patch_multiproc_worker_global_rank_env() -> None:
    """Set TorchTPU binding env on spawned workers before torch_tpu import.

    Native vLLM DP passes per-engine rank/local_rank to workers and computes
    global torch.distributed rank later from ParallelConfig. TorchTPU currently
    reads RANK/LOCAL_RANK/WORLD_SIZE earlier for physical chip/PjRt binding, so
    this shim exposes the unified single-host DP*TP slice to TorchTPU without
    changing the worker rank arguments passed through vLLM.
    """
    import os as _os

    from vllm.v1.executor.multiproc_executor import WorkerProc

    from vllm_torchtpu.worker.tpu_rank_binding import get_tpu_worker_binding

    if getattr(WorkerProc, "_tpu_global_rank_env_patch", False):
        return
    _orig = WorkerProc.make_worker_process

    def _wrapped(vllm_config, local_rank, rank, *args, **kwargs):
        pc = vllm_config.parallel_config
        binding = get_tpu_worker_binding(pc, rank, local_rank, env=_os.environ)
        _os.environ.update(binding.as_env())
        # VLLM_TPU_DEVICE_ORDER lists a physical device per local rank, so
        # consecutive ranks (a two-device pipeline stage) can sit on different
        # chips; TorchTPU reads TPU_VISIBLE_DEVICES at import, hence set here.
        order = envs.VLLM_TPU_DEVICE_ORDER
        if order:
            device = [int(x) for x in order.split(",")][binding.local_rank]
            _os.environ["TPU_VISIBLE_CHIPS"] = str(device)
            _os.environ["TPU_VISIBLE_DEVICES"] = str(device)
            logger.info(
                "Applied TPU patch: worker spawn env local rank %d -> device %d "
                "(VLLM_TPU_DEVICE_ORDER=%s)",
                binding.local_rank,
                device,
                order,
            )
        logger.info(
            "Applied TPU patch: worker spawn env RANK=%d LOCAL_RANK=%d "
            "WORLD_SIZE=%d LOCAL_WORLD_SIZE=%d "
            "(rank=%d local_rank=%d dp_rank=%d dp_size=%d offset=%d "
            "init_local_rank=%d)",
            binding.rank,
            binding.local_rank,
            binding.world_size,
            binding.local_world_size,
            rank,
            local_rank,
            binding.dp_rank,
            binding.dp_size,
            binding.local_rank_offset,
            binding.init_local_rank,
        )
        return _orig(vllm_config, local_rank, rank, *args, **kwargs)

    WorkerProc.make_worker_process = staticmethod(_wrapped)
    WorkerProc._tpu_global_rank_env_patch = True
    logger.info("Applied TPU patch: MultiprocExecutor worker binding env.")


def _run_engine_core_with_tpu_patches(*args, **kwargs):
    from vllm_torchtpu import patch_registry

    patch_registry.apply("engine_core")
    from vllm_torchtpu.core.pp_chunk_scheduler import patch_engine_core_for_pp_chunks
    from vllm_torchtpu.core.pp_lazy_step import patch_engine_core_for_lazy_push
    from vllm_torchtpu.distributed.pp_push import patch_executor_for_pp_wave

    patch_executor_for_pp_wave(kwargs.get("vllm_config"))
    patch_engine_core_for_pp_chunks(kwargs.get("vllm_config"))
    patch_engine_core_for_lazy_push(kwargs.get("vllm_config"))

    from vllm.v1.engine.core import EngineCoreProc

    original_run = EngineCoreProc._tpu_original_run_engine_core
    return original_run(*args, **kwargs)


def _patch_vllm_offloading_config_build() -> None:
    """Adapt vLLM's OffloadingConfig construction to TPU PCP.

    ``build_offloading_config`` scales each group's ``tokens_per_block`` by
    DCP only (it records ``pcp_size`` in the parallel config but never
    folds PCP into block spans). TorchTPU PCP scheduler blocks
    span all PCP ranks (see ``_patch_vllm_hybrid_pcp_block_sizes``), so
    ``tokens_per_block`` must scale by PCP as well and ``tokens_per_hash``
    must come from the runtime (PCP-patched) resolve; otherwise the
    offloading manager would track sub-scheduler-block chunks, and on PCP
    hybrid configs ``build_offloading_config`` trips its own
    ``tokens_per_block % tokens_per_hash`` assert (the PCP-patched resolve
    returns logical hash sizes while the group sizes stay rank-local).
    """
    import sys
    from dataclasses import replace

    from vllm.distributed.kv_transfer.kv_connector.v1.offloading import (
        config as offloading_config_module,
    )

    if getattr(offloading_config_module, "_tpu_offloading_config_patch", False):
        return

    original_build = offloading_config_module.build_offloading_config

    def build_offloading_config_tpu(vllm_config, kv_cache_config):
        pcp = vllm_config.parallel_config.prefill_context_parallel_size
        if pcp <= 1:
            return original_build(vllm_config, kv_cache_config)

        # Run the original against rank-local block sizes (the
        # unpatched resolve, so its tokens_per_block % tokens_per_hash
        # assert compares rank-local to rank-local), then scale each
        # group's tokens_per_block to the logical all-PCP-rank span.
        from vllm.v1.core import kv_cache_utils

        unpatched_resolve = getattr(
            kv_cache_utils,
            "_tpu_original_resolve_kv_cache_block_sizes",
            kv_cache_utils.resolve_kv_cache_block_sizes,
        )
        saved_resolve = offloading_config_module.resolve_kv_cache_block_sizes
        offloading_config_module.resolve_kv_cache_block_sizes = unpatched_resolve
        try:
            config = original_build(vllm_config, kv_cache_config)
        finally:
            offloading_config_module.resolve_kv_cache_block_sizes = saved_resolve
        # tokens_per_hash must match the granularity the scheduler
        # actually hashes Request.block_hashes at -- the runtime
        # (PCP-patched) resolve: logical under hybrid PCP, rank-local
        # otherwise.
        _, tokens_per_hash = kv_cache_utils.resolve_kv_cache_block_sizes(
            kv_cache_config, vllm_config
        )
        config = replace(
            config,
            groups=tuple(
                replace(group, tokens_per_block=group.tokens_per_block * pcp)
                for group in config.groups
            ),
            cache=replace(config.cache, tokens_per_hash=tokens_per_hash),
        )
        for group in config.groups:
            assert group.tokens_per_block % tokens_per_hash == 0, (
                f"tokens_per_block={group.tokens_per_block} not "
                f"divisible by tokens_per_hash={tokens_per_hash} after "
                f"PCP scaling (pcp={pcp})"
            )
        return config

    offloading_config_module.build_offloading_config = build_offloading_config_tpu
    offloading_config_module._tpu_offloading_config_patch = True

    # OffloadingConnector imports the function directly; rebind if the
    # module is already loaded.
    connector_module = sys.modules.get(
        "vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector"
    )
    if connector_module is not None:
        connector_module.build_offloading_config = build_offloading_config_tpu

    logger.info("Applied TPU patch: PCP-aware OffloadingConfig build.")


def _patch_vllm_kimi_kda_layer_counts() -> None:
    """Patch ModelConfig.get_num_layers_by_block_type to report accurate counts
    for Kimi models with KDA using linear_attn_config (e.g. Kimi-K3).
    """
    from typing import Literal

    from vllm.config.model import ModelConfig

    if getattr(ModelConfig, "_tpu_kimi_kda_layer_counts_patched", False):
        return

    original_get_num_layers = ModelConfig.get_num_layers_by_block_type

    def patched_get_num_layers_by_block_type(
        self,
        parallel_config,
        block_type: Literal["attention", "linear_attention", "mamba"] = "attention",
    ) -> int:
        try:
            return original_get_num_layers(self, parallel_config, block_type)
        except ValueError:
            pass

        start, end = self.get_layers_start_end_indices(parallel_config)
        linear_attn_config = getattr(self.hf_text_config, "linear_attn_config", None)
        if linear_attn_config is not None and block_type == "attention":
            kda_layers = set(linear_attn_config.get("kda_layers", []))
            return sum((idx + 1) not in kda_layers for idx in range(start, end))

        return original_get_num_layers(self, parallel_config, block_type)

    ModelConfig.get_num_layers_by_block_type = patched_get_num_layers_by_block_type
    ModelConfig._tpu_kimi_kda_layer_counts_patched = True
    logger.info(
        "Applied TPU patch: accurate layer counts for Kimi-K3 / KDA hybrid models."
    )


def _patch_vllm_hybrid_pcp_block_sizes() -> None:
    """Resolve full-attention + Mamba block sizes under TPU PCP.

    Upstream vLLM rejects all multi-group KV cache configs when
    ``pcp_world_size > 1`` because it cannot infer the scheduler block size for
    mixed cache types. TorchTPU's Qwen3.5 path has a narrower, well-defined
    layout: each physical cache page is rank-local, while the scheduler and
    cache coordinator operate on logical pages spanning all PCP ranks. Present
    those logical page sizes to the generic hybrid coordinator instead of
    rejecting the config.
    """
    import math
    import sys
    from dataclasses import replace

    from vllm.utils.math_utils import cdiv
    from vllm.v1.core import kv_cache_coordinator, kv_cache_utils
    from vllm.v1.kv_cache_interface import (
        AttentionSpec,
        FullAttentionSpec,
        KVCacheSpec,
        MambaSpec,
        UniformTypeKVCacheSpecs,
    )

    def full_attention_max_memory_usage_bytes(self, vllm_config):
        # Specs describe physical, rank-local pages. Capacity admission and
        # auto-fit run before the coordinator converts them to logical PCP
        # pages, so account for context sharding here as well as for DCP.
        parallel = vllm_config.parallel_config
        cp_size = max(
            parallel.prefill_context_parallel_size,
            parallel.decode_context_parallel_size,
        )
        num_pages = cdiv(
            vllm_config.model_config.max_model_len, self.block_size * cp_size
        )
        return num_pages * self.page_size_bytes

    FullAttentionSpec.max_memory_usage_bytes = full_attention_max_memory_usage_bytes

    already_patched = getattr(
        kv_cache_utils, "_tpu_hybrid_pcp_block_sizes_patch", False
    )
    original_resolve = getattr(
        kv_cache_utils,
        "_tpu_original_resolve_kv_cache_block_sizes",
        kv_cache_utils.resolve_kv_cache_block_sizes,
    )

    def _iter_leaf_specs(spec: KVCacheSpec):
        if isinstance(spec, UniformTypeKVCacheSpecs):
            yield from spec.kv_cache_specs.values()
        else:
            yield spec

    def _has_spec_type(spec: KVCacheSpec, spec_type: type) -> bool:
        return any(isinstance(leaf, spec_type) for leaf in _iter_leaf_specs(spec))

    def _is_attention_mamba_hybrid(groups) -> bool:
        return any(
            _has_spec_type(group.kv_cache_spec, MambaSpec) for group in groups
        ) and any(
            _has_spec_type(group.kv_cache_spec, AttentionSpec) for group in groups
        )

    def _with_effective_block_size(
        spec: KVCacheSpec, pcp: int, dcp: int = 1
    ) -> KVCacheSpec:
        """Restate a rank-local spec in the logical (all-ranks) page size."""
        assert not (pcp > 1 and dcp > 1), (
            f"Concurrent PCP and DCP are not supported: got pcp={pcp}, dcp={dcp}."
        )
        if isinstance(spec, UniformTypeKVCacheSpecs):
            effective_specs = {
                name: _with_effective_block_size(leaf, pcp, dcp)
                for name, leaf in spec.kv_cache_specs.items()
            }
            effective_block_size = math.lcm(
                *(leaf.block_size for leaf in effective_specs.values())
            )
            return replace(
                spec, block_size=effective_block_size, kv_cache_specs=effective_specs
            )
        if isinstance(spec, AttentionSpec):
            # Attention KV cache is partitioned along the 1D Context Parallel (CP)
            # axis during prefill (PCP) or decode (DCP). Because PCP and DCP share
            # the same physical axis (rather than forming a 2D grid), the effective
            # sharding factor is max(pcp, dcp).
            cp_world_size = max(pcp, dcp)
            return replace(spec, block_size=spec.block_size * cp_world_size)
        if isinstance(spec, MambaSpec):
            # Mamba/GDN recurrent state has a fixed per-sequence footprint that
            # is partitioned across ranks during prefill (PCP), but maintains
            # full per-rank state during decode and is not sharded by DCP.
            return replace(spec, block_size=spec.block_size * pcp)
        return spec

    def _effective_block_size(spec: KVCacheSpec, pcp: int, dcp: int = 1) -> int:
        return _with_effective_block_size(spec, pcp, dcp).block_size

    def resolve_kv_cache_block_sizes(kv_cache_config, vllm_config):
        cache_config = vllm_config.cache_config
        parallel_config = vllm_config.parallel_config
        dcp = parallel_config.decode_context_parallel_size
        pcp = parallel_config.prefill_context_parallel_size
        groups = kv_cache_config.kv_cache_groups

        if len(groups) <= 1 or dcp != 1 or pcp == 1:
            return original_resolve(kv_cache_config, vllm_config)

        if not _is_attention_mamba_hybrid(groups):
            return original_resolve(kv_cache_config, vllm_config)

        group_block_sizes = [
            _effective_block_size(group.kv_cache_spec, pcp) for group in groups
        ]
        scheduler_block_size = math.lcm(*group_block_sizes)

        connector_enabled = vllm_config.kv_transfer_config is not None
        if not (cache_config.enable_prefix_caching or connector_enabled):
            hash_block_size = scheduler_block_size
        elif cache_config.prefix_match_unit is None:
            hash_block_size = math.gcd(*group_block_sizes)
        else:
            hash_block_size = cache_config.prefix_match_unit
            if any(bs % hash_block_size != 0 for bs in group_block_sizes):
                raise ValueError(
                    f"Invalid prefix_match_unit={hash_block_size}; all logical "
                    "PCP KV cache group block sizes must be divisible by "
                    "prefix_match_unit. Got "
                    f"block sizes={group_block_sizes}."
                )

        logger.info(
            "Applied TPU PCP hybrid KV cache block-size resolution: "
            "pcp=%d group_effective_block_sizes=%s scheduler_block_size=%d "
            "hash_block_size=%d",
            pcp,
            group_block_sizes,
            scheduler_block_size,
            hash_block_size,
        )
        return scheduler_block_size, hash_block_size

    patched_resolve = (
        kv_cache_utils.resolve_kv_cache_block_sizes
        if already_patched
        else resolve_kv_cache_block_sizes
    )
    if not already_patched:
        kv_cache_utils._tpu_original_resolve_kv_cache_block_sizes = original_resolve
        kv_cache_utils.resolve_kv_cache_block_sizes = patched_resolve
        kv_cache_utils._tpu_hybrid_pcp_block_sizes_patch = True

    coordinator_factory = kv_cache_coordinator.get_kv_cache_coordinator
    if not getattr(coordinator_factory, "_tpu_hybrid_pcp_coordinator_patch", False):
        original_factory = coordinator_factory

        def _get_coordinator_with_logical_pcp_blocks(
            kv_cache_config,
            max_model_len,
            max_in_flight_tokens,
            use_eagle,
            enable_caching,
            enable_kv_cache_events,
            dcp_world_size,
            pcp_world_size,
            scheduler_block_size,
            hash_block_size,
            metrics_collector=None,
            num_prefill_lookahead=0,
            allow_partial_hash_hits: bool = True,
            **kwargs,
        ):
            groups = kv_cache_config.kv_cache_groups
            logical_pcp_world_size = pcp_world_size
            if pcp_world_size == 1 and _is_attention_mamba_hybrid(groups):
                # vLLM 0.26's Scheduler already folds PCP into
                # scheduler_block_size, but hard-codes pcp_world_size=1 when
                # constructing KVCacheManager. Recover the folded PCP factor
                # from the logical scheduler block and the rank-local specs.
                physical_scheduler_block_size = math.lcm(
                    *(
                        _effective_block_size(group.kv_cache_spec, 1, dcp_world_size)
                        for group in groups
                    )
                )
                if scheduler_block_size % physical_scheduler_block_size != 0:
                    raise ValueError(
                        "Logical scheduler block size must be divisible by "
                        "the physical hybrid KV cache block size. Got "
                        f"scheduler_block_size={scheduler_block_size}, "
                        "physical_scheduler_block_size="
                        f"{physical_scheduler_block_size}."
                    )
                logical_pcp_world_size = (
                    scheduler_block_size // physical_scheduler_block_size
                )
                if logical_pcp_world_size > 1:
                    logger.info(
                        "Inferred PCP world size %d from logical scheduler "
                        "block size %d and physical block size %d.",
                        logical_pcp_world_size,
                        scheduler_block_size,
                        physical_scheduler_block_size,
                    )

            if logical_pcp_world_size > 1 and _is_attention_mamba_hybrid(groups):
                if dcp_world_size != 1:
                    raise ValueError(
                        "TPU hybrid PCP cache coordination does not support DCP."
                    )
                # The generic coordinator only needs logical token
                # granularity. Encoding PCP into each group spec and then
                # passing pcp_world_size=1 avoids multiplying block sizes
                # twice and keeps its prefix-cache lookup, hashing, and
                # allocation invariants internally consistent.
                kv_cache_config = replace(
                    kv_cache_config,
                    kv_cache_groups=[
                        replace(
                            group,
                            kv_cache_spec=_with_effective_block_size(
                                group.kv_cache_spec, logical_pcp_world_size
                            ),
                        )
                        for group in kv_cache_config.kv_cache_groups
                    ],
                )
                pcp_world_size = 1

            return original_factory(
                kv_cache_config,
                max_model_len,
                max_in_flight_tokens,
                use_eagle,
                enable_caching,
                enable_kv_cache_events,
                dcp_world_size,
                pcp_world_size,
                scheduler_block_size,
                hash_block_size,
                metrics_collector,
                num_prefill_lookahead=num_prefill_lookahead,
                allow_partial_hash_hits=allow_partial_hash_hits,
                **kwargs,
            )

        _get_coordinator_with_logical_pcp_blocks._tpu_hybrid_pcp_coordinator_patch = (
            True
        )
        kv_cache_coordinator.get_kv_cache_coordinator = (
            _get_coordinator_with_logical_pcp_blocks
        )

    # Normalize before the factory selects a prefix-cached or no-prefix
    # coordinator. Patching only HybridKVCacheCoordinator misses the latter,
    # so runtime admission would count unsharded pages and stall long prompts.
    from vllm.v1.core import kv_cache_manager

    kv_cache_manager.get_kv_cache_coordinator = (
        kv_cache_coordinator.get_kv_cache_coordinator
    )

    from vllm.v1.engine.core import EngineCoreProc

    # EngineCore imports the function directly, so patch module bindings as
    # well as the source module.
    for module_name in ("vllm.v1.engine.core", "vllm.v1.kv_offload.base"):
        module = sys.modules.get(module_name)
        if module is not None:
            module.resolve_kv_cache_block_sizes = patched_resolve

    # Upstream _promote_local_kv_cache_specs asserts that all workers share a
    # uniform KV cache spec across groups, raising ValueError when encountering
    # heterogeneous layer configurations (e.g. hybrid attention/Mamba models,
    # pure linear state layers, or disaggregated PCP configurations with
    # per-group block sizing). We patch it to catch ValueError and return an
    # empty dict so initialization falls back cleanly to the TPU hybrid/PCP
    # cache coordinator without crashing.
    original_promote = getattr(kv_cache_utils, "_promote_local_kv_cache_specs", None)
    if original_promote is not None and not getattr(
        original_promote, "_tpu_hybrid_promote_patch", False
    ):

        def patched_promote(specs):
            try:
                return original_promote(specs)
            except ValueError:
                return {}

        patched_promote._tpu_hybrid_promote_patch = True
        kv_cache_utils._promote_local_kv_cache_specs = patched_promote
        for module_name in ("vllm.v1.engine.core", "vllm.v1.core.kv_cache_utils"):
            mod = sys.modules.get(module_name)
            if mod is not None and hasattr(mod, "_promote_local_kv_cache_specs"):
                mod._promote_local_kv_cache_specs = patched_promote

    if not getattr(EngineCoreProc, "_tpu_engine_core_patch_wrapper", False):
        EngineCoreProc._tpu_original_run_engine_core = EngineCoreProc.run_engine_core
        EngineCoreProc.run_engine_core = staticmethod(_run_engine_core_with_tpu_patches)
        EngineCoreProc._tpu_engine_core_patch_wrapper = True

    if not already_patched:
        logger.info("Applied TPU patch: hybrid full-attention + Mamba PCP block sizes.")


# CPU KV offloading connectors (vLLM's and the Raiden store-backed one). They
# hash at the scheduler's hash unit and chunk at the logical block.
_OFFLOADING_CONNECTORS = frozenset(
    {"OffloadingConnector", "TPURaidenOffloadingConnector"}
)


def _is_offloading_connector(kv_transfer_config) -> bool:
    return getattr(kv_transfer_config, "kv_connector", None) in _OFFLOADING_CONNECTORS


def _patch_vllm_offload_fine_prefix_alignment() -> None:
    """Load offloaded chunks only from a chunk-aligned GPU prefix hit.

    With DCP fine prefix hits a GPU hit can end inside a logical block. The
    manager then copy-on-writes that shared block into a fresh one, and the
    copy is applied by the runner at the next forward. An offload load of the
    same chunk would DMA into that fresh block concurrently, and whichever
    lands last wins -- the copy carries the other sequence's tail past the hit
    point. A hit that ends mid-block means the prompt diverged inside that
    block (eviction frees whole blocks, tail first), so the offload tier holds
    nothing past it either; skip the lookup there instead of racing.
    """
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
        OffloadingConnectorScheduler,
    )

    if getattr(OffloadingConnectorScheduler, "_tpu_fine_prefix_alignment", False):
        return
    original_lookup = OffloadingConnectorScheduler._lookup

    def _lookup(self, req_status, *args, **kwargs):
        local_tokens = req_status.num_locally_computed_tokens
        for group_config in self.config.kv_group_configs:
            if (
                group_config.hashes_per_chunk > 1
                and local_tokens % group_config.tokens_per_chunk
            ):
                return 0
        return original_lookup(self, req_status, *args, **kwargs)

    OffloadingConnectorScheduler._lookup = _lookup
    OffloadingConnectorScheduler._tpu_fine_prefix_alignment = True
    logger.info(
        "Applied TPU patch: chunk-aligned offload loads under fine prefix hits."
    )


def _dcp_fine_prefix_match_unit(kv_cache_config, vllm_config) -> int | None:
    """Return the fine prefix-hit unit for a single-group DCP model, or None.

    Opt-in: only a single full-attention (incl. MLA) group under DCP with
    prefix caching, no PCP, no KV connector other than a CPU offloading one,
    and an explicit ``--prefix-match-unit`` smaller than the logical block
    (block_size * dcp). Everything else keeps upstream behavior (hash ==
    scheduler block).

    Offloading connectors chunk at the logical block and key each chunk by its
    last fine hash, so they work with a finer hash unit; see
    ``_patch_vllm_offload_fine_prefix_alignment`` for the one load case that
    needs care.
    """
    from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
    from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs
    from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

    cache_config = vllm_config.cache_config
    parallel_config = vllm_config.parallel_config
    unit = cache_config.prefix_match_unit
    dcp = parallel_config.decode_context_parallel_size
    groups = kv_cache_config.kv_cache_groups
    if (
        unit is None
        or not cache_config.enable_prefix_caching
        or dcp <= 1
        or parallel_config.prefill_context_parallel_size != 1
        # P/D connectors consume block hashes at block granularity.
        or (
            vllm_config.kv_transfer_config is not None
            and not _is_offloading_connector(vllm_config.kv_transfer_config)
        )
        or len(groups) != 1
    ):
        return None
    spec = groups[0].kv_cache_spec
    leaves = (
        spec.kv_cache_specs.values()
        if isinstance(spec, UniformTypeKVCacheSpecs)
        else (spec,)
    )
    # Only layers managed by FullAttentionManager (full attention and MLA,
    # incl. the DSA indexer cache), which implements fine-grained lookup and
    # partial-hit CoW. SWA / R-SWA / chunked-local / mamba are excluded.
    if not all(
        KVCacheSpecRegistry.get_manager_class(leaf) is FullAttentionManager
        for leaf in leaves
    ):
        return None
    logical_block = cache_config.block_size * dcp
    if unit >= logical_block:
        return None
    if logical_block % unit != 0:
        raise ValueError(
            f"Invalid prefix_match_unit={unit}; the DCP logical block size "
            f"block_size*dcp={logical_block} must be divisible by it."
        )
    return unit


def _patch_vllm_dcp_fine_prefix_hits() -> None:
    """Prefix-cache hits finer than the DCP logical block (single group).

    Under DCP a single-group model's manager block is ``block_size * dcp``
    tokens (e.g. 256 * 16 = 4096), and upstream hashes at that granularity,
    so a shared prefix only hits in 4096-token steps. With
    ``--prefix-match-unit U`` (U < block_size * dcp) this patch:

    1. resolves ``(scheduler_block_size, hash_block_size)`` to
       ``(block_size * dcp, U)`` (request hashing then runs every U tokens);
    2. lets ``UnitaryKVCacheCoordinator`` accept ``hash < block`` and look up
       hits at U alignment. ``FullAttentionManager`` already implements the
       fine-grained probe, the prompt-tail partial entry, the one-hash-unit
       EAGLE/MTP drop, and copy-on-write of the shared tail block; the
       scheduler forwards those CoW pairs as ``kv_cache_block_copies``,
       which the TPU runner applies per layer (see
       ``TPUModelRunner._apply_dense_kv_cache_block_copies``).

    The physical block size and allocation are unchanged.
    """
    import functools
    import inspect
    import sys

    from vllm.v1.core import kv_cache_coordinator, kv_cache_utils

    # (1) Block-size resolution. Wrap whatever is installed (the hybrid PCP
    # wrapper runs first in the registry) and republish the direct-name
    # imports, as EngineCore imports the function by name.
    resolve = kv_cache_utils.resolve_kv_cache_block_sizes
    if not getattr(resolve, "_tpu_dcp_fine_prefix_patch", False):
        previous_resolve = resolve

        @functools.wraps(previous_resolve)
        def resolve_kv_cache_block_sizes(kv_cache_config, vllm_config):
            unit = _dcp_fine_prefix_match_unit(kv_cache_config, vllm_config)
            if unit is None:
                if (
                    vllm_config.cache_config.prefix_match_unit is not None
                    and vllm_config.parallel_config.decode_context_parallel_size > 1
                    and len(kv_cache_config.kv_cache_groups) == 1
                ):
                    logger.warning(
                        "prefix_match_unit=%s is not applied under DCP for this "
                        "config (needs prefix caching, no PCP, no KV connector, "
                        "a single full-attention/MLA group, and a unit smaller "
                        "than block_size*dcp); prefix hits stay block-aligned.",
                        vllm_config.cache_config.prefix_match_unit,
                    )
                    orig_unit = vllm_config.cache_config.prefix_match_unit
                    try:
                        vllm_config.cache_config.prefix_match_unit = None
                        return previous_resolve(kv_cache_config, vllm_config)
                    finally:
                        vllm_config.cache_config.prefix_match_unit = orig_unit
                return previous_resolve(kv_cache_config, vllm_config)
            dcp = vllm_config.parallel_config.decode_context_parallel_size
            scheduler_block_size = vllm_config.cache_config.block_size * dcp
            logger.info(
                "DCP fine-grained prefix hits: scheduler_block_size=%d "
                "hash_block_size=%d (dcp=%d)",
                scheduler_block_size,
                unit,
                dcp,
            )
            return scheduler_block_size, unit

        resolve_kv_cache_block_sizes._tpu_dcp_fine_prefix_patch = True
        kv_cache_utils.resolve_kv_cache_block_sizes = resolve_kv_cache_block_sizes
        resolve = resolve_kv_cache_block_sizes
    # The offloading config builder imports it by name too; its tokens_per_hash
    # must match the scheduler's hash unit or offload keys land on the wrong
    # hash boundaries.
    for module_name in (
        "vllm.v1.engine.core",
        "vllm.v1.kv_offload.base",
        "vllm.distributed.kv_transfer.kv_connector.v1.offloading.config",
    ):
        module = sys.modules.get(module_name)
        if module is not None and hasattr(module, "resolve_kv_cache_block_sizes"):
            module.resolve_kv_cache_block_sizes = resolve

    # (2) Coordinator. Patch the class (not the factory, which the hybrid PCP
    # patch owns) so a hash finer than the manager block is accepted only in
    # the DCP single-group case; otherwise the upstream __init__ runs as-is.
    cls = kv_cache_coordinator.UnitaryKVCacheCoordinator
    if getattr(cls, "_tpu_dcp_fine_prefix_patch", False):
        return
    original_init = cls.__init__
    original_find = cls.find_longest_cache_hit
    base_init = kv_cache_coordinator.KVCacheCoordinator.__init__
    # Bind by name instead of restating upstream's parameter list, so a new
    # upstream parameter is forwarded rather than breaking this wrapper.
    init_signature = inspect.signature(original_init)

    @functools.wraps(original_init)
    def __init__(self, *args, **kwargs):
        bound = init_signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        init_args = dict(bound.arguments)
        del init_args["self"]
        kv_cache_config = init_args["kv_cache_config"]
        hash_block_size = init_args["hash_block_size"]
        num_prefill_lookahead = init_args.get("num_prefill_lookahead", 0)
        fine = (
            init_args["enable_caching"]
            and init_args["dcp_world_size"] > 1
            and init_args["pcp_world_size"] == 1
            and len(kv_cache_config.kv_cache_groups) == 1
            and hash_block_size < init_args["scheduler_block_size"]
        )
        if not fine:
            original_init(self, *args, **kwargs)
            return
        # Upstream's __init__ forwards all of its arguments to the base
        # __init__, then sets the attributes below. Mirror that minus its
        # hash_block_size == block_size assert.
        base_init(self, **init_args)
        manager = self.single_type_managers[0]
        self.kv_cache_spec = kv_cache_config.kv_cache_groups[0].kv_cache_spec
        self.dcp_world_size = manager.dcp_world_size
        self.pcp_world_size = init_args["pcp_world_size"]
        self.block_size = manager.block_size
        self.hash_block_size = hash_block_size
        manager.use_eagle = 0 in self.eagle_group_ids
        if not type(manager).supports_fine_grained_hash_lookup:
            raise ValueError(
                f"{type(manager).__name__} does not support prefix hits finer "
                "than its block; unset --prefix-match-unit."
            )
        if self.block_size % hash_block_size != 0:
            raise ValueError(
                f"prefix_match_unit={hash_block_size} must divide the DCP "
                f"manager block size {self.block_size}."
            )
        # EAGLE/MTP drops one hash unit (not a whole block) from each hit; it
        # must still cover the multi-module prefill lookahead.
        if self.eagle_group_ids and hash_block_size < num_prefill_lookahead:
            raise ValueError(
                f"prefix_match_unit={hash_block_size} must be >= "
                f"num_speculative_tokens={num_prefill_lookahead}."
            )
        self.enable_partial_hash_hits = True
        # Lets cache_blocks' reachability mask and the partial-tail entry use
        # hash-unit boundaries.
        manager.cache_hit_alignment_tokens = hash_block_size

    @functools.wraps(original_find)
    def find_longest_cache_hit(self, block_hashes, max_cache_hit_length):
        if not self.enable_partial_hash_hits:
            return original_find(self, block_hashes, max_cache_hit_length)
        # Same as upstream but aligned to the hash unit, which switches
        # FullAttentionManager to its fine-grained probe; a hit ending inside
        # a block is then CoW-copied by the manager on allocation.
        hit_blocks, hit_length = self.single_type_managers[0].find_longest_cache_hit(
            block_hashes=block_hashes,
            max_length=max_cache_hit_length,
            kv_cache_group_ids=[0],
            block_pool=self.block_pool,
            kv_cache_spec=self.kv_cache_spec,
            drop_eagle_block=0 in self.eagle_group_ids,
            alignment_tokens=self.hash_block_size,
            dcp_world_size=self.dcp_world_size,
            pcp_world_size=self.pcp_world_size,
        )
        return hit_blocks, hit_length, 0

    cls.__init__ = __init__
    cls.find_longest_cache_hit = find_longest_cache_hit
    cls._tpu_dcp_fine_prefix_patch = True
    logger.info("Applied TPU patch: DCP single-group fine-grained prefix hits.")


def _patch_vllm_merge_multimodal_embeddings() -> None:
    """Patch vLLM's _merge_multimodal_embeddings to use static indexing on TPU.

    PyTorch boolean indexing (inputs[mask] = values) internally lowers to
    `nonzero()` + `index_put_`, which creates dynamic coordinate shapes
    parameterized by the number of True entries. On TPU, this causes
    `tt_jit_index_put__nonzero` to recompile on every request with a different
    image token count (~15s compilation each). We replace it with static prefix-sum
    indexing and `torch.where`, which avoids `nonzero` and eliminates recompilation.
    """
    import sys

    import torch
    import vllm.model_executor.models.utils as vllm_utils
    from vllm.multimodal import NestedTensors

    if not getattr(vllm_utils, "_tpu_static_merge_mm_patch", False):
        orig_merge = vllm_utils._merge_multimodal_embeddings

        def static_merge_multimodal_embeddings(
            inputs_embeds: torch.Tensor,
            multimodal_embeddings: NestedTensors,
            is_multimodal: torch.Tensor,
        ) -> torch.Tensor:
            if len(multimodal_embeddings) == 0:
                return inputs_embeds

            mm_embeds_flat = vllm_utils._flatten_embeddings(multimodal_embeddings)
            if mm_embeds_flat.numel() == 0:
                return inputs_embeds

            mm_embeds_flat = mm_embeds_flat.to(
                dtype=inputs_embeds.dtype, device=inputs_embeds.device
            )

            idx = torch.cumsum(is_multimodal, dim=0, dtype=torch.int32) - 1
            idx = torch.clamp(idx, 0, mm_embeds_flat.shape[0] - 1)
            gathered = mm_embeds_flat[idx]
            condition = is_multimodal.unsqueeze(-1)
            return torch.where(condition, gathered, inputs_embeds)

        vllm_utils._original_merge_multimodal_embeddings = orig_merge
        vllm_utils._merge_multimodal_embeddings = static_merge_multimodal_embeddings
        vllm_utils._tpu_static_merge_mm_patch = True
        logger.info("Applied TPU patch: static _merge_multimodal_embeddings")

    # Re-run the sweep across imported model modules on every invocation so that
    # models imported after the first call also have their bindings updated.
    patched_fn = vllm_utils._merge_multimodal_embeddings
    for mod_name, mod in list(sys.modules.items()):
        if (
            mod_name.startswith("vllm.model_executor.models")
            and hasattr(mod, "_merge_multimodal_embeddings")
            and mod._merge_multimodal_embeddings is not patched_fn
        ):
            mod._merge_multimodal_embeddings = patched_fn


def _patch_vllm_reset_compile_wrapper() -> None:
    """Reset the compile wrapper on every decoder layer, not just the model.

    `support_torch_compile` makes each decorated module its own compile
    wrapper, so compiling per layer puts a wrapper on all 40+ layers. vLLM's
    `reset_compile_wrapper` only looks at the object it is handed and that
    object's `.model` attribute; finding no wrapper there, it returns and
    resets nothing.
    """
    import torch
    import vllm.compilation.wrapper as wrapper_mod
    from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper

    original_reset = wrapper_mod.reset_compile_wrapper
    if getattr(original_reset, "_tpu_reset_compile_wrapper_patch", False):
        return

    def reset_compile_wrapper_tpu(model: torch.nn.Module) -> None:
        if model is None:
            return
        if isinstance(model, TorchCompileWithNoGuardsWrapper):
            original_reset(model)
            return

        found_wrapper = False
        for m in model.modules():
            if isinstance(m, TorchCompileWithNoGuardsWrapper):
                found_wrapper = True
                original_reset(m)

        if not found_wrapper:
            original_reset(model)

    reset_compile_wrapper_tpu._tpu_reset_compile_wrapper_patch = True
    wrapper_mod.reset_compile_wrapper = reset_compile_wrapper_tpu
    logger.info(
        "Applied TPU patch: recursive reset_compile_wrapper for per-layer compilation."
    )


def _patch_vllm_piecewise_backend() -> None:
    """Pick the compiled token-count bucket from a tensor's leading dimension.

    A graph is compiled once per bucket of token counts, and vLLM decides
    which one a call needs by reading an argument that is a `SymInt`. A layer
    whose arguments are all tensors has no such argument, so vLLM assumes only
    one bucket was compiled and asserts when several were.
    """
    import torch
    from vllm.compilation.piecewise_backend import PiecewiseBackend

    original_call = PiecewiseBackend.__call__
    if getattr(original_call, "_tpu_piecewise_backend_patch", False):
        return

    def patched_call(self, *args, **kwargs):
        if self.sym_shape_indices:
            runtime_shape = args[self.sym_shape_indices[0]]
            range_entry = self._find_range_for_shape(runtime_shape)
            assert range_entry is not None, (
                f"Shape: {runtime_shape} out of considered ranges: "
                f"{self.compile_ranges}"
            )
        elif len(self.range_entries) > 1 or (
            self.compile_sizes and len(self.compile_sizes) > 1
        ):
            # Dynamic token count from tensor dimension 0
            first_tensor = next((x for x in args if isinstance(x, torch.Tensor)), None)
            if first_tensor is not None:
                runtime_shape = first_tensor.shape[0]
                range_entry = self._find_range_for_shape(runtime_shape)
                assert range_entry is not None, (
                    f"Shape: {runtime_shape} out of considered ranges: "
                    f"{self.compile_ranges}"
                )
            else:
                compiled_entries = [
                    re for re in self.range_entries.values() if re.compiled
                ]
                assert len(compiled_entries) == 1, (
                    f"Expected exactly one compiled range_entry for static shape "
                    f"compilation, but found {len(compiled_entries)}"
                )
                range_entry = compiled_entries[0]
        else:
            compiled_entries = [re for re in self.range_entries.values() if re.compiled]
            assert len(compiled_entries) == 1, (
                f"Expected exactly one compiled range_entry for static shape "
                f"compilation, but found {len(compiled_entries)}"
            )
            range_entry = compiled_entries[0]

        assert range_entry.compiled, (
            "All ranges should be compiled or loaded up front in "
            "PiecewiseBackend.__init__. "
            f"range_entry={range_entry.compile_range}"
        )
        return range_entry.runnable(*args, **kwargs)

    patched_call._tpu_piecewise_backend_patch = True
    PiecewiseBackend.__call__ = patched_call
    logger.info("Applied TPU patch: PiecewiseBackend tensor-dim-0 bucket dispatch.")


def _patch_vllm_compile_prefix_isolation() -> None:
    """Give each support_torch_compile'd instance its own cache identity.

    vLLM's shared `compile_prefix=""` is only safe when all instances of a
    class close over identically-shaped buffers; DSv4's KV cache overlay
    breaks that, and one layer's executable can be reused by another (XLA
    e0102). Keying on self.prefix is a no-op for shape-uniform models.
    """
    import itertools

    from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper

    original_init = TorchCompileWithNoGuardsWrapper.__init__
    if getattr(original_init, "_tpu_compile_prefix_isolation_patch", False):
        return

    counter = itertools.count()

    def patched_init(self, compile_prefix: str = "", is_encoder: bool = False) -> None:
        if not compile_prefix and not is_encoder:
            prefix = getattr(self, "prefix", None)
            if prefix:
                compile_prefix = prefix
            else:
                # No self.prefix: fall back to a counter. Stable within a
                # process but not across restarts, so the cache won't persist.
                compile_prefix = f"_tpu_instance_{next(counter)}"
                logger.warning(
                    "compile_prefix isolation: %s has no self.prefix; using "
                    "non-deterministic fallback %s (its on-disk compile "
                    "cache will not persist across restarts).",
                    type(self).__name__,
                    compile_prefix,
                )
        original_init(self, compile_prefix=compile_prefix, is_encoder=is_encoder)

    patched_init._tpu_compile_prefix_isolation_patch = True
    TorchCompileWithNoGuardsWrapper.__init__ = patched_init
    logger.info(
        "Applied TPU patch: per-instance compile-artifact cache isolation "
        "(compile_prefix keyed off self.prefix)."
    )


def _patch_dflash_bypass_v2_runner_check() -> None:
    """Bypass V2 model runner check in DFlash for TPU execution.

    Upstream ``qwen3_dflash._resolve_layer_attention`` enforces
    ``use_v2_model_runner`` for mixed SWA/full attention DFlash drafters.
    TPU uses TPUModelRunner which handles mixed attention without Triton V2 runner.
    """
    try:
        from vllm.model_executor.models import qwen3_dflash
    except ImportError:
        return

    if getattr(qwen3_dflash, "_tpu_bypass_v2_runner_patch", False):
        return

    original_resolve = qwen3_dflash._resolve_layer_attention
    _FLAG = "use_v2_model_runner"

    def patched_resolve(config, layer_idx: int):
        # Force the flag True for the duration of the upstream call only, by
        # swapping the class attribute and restoring it. Deliberately not
        # `unittest.mock.patch`: a test helper has no business in the load
        # path, and — more usefully — looking the attribute up explicitly
        # means that if upstream ever moves or renames it, this raises here
        # instead of silently no-op'ing and letting the original V2 check
        # reject the drafter with an unrelated-looking error.
        from vllm.config import get_current_vllm_config

        cfg_cls = type(get_current_vllm_config())
        if not hasattr(cfg_cls, _FLAG):
            raise RuntimeError(
                f"VllmConfig no longer defines {_FLAG!r}; the TPU DFlash "
                "bypass needs updating for this vLLM version."
            )

        # Restore exactly what was there: if the flag is inherited rather than
        # defined on this class, putting it back with setattr would leave a
        # permanent shadow on the subclass.
        had_own = _FLAG in cfg_cls.__dict__
        original_flag = cfg_cls.__dict__.get(_FLAG)
        setattr(cfg_cls, _FLAG, property(lambda self: True))
        try:
            return original_resolve(config, layer_idx)
        finally:
            if had_own:
                setattr(cfg_cls, _FLAG, original_flag)
            else:
                delattr(cfg_cls, _FLAG)

    qwen3_dflash._resolve_layer_attention = patched_resolve
    qwen3_dflash._tpu_bypass_v2_runner_patch = True
    logger.info("Applied TPU patch: bypass DFlash V2 runner check on TPU.")


def _patch_vllm_config_triton_tpu() -> None:
    """Bypass V2 model runner Triton check on TPU.

    The config forces use_v2_model_runner=True for hybrid DFlash drafters.
    When it validates V2, it crashes on TPU because HAS_TRITON is False.
    TPU uses TPUModelRunner instead, so we can safely bypass this.
    """
    from vllm.config.vllm import VllmConfig

    if getattr(VllmConfig, "_tpu_triton_patch_applied", False):
        return

    original_validate = VllmConfig._validate_v2_model_runner

    def patched_validate(self):
        if (
            getattr(self, "device_config", None)
            and self.device_config.device_type == "tpu"
        ):
            return
        return original_validate(self)

    VllmConfig._validate_v2_model_runner = patched_validate
    VllmConfig._tpu_triton_patch_applied = True
    logger.info("Applied TPU patch: bypass V2 model runner Triton check.")


def _patch_vllm_force_v1_runner_tpu() -> None:
    """Keep TPU's V1 scheduler contract and validate its own capabilities.

    vLLM 0.29 defaults to GPU Model Runner V2 when Triton is available and
    rejects GPU V2-only features when V1 is selected. TPUModelRunner owns
    PCP, DSpark, and mixed-attention DFlash implementations, but consumes
    V1-shaped scheduler output. Keep that default regardless of Triton and
    remove only those implemented features from the native V1 rejection
    list. The other native checks and TPU's combination checks still apply.
    An explicit VLLM_USE_V2_MODEL_RUNNER retains upstream precedence.
    """
    from collections.abc import Callable
    from typing import Any, cast

    from vllm.config.vllm import VllmConfig

    if getattr(VllmConfig, "_tpu_force_v1_runner_patch", False):
        return

    original_prop = VllmConfig.use_v2_model_runner
    if not isinstance(original_prop, property):
        logger.warning(
            "VllmConfig.use_v2_model_runner is no longer a property; skipping "
            "the TPU V1-runner patch. TPU execution may misbehave until this "
            "is updated for the current vLLM version."
        )
        return

    original_fget = cast(Callable[[Any], bool], original_prop.fget)

    def patched_use_v2(self):
        import vllm.envs as envs

        if envs.VLLM_USE_V2_MODEL_RUNNER is not None:
            return original_fget(self)
        device_config = getattr(self, "device_config", None)
        if device_config is not None and device_config.device_type == "tpu":
            return False
        return original_fget(self)

    original_unsupported = VllmConfig._get_v1_model_runner_unsupported_features

    def patched_unsupported(self):
        unsupported = original_unsupported(self)
        device_config = getattr(self, "device_config", None)
        if device_config is None or device_config.device_type != "tpu":
            return unsupported
        # These are limitations of the upstream GPU V1 runner. Do not bypass
        # the validator: adaptive verification, DFlash2, and future features
        # must remain rejected until TPU implements them.
        implemented = {
            "prefill context parallel",
            "dspark speculative decoding",
            "mixed sliding/full dflash drafts",
        }
        return [feature for feature in unsupported if feature not in implemented]

    VllmConfig._get_v1_model_runner_unsupported_features = patched_unsupported
    VllmConfig.use_v2_model_runner = property(patched_use_v2)
    VllmConfig._tpu_force_v1_runner_patch = True
    logger.info("Applied TPU patch: force V1 model runner semantics on TPU.")


if "proxy" in envs.JAX_PLATFORMS:
    logger.info("Running vLLM on TPU via Pathways proxy.")
    # Must run pathwaysutils.initialize() before any JAX operations
    try:
        import traceback

        import pathwaysutils
        import vllm
        from vllm.platforms import (
            resolve_current_platform_cls_qualname,
            resolve_obj_by_qualname,
        )

        pathwaysutils.initialize()
        logger.info("Module pathwaysutils is imported.")

        # Pathways requires eager resolution of vllm.current_platform instead of
        # lazy resolution in the normal code path. Since this part involves
        # global topology discovery across multiple hosts, the platform
        # resolution must happen before other components are loaded.
        logger.info("Eagerly resolving vLLM current_platform for Pathways.")
        platform_cls_qualname = resolve_current_platform_cls_qualname()
        resolved_platform_instance = resolve_obj_by_qualname(platform_cls_qualname)()
        vllm.platforms._current_platform = resolved_platform_instance
        vllm.platforms._init_trace = "".join(traceback.format_stack())
        logger.info(
            "vLLM platform resolved to: %s",
            resolved_platform_instance.__class__.__name__,
        )

    except Exception as e:
        logger.error(
            "Error occurred while importing pathwaysutils or logging TPU info: %s", e
        )


def _patch_vllm_block_pool_lifo_free() -> None:
    """Restore pre-#48017 LIFO block reuse when prefix caching is off.

    vllm a82f1b38 switched hashless freed blocks from prepend (LIFO reuse
    of a small hot set of block ids) to append (FIFO) when prefix caching
    is disabled. On TPU v7x the FIFO order cycles the entire KV block-id
    space under sustained load and an indexed SparseCore-offloaded op on
    the paged-KV path traps with E0200 RuntimeUnexpectedCoreHalt (UserFatal
    on SC2). Bisect evidence: docs/sc2-sparsecore-halt-2026-08.md on branch
    mhhua/sc2-sparsecore-halt-evidence, vllm-torchtpu-dev builds 136-146.
    Behavior with prefix caching enabled is unchanged. Remove once the
    SparseCore index handling is fixed in the TPU runtime.
    """
    from vllm.v1.core.block_pool import BlockPool

    if BlockPool.__dict__.get("_tpu_lifo_free_patch", False):
        return

    def free_blocks(self, ordered_blocks) -> None:
        blocks_with_hash = []
        blocks_without_hash = []
        for block in ordered_blocks:
            block.ref_cnt -= 1
            if block.ref_cnt == 0 and not block.is_null:
                if block.block_hash is None:
                    blocks_without_hash.append(block)
                else:
                    blocks_with_hash.append(block)
        self.free_block_queue.prepend_n(blocks_without_hash)
        self.free_block_queue.append_n(blocks_with_hash)

    BlockPool.free_blocks = free_blocks
    BlockPool._tpu_lifo_free_patch = True
    logger.info("Applied LIFO free-block patch (pre-vllm#48017 order) to BlockPool")


def _patch_vllm_same_step_prefix_hits() -> None:
    """Keep prefix-cache hits off blocks the current step has not written.

    vLLM publishes a block's hash when the block is allocated, before the
    forward that fills it runs, so a request admitted later in the same
    `schedule()` can hit a block an earlier request fills in that very step.
    GPU attention writes the whole batch's KV before any request reads it,
    which makes that hit safe there. The TPU ragged paged attention kernel
    writes each sequence's new KV as it walks the batch instead: a sequence
    ordered before the one filling the block DMAs the block from HBM
    unwritten and attends over stale KV. Identical prompts admitted together
    (n > 1 sampling, a repeated prompt) then decode garbage.

    Blocks hashed during the current `schedule()` therefore count as misses,
    and a request that would have hit one recomputes that prefix itself.
    From the next `schedule()` on they hit normally: the worker runs the
    steps' forwards in order, so the step that fills a block has run before
    any later step reads it, with or without async scheduling.
    """
    import functools

    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.sched.scheduler import Scheduler

    if getattr(BlockPool.get_cached_block, "_tpu_same_step_prefix_hit_patch", False):
        return

    original_insert_block_hash = BlockPool._insert_block_hash
    original_get_cached_block = BlockPool.get_cached_block
    original_schedule = Scheduler.schedule

    def unwritten_block_ids(pool) -> set[int]:
        return pool.__dict__.setdefault("_tpu_unwritten_block_ids", set())

    # Every path that makes a block hittable (full blocks, partial blocks,
    # re-pointed hashes) goes through here.
    @functools.wraps(original_insert_block_hash)
    def _insert_block_hash(self, block_hash_with_group_id, block, num_tokens):
        original_insert_block_hash(self, block_hash_with_group_id, block, num_tokens)
        unwritten_block_ids(self).add(block.block_id)

    @functools.wraps(original_get_cached_block)
    def get_cached_block(self, block_hash, kv_cache_group_ids):
        blocks = original_get_cached_block(self, block_hash, kv_cache_group_ids)
        unwritten = self.__dict__.get("_tpu_unwritten_block_ids")
        if (
            blocks
            and unwritten
            and any(block.block_id in unwritten for block in blocks)
        ):
            return None
        return blocks

    @functools.wraps(original_schedule)
    def schedule(self, *args, **kwargs):
        unwritten_block_ids(self.kv_cache_manager.block_pool).clear()
        return original_schedule(self, *args, **kwargs)

    for wrapper in (_insert_block_hash, get_cached_block, schedule):
        wrapper._tpu_same_step_prefix_hit_patch = True
    BlockPool._insert_block_hash = _insert_block_hash
    BlockPool.get_cached_block = get_cached_block
    Scheduler.schedule = schedule
    logger.info(
        "Applied TPU patch: no prefix-cache hits on blocks the "
        "current step has not written."
    )


def _patch_vllm_vocab_parallel_embedding() -> None:
    """Register shard index bounds as module buffers in VocabParallelEmbedding.

    Prevents TorchTPU from baking rank-specific integer literals into the MLIR
    graph as scalar constants, ensuring identical CompilationCacheKeys across
    all TP ranks.
    """
    import torch
    import vllm.model_executor.layers.vocab_parallel_embedding as vpe

    if getattr(
        vpe.VocabParallelEmbedding, "_tpu_vocab_parallel_embedding_patch", False
    ):
        return

    original_init = vpe.VocabParallelEmbedding.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if not hasattr(self, "org_vocab_start_index"):
            self.register_buffer(
                "org_vocab_start_index",
                torch.tensor(
                    self.shard_indices.org_vocab_start_index, dtype=torch.int64
                ),
                persistent=False,
            )
            self.register_buffer(
                "org_vocab_end_index",
                torch.tensor(self.shard_indices.org_vocab_end_index, dtype=torch.int64),
                persistent=False,
            )
            self.register_buffer(
                "num_org_vocab_padding",
                torch.tensor(
                    self.shard_indices.num_org_vocab_padding, dtype=torch.int64
                ),
                persistent=False,
            )
            self.register_buffer(
                "added_vocab_start_index",
                torch.tensor(
                    self.shard_indices.added_vocab_start_index, dtype=torch.int64
                ),
                persistent=False,
            )
            self.register_buffer(
                "added_vocab_end_index",
                torch.tensor(
                    self.shard_indices.added_vocab_end_index, dtype=torch.int64
                ),
                persistent=False,
            )

    def patched_forward(self, input_):
        input_mask = None
        if self.tp_size > 1:
            masked_input, input_mask = vpe.get_masked_input_and_mask(
                input_,
                getattr(
                    self,
                    "org_vocab_start_index",
                    self.shard_indices.org_vocab_start_index,
                ),
                getattr(
                    self, "org_vocab_end_index", self.shard_indices.org_vocab_end_index
                ),
                getattr(
                    self,
                    "num_org_vocab_padding",
                    self.shard_indices.num_org_vocab_padding,
                ),
                getattr(
                    self,
                    "added_vocab_start_index",
                    self.shard_indices.added_vocab_start_index,
                ),
                getattr(
                    self,
                    "added_vocab_end_index",
                    self.shard_indices.added_vocab_end_index,
                ),
            )
        else:
            masked_input = input_
        output_parallel = self.quant_method.embedding(self, masked_input.long())
        if self.tp_size > 1:
            assert input_mask is not None
            output_parallel.masked_fill_(input_mask.unsqueeze(-1), 0)
            from vllm.distributed import tensor_model_parallel_all_reduce

            return tensor_model_parallel_all_reduce(output_parallel)
        return output_parallel

    def patched_get_masked_input_and_mask(
        input_: torch.Tensor,
        org_vocab_start_index: int | torch.Tensor,
        org_vocab_end_index: int | torch.Tensor,
        num_org_vocab_padding: int | torch.Tensor,
        added_vocab_start_index: int | torch.Tensor,
        added_vocab_end_index: int | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = input_.device
        dtype = input_.dtype

        def _to_tensor(val: int | torch.Tensor) -> torch.Tensor:
            if isinstance(val, torch.Tensor):
                return val.to(device=device)
            return torch.tensor(val, dtype=dtype, device=device)

        start = _to_tensor(org_vocab_start_index)
        end = _to_tensor(org_vocab_end_index)
        num_padding = _to_tensor(num_org_vocab_padding)
        added_start = _to_tensor(added_vocab_start_index)
        added_end = _to_tensor(added_vocab_end_index)

        org_vocab_mask = (input_ >= start) & (input_ < end)
        added_vocab_mask = (input_ >= added_start) & (input_ < added_end)
        added_offset = added_start - (end - start) - num_padding
        valid_offset = (start * org_vocab_mask) + (added_offset * added_vocab_mask)
        vocab_mask = org_vocab_mask | added_vocab_mask
        input_ = vocab_mask * (input_ - valid_offset)
        return input_, ~vocab_mask

    vpe.get_masked_input_and_mask = patched_get_masked_input_and_mask
    vpe.VocabParallelEmbedding.__init__ = patched_init
    vpe.VocabParallelEmbedding.forward = patched_forward
    vpe.VocabParallelEmbedding._tpu_vocab_parallel_embedding_patch = True
    logger.info(
        "Applied TPU patch: register shard index bounds as buffers in "
        "VocabParallelEmbedding"
    )
