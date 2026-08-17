# SPDX-License-Identifier: Apache-2.0

# Ensure environment overrides are applied before any other imports,
# especially torch_tpu which might read them at import time.
import vllm_torchtpu.env_override  # noqa: F401  # isort: skip

import os
import time
from typing import Dict, Tuple
from urllib.parse import urlparse

import torch
import torch.profiler
import torch_tpu  # noqa: F401
from torch_tpu._internal.profiler import TpuProfilerConfig
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed.kv_transfer import (ensure_kv_transfer_initialized,
                                          get_kv_transfer_group,
                                          has_kv_transfer_group)
from vllm.distributed.parallel_state import (ensure_model_parallel_initialized,
                                             get_pp_group,
                                             get_tensor_model_parallel_rank,
                                             init_distributed_environment)
from vllm.v1 import utils as vllm_utils
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.worker_base import CompilationTimes, WorkerBase

import vllm_torchtpu.distributed.utils as dist_utils
from vllm_torchtpu import envs, profiler_trace, utils
from vllm_torchtpu.distributed import jax_parallel_state
from vllm_torchtpu.distributed.pcp_rank_order import (
    pcp_topology_order, resolve_pcp_topology_order, verify_pcp_topology_order)
from vllm_torchtpu.layers.vllm.attention import TPU_STR_DTYPE_TO_TORCH_DTYPE
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner
from vllm_torchtpu.worker.tpu_rank_binding import get_tpu_worker_binding

logger = init_logger(__name__)
DEBUG_TPU_LOCAL_RANK_OFFSET_ENV = "DEBUG_TPU_LOCAL_RANK_OFFSET"


def _get_kv_connector_handshake_metadata_key(metadata=None) -> tuple[int, int]:
    transfer_rank = getattr(metadata, "transfer_rank", None)
    rank = (int(transfer_rank) if transfer_rank is not None else int(
        get_tensor_model_parallel_rank()))
    return get_pp_group().rank_in_group, rank


def _debug_tpu_local_rank_offset() -> int:
    value = os.environ.get(DEBUG_TPU_LOCAL_RANK_OFFSET_ENV, "0") or "0"
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{DEBUG_TPU_LOCAL_RANK_OFFSET_ENV} must be an int, "
                         f"got {value!r}") from exc


def _configure_tpu_process_env(rank: int, local_rank: int, world_size: int,
                               local_world_size: int) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(local_world_size)
    if world_size > 1:
        os.environ["WORLD_SIZE"] = str(world_size)
    else:
        os.environ.pop("WORLD_SIZE", None)


class TPUWorker(WorkerBase):

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
        devices=None,
        ip: str = "localhost",
        prev_worker_ip: str = "localhost",
    ):
        # Re-apply patches that were done in check_and_update_config.
        # Workers may be spawned (not forked) so module-level patches
        # from the parent process are lost.
        from vllm_torchtpu.platforms.tpu_platform import apply_tpu_patches
        apply_tpu_patches()

        super().__init__(vllm_config=vllm_config,
                         local_rank=local_rank,
                         rank=rank,
                         distributed_init_method=distributed_init_method,
                         is_driver_worker=is_driver_worker)
        # WorkerBase initializes self.device=None which makes vLLM's
        # MultiprocExecutor.async_output_busy_loop call
        # current_platform.set_device(None) → TPU has no torch device-switching
        # API (pinning is via TPU_VISIBLE_CHIPS env var). Drop the attribute
        # so hasattr(self.worker, "device") returns False and the call skips.
        if hasattr(self, "device"):
            del self.device

        # TPU compilation requires static shapes, so clear dynamic compile
        # ranges vLLM might have added.
        self.compilation_config.compile_ranges_endpoints = []

        # TPU-specific extras not in WorkerBase.
        self.devices = devices if devices is not None else []
        self.device_ranks = set(device.id for device in self.devices)
        self.prev_worker_ip = prev_worker_ip

        if self.cache_config.cache_dtype == "auto":
            model_dtype = self.model_config.dtype
            self.kv_cache_dtype = (TPU_STR_DTYPE_TO_TORCH_DTYPE[model_dtype]
                                   if isinstance(model_dtype, str) else
                                   model_dtype)
        else:
            self.kv_cache_dtype = TPU_STR_DTYPE_TO_TORCH_DTYPE[
                self.cache_config.cache_dtype]

        # TPU profiler: every worker captures the chips it owns into its own
        # sandbox, and the captures are merged into one run directory on stop
        # (see profiler_trace). A worker can only trace its own chips, so
        # capturing on rank 0 alone would miss the rest of the slice.
        self.profile_dir: str | None = None
        self.profile_context = None
        self.profile_run_dir: str | None = None
        self.profile_capture_dir: str | None = None
        self.profile_canonical_ts: str | None = None
        self.profile_session_key: str | None = None
        self.profile_session_index = 0
        # Rank/world used to label and merge traces. init_device() upgrades
        # these to the slice-global TPU rank, which — unlike parallel_config's
        # TPxPP-scoped rank — is unique across DP replicas too.
        self.profile_rank = self.rank
        self.profile_world_size = self.parallel_config.world_size
        torch_profiler_dir = self.vllm_config.profiler_config.torch_profiler_dir
        if torch_profiler_dir:
            self.profile_dir = torch_profiler_dir
            logger.info("Profiling enabled. Traces will be saved to: %s",
                        self.profile_dir)

        # step_counter is used to calc uuid for intermediate tensor transfer.
        self.step_counter = 0

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

    def init_device(self):
        # vLLM's MultiprocExecutor passes per-engine rank/local_rank as
        # constructor args. Native vLLM DP keeps those values local to the DP
        # engine and lets init_distributed_environment() derive the global
        # distributed rank from ParallelConfig.data_parallel_rank.
        #
        # TorchTPU currently also reads torchrun-style env vars for physical
        # chip/PjRt binding. Keep that as a compatibility shim only: env rank
        # values describe the unified TPU slice, while the vLLM distributed
        # init below still receives upstream-shaped local rank/world args.
        # TORCH_TPU_SLICEBUILDER_ADDRESSES and TORCH_TPU_TOPOLOGY are
        # already inherited from the parent process (set by
        # prepare_tpu_environment() in tpu_platform.py).
        pc = self.parallel_config
        dp_size = utils.get_dp_size(pc)
        pcp_size = pc.prefill_context_parallel_size
        binding = get_tpu_worker_binding(pc,
                                         self.rank,
                                         self.local_rank,
                                         env=os.environ,
                                         use_spawned_pcp_local_rank=True)
        # Slice-global rank: unique per worker process across DP replicas and
        # TP/PP ranks, which is what trace filenames must be keyed on. Set
        # before TPUModelRunner is built below, since the phased profiler
        # takes these at construction time.
        self.profile_rank = binding.rank
        self.profile_world_size = binding.world_size
        if dp_size > 1:
            per_engine_world = pc.world_size
            dp_rank = binding.dp_rank

            # TorchTPU uses independent per-engine tpu_dist worlds unless EP
            # needs a unified DP*TP world for expert collectives.
            if pc.enable_expert_parallel:
                pc.data_parallel_size = dp_size
                pc.data_parallel_rank = dp_rank
                if pc.data_parallel_rank_local is None:
                    pc.data_parallel_rank_local = dp_rank
            else:
                pc.data_parallel_size = 1
                # ParallelConfig's own validator requires
                # data_parallel_size_local <= data_parallel_size. vLLM core's
                # own analogous single-engine collapse (run_engine_core's
                # non-MoE DP path) sets both fields together for the same
                # reason: https://github.com/vllm-project/vllm/blob/main/vllm/config/parallel.py#L1044
                pc.data_parallel_size_local = 1
                pc.data_parallel_rank = 0
                pc.data_parallel_rank_local = 0

            master_addr = os.environ.get("TORCH_TPU_DP_MASTER_ADDR",
                                         "localhost")
            master_port = os.environ["TORCH_TPU_DP_MASTER_PORT"]
            os.environ.update(binding.as_env())
            os.environ["MASTER_ADDR"] = str(master_addr)
            os.environ["MASTER_PORT"] = str(master_port)

            if pc.enable_expert_parallel:
                # Setting data_parallel_size above puts
                # init_distributed_environment() on its DP path, where it
                # discards the rank and the init method it was handed and
                # recomputes both: rank becomes
                # `data_parallel_rank * world_size + rank`, and the rendezvous
                # becomes data_parallel_master_ip plus a port popped from the
                # DP port list. Both predate this worker having an
                # executor-assigned slice placement, and both are wrong here.
                #
                # The address is wrong outright: data_parallel_master_ip
                # defaults to 127.0.0.1, so each host's workers rendezvous on
                # their own loopback and the two halves of the slice never
                # meet -- init_process_group then blocks until the engine
                # start timeout. Point it at the host holding slice rank 0,
                # which every worker derived identically in Step 7 of the
                # executor. The port needs no such fixup: it is popped from a
                # list built once on the driver and carried unchanged into
                # every worker, so all of them already agree on it. Leave
                # data_parallel_master_port alone -- MASTER_PORT below is
                # torch_tpu's own slice rendezvous, and forcing it here would
                # collide with that.
                pc.data_parallel_master_ip = master_addr
                # Ray hands each engine whichever chips were free, so slice
                # ranks do not run dp-rank-major and binding.rank is not
                # `dp_rank * per_engine_world + self.rank`. Pre-subtract the
                # offset the rewrite is about to add, so it lands back on the
                # slice rank this worker is actually bound to. The subtraction
                # can go negative for an engine placed early in the slice;
                # that value is consumed by the rewrite before torch sees it.
                init_rank = binding.rank - dp_rank * per_engine_world
                init_world = binding.init_world_size
                dist_init_method = self.distributed_init_method
            else:
                # Dense DP has no cross-rank collective, so each replica
                # should bootstrap its own isolated torch_tpu slice instead
                # of sharing check_and_update_config's full DP*TP-wide one
                # (RANK/WORLD_SIZE above are already replica-local via
                # binding.as_env()). Narrow the inherited full-slice
                # TORCH_TPU_SLICEBUILDER_ADDRESSES down to just this
                # replica's own per_engine_world workers -- narrowing
                # (rather than each worker picking fresh ports
                # independently) keeps the address list agreed upon by
                # every worker in this replica, since it's sliced
                # deterministically out of the same parent-established list
                # every sibling worker also inherited.
                full_sb_addresses = os.environ.get(
                    "TORCH_TPU_SLICEBUILDER_ADDRESSES", "")
                sb_addresses = full_sb_addresses.split(
                    ",") if full_sb_addresses else []
                start = dp_rank * per_engine_world
                my_sb_addresses = sb_addresses[start:start + per_engine_world]
                if len(my_sb_addresses) == per_engine_world:
                    os.environ["TORCH_TPU_SLICEBUILDER_ADDRESSES"] = ",".join(
                        my_sb_addresses)
                else:
                    logger.warning(
                        "Expected %d slicebuilder addresses for dp_rank=%d "
                        "at offset %d, found %d in inherited list %r; "
                        "leaving TORCH_TPU_SLICEBUILDER_ADDRESSES as-is.",
                        per_engine_world, dp_rank, start, len(my_sb_addresses),
                        full_sb_addresses)
                from vllm_torchtpu.platforms.tpu_platform import TpuPlatform
                os.environ["TORCH_TPU_TOPOLOGY"] = \
                    TpuPlatform._get_tpu_topology(per_engine_world)
                # Request this worker's own physical chip explicitly,
                # instead of relying on the flattened-rank chip binding EP
                # uses, so independent replicas don't contend for the same
                # chips. Both TPU_VISIBLE_CHIPS and TPU_VISIBLE_DEVICES must
                # be set -- TorchTPU only honors the chip pin under the
                # RANK/WORLD_SIZE distributed bootstrap when both agree.
                os.environ["TPU_VISIBLE_CHIPS"] = str(
                    binding.native_local_rank)
                os.environ["TPU_VISIBLE_DEVICES"] = str(
                    binding.native_local_rank)
                os.environ["ALLOW_MULTIPLE_LIBTPU_LOAD"] = "1"
                init_rank = binding.init_rank
                init_world = binding.init_world_size
                dist_init_method = self.distributed_init_method
            local_rank_for_init = binding.init_local_rank
            dist_world_size = binding.world_size
            logger.info(
                "TPU DP worker: dp_size=%d dp_rank=%d per_engine_world=%d "
                "self.rank=%d self.local_rank=%d -> tpu_rank=%d "
                "tpu_local_rank=%d global_world=%d "
                "init_local_rank=%d "
                "(vllm_init_rank=%d vllm_init_world=%d)",
                dp_size,
                dp_rank,
                per_engine_world,
                self.rank,
                self.local_rank,
                binding.rank,
                binding.local_rank,
                binding.world_size,
                binding.init_local_rank,
                init_rank,
                init_world,
            )
        else:
            dist_init_method = self.distributed_init_method
            parsed = urlparse(dist_init_method)
            if parsed.scheme != "tcp" or not parsed.hostname or not parsed.port:
                raise ValueError(
                    "Expected tcp://<host>:<port> distributed_init_method, "
                    f"got: {dist_init_method!r}")
            if pcp_size > 1:
                logger.info(
                    "PCP native-rank TPU binding | rank=%d "
                    "native_local_rank=%d local_rank_env=%d "
                    "tpu_local_rank_env=%d local_world=%s "
                    "tpu_local_world=%d",
                    self.rank,
                    self.local_rank,
                    binding.init_local_rank,
                    binding.local_rank,
                    binding.world_size,
                    binding.local_world_size,
                )
                init_rank = binding.init_rank
                init_world = binding.init_world_size
                local_rank_for_init = binding.init_local_rank
                dist_world_size = binding.world_size
                os.environ.update(binding.as_env())
            else:
                local_world = os.environ.get(
                    "LOCAL_WORLD_SIZE") or pc.world_size
                local_rank_for_init = int(self.local_rank)
                debug_local_rank_offset = _debug_tpu_local_rank_offset()
                tpu_local_rank_env = (local_rank_for_init +
                                      debug_local_rank_offset)
                tpu_local_world = int(local_world)
                if debug_local_rank_offset:
                    tpu_local_world = max(
                        tpu_local_world,
                        debug_local_rank_offset + pc.world_size)
                    logger.info(
                        "DEBUG TPU local rank offset applied: rank=%d "
                        "local_rank=%d %s=%d -> tpu_local_rank=%d "
                        "tpu_local_world=%d", self.rank, self.local_rank,
                        DEBUG_TPU_LOCAL_RANK_OFFSET_ENV,
                        debug_local_rank_offset, tpu_local_rank_env,
                        tpu_local_world)
                init_rank = self.rank
                init_world = pc.world_size
                dist_world_size = pc.world_size
                _configure_tpu_process_env(self.rank, tpu_local_rank_env,
                                           pc.world_size, tpu_local_world)
            if int(os.environ.get("TPU_LOCAL_RANK_OFFSET", "0") or "0"):
                logger.info(
                    "TPU local-rank offset binding | rank=%d local_rank=%d "
                    "offset=%d -> tpu_local_rank=%d tpu_local_world=%d",
                    self.rank,
                    self.local_rank,
                    int(os.environ.get("TPU_LOCAL_RANK_OFFSET", "0") or "0"),
                    binding.local_rank,
                    binding.local_world_size,
                )
            os.environ.setdefault("MASTER_ADDR", parsed.hostname)
            os.environ.setdefault("MASTER_PORT", str(parsed.port))

        if not self.devices:
            self.devices = [torch.device("tpu")]

        from vllm.platforms import current_platform
        dist_backend = current_platform.get_worker_distributed_backend(
            dist_world_size)

        with set_current_vllm_config(self.vllm_config):
            init_distributed_environment(
                world_size=init_world,
                rank=init_rank,
                local_rank=local_rank_for_init,
                distributed_init_method=dist_init_method,
                backend=dist_backend,
            )
        # Ring order is the PCP group's rank order, and the group is built
        # below. topology_aware_mesh needs open chips and a live process
        # group, so this is the first point it can be asked -- and the last
        # point the answer can still be applied.
        group_ranks_by_name = resolve_pcp_topology_order(self.vllm_config)
        with set_current_vllm_config(self.vllm_config), \
                pcp_topology_order(group_ranks_by_name):
            ensure_model_parallel_initialized(
                tensor_model_parallel_size=self.parallel_config.
                tensor_parallel_size,
                pipeline_model_parallel_size=self.parallel_config.
                pipeline_parallel_size,
                prefill_context_model_parallel_size=pcp_size,
            )
        # The patch above substitutes rank lists on the way in; this reads the
        # built groups back, so a patch that silently stopped applying fails
        # here rather than costing a few percent unnoticed.
        verify_pcp_topology_order(group_ranks_by_name)

        # TODO: Enable PP support. The old JAX-based PP init
        # (jax_parallel_state.init_pp_distributed_environment) was removed
        # during the torch_tpu migration. PP will need KV transfer init
        # and proper rank assignment via torch.distributed.
        is_first_rank = True
        is_last_rank = True
        if self.parallel_config.pipeline_parallel_size > 1:
            is_first_rank = self.rank == 0
            is_last_rank = (
                self.rank == self.parallel_config.pipeline_parallel_size - 1)

        # TODO: Fix device assignment
        self.model_runner = TPUModelRunner(
            self.vllm_config,
            self.devices[0],
            profiler_rank=self.profile_rank,
            profiler_world_size=self.profile_world_size)
        logger.info(f"Init worker | "
                    f"rank={self.rank} | "
                    f"is_first_rank={is_first_rank} | "
                    f"is_last_rank={is_last_rank} | "
                    f"node_id={dist_utils.get_node_id()} | "
                    f"is_driver_worker={self.is_driver_worker} | ")
        # f"hbm={utils.hbm_usage_gb(self.devices)}GiB")
        vllm_utils.report_usage_stats(self.vllm_config)

    def initialize_pp_transfer_connect(self):
        if self.rank == 0:
            return
        jax_parallel_state.connect(self.prev_worker_ip, self.rank - 1)

    def determine_available_memory(self) -> int:
        # To determine a reasonable kv_cache_memory_bytes for a specific vllm
        # setup, users can run vllm without this flag, and collect the
        # total_hbm_avail_gb value logged later in this function.
        if kv_cache_memory_bytes := self.cache_config.kv_cache_memory_bytes:
            msg = ("kv_cache_memory_bytes in cache_config is specified."
                   "This does not respect the gpu_memory_utilization config. "
                   "Only use kv_cache_memory_bytes config "
                   "when you want manual control of KV cache memory size. "
                   "If OOM'ed, check the difference of initial free "
                   "memory between the current run and the previous run "
                   "where kv_cache_memory_bytes is suggested and update it "
                   "correspondingly.")
            logger.info(msg)
            return kv_cache_memory_bytes

        # VLLM directive of the percentage of HBM memory the model executor can use
        self.model_runner.profile_run(self.model_runner.max_num_tokens)

        gpu_memory_utilization = self.cache_config.gpu_memory_utilization
        budget = utils.compute_hbm_budget(self.devices, gpu_memory_utilization)

        # Some KV connectors allocate HBM AFTER profile_run completes — e.g.,
        # an offload spec with a host<->device transfer staging buffer, or a
        # P/D disagg connector with NCCL receive buffers. vLLM's normal
        # accounting misses those bytes. Ask the active connector how much
        # to reserve and subtract from the KV-cache budget. Returns 0 if no
        # connector needs a reserve. The connector owns the size formula.
        kv_connector_hbm_reserve = utils.estimate_kv_connector_hbm_reserve(
            self.vllm_config)
        available = budget.available - kv_connector_hbm_reserve

        total_hbm_limit_gb = round(budget.total_limit / utils.GBYTES, 2)
        total_hbm_limit_cap_gb = round(budget.cap / utils.GBYTES, 2)
        total_hbm_used_gb = round(budget.total_used / utils.GBYTES, 2)
        kv_connector_hbm_reserve_gb = round(
            kv_connector_hbm_reserve / utils.GBYTES, 2)
        total_hbm_avail_gb = round(available / utils.GBYTES, 2)
        logger.info(f"Memory statistics | "
                    f"{total_hbm_limit_gb=}GiB | "
                    f"{total_hbm_limit_cap_gb=}GiB | "
                    f"{total_hbm_used_gb=}GiB | "
                    f"{kv_connector_hbm_reserve_gb=}GiB | "
                    f"{total_hbm_avail_gb=}GiB")

        if available <= 0:
            raise ValueError(f"{total_hbm_used_gb=}GiB exceeds "
                             f"{total_hbm_limit_cap_gb=}GiB by "
                             f"{-total_hbm_avail_gb}GiB. Please consider "
                             f"increasing --gpu-memory-utilization from "
                             f"{gpu_memory_utilization} to a larger value.")
        return available

    def execute_model(self, scheduler_output):
        return self.model_runner.execute_model(scheduler_output)

    def sample_tokens(self, grammar_output):
        return self.model_runner.sample_tokens(grammar_output)

    def take_draft_token_ids(self):
        return self.model_runner.take_draft_token_ids()

    def execute_dummy_batch(self) -> None:
        """Run an idle DP step with the same collective pattern as active DP."""
        runner = self.model_runner
        bucket, target_num_chunks = runner._dp_coordinated_step(0, 0)
        if bucket is None:
            # TPU compiles exact token buckets, so the idle-engine dummy must
            # use one of the precompiled model-forward shapes.
            dummy_tokens = runner.num_tokens_paddings[0]
            runner._dummy_run(dummy_tokens,
                              runner.num_reqs_max_model_len,
                              runner.max_num_blocks_per_req,
                              use_max_model_len=True)
            return
        # Single DP-EP pairing entry: all target dummy forwards, then all draft
        # dummy forwards, in the runner that owns the state.
        runner._run_dp_idle_pairing(bucket, target_num_chunks)

    def profile(self,
                is_start: bool = True,
                profile_prefix: str | None = None):
        if envs.USE_PHASED_PROFILER:
            # A phased run captures one trace per inference phase rather than
            # a single continuous one, driven from TPUModelRunner since phase
            # detection needs per-step batch composition. profile() still owns
            # arming/disarming it, so both profilers share one trigger.
            if is_start:
                self.model_runner.start_phased_profiling(profile_prefix)
            else:
                self.model_runner.stop_phased_profiling()
            return

        if self.profile_dir is None:
            logger.warning("Profile directory is not set. Skipping profiling.")
            return

        if is_start:
            if self.profile_context is not None:
                logger.warning(
                    "Profiler is already running. Ignoring start request.")
                return
            from vllm_torchtpu.tracing.options import \
                resolve_profile_dir_and_opts
            profile_dir, standard_opts, advanced_opts = resolve_profile_dir_and_opts(
                self.profile_dir, profile_prefix)
            os.makedirs(profile_dir, exist_ok=True)
            # All ranks capture concurrently, so each writes into its own
            # sandbox under a run directory they agree on; stop merges them.
            self.profile_run_dir = profile_dir
            self.profile_session_key = self._next_profile_session_key()
            self.profile_canonical_ts = profiler_trace.resolve_canonical_dst_ts(
                profile_dir,
                self.profile_rank,
                session_key=self.profile_session_key)
            self.profile_capture_dir = profiler_trace.rank_capture_dir(
                profile_dir, self.profile_rank)
            os.makedirs(self.profile_capture_dir, exist_ok=True)

            logger.info("Starting TorchTPU profiler trace at %s...",
                        self.profile_capture_dir)
            handler = torch.profiler.tensorboard_trace_handler(
                dir_name=self.profile_capture_dir, use_gzip=True)
            config = TpuProfilerConfig(
                run_dir=self.profile_capture_dir,
                host_tracer_level=standard_opts["host_tracer_level"],
                device_tracer_level=standard_opts["device_tracer_level"],
                python_tracer_level=standard_opts["python_tracer_level"],
                experimental_options=advanced_opts,
            )
            self.profile_context = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.PrivateUse1,
                ],
                on_trace_ready=handler,
                experimental_config=config,
            )
            try:
                self.profile_context.__enter__()
            except Exception:
                self.profile_context = None
                raise
        else:
            if self.profile_context is None:
                logger.warning(
                    "Profiler context is not set. Cannot stop profiler.")
                return
            logger.info("Stopping TorchTPU profiler trace...")
            try:
                # The trace is written by the on_trace_ready handler during
                # __exit__, so it is on disk once this returns.
                self.profile_context.__exit__(None, None, None)
            finally:
                self.profile_context = None
            # Merge into the run directory this capture was started under,
            # which is what the sandbox path is relative to, rather than
            # re-deriving it from the stop call's prefix.
            profile_dir = self.profile_run_dir
            if self.profile_capture_dir and self.profile_canonical_ts:
                profiler_trace.merge_rank_capture(
                    self.profile_capture_dir,
                    profile_dir,
                    self.profile_canonical_ts,
                    self.profile_rank,
                    world_size=self.profile_world_size,
                )
                if self.profile_rank == 0:
                    profiler_trace.clear_canonical_ts_marker(
                        profile_dir, self.profile_session_key)
            logger.info("Profiler trace saved to %s", profile_dir)
            self.profile_run_dir = None
            self.profile_capture_dir = None
            self.profile_canonical_ts = None
            self.profile_session_key = None

    def _next_profile_session_key(self) -> str:
        """Marker key for the start/stop cycle that is beginning.

        Every worker sees the same sequence of profile RPCs, so the index
        keeps back-to-back runs into the same directory from reusing an
        earlier run's marker.
        """
        key = (f"{profiler_trace.profile_session_id()}"
               f"_{self.profile_session_index}")
        self.profile_session_index += 1
        return key

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        from vllm_torchtpu.platforms.tpu_platform import \
            _apply_model_specific_patches
        _apply_model_specific_patches(self.model_config)
        self.model_runner.load_model()

    def compile_or_warm_up_model(self) -> CompilationTimes:
        start = time.perf_counter()
        self.model_runner.capture_model()
        compilation_time = time.perf_counter() - start
        self.compilation_config.compilation_time = compilation_time
        return CompilationTimes(language_model=compilation_time, encoder=0.0)

    def reload_kernels(self, modules: list[str] | None = None) -> dict:
        """Kernel-iteration hot-reload: re-import the reloadable kernel
        sources and swap the live callables behind the registered custom ops.

        Invoked on every worker via collective_rpc from the /reload_kernel
        dev endpoint. Requires TPU_KERNEL_ITER_MODE=1: the ops execute
        eagerly outside the cached compiled pieces, so swapping them cannot
        leave stale compiled graphs. No eager rewarm is performed — the
        serving path builds the device program around the swapped kernel
        with its own identity, so a pre-warm compiles a program serving
        would not reuse; the first request after the swap pays the single
        kernel compile instead.
        """
        if not envs.TPU_KERNEL_ITER_MODE:
            raise RuntimeError(
                "reload_kernels requires TPU_KERNEL_ITER_MODE=1")
        from vllm_torchtpu.compilation import kernel_reload

        start = time.perf_counter()
        stats = kernel_reload.reload_kernels(modules)
        stats["rank"] = self.rank
        stats["total_s"] = round(time.perf_counter() - start, 3)
        logger.info("Kernel hot-reload done: %s", stats)
        return stats

    def get_model(self):
        return self.model_runner.get_model()

    def get_supported_tasks(self):
        return self.model_runner.get_supported_tasks()

    def get_kv_cache_spec(self):
        return self.model_runner.get_kv_cache_spec()

    def get_num_gpu_blocks_override(self) -> int | None:
        return self.vllm_config.cache_config.num_gpu_blocks_override

    def get_kv_prewarm_shapes(self) -> list[int]:
        return self.model_runner.get_kv_prewarm_shapes()

    def prewarm_kv_offload_shape(self, p: int) -> None:
        self.model_runner.prewarm_kv_offload_shape(p)

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        self.cache_config.num_gpu_blocks = kv_cache_config.num_blocks
        ensure_kv_transfer_initialized(self.vllm_config, kv_cache_config)
        self.model_runner.initialize_kv_cache(kv_cache_config)

    def get_node_kv_ip_port(self) -> tuple[int, str, int]:
        node_id = dist_utils.get_node_id()
        ip = dist_utils.get_host_ip()
        tp_size = self.parallel_config.tensor_parallel_size
        n_cfg = dist_utils.get_transfer_channel_number()
        n_channels = tp_size if n_cfg <= 0 else min(n_cfg, tp_size)
        n_channels = max(1, n_channels)
        base_port = int(
            dist_utils.get_kv_transfer_port()) + node_id * n_channels
        return (node_id, ip, base_port)

    def sync_weights(self,
                     updated_weights,
                     mappings: Dict[str, Tuple[str, Tuple[str]]],
                     transpose_keys: Dict[str, Tuple[int]],
                     reshard_fn=None) -> None:
        return self.model_runner._sync_weights(
            updated_weights=updated_weights,
            mappings=mappings,
            transpose_keys=transpose_keys,
            reshard_fn=reshard_fn,
        )

    def get_kv_connector_handshake_metadata(self) -> dict | None:
        if not has_kv_transfer_group():
            return None
        connector = get_kv_transfer_group()
        metadata = connector.get_handshake_metadata()
        if metadata is None:
            return None
        return {_get_kv_connector_handshake_metadata_key(metadata): metadata}
