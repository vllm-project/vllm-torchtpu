# SPDX-License-Identifier: Apache-2.0

# Ensure environment overrides are applied before any other imports,
# especially torch_tpu which might read them at import time.
import vllm_torchtpu.env_override  # noqa: F401  # isort: skip

import importlib
import os
import time
from typing import Dict, Tuple
from urllib.parse import urlparse

import torch
import torch_tpu  # noqa: F401
from torch_tpu._internal.profiler import profiler_api
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed.kv_transfer import (ensure_kv_transfer_initialized,
                                          get_kv_transfer_group,
                                          has_kv_transfer_group)
from vllm.distributed.parallel_state import (ensure_model_parallel_initialized,
                                             get_tensor_model_parallel_rank,
                                             init_distributed_environment)
from vllm.v1 import utils as vllm_utils
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.worker_base import CompilationTimes, WorkerBase

import vllm_torchtpu.distributed.utils as dist_utils
from vllm_torchtpu import envs, utils
from vllm_torchtpu.distributed import jax_parallel_state
from vllm_torchtpu.layers.vllm.attention import TPU_STR_DTYPE_TO_TORCH_DTYPE
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner
from vllm_torchtpu.worker.tpu_rank_binding import get_tpu_worker_binding

logger = init_logger(__name__)
DEBUG_TPU_LOCAL_RANK_OFFSET_ENV = "DEBUG_TPU_LOCAL_RANK_OFFSET"


def _get_kv_connector_handshake_metadata_key() -> int:
    return int(get_tensor_model_parallel_rank())


def _debug_tpu_local_rank_offset() -> int:
    value = os.environ.get(DEBUG_TPU_LOCAL_RANK_OFFSET_ENV, "0") or "0"
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{DEBUG_TPU_LOCAL_RANK_OFFSET_ENV} must be an int, "
                         f"got {value!r}") from exc


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

        if envs.MODEL_IMPL_TYPE != "vllm":
            raise ValueError("Only vLLM models are supported")

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

        # TPU profiler: only on rank 0 single-host, or every PP worker.
        self.profile_dir: str | None = None
        self.profile_context = None
        torch_profiler_dir = os.getenv("VLLM_TORCH_PROFILER_DIR")
        pp_size = self.parallel_config.pipeline_parallel_size
        if torch_profiler_dir and pp_size == 1 and self.rank < 1 and (
                not self.devices or 0 in self.device_ranks):
            # Only 1 active profiler session per server is allowed.
            self.profile_dir = torch_profiler_dir
            logger.info("Profiling enabled. Traces will be saved to: %s",
                        self.profile_dir)
        elif pp_size > 1 and torch_profiler_dir:
            # PP uses MPMD: profile every worker.
            self.profile_dir = os.path.join(
                torch_profiler_dir,
                f"pprank_{self.rank}_ppworldsize_{pp_size}")
            os.makedirs(self.profile_dir, exist_ok=True)

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
        pcp_size = getattr(pc, "prefill_context_parallel_size", 1)
        if not isinstance(pcp_size, int):
            pcp_size = 1
        binding = get_tpu_worker_binding(pc,
                                         self.rank,
                                         self.local_rank,
                                         env=os.environ,
                                         use_spawned_pcp_local_rank=True)
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
                pc.data_parallel_rank = 0
                pc.data_parallel_rank_local = 0

            master_addr = os.environ.get("TORCH_TPU_DP_MASTER_ADDR",
                                         "localhost")
            master_port = os.environ["TORCH_TPU_DP_MASTER_PORT"]
            os.environ.update(binding.as_env())
            os.environ["MASTER_ADDR"] = str(master_addr)
            os.environ["MASTER_PORT"] = str(master_port)

            if pc.enable_expert_parallel:
                init_rank = binding.init_rank
                init_world = binding.init_world_size
                dist_init_method = self.distributed_init_method
            else:
                init_rank = binding.init_rank
                init_world = binding.init_world_size
                dist_init_method = self.distributed_init_method
            local_rank_for_init = binding.init_local_rank
            dist_world_size = binding.world_size
            logger.info(
                "TPU DP worker: dp_size=%d dp_rank=%d per_engine_world=%d "
                "self.rank=%d self.local_rank=%d -> tpu_rank=%d "
                "tpu_local_rank=%d global_world=%d "
                "init_local_rank=%d pcp_remap=%s source=%s "
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
                binding.pcp_local_rank_remap,
                binding.pcp_remap_source,
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
                    "tpu_local_world=%d remap=%s source=%s",
                    self.rank,
                    self.local_rank,
                    binding.init_local_rank,
                    binding.local_rank,
                    binding.world_size,
                    binding.local_world_size,
                    binding.pcp_local_rank_remap,
                    binding.pcp_remap_source,
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
                os.environ["RANK"] = str(self.rank)
                os.environ["LOCAL_RANK"] = str(tpu_local_rank_env)
                os.environ["WORLD_SIZE"] = str(pc.world_size)
                os.environ["LOCAL_WORLD_SIZE"] = str(tpu_local_world)
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
        with set_current_vllm_config(self.vllm_config):
            ensure_model_parallel_initialized(
                tensor_model_parallel_size=self.parallel_config.
                tensor_parallel_size,
                pipeline_model_parallel_size=self.parallel_config.
                pipeline_parallel_size,
                prefill_context_model_parallel_size=pcp_size,
            )

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
        self.model_runner = TPUModelRunner(self.vllm_config, self.devices[0])
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

    def _estimate_kv_connector_hbm_reserve(self) -> int:
        """HBM bytes the active KV offload spec needs reserved upfront.

        The configured spec is resolved dynamically from
        kv_connector_extra_config (same lookup as vllm's OffloadingSpecFactory).
        Any spec class implementing
        `estimate_hbm_reserve_bytes(vllm_config) -> int` is asked for its
        reserve; tpu_worker stays agnostic to connector type and spec class.
        Returns 0 when no connector is configured, the spec can't be resolved,
        or the spec doesn't advertise an HBM reserve.
        """
        kv_tc = self.vllm_config.kv_transfer_config
        if kv_tc is None:
            return 0
        extra = kv_tc.kv_connector_extra_config or {}
        spec_name = extra.get("spec_name")
        spec_module_path = extra.get("spec_module_path")
        if not spec_name or not spec_module_path:
            return 0
        try:
            spec_module = importlib.import_module(spec_module_path)
            spec_cls = getattr(spec_module, spec_name)
        except (ImportError, AttributeError):
            return 0
        estimator = getattr(spec_cls, "estimate_hbm_reserve_bytes", None)
        if estimator is None:
            return 0
        return estimator(self.vllm_config)

    def determine_available_memory(self) -> int:
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
        kv_connector_hbm_reserve = self._estimate_kv_connector_hbm_reserve()
        available = budget.available - kv_connector_hbm_reserve

        total_hbm_limit_gb = round(budget.total_limit / utils.GBYTES, 2)
        total_hbm_limit_cap_gb = round(budget.cap / utils.GBYTES, 2)
        total_hbm_used_gb = round(budget.total_used / utils.GBYTES, 2)
        kv_cache_headroom_gb = round(budget.headroom / utils.GBYTES, 2)
        kv_connector_hbm_reserve_gb = round(
            kv_connector_hbm_reserve / utils.GBYTES, 2)
        total_hbm_avail_gb = round(available / utils.GBYTES, 2)
        logger.info(f"Memory statistics | "
                    f"{total_hbm_limit_gb=}GiB | "
                    f"{total_hbm_limit_cap_gb=}GiB | "
                    f"{total_hbm_used_gb=}GiB | "
                    f"{kv_cache_headroom_gb=}GiB | "
                    f"{kv_connector_hbm_reserve_gb=}GiB | "
                    f"{total_hbm_avail_gb=}GiB")

        if available <= 0:
            raise ValueError(f"{total_hbm_used_gb=}GiB exceeds "
                             f"{total_hbm_limit_cap_gb=}GiB by "
                             f"{-total_hbm_avail_gb}GiB. Please consider "
                             f"increasing --gpu-memory-utilization from "
                             f"{gpu_memory_utilization} to a larger value, "
                             "or decreasing TPU_KV_CACHE_HEADROOM_MIB if "
                             "this run has a known smaller TPU runtime "
                             "headroom requirement.")
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
        for _ in range(target_num_chunks):
            runner._run_dp_dummy_chunk(bucket)

    def profile(self,
                is_start: bool = True,
                profile_prefix: str | None = None):
        if self.profile_dir is None:
            logger.warning("Profile directory is not set. Skipping profiling.")
            return

        profile_dir = self.profile_dir
        if profile_prefix:
            profile_dir = os.path.join(profile_dir, profile_prefix)
            os.makedirs(profile_dir, exist_ok=True)

        if is_start:
            logger.info(
                f"Starting TorchTPU profiler trace at {profile_dir}...")
            handler = profiler_api.xprof_trace_handler(dir_name=profile_dir)
            self.profile_context = profiler_api.profile(activities=[
                profiler_api.ProfilerActivity.CPU,
                profiler_api.ProfilerActivity.TPU
            ],
                                                        on_trace_ready=handler)
            self.profile_context.__enter__()
        else:
            if self.profile_context is not None:
                logger.info("Stopping TorchTPU profiler trace...")
                self.profile_context.__exit__(None, None, None)
                logger.info(f"Profiler trace saved to {profile_dir}")
            else:
                logger.warning(
                    "Profiler context is not set. Cannot stop profiler.")

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        self.model_runner.load_model()

    def compile_or_warm_up_model(self) -> CompilationTimes:
        start = time.perf_counter()
        self.model_runner.capture_model()
        compilation_time = time.perf_counter() - start
        self.compilation_config.compilation_time = compilation_time
        return CompilationTimes(language_model=compilation_time, encoder=0.0)

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
        return {_get_kv_connector_handshake_metadata_key(): metadata}
