# SPDX-License-Identifier: Apache-2.0
"""TpuMultiprocExecutor: MultiprocExecutor with per-shape KV offload prewarm.

Splits H2D Pallas kernel prewarm into one collective_rpc call per power-of-2
shape so each shape gets its own 60-second shm_broadcast window (~4s per shape)
instead of all 14 shapes sharing a single window and timing out.
"""

from vllm.logger import init_logger
from vllm.v1.executor.multiproc_executor import MultiprocExecutor
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec

logger = init_logger(__name__)

# Workers size the KV pool from their OWN measured free HBM, so their block
# counts never agree exactly. The spread that matters is relative: a few tens of
# MiB of allocator jitter is a couple of blocks when a block is megabytes (Kimi
# K3: ~14 MiB/block) but hundreds of blocks when a block is ~140 KiB
# (Kimi-Linear-48B, where 8 workers landed 551348..551506 -- a spread of 158
# against this absolute tolerance of 4). Accept whichever is larger, so the
# check still catches a worker that genuinely disagrees about the budget.
_NUM_BLOCKS_OVERRIDE_TOL = 4
_NUM_BLOCKS_OVERRIDE_REL_TOL = 0.005


class TpuMultiprocExecutor(MultiprocExecutor):

    def _init_executor(self) -> None:
        # EngineCore does not run platform patch setup before spawning workers.
        from vllm_torchtpu import _patch_multiproc_worker_global_rank_env
        _patch_multiproc_worker_global_rank_env()
        super()._init_executor()

    def get_kv_cache_specs(self) -> list[dict[str, KVCacheSpec]]:
        specs = super().get_kv_cache_specs()

        if self.vllm_config.cache_config.num_gpu_blocks_override is None:
            # Compact-mamba sizing sets `num_gpu_blocks_override` on the worker's
            # cache_config during the RPC above; workers are separate processes, so
            # copy it to the engine-side config here.
            overrides = [
                ovr
                for ovr in self.collective_rpc("get_num_gpu_blocks_override")
                if ovr is not None
            ]
            if overrides:
                lo, hi = min(overrides), max(overrides)
                tol = max(_NUM_BLOCKS_OVERRIDE_TOL,
                          int(lo * _NUM_BLOCKS_OVERRIDE_REL_TOL))
                if hi - lo > tol:
                    raise ValueError(
                        "[kv-sizing] workers disagree on the compact-mamba "
                        f"attention block count by {hi - lo} blocks "
                        f"(min={lo}, max={hi}, tolerance={tol}). Each worker "
                        "sizes the pool from its own measured free HBM, so a "
                        "spread this large means the workers do not have "
                        "comparable HBM budgets -- check the per-worker "
                        "'Compact-mamba KV cache:' lines in "
                        "tpu_runner._maybe_set_compact_mamba_num_blocks_"
                        f"override for the outlier. Per-worker values: "
                        f"{sorted(overrides)}")
                logger.info(
                    "[kv-sizing] compact-mamba attention blocks across %d "
                    "workers: min=%d max=%d spread=%d (tolerance=%d); "
                    "pinning num_gpu_blocks_override=%d", len(overrides), lo,
                    hi, hi - lo, tol, lo)
                self.vllm_config.cache_config.num_gpu_blocks_override = lo

        return specs

    def initialize_from_config(self,
                               kv_cache_configs: list[KVCacheConfig]) -> None:
        super().initialize_from_config(kv_cache_configs)
        # Per-shape H2D prewarm: each shape is its own collective_rpc call with
        # its own 60-second shm_broadcast window.  Workers that have no KV
        # offload spec return [] from get_kv_prewarm_shapes() — a no-op.
        shapes = self.collective_rpc("get_kv_prewarm_shapes")
        if not shapes or not shapes[0]:
            return
        for p in shapes[0]:
            self.collective_rpc("prewarm_kv_offload_shape", args=(p, ))
