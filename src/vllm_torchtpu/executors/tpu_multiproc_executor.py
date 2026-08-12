# SPDX-License-Identifier: Apache-2.0
"""TpuMultiprocExecutor: MultiprocExecutor with per-shape KV offload prewarm.

Splits H2D Pallas kernel prewarm into one collective_rpc call per power-of-2
shape so each shape gets its own 60-second shm_broadcast window (~4s per shape)
instead of all 14 shapes sharing a single window and timing out.
"""

from vllm.v1.executor.multiproc_executor import MultiprocExecutor
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec

from vllm_torchtpu.executors.kv_block_override import \
    reconcile_num_gpu_blocks_override


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
            agreed = reconcile_num_gpu_blocks_override(
                self.collective_rpc("get_num_gpu_blocks_override"))
            if agreed is not None:
                self.vllm_config.cache_config.num_gpu_blocks_override = agreed

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
