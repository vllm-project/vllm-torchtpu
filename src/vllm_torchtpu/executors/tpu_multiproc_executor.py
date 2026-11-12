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
"""TpuMultiprocExecutor: MultiprocExecutor with per-shape KV offload prewarm.

Splits H2D Pallas kernel prewarm into one collective_rpc call per power-of-2
shape so each shape gets its own 60-second shm_broadcast window (~4s per shape)
instead of all 14 shapes sharing a single window and timing out.
"""

from vllm.v1.executor.multiproc_executor import MultiprocExecutor
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec

from vllm_torchtpu.executors.kv_block_override import (
    adopt_worker_num_gpu_blocks_override,
)


class TpuMultiprocExecutor(MultiprocExecutor):
    def _init_executor(self) -> None:
        # EngineCore does not run platform patch setup before spawning workers.
        from vllm_torchtpu import _patch_multiproc_worker_global_rank_env

        _patch_multiproc_worker_global_rank_env()
        super()._init_executor()

    def _can_start_workers_concurrently(self, inherited_fds: list[int] | None) -> bool:
        # The rank-binding patch writes each worker's binding into os.environ
        # right before spawning it, so the starts must not overlap.
        return False

    def get_kv_cache_specs(self) -> list[dict[str, KVCacheSpec]]:
        specs = super().get_kv_cache_specs()
        adopt_worker_num_gpu_blocks_override(self)
        return specs

    def determine_available_memory(self) -> list[int]:
        available = super().determine_available_memory()
        adopt_worker_num_gpu_blocks_override(self)
        return available

    def initialize_from_config(self, kv_cache_configs: list[KVCacheConfig]) -> None:
        super().initialize_from_config(kv_cache_configs)
        # Per-shape H2D prewarm: each shape is its own collective_rpc call with
        # its own 60-second shm_broadcast window.  Workers that have no KV
        # offload spec return [] from get_kv_prewarm_shapes() — a no-op.
        shapes = self.collective_rpc("get_kv_prewarm_shapes")
        if not shapes or not shapes[0]:
            return
        for p in shapes[0]:
            self.collective_rpc("prewarm_kv_offload_shape", args=(p,))
