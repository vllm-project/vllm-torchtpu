# SPDX-License-Identifier: Apache-2.0
"""TpuMultiprocExecutor: MultiprocExecutor with per-shape KV offload prewarm.

Splits H2D Pallas kernel prewarm into one collective_rpc call per power-of-2
shape so each shape gets its own 60-second shm_broadcast window (~4s per shape)
instead of all 14 shapes sharing a single window and timing out.
"""

import os

from vllm.logger import init_logger
from vllm.v1.executor.multiproc_executor import MultiprocExecutor
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec

from vllm_torchtpu.executors.kv_block_override import (
    adopt_worker_num_gpu_blocks_override,
)

logger = init_logger(__name__)


def _patch_spawn_pipe_size() -> None:
    """Let ``spawn`` start every worker without waiting for the previous one.

    ``multiprocessing``'s spawn launcher writes the pickled process object
    (here the whole VllmConfig) into a pipe the child reads only after
    importing the parent's ``__main__`` (the ``vllm`` entry point, ~6 s).
    Past the 64 KiB default pipe buffer that write blocks, so ``proc.start()``
    returns only once the child has imported, and the executor's spawn loop
    starts the workers ~6 s apart: ~48 s for 8 workers before the last one even
    begins TPU init. Growing the launcher's pipes (to the 1 MiB unprivileged
    limit) lets the write complete at once, so the workers import in parallel.
    Each worker still gets its own RANK/LOCAL_RANK env: the spawn itself, which
    snapshots the env, stays in the loop's order.
    """
    import fcntl
    import os
    from multiprocessing import popen_spawn_posix

    if getattr(popen_spawn_posix, "_tpu_pipe_patch", False):
        return
    F_SETPIPE_SZ = getattr(fcntl, "F_SETPIPE_SZ", 1031)

    class _OsWithBigPipes:
        def __getattr__(self, name):
            return getattr(os, name)

        @staticmethod
        def pipe():
            r, w = os.pipe()
            for size in (1 << 20, 1 << 19, 1 << 18):
                try:
                    fcntl.fcntl(w, F_SETPIPE_SZ, size)
                    break
                except OSError:
                    continue
            return r, w

    popen_spawn_posix.os = _OsWithBigPipes()
    popen_spawn_posix._tpu_pipe_patch = True

    from multiprocessing import reduction

    orig_dump = reduction.dump

    def dump(obj, file, protocol=None):
        orig_dump(obj, file, protocol)
        if type(obj).__name__.endswith("Process"):
            try:
                logger.info(
                    "Parallel worker spawn: %s payload %d bytes so far",
                    getattr(obj, "name", "?"),
                    file.tell(),
                )
            except Exception:  # noqa: BLE001 - logging only
                pass

    reduction.dump = dump


class TpuMultiprocExecutor(MultiprocExecutor):
    def _init_executor(self) -> None:
        # EngineCore does not run platform patch setup before spawning workers.
        from vllm_torchtpu import _patch_multiproc_worker_global_rank_env

        _patch_multiproc_worker_global_rank_env()
        if os.environ.get("VLLM_TPU_PARALLEL_WORKER_SPAWN", "0") in ("1", "true"):
            _patch_spawn_pipe_size()
        super()._init_executor()

    def get_kv_cache_specs(self) -> list[dict[str, KVCacheSpec]]:
        specs = super().get_kv_cache_specs()
        adopt_worker_num_gpu_blocks_override(self)
        return specs

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
