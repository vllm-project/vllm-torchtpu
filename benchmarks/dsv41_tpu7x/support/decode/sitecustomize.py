import os, sys
for _k, _v in {
    "TPU_MOE_SKIP_PADDED_TOKENS": "0",
    "VLLM_DSV41_CED": "0",
    "VLLM_DSV41_HYBRID_CED_DECODE": "0",
    "VLLM_DSV41_MOE_EP_V2_ALL39": "0",
    "VLLM_DSV41_TP_SHARD_ENGRAM": "0",
    "VLLM_DSV41_CSA_UNION": "0",
    "VLLM_DSV41_CSA_UNION_MAX_TOKENS": "0",
    "ENABLE_CONNECTOR": "1",
    "ENABLE_DECODE_BENCH_CONNECTOR": "1",
}.items():
    os.environ.setdefault(_k, _v)
import os, sys
os.environ.setdefault("VLLM_DSV41_CSA_UNION", "0")
os.environ.setdefault("VLLM_TPU_FAST_TOKEN_SUBSTITUTION", "1")
os.environ.setdefault("VLLM_DSV41_ENGRAM_H2D_PREFIX", "0")
os.environ.setdefault(
    "LIBTPU_INIT_ARGS",
    "--xla_tpu_enable_diagnostic_coordinator=false --xla_tpu_no_crash_on_oom=false --xla_tpu_scheduler_percent_shared_memory_limit=100 --xla_tpu_enable_scheduler_memory_pressure_tracking=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_sparse_core_compute_offload=false --xla_tpu_offload_gather_to_sparsecore=false --xla_tpu_enable_offloading_gather_to_sparsecore=false --xla_tpu_offload_all_supported_gathers_to_sparsecore=false --xla_tpu_enable_offloading_scatter_to_sparsecore=false --xla_tpu_enable_offloading_sort_to_sparsecore=false --xla_tpu_enable_offloading_copy_to_sparsecore=false --xla_tpu_enable_offloading_reshape_to_sparsecore=false --xla_tpu_enable_offloading_reduce_to_sparsecore=false --xla_tpu_num_sparse_cores_for_gather_offloading=0 --xla_tpu_num_sparse_cores_for_scatter_offloading=0 --xla_tpu_num_sparse_cores_for_sort_offloading=0 --xla_tpu_enable_sparse_core_collective_offload_all_gather=false --xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=false --xla_tpu_enable_sparse_core_collective_offload_3d_all_gather=false --xla_tpu_enable_sparse_core_collective_offload_all_reduce=false --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_enable_sparse_core_collective_offload_nd_reduce_scatter=false --xla_tpu_enable_sparse_core_collective_offload_dense_all_to_all=false --xla_tpu_enable_sparse_core_collective_offload_ragged_all_to_all=false --xla_tpu_enable_sparse_core_async_all_to_all=false --xla_tpu_use_single_sparse_core_for_reduce_scatter_offload=false",
)
for _p in ("/workspace/vllm-torchtpu/src", "/workspace/vllm"):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

if os.environ.get("VLLM_LOG_POST_WARMUP_HBM") == "1":
    import importlib.abc
    class _HbmLogFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname == "vllm_torchtpu.worker.tpu_worker":
                sys.meta_path.remove(self)
                for finder in sys.meta_path:
                    spec = finder.find_spec(fullname, path, target)
                    if spec is not None and spec.loader is not None:
                        orig_exec = spec.loader.exec_module
                        def _exec_mod(module):
                            orig_exec(module)
                            orig_cw = module.TPUWorker.compile_or_warm_up_model
                            def _wrapped_cw(self, *a, **kw):
                                res = orig_cw(self, *a, **kw)
                                try:
                                    import torch
                                    free_b, total_b = torch.accelerator.get_memory_info()
                                    used_b = total_b - free_b
                                    print(
                                        f"[POST_WARMUP_HBM] rank={self.rank} "
                                        f"free_bytes={free_b} total_bytes={total_b} "
                                        f"free_gib={free_b / (1024**3):.3f} "
                                        f"used_gib={used_b / (1024**3):.3f} "
                                        f"total_gib={total_b / (1024**3):.3f}",
                                        flush=True,
                                    )
                                except Exception as e:
                                    print(f"[POST_WARMUP_HBM] rank={getattr(self, 'rank', -1)} err={e}", flush=True)
                                return res
                            module.TPUWorker.compile_or_warm_up_model = _wrapped_cw
                        spec.loader.exec_module = _exec_mod
                        return spec
            return None
    sys.meta_path.insert(0, _HbmLogFinder())
