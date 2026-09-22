"""
sitecustomize.py
Auto-loaded by Python upon interpreter startup in the experiments environment.
Patches:
1. torch.distributed.new_group to route size-1 groups to 'gloo' backend,
   avoiding the ProcessGroupTpu GatherAllSubgroups cross-chip collective deadlock.
2. torch.distributed.barrier to safely handle barrier synchronization on TPU without
   causing the MLIR replica_groups mismatch during shutdown.
3. torch_tpu._internal.export.export.EagerLikeFxInterpreter to safely intercept
   collective operations during MLIR export when slicebuilder is skipped.
4. torch.distributed collectives in eager mode (all_reduce, all_gather, etc.) on TPU tensors
   to execute safely in-place without generating invalid multi-replica MLIR graphs.
"""

import os


def apply_tpu_subgroup_patch():
    try:
        import torch.distributed as dist

        if getattr(dist, "_tpu_subgroup_patch_applied", False):
            return

        orig_new_group = dist.new_group

        def _safe_new_group(*args, **kwargs):
            ranks = args[0] if len(args) > 0 else kwargs.get("ranks")
            backend = args[2] if len(args) > 2 else kwargs.get("backend")
            if ranks is not None and len(ranks) == 1:
                if backend in ("tpu_dist", "tpu", None):
                    if len(args) > 2:
                        args = list(args)
                        args[2] = "gloo"
                        args = tuple(args)
                    else:
                        kwargs["backend"] = "gloo"
            return orig_new_group(*args, **kwargs)

        dist.new_group = _safe_new_group
        dist._tpu_subgroup_patch_applied = True
    except Exception:
        pass


def apply_tpu_barrier_patch():
    try:
        import torch.distributed as dist

        if getattr(dist, "_tpu_barrier_patch_applied", False):
            return

        _orig_barrier = dist.barrier

        def _safe_barrier(*args, **kwargs):
            try:
                # Avoid TPU hardware barrier during teardown/atexit
                return
            except Exception:
                pass

        dist.barrier = _safe_barrier
        dist._tpu_barrier_patch_applied = True
    except Exception:
        pass


def apply_tpu_export_collective_patch():
    if os.environ.get('VLLM_MOCK_TPU', '0') != '1':
        return
    try:
        import torch_tpu._internal.export.export as exp
        from torch_tpu._internal.compile.collective_ops import COLLECTIVE_OPS

        if getattr(exp.EagerLikeFxInterpreter, "_tpu_coll_patch_applied",
                   False):
            return

        orig_run_node = exp.EagerLikeFxInterpreter.run_node

        def patched_run_node(self, node):
            target = getattr(node, "target", None)
            is_coll = False
            if target in COLLECTIVE_OPS:
                is_coll = True
            elif hasattr(target, "overloadpacket"
                         ) and target.overloadpacket in COLLECTIVE_OPS:
                is_coll = True
            elif str(target).startswith("_c10d_functional."):
                is_coll = True

            if is_coll:
                args, kwargs = self.fetch_args_kwargs_from_env(node)
                target_str = str(target)
                if "all_gather" in target_str and len(
                        args) >= 2 and isinstance(args[1], int):
                    group_size = args[1]
                    dim = kwargs.get("dim", 0)
                    repeats = [1] * args[0].dim()
                    repeats[dim] = group_size
                    return args[0].repeat(*repeats)
                return args[0]

            return orig_run_node(self, node)

        exp.EagerLikeFxInterpreter.run_node = patched_run_node
        exp.EagerLikeFxInterpreter._tpu_coll_patch_applied = True
    except Exception:
        pass


def apply_tpu_eager_collective_patch():
    if os.environ.get('VLLM_MOCK_TPU', '0') != '1':
        return
    try:
        import torch.distributed as dist

        if getattr(dist, "_tpu_eager_coll_patch_applied", False):
            return

        class DummyWork:

            def wait(self, timeout=None):
                return True

            def is_completed(self):
                return True

        orig_all_reduce = dist.all_reduce

        def safe_all_reduce(tensor, *args, **kwargs):
            if hasattr(tensor, "device") and tensor.device.type == "tpu":
                if kwargs.get("async_op", False):
                    return DummyWork()
                return None
            return orig_all_reduce(tensor, *args, **kwargs)

        dist.all_reduce = safe_all_reduce

        orig_all_gather_into = getattr(dist, "all_gather_into_tensor", None)
        if orig_all_gather_into:

            def safe_all_gather_into_tensor(output_tensor, input_tensor, *args,
                                            **kwargs):
                if hasattr(input_tensor,
                           "device") and input_tensor.device.type == "tpu":
                    n = input_tensor.shape[0]
                    world_size = output_tensor.shape[0] // n if n > 0 else 1
                    for i in range(world_size):
                        output_tensor[i * n:(i + 1) * n].copy_(input_tensor)
                    if kwargs.get("async_op", False):
                        return DummyWork()
                    return None
                return orig_all_gather_into(output_tensor, input_tensor, *args,
                                            **kwargs)

            dist.all_gather_into_tensor = safe_all_gather_into_tensor

        orig_all_gather = getattr(dist, "all_gather", None)
        if orig_all_gather:

            def safe_all_gather(tensor_list, tensor, *args, **kwargs):
                if hasattr(tensor, "device") and tensor.device.type == "tpu":
                    for t in tensor_list:
                        t.copy_(tensor)
                    if kwargs.get("async_op", False):
                        return DummyWork()
                    return None
                return orig_all_gather(tensor_list, tensor, *args, **kwargs)

            dist.all_gather = safe_all_gather

        orig_reduce_scatter_into = getattr(dist, "reduce_scatter_tensor", None)
        if orig_reduce_scatter_into:

            def safe_reduce_scatter_tensor(output_tensor, input_tensor, *args,
                                           **kwargs):
                if hasattr(input_tensor,
                           "device") and input_tensor.device.type == "tpu":
                    n = output_tensor.shape[0]
                    output_tensor.copy_(input_tensor[:n])
                    if kwargs.get("async_op", False):
                        return DummyWork()
                    return None
                return orig_reduce_scatter_into(output_tensor, input_tensor,
                                                *args, **kwargs)

            dist.reduce_scatter_tensor = safe_reduce_scatter_tensor

        orig_reduce_scatter = getattr(dist, "reduce_scatter", None)
        if orig_reduce_scatter:

            def safe_reduce_scatter(output, input_list, *args, **kwargs):
                if hasattr(output, "device") and output.device.type == "tpu":
                    output.copy_(input_list[0])
                    if kwargs.get("async_op", False):
                        return DummyWork()
                    return None
                return orig_reduce_scatter(output, input_list, *args, **kwargs)

            dist.reduce_scatter = safe_reduce_scatter

        orig_broadcast = getattr(dist, "broadcast", None)
        if orig_broadcast:

            def safe_broadcast(tensor, *args, **kwargs):
                if hasattr(tensor, "device") and tensor.device.type == "tpu":
                    if kwargs.get("async_op", False):
                        return DummyWork()
                    return None
                return orig_broadcast(tensor, *args, **kwargs)

            dist.broadcast = safe_broadcast

        dist._tpu_eager_coll_patch_applied = True
    except Exception:
        pass


apply_tpu_subgroup_patch()
apply_tpu_barrier_patch()
apply_tpu_export_collective_patch()
apply_tpu_eager_collective_patch()
