import ctypes
import gc
from typing import Any, Callable, Optional

import torch
from torch.nn import Parameter


def get_cpu_weight_loader_hook(layer: torch.nn.Module,
                               orig_loader: Optional[Callable],
                               tp_size: int,
                               tp_rank: int,
                               is_param_transposed: bool = True) -> Callable:
    """
    Returns a custom weight loader hook for TPU to intercept specific Compressed Tensors
    MoE weights and load them into a CPU scratchpad first, preventing TPU fragmentation.

    If `is_param_transposed` is True, the layer's parameter is assumed to be in
    [E, K, N] format, and the untransposed scratchpad will be [E, N, K].
    """

    def _cpu_weight_loader_hook(param: torch.nn.Parameter,
                                loaded_weight: torch.Tensor, weight_name: str,
                                shard_id: str, expert_id: int, *args: Any,
                                **kwargs: Any) -> bool:
        local_expert_id = layer._map_global_expert_id_to_local_expert_id(
            expert_id)
        if local_expert_id == -1:
            return False
        expert_id = local_expert_id

        is_main_weight = ("w13_weight_packed" in weight_name
                          or "w2_weight_packed" in weight_name
                          or "w13_weight_scale" in weight_name
                          or "w2_weight_scale" in weight_name)

        # Special unpacking operations are not needed for other weights, so we can
        # just use the original weight loader.
        if not is_main_weight:
            if orig_loader is not None:
                return orig_loader(param, loaded_weight, weight_name, shard_id,
                                   expert_id, *args, **kwargs)
            return False

        if not hasattr(param, "_cpu_scratch"):
            if is_param_transposed:
                untransposed_shape = (param.shape[0], param.shape[2],
                                      param.shape[1])
            else:
                untransposed_shape = param.shape

            param._cpu_scratch = Parameter(torch.empty(untransposed_shape,
                                                       dtype=param.dtype,
                                                       device="cpu"),
                                           requires_grad=False)

        cpu_scratch = param._cpu_scratch

        if "w13_weight_packed" in weight_name:
            # Shard along dim 0
            loaded_per_rank = loaded_weight.shape[0] // tp_size
            start = loaded_per_rank * tp_rank
            weight_shard = loaded_weight[start:start + loaded_per_rank]

            shard_size = cpu_scratch.shape[1] // 2
            offset = 0 if shard_id == "w1" else shard_size
            cpu_scratch.data[expert_id, offset:offset +
                             weight_shard.shape[0], :].copy_(weight_shard)
            return True

        elif "w2_weight_packed" in weight_name:
            # Shard along dim 1
            loaded_per_rank = loaded_weight.shape[1] // tp_size
            start = loaded_per_rank * tp_rank
            weight_shard = loaded_weight[:, start:start + loaded_per_rank]

            cpu_scratch.data[expert_id].copy_(weight_shard)
            return True

        elif "w13_weight_scale" in weight_name:
            # Shard along dim 0
            loaded_per_rank = loaded_weight.shape[0] // tp_size
            start = loaded_per_rank * tp_rank
            weight_shard = loaded_weight[start:start + loaded_per_rank, :]

            shard_size = cpu_scratch.shape[1] // 2
            offset = 0 if shard_id == "w1" else shard_size
            cpu_scratch.data[expert_id, offset:offset +
                             weight_shard.shape[0], :].copy_(weight_shard)
            return True

        elif "w2_weight_scale" in weight_name:
            # Shard along dim 1
            loaded_per_rank = loaded_weight.shape[1] // tp_size
            start = loaded_per_rank * tp_rank
            weight_shard = loaded_weight[:, start:start + loaded_per_rank]

            cpu_scratch.data[expert_id].copy_(weight_shard)
            return True

        # Fallback to default loader (e.g., for helper tensors)
        if orig_loader is not None:
            return orig_loader(cpu_scratch, loaded_weight, weight_name,
                               shard_id, expert_id, *args, **kwargs)
        return False

    return _cpu_weight_loader_hook


def release_memory_to_os() -> None:
    """Forces Python GC and instructs the C allocator to return free memory to the OS."""
    gc.collect()
    try:
        ctypes.CDLL(None).malloc_trim(0)
    except Exception:
        pass
