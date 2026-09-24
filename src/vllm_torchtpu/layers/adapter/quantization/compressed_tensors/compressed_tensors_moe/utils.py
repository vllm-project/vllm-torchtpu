import ctypes
import gc
from collections.abc import Callable
from typing import Any

import torch
from torch.nn import Parameter


def get_cpu_weight_loader_hook(
    layer: torch.nn.Module,
    orig_loader: Callable | None,
    tp_size: int,
    tp_rank: int,
    is_param_transposed: bool = True,
) -> Callable:
    """
    Returns a custom weight loader hook for TPU to intercept specific Compressed Tensors
    MoE weights and load them into a CPU scratchpad first, preventing TPU fragmentation.

    If `is_param_transposed` is True, the layer's parameter is assumed to be in
    [E, K, N] format, and the untransposed scratchpad will be [E, N, K].
    """

    def _cpu_weight_loader_hook(
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
        *args: Any,
        **kwargs: Any,
    ) -> bool:
        local_expert_id = layer._map_global_expert_id_to_local_expert_id(expert_id)
        if local_expert_id == -1:
            return False
        expert_id = local_expert_id

        is_main_weight = (
            "w13_weight_packed" in weight_name
            or "w2_weight_packed" in weight_name
            or "w13_weight_scale" in weight_name
            or "w2_weight_scale" in weight_name
        )

        # Special unpacking operations are not needed for other weights, so we can
        # just use the original weight loader.
        if not is_main_weight:
            if orig_loader is not None:
                return orig_loader(
                    param,
                    loaded_weight,
                    weight_name,
                    shard_id,
                    expert_id,
                    *args,
                    **kwargs,
                )
            return False

        if not hasattr(param, "_cpu_scratch"):
            shape = getattr(param, "_orig_shape", param.shape)
            if is_param_transposed:
                untransposed_shape = (shape[0], shape[2], shape[1])
            else:
                untransposed_shape = shape

            param._cpu_scratch = Parameter(
                torch.empty(untransposed_shape, dtype=param.dtype, device="cpu"),
                requires_grad=False,
            )

        cpu_scratch = param._cpu_scratch

        if "w13_weight_packed" in weight_name:
            # Shard along dim 0
            loaded_per_rank = loaded_weight.shape[0] // tp_size
            start = loaded_per_rank * tp_rank
            weight_shard = loaded_weight[start : start + loaded_per_rank]

            shard_size = cpu_scratch.shape[1] // 2
            offset = 0 if shard_id == "w1" else shard_size
            cpu_scratch.data[
                expert_id, offset : offset + weight_shard.shape[0], :
            ].copy_(weight_shard)
            return True

        elif "w2_weight_packed" in weight_name:
            # Shard along dim 1
            loaded_per_rank = loaded_weight.shape[1] // tp_size
            start = loaded_per_rank * tp_rank
            weight_shard = loaded_weight[:, start : start + loaded_per_rank]

            cpu_scratch.data[expert_id].copy_(weight_shard)
            return True

        elif "w13_weight_scale" in weight_name:
            # Shard along dim 0
            loaded_per_rank = loaded_weight.shape[0] // tp_size
            start = loaded_per_rank * tp_rank
            weight_shard = loaded_weight[start : start + loaded_per_rank, :]

            shard_size = cpu_scratch.shape[1] // 2
            offset = 0 if shard_id == "w1" else shard_size
            cpu_scratch.data[
                expert_id, offset : offset + weight_shard.shape[0], :
            ].copy_(weight_shard)
            return True

        elif "w2_weight_scale" in weight_name:
            # Shard along dim 1
            loaded_per_rank = loaded_weight.shape[1] // tp_size
            start = loaded_per_rank * tp_rank
            weight_shard = loaded_weight[:, start : start + loaded_per_rank]

            cpu_scratch.data[expert_id].copy_(weight_shard)
            return True

        # Fallback to default loader (e.g., for helper tensors)
        if orig_loader is not None:
            return orig_loader(
                cpu_scratch,
                loaded_weight,
                weight_name,
                shard_id,
                expert_id,
                *args,
                **kwargs,
            )
        return False

    return _cpu_weight_loader_hook


def release_memory_to_os() -> None:
    """Forces Python GC and instructs the C allocator to return free memory to the OS."""
    gc.collect()
    try:
        ctypes.CDLL(None).malloc_trim(0)
    except Exception:
        pass


def quantize_to_fp4_e2m1_nibbles(tensor_scaled: torch.Tensor) -> torch.Tensor:
    """Vectorized projection of scaled float values into 4-bit FP4 E2M1 indices (0..15)."""
    abs_x = tensor_scaled.abs()
    thresholds = torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
        dtype=tensor_scaled.dtype,
        device=tensor_scaled.device,
    )
    mag = torch.bucketize(abs_x, thresholds).to(torch.uint8)
    sign = torch.where(
        tensor_scaled < 0,
        torch.tensor(8, dtype=torch.uint8, device=tensor_scaled.device),
        torch.tensor(0, dtype=torch.uint8, device=tensor_scaled.device),
    )
    return mag | sign


def requantize_int4_weights(
    w_packed: torch.Tensor,
    scale_f: torch.Tensor,
    requant_block_size: int,
    in_group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    INT4 MoE block-wise requantization on device:
    [..., N, K/8] -> [..., N, K/8] for weights
    [..., N, K/in_group] -> [..., N, K/requant_block] for scales
    Executes directly on device (TPU / CUDA / CPU) with vectorized PyTorch operations.
    """
    orig_scale_dtype = scale_f.dtype
    device = w_packed.device
    out_dim_prefix = w_packed.shape[:-1]
    k_div_8 = w_packed.shape[-1]
    size_k = k_div_8 * 8
    num_in_blocks = scale_f.shape[-1]
    in_block_size = size_k // num_in_blocks

    effective_requant_block = min(requant_block_size, size_k)

    groups_per_block = effective_requant_block // in_block_size
    num_out_blocks = size_k // effective_requant_block

    # 1. Vectorized Unpack to Signed INT4 [-8..7]
    shifts = torch.tensor(
        [0, 4, 8, 12, 16, 20, 24, 28], dtype=torch.int32, device=device
    )
    p_reshaped = w_packed.reshape(*out_dim_prefix, num_in_blocks, in_block_size // 8, 1)
    w_u4 = ((p_reshaped >> shifts) & 0xF).to(torch.int8)
    w_s4 = (w_u4 - 8).reshape(
        *out_dim_prefix, num_out_blocks, groups_per_block, in_block_size
    )

    # 2. Real float max per in_block and out_block
    max_abs_per_in_block = w_s4.abs().amax(dim=-1).to(torch.float32)
    scale_f_blocked = scale_f.reshape(
        *out_dim_prefix, num_out_blocks, groups_per_block
    ).to(torch.float32)
    real_max_per_in_block = max_abs_per_in_block * scale_f_blocked
    abs_max = real_max_per_in_block.amax(dim=-1, keepdim=True)

    new_scale = abs_max / 7.0
    safe_scale_inv = torch.where(
        new_scale == 0, torch.zeros_like(new_scale), 1.0 / new_scale
    )

    # 3. Multiplier per in_block and requantize
    mult = (scale_f_blocked * safe_scale_inv).unsqueeze(-1)
    w_s4_new = torch.clamp(torch.round(w_s4.to(torch.float32) * mult), -8, 7).to(
        torch.int32
    )
    w_u4_new = (w_s4_new + 8) & 0xF

    # 4. Fast Pack into int32: [..., N, K_div_8]
    w_quant = w_u4_new.reshape(*out_dim_prefix, k_div_8, 8)
    packed_out = (w_quant << shifts).sum(dim=-1, dtype=torch.int32)

    return packed_out, new_scale.squeeze(-1).to(orig_scale_dtype)


def requantize_int4_to_fp4_weights(
    w_packed: torch.Tensor,
    scale_f: torch.Tensor,
    requant_block_size: int,
    in_group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Requantizes INT4 weights [..., N, K/8] to FP4 (E2M1) packed into torch.float8_e4m3fn [..., N, K/2].
    [..., N, K/8] -> [..., N, K/2] for weights (2 FP4 per byte)
    [..., N, K/in_group] -> [..., N, K/requant_block] for scales
    """
    # If tensor contains multiple items (e.g. experts along dim 0), process per-expert to keep peak memory minimal
    if w_packed.dim() >= 3 and w_packed.shape[0] > 1:
        num_items = w_packed.shape[0]
        out_packed_list = []
        out_scale_list = []
        for i in range(num_items):
            p_i, s_i = requantize_int4_to_fp4_weights(
                w_packed[i], scale_f[i], requant_block_size, in_group_size
            )
            out_packed_list.append(p_i)
            out_scale_list.append(s_i)
        return torch.stack(out_packed_list, dim=0), torch.stack(out_scale_list, dim=0)

    orig_scale_dtype = scale_f.dtype
    device = w_packed.device
    out_dim_prefix = w_packed.shape[:-1]
    k_div_8 = w_packed.shape[-1]
    size_k = k_div_8 * 8
    num_in_blocks = scale_f.shape[-1]
    in_block_size = size_k // num_in_blocks

    effective_requant_block = min(requant_block_size, size_k)
    groups_per_block = max(1, effective_requant_block // in_block_size)
    num_out_blocks = size_k // effective_requant_block

    # 1. Vectorized Unpack to Signed INT4 [-8..7]
    shifts = torch.tensor(
        [0, 4, 8, 12, 16, 20, 24, 28], dtype=torch.int32, device=device
    )
    p_reshaped = w_packed.reshape(*out_dim_prefix, num_in_blocks, in_block_size // 8, 1)
    w_u4 = ((p_reshaped >> shifts) & 0xF).to(torch.int8)
    w_s4 = (w_u4 - 8).reshape(
        *out_dim_prefix, num_out_blocks, groups_per_block, in_block_size
    )

    # 2. Dequantize to float
    scale_f_blocked = scale_f.reshape(
        *out_dim_prefix, num_out_blocks, groups_per_block
    ).to(torch.float32)
    w_float = (w_s4.to(torch.float32) * scale_f_blocked.unsqueeze(-1)).reshape(
        *out_dim_prefix, num_out_blocks, effective_requant_block
    )

    # 3. Compute FP4 Scale per block (max magnitude / 6.0)
    abs_max = w_float.abs().amax(dim=-1, keepdim=True)
    fp4_scale = abs_max / 6.0
    safe_scale_inv = torch.where(
        fp4_scale == 0, torch.zeros_like(fp4_scale), 1.0 / fp4_scale
    )

    # 4. Quantize to FP4 E2M1 nibbles (0..15)
    w_scaled = w_float * safe_scale_inv
    nibbles = quantize_to_fp4_e2m1_nibbles(w_scaled).reshape(
        *out_dim_prefix, size_k // 2, 2
    )

    # 5. Pack 2 FP4 nibbles into 1 byte (torch.float8_e4m3fn carrier)
    packed_u8 = (nibbles[..., 0] & 0x0F) | ((nibbles[..., 1] & 0x0F) << 4)
    packed_fp8 = packed_u8.view(torch.float8_e4m3fn)

    return packed_fp8, fp4_scale.squeeze(-1).to(orig_scale_dtype)


def requantize_and_transpose_int4_weights(
    w_packed: torch.Tensor,
    scale_f: torch.Tensor,
    requant_block_size: int,
    sign_xor_val: int,
    in_group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Requantizes INT4 weights and returns transposed & XORed tensors matching gmm_v2 layout."""
    w_req, s_req = requantize_int4_weights(
        w_packed, scale_f, requant_block_size, in_group_size
    )
    w_req = w_req.bitwise_xor(sign_xor_val)
    w_out = w_req.transpose(-2, -1).contiguous() if w_req.dim() >= 2 else w_req
    s_out = s_req.transpose(-2, -1).contiguous() if s_req.dim() >= 2 else s_req
    return w_out, s_out


def requantize_and_transpose_int4_to_fp4_weights(
    w_packed: torch.Tensor,
    scale_f: torch.Tensor,
    requant_block_size: int,
    in_group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Requantizes INT4 weights to FP4 (torch.float8_e4m3fn carrier) and returns transposed tensors matching gmm_v2 layout."""
    w_req, s_req = requantize_int4_to_fp4_weights(
        w_packed, scale_f, requant_block_size, in_group_size
    )
    w_out = w_req.transpose(-2, -1).contiguous() if w_req.dim() >= 2 else w_req
    s_out = s_req.transpose(-2, -1).contiguous() if s_req.dim() >= 2 else s_req
    return w_out, s_out
