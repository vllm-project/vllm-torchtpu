# SPDX-License-Identifier: Apache-2.0
"""Decode rank-local Muse NVFP4 weights on CPU without requantization."""

import torch


def expert_ep_slice(tensor: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    """Copy a contiguous block of whole experts, including their scales."""
    if tensor.shape[0] % world_size:
        raise ValueError("Muse expert count must be divisible by EP")
    count = tensor.shape[0] // world_size
    return tensor.narrow(0, rank * count, count).clone()


def expert_tp_slice(
    tensor: torch.Tensor, name: str, rank: int, world_size: int
) -> torch.Tensor:
    """Copy only the intermediate-dimension partition; keep all expert IDs."""
    if name in ("gate_up_proj", "gate_up_proj_weight_scale"):
        gate, up = tensor.chunk(2, dim=1)
        if gate.shape[1] % world_size:
            raise ValueError("Muse gate/up dimension must be divisible by TP")
        width = gate.shape[1] // world_size
        return torch.cat(
            (
                gate[:, rank * width : (rank + 1) * width],
                up[:, rank * width : (rank + 1) * width],
            ),
            dim=1,
        ).contiguous()
    if name in ("down_proj", "down_proj_weight_scale"):
        if tensor.shape[-1] % world_size:
            raise ValueError("Muse down-projection blocks must be divisible by TP")
        width = tensor.shape[-1] // world_size
        return tensor[..., rank * width : (rank + 1) * width].clone()
    if name in ("gate_up_proj_weight_scale_2", "down_proj_weight_scale_2"):
        return tensor.clone()
    raise ValueError(f"Unknown Muse expert tensor: {name}")


def dequantize_nvfp4(
    packed: torch.Tensor, block_scale: torch.Tensor, global_scale: torch.Tensor
) -> torch.Tensor:
    """Linear-layout E2M1 * block-16 E4M3 * global scale, in FP32."""
    if packed.dtype != torch.uint8:
        raise ValueError("NVFP4 weights must contain packed uint8 bytes")
    if packed.shape[-1] != block_scale.shape[-1] * 8:
        raise ValueError("NVFP4 block scales must cover 16 unpacked values")
    if packed.shape[:-1] != block_scale.shape[:-1]:
        raise ValueError("NVFP4 scale and weight row shapes disagree")
    lut = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        dtype=torch.float32,
        device=packed.device,
    )
    indices = torch.stack((packed & 15, packed >> 4), dim=-1).long()
    values = lut[indices].flatten(-2)
    scales = block_scale.float() * global_scale.float()
    return (values.reshape(*block_scale.shape, 16) * scales.unsqueeze(-1)).flatten(-2)


def decode_expert_weight(parts: dict[str, torch.Tensor], name: str) -> torch.Tensor:
    """Return BF16 [E, K, N], decoding one expert at a time to bound CPU memory."""
    packed = parts[name]
    out_shape = (packed.shape[0], packed.shape[-1] * 2, packed.shape[1])
    result = torch.empty(out_shape, dtype=torch.bfloat16, device="cpu")
    scales = parts[name + "_weight_scale"]
    globals_ = parts[name + "_weight_scale_2"]
    for e in range(packed.shape[0]):
        if name == "gate_up_proj":
            if globals_.shape != (packed.shape[0], 2):
                raise ValueError(
                    "Muse gate/up require separate per-expert global scales"
                )
            gate, up = packed[e].chunk(2, dim=0)
            sg, su = scales[e].chunk(2, dim=0)
            decoded = torch.cat(
                (
                    dequantize_nvfp4(gate, sg, globals_[e, 0]),
                    dequantize_nvfp4(up, su, globals_[e, 1]),
                ),
                dim=0,
            )
        else:
            if globals_.shape != (packed.shape[0],):
                raise ValueError(
                    "Muse down projection requires one global scale per expert"
                )
            decoded = dequantize_nvfp4(packed[e], scales[e], globals_[e])
        result[e].copy_(decoded.t())
    return result


def combine_nvfp4_scales(parts: dict[str, torch.Tensor], name: str) -> torch.Tensor:
    """Keep the CPU decoder's FP32 scale product, including separate gate/up."""
    scales = parts[name + "_weight_scale"].float()
    global_scale = parts[name + "_weight_scale_2"].float()
    if name == "gate_up_proj":
        if global_scale.shape != (scales.shape[0], 2):
            raise ValueError("Muse gate/up require separate per-expert global scales")
        gate, up = scales.chunk(2, dim=1)
        return torch.cat(
            (
                gate * global_scale[:, 0, None, None],
                up * global_scale[:, 1, None, None],
            ),
            dim=1,
        )
    if global_scale.shape != (scales.shape[0],):
        raise ValueError("Muse down projection requires one global scale per expert")
    return scales * global_scale[:, None, None]


def selected_packed_expert_outputs(
    x: torch.Tensor,
    ids: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    w13_global: torch.Tensor | None = None,
    w2_global: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference decoder for selected experts; TPU serving uses tiled GMMs."""
    tokens, top_k = ids.shape
    flat_ids = ids.flatten()

    def selected(weight, scale, global_scale):
        scale = scale.index_select(0, flat_ids)
        if global_scale is not None:
            scale = scale.view(torch.float8_e4m3fn).float()
            global_scale = global_scale.index_select(0, flat_ids).reshape(
                flat_ids.numel(), -1
            )
            global_scale = global_scale.repeat_interleave(
                scale.shape[1] // global_scale.shape[1], dim=-1
            )
            scale = scale * global_scale[:, :, None]
        return (
            dequantize_nvfp4(
                weight.index_select(0, flat_ids), scale, scale.new_ones(())
            )
            .to(x.dtype)
            .transpose(1, 2)
        )

    inputs = x[:, None, :].expand(-1, top_k, -1).reshape(-1, 1, x.shape[-1])
    gate, up = torch.bmm(inputs, selected(w13, w13_scale, w13_global)).chunk(2, dim=-1)
    activated = (torch.nn.functional.silu(gate.float()) * up.float()).to(x.dtype)
    rows = torch.bmm(activated, selected(w2, w2_scale, w2_global))
    return rows.reshape(tokens, top_k, x.shape[-1])
