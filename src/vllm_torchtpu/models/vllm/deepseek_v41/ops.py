"""RoPE cache tables and tensor rotary operations."""

import math

import torch


def compact_rope_caches(attentions, max_model_len):
    """Own compact serving tables without changing upstream's shared YaRN state.

    Frequencies and their original context stay unchanged. Only rows reachable
    by this engine are copied; layers sharing an upstream table share one copy.
    """
    if max_model_len <= 0:
        raise ValueError("max_model_len must be positive")
    compact = {}
    for attention in attentions:
        original = attention.rotary_emb.cos_sin_cache
        key = id(original)
        if key not in compact:
            compact[key] = (
                original[:max_model_len].clone()
                if original.shape[0] > max_model_len
                else original
            )
        attention._tpu_rope_cache = compact[key]


def rotary(x, positions, config, *, compressed, inverse=False):
    dim = config["qk_rope_head_dim"]
    if dim < 2 or dim % 2 or dim > x.shape[-1]:
        raise ValueError("invalid rotary dimension")
    base = config["compress_rope_theta"] if compressed else config["rope_theta"]
    frequencies = 1 / base ** (
        torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / dim
    )
    yarn = (
        (config.get("rope_scaling") or config.get("rope_parameters"))
        if compressed
        else None
    )
    if yarn and yarn["original_max_position_embeddings"] > 0:
        original = yarn["original_max_position_embeddings"]

        def correction(rotations):
            return (
                dim
                * math.log(original / (rotations * 2 * math.pi))
                / (2 * math.log(base))
            )

        low = max(math.floor(correction(yarn["beta_fast"])), 0)
        high = min(math.ceil(correction(yarn["beta_slow"])), dim - 1)
        ramp = (
            (torch.arange(dim // 2, device=x.device) - low) / max(high - low, 1e-3)
        ).clamp(0, 1)
        frequencies = frequencies * (1 - ramp + ramp / yarn["factor"])
    angles = positions.to(device=x.device, dtype=torch.float32)[:, None] * frequencies
    shape = (1, len(positions), *((1,) * (x.ndim - 3)), dim // 2)
    cosine, sine = angles.cos().reshape(shape), angles.sin().reshape(shape)
    if inverse:
        sine = -sine
    pairs = x[..., -dim:].float().unflatten(-1, (-1, 2))
    real, imag = pairs[..., 0], pairs[..., 1]
    rotated = torch.stack(
        (real * cosine - imag * sine, real * sine + imag * cosine), -1
    ).flatten(-2)
    return torch.cat((x[..., :-dim], rotated.to(x.dtype)), -1)
