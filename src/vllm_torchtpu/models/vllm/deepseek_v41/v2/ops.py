"""Scoped upstream device bindings and V4.1 tensor helpers."""

import math
from contextlib import contextmanager
from threading import RLock

import torch

_lock = RLock()


@contextmanager
def device_hooks(module, **replacements):
    with _lock:
        previous = {name: getattr(module, name) for name in replacements}
        try:
            for name, value in replacements.items():
                setattr(module, name, value)
            yield
        finally:
            for name, value in previous.items():
                setattr(module, name, value)


def bind_device_ops(function, **operations):
    """Reuse upstream bytecode with per-adapter device function bindings.

    Unlike runtime module monkeypatches, bindings are immutable during forward
    and TorchDynamo can trace them without locks or global mutation.
    """
    import sys
    from types import FunctionType, ModuleType

    # Dynamo guards resolve global functions by module name. A copied globals
    # dict with the upstream name would make guards inspect the GPU operation.
    name = f"{__name__}.{function.__qualname__.replace('.', '_')}"
    namespace = ModuleType(name)
    namespace.__dict__.update(function.__globals__ | operations | {"__name__": name})
    sys.modules[name] = namespace
    bound = FunctionType(
        function.__code__,
        namespace.__dict__,
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )
    bound.__kwdefaults__ = function.__kwdefaults__
    bound.__annotations__ = function.__annotations__
    bound.__qualname__ = function.__qualname__
    setattr(namespace, function.__name__, bound)
    return bound


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


def round_fp8_activation(x, block_size=32):
    if x.shape[-1] % block_size:
        raise ValueError("FP8 block size must divide contraction dimension")
    blocks = x.float().unflatten(-1, (-1, block_size))
    scales = torch.pow(
        2.0, torch.ceil(torch.log2(blocks.abs().amax(-1).clamp_min(1e-4) / 448))
    )
    values = (blocks / scales.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
    return (values.float() * scales.unsqueeze(-1)).flatten(-2).to(x.dtype)


def rms_norm(x, weight, eps):
    normalized = x.float() * torch.rsqrt(
        x.float().square().mean(-1, keepdim=True) + eps
    )
    return (normalized * weight.float()).to(x.dtype)
