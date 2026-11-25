# SPDX-License-Identifier: Apache-2.0
"""Debug-only per-layer digests of the target's hidden states.

``VLLM_TPU_LAYER_DIGEST="T:N"``: every target decoder layer stores the digests
of four of its tensors in a module buffer during the forward
(``POINTS``: the mixer input after the HC mix, the attention or GDN output,
the MoE output and the layer output). After a step of exactly ``T`` scheduled
tokens the runner reads the buffers back on every rank and logs one
``LAYER_DIGEST`` line, for the first ``N`` such steps. A digest is a pair of
wrapping int32 sums of the tensor's raw 16- or 32-bit words, plain and
weighted by word position, so tensors that differ in any bit almost surely
differ in it. Padded rows enter the digest: compare steps whose tokens fill
their bucket. With the flag unset nothing is registered and the forward is
unchanged.
"""

import os

import torch

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

POINTS = ("mix", "mixer", "moe", "out")
# Largest prime below 2**16: the position weights stay small and odd.
_WEIGHT_MOD = 65521

_state = {"taken": 0}


def parse(value: str | None = None) -> tuple[int, int] | None:
    """``(tokens, steps)`` of VLLM_TPU_LAYER_DIGEST, or None when unset."""
    value = envs.VLLM_TPU_LAYER_DIGEST if value is None else value
    if not value:
        return None
    tokens, _, steps = value.partition(":")
    return int(tokens), int(steps or 1)


def digest(x: torch.Tensor) -> torch.Tensor:
    """int32 ``[2]``: wrapping sums of ``x``'s raw words, plain and weighted."""
    flat = x.contiguous().reshape(-1)
    words = flat.view(torch.int16 if flat.element_size() == 2 else torch.int32)
    words = words.to(torch.int32)
    weights = (
        torch.arange(words.numel(), dtype=torch.int32, device=words.device)
        % _WEIGHT_MOD
        + 1
    )
    return torch.stack(
        [words.sum(dtype=torch.int32), (words * weights).sum(dtype=torch.int32)]
    )


def _decoder_layers(model):
    backbone = getattr(model, "model", None)
    return [
        layer
        for layer in getattr(backbone, "layers", ())
        if hasattr(layer, "attn_hyper_connection")
    ]


def bind(target_model) -> int:
    """Registers the digest buffer on every target decoder layer of this
    stage; returns the count."""
    if parse() is None:
        return 0
    layers = _decoder_layers(target_model)
    for layer in layers:
        device = next(layer.parameters()).device
        buf = torch.full((len(POINTS), 2), -1, dtype=torch.int32, device=device)
        layer.register_buffer("_layer_digest", buf, persistent=False)
    logger.info("Layer digest: %d target layers, points %s", len(layers), POINTS)
    return len(layers)


def _rank() -> int:
    try:
        from vllm.distributed import get_tensor_model_parallel_rank

        return int(get_tensor_model_parallel_rank())
    except Exception:  # noqa: BLE001
        return int(os.environ.get("RANK", "0"))


def _word(pair) -> int:
    """One 32-bit word per point for the log line."""
    return (int(pair[0]) ^ (int(pair[1]) * 0x9E3779B1)) & 0xFFFFFFFF


def after_forward(model, num_tokens: int, chunk_index: int) -> str | None:
    """Logs the digests after a step of the configured token count; returns
    the logged line (None when nothing was logged)."""
    config = parse()
    if config is None or chunk_index != 0:
        return None
    tokens, steps = config
    if num_tokens != tokens or _state["taken"] >= steps:
        return None
    _state["taken"] += 1
    layers = [
        layer
        for layer in _decoder_layers(model)
        if getattr(layer, "_layer_digest", None) is not None
    ]
    if not layers:
        return None
    values = torch.stack([layer._layer_digest for layer in layers]).cpu().tolist()
    fields = " ".join(
        f"{layer.layer_index}:" + ".".join(f"{_word(p):08x}" for p in pairs)
        for layer, pairs in zip(layers, values)
    )
    line = (
        f"LAYER_DIGEST n={_state['taken']} rank={_rank()} tokens={num_tokens} "
        f"points={','.join(POINTS)} {fields}"
    )
    logger.info("%s", line)
    return line
