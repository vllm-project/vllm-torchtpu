# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi text model for the TorchTPU vLLM plugin."""

import importlib

# Resolved on first attribute access (PEP 562) rather than at import time, so
# that importing one layer does not pull in the others. The KDA layer and its
# accuracy tests need `attention` alone; eagerly importing `model` and `moe`
# would make them depend on the MoE layer's vLLM surface as well. Model
# registration goes through the "<module>:<name>" string path in
# `models/vllm/__init__.py`, so nothing here needs to be imported up front.
_EXPORTS = {
    "KimiDeltaAttention": "attention",
    "MultiHeadLatentAttention": "attention",
    "KimiLinearForCausalLM": "model",
    "KimiModel": "model",
    "KimiMoE": "moe",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f".{module}", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_EXPORTS})
