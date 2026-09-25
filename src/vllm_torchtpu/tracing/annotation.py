# Copyright 2026 Google LLC
"""PyTorch Kineto and TPU profiler trace annotation utilities."""

import contextlib
from typing import Any

import torch


def is_trace_annotation_enabled() -> bool:
    """Checks whether PyTorch Kineto profiler session is active."""
    return torch.autograd._profiler_enabled()


class TraceAnnotation(contextlib.ContextDecorator):
    def __init__(self, name: str, **kwargs: Any) -> None:
        self.name = name
        self.kwargs = kwargs
        self._rf = None

    @classmethod
    def is_enabled(cls) -> bool:
        return is_trace_annotation_enabled()

    def __enter__(self):
        if self.is_enabled():
            name = self.name
            if self.kwargs:
                args_str = ",".join(f"{k}={v}" for k, v in self.kwargs.items())
                name = f"{name}#{args_str}#"
            self._rf = torch.profiler.record_function(name)
            self._rf.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._rf is not None:
            self._rf.__exit__(exc_type, exc_val, exc_tb)
            self._rf = None
