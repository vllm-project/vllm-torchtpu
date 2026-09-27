# SPDX-License-Identifier: Apache-2.0

from typing import Any


class _LazyVllmLogger:
    def __init__(self, name: str) -> None:
        self._name = "vllm." + name
        self._logger = None

    def _resolve(self):
        if self._logger is None:
            from vllm.logger import init_logger as init_vllm_logger

            self._logger = init_vllm_logger(self._name)
        return self._logger

    def __getattr__(self, name: str) -> Any:
        return getattr(self._resolve(), name)


def init_logger(name: str) -> _LazyVllmLogger:
    return _LazyVllmLogger(name)
