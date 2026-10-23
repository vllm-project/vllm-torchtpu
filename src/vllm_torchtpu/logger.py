# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
