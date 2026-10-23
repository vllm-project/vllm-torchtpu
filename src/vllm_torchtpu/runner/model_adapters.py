# Copyright 2026 Google LLC
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
"""Optional model-owned runner extension, selected once before weight loading.

The ordinary runner/cache path is unchanged when selection returns None.
Adapters own input snapshots, dummy/profile geometry and cache lifecycles;
model forward receives explicit kwargs rather than consulting runner state.
"""

from dataclasses import dataclass
from typing import Any, Protocol

import torch


@dataclass(frozen=True)
class ModelInputSnapshot:
    ngram_context: torch.Tensor
    row_start: int
    num_requests: int
    num_tokens: int
    attention_metadata: dict[str, Any]
    input_ids: torch.Tensor


class ModelAdapter(Protocol):
    def bind_model(self, model): ...

    def begin_step(self) -> torch.Tensor: ...

    def prepare_inputs(self, snapshot: ModelInputSnapshot) -> dict[str, Any]: ...

    def dummy_run(
        self, num_tokens, num_reqs, profile_prefix=None, *, dp_lockstep=False
    ) -> float | None: ...

    def get_kv_cache_spec(self) -> dict[str, Any]: ...

    def initialize_kv_cache(self, config): ...


def select_model_adapter(runner) -> ModelAdapter | None:
    return None
