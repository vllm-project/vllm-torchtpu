# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""Kimi-K3.1's Hugging Face config: Kimi-K3's, under its own model type."""

from typing import Any

from vllm.transformers_utils.configs.kimi_k3 import KimiK3Config


class KimiK31Config(KimiK3Config):
    model_type = "kimi_k31"

    # transformers generates an __init__ for a config subclass without its
    # own, which would skip Kimi-K3's sub-config parsing.
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
