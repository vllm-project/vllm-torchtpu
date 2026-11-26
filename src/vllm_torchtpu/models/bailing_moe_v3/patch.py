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
"""Install Bailing's TPU KDA layer after platform activation."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import ModelConfig


def maybe_patch_bailing_moe_v3(
    model_config: "ModelConfig | None" = None,
) -> bool | None:
    if model_config is None or model_config.architecture not in (
        "BailingMoeV3ForCausalLM",
        "BailingMoeV3VLForConditionalGeneration",
    ):
        return False

    from vllm_torchtpu.models.bailing_moe_v3 import attention  # noqa: F401
