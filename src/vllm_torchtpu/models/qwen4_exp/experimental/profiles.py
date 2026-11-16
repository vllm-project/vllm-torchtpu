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
"""Named experimental Qwen3.8-Flash recipes: profile -> {slot: variants by preference}.

Every profile pins every slot of `variants.VARIANTS`. `DEFAULT_PROFILE` is
the proven-best served recipe (agentx mixed 256K: TP8, EP, MTP K=3, the
aggregated infx-qwen38-flash-agentx curve); `vllm serve` uses it with no
extra flags. Pick another with
``--additional-config '{"model_plan": {"profile": "<name>"}}'``.
"""

DEFAULT_PROFILE = "agentx-mixed-256k"

_MIXED = {
    "decode_batch": ("large", "small"),
    "c48_stack": ("off",),
    "pool_budget": ("fixed_61g", "auto"),
}

PROFILES: dict[str, dict[str, tuple[str, ...]]] = {
    "agentx-mixed-256k": dict(_MIXED),
    # The c48 point of the same curve: stack B (DCP sub-features, PLE host
    # overlap) on top of the large-batch stack; nothing in the config tells
    # c48 from c32 (both max_num_seqs 32), so it is a profile.
    "agentx-c48": {**_MIXED, "c48_stack": ("on",)},
}
