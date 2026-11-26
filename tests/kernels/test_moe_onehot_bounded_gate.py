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
"""TPU_MOE_ONEHOT_BOUNDED_SUBLANE_TOKENS: the bounded one-hot kernels' shape
gate admits the sublane-tiled decode buckets only with the key on."""

import pytest

pytest.importorskip("jax")

from vllm_torchtpu import envs  # noqa: E402
from vllm_torchtpu.kernels.megablox import moe_onehot_bounded as mob  # noqa: E402

HIDDEN, TOPK = 6144, 8


@pytest.mark.parametrize(
    "tokens,off,on",
    [(256, True, True), (384, True, True), (320, False, True), (416, False, True),
     (192, False, False), (328, False, False)],
)
def test_gate(monkeypatch, tokens, off, on):
    monkeypatch.setattr(mob, "_col_chunk", lambda *a: 128)
    for flag, want in ((False, off), (True, on)):
        monkeypatch.setattr(envs, "TPU_MOE_ONEHOT_BOUNDED_SUBLANE_TOKENS", flag, raising=False)
        assert mob.can_use_bounded_onehot(tokens, tokens * TOPK, HIDDEN) is want
