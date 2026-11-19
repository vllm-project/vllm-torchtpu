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
"""TPU_TOKEN_BUCKET_EXTRA accepts ":" (and ";") as well as "," between values.

Recipes pass server env as a comma-separated KEY=VALUE list, so a multi-value
bucket list has to reach the server as e.g. ``48:80:96:112``.
"""

import pytest

from vllm_torchtpu import envs
from vllm_torchtpu.platforms.tpu_platform import _get_token_paddings

FULL = [48, 80, 96, 112, 160, 192, 224]


@pytest.mark.parametrize(
    "raw, want",
    [
        ("48:80:96:112:160:192:224", FULL),
        ("48,80,96,112,160,192,224", FULL),
        ("48;80;96;112;160;192;224", FULL),
        ("48:80,96;112", [48, 80, 96, 112]),
        ("96", [96]),
        (" 48 : 96 ", [48, 96]),
        ("48::96:", [48, 96]),
        ("", []),
    ],
)
def test_token_bucket_extra_separators(monkeypatch, raw, want):
    monkeypatch.setenv("TPU_TOKEN_BUCKET_EXTRA", raw)
    assert envs.TPU_TOKEN_BUCKET_EXTRA == want


def test_token_bucket_extra_unset(monkeypatch):
    monkeypatch.delenv("TPU_TOKEN_BUCKET_EXTRA", raising=False)
    assert envs.TPU_TOKEN_BUCKET_EXTRA == []


def test_colon_list_reaches_the_compiled_paddings(monkeypatch):
    # The decode's max-num-batched-tokens is 2048; the served buckets are powers of
    # two from 16 plus the extras.
    monkeypatch.setenv("TPU_TOKEN_BUCKET_EXTRA", "48:80:96:112:160:192:224")
    paddings = _get_token_paddings(
        min_token_size=16,
        max_token_size=2048,
        extra_bucket_sizes=list(envs.TPU_TOKEN_BUCKET_EXTRA),
    )
    assert paddings == [16, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256, 512, 1024, 2048]


def test_comma_free_single_value_is_unchanged(monkeypatch):
    # CP2's served value (one bucket) parses as before the change.
    monkeypatch.setenv("TPU_TOKEN_BUCKET_EXTRA", "96")
    paddings = _get_token_paddings(16, 2048, list(envs.TPU_TOKEN_BUCKET_EXTRA))
    assert paddings == [16, 32, 64, 96, 128, 256, 512, 1024, 2048]
