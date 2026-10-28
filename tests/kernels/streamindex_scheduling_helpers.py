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
"""Shared StreamIndex test inputs."""

import numpy as np
import torch


def fixture_inputs(tokens):
    """Strictly ordered BF16 keys, fragmented pages, two live and six idle slots."""
    pages = 80
    rng = np.random.default_rng(42)
    tables = np.zeros((8, pages), np.int32)
    tables[:2] = rng.permutation(np.arange(1, 2 * pages + 1)).reshape(2, pages)
    values = (np.arange(pages * 128, dtype=np.uint16) + 0x2000).copy()
    cache_words = np.zeros((2 * pages + 1, 128, 256), np.uint16)
    cache_words[:, :, 128] = 0x3F80
    cache_words[0] = 0x3E80
    for request in range(2):
        for logical, physical in enumerate(tables[request]):
            cache_words[physical, :, 0] = values[logical * 128 : (logical + 1) * 128]
    cache = torch.from_numpy(cache_words.copy()).view(torch.bfloat16)
    cache = cache.reshape(2 * pages + 1, 128, 1, 256)
    query = torch.zeros((tokens, 4, 128), dtype=torch.bfloat16)
    query[..., 0] = 1
    counts = np.array([tokens // 2, tokens - tokens // 2] + [0] * 6, np.int32)
    starts = np.r_[0, counts.cumsum()].astype(np.int32)
    context = np.array([4096, 2048] + [0] * 6, np.int32)
    requests = np.repeat(np.arange(8), counts).astype(np.int32)
    lengths = np.concatenate(
        [
            context[request] + np.arange(1, count + 1)
            for request, count in enumerate(counts)
        ]
    ).astype(np.int32)
    return tuple(
        [query, cache]
        + [
            torch.from_numpy(x.copy())
            for x in (lengths, tables, requests, starts, context)
        ]
    )
