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
"""Pool page geometry with a token-split full-attention group."""

from types import SimpleNamespace as NS

import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

from vllm_torchtpu.kv_cache_materializer import _pool_attention_geometry


def _fa(block_size, tokens_per_state=1):
    return FullAttentionSpec(
        block_size,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        tokens_per_state=tokens_per_state,
    )


def _config():
    mamba = MambaSpec(block_size=256, shapes=((4, 128),), dtypes=(torch.bfloat16,))
    return NS(kv_cache_groups=[NS(kv_cache_spec=mamba)])


def _group(gid, spec):
    return NS(kv_cache_spec=spec, kv_cache_group_id=gid, backend="pallas")


def test_split_spec_is_described_by_its_stored_rows():
    split = _fa(512, tokens_per_state=2)
    spec, _, kernel = _pool_attention_geometry(
        _config(), [[_group(0, split)]], {0: 512}
    )
    assert (spec.block_size, spec.tokens_per_state, kernel) == (256, 1, 256)
    assert spec.page_size_bytes == split.page_size_bytes


def test_split_spec_prefers_an_unsplit_group():
    split, plain = _fa(512, tokens_per_state=2), _fa(256)
    spec, _, kernel = _pool_attention_geometry(
        _config(), [[_group(0, split)], [_group(1, plain)]], {0: 512, 1: 256}
    )
    assert spec is plain and kernel == 256


def test_unsplit_first_group_keeps_its_geometry():
    plain = _fa(256)
    spec, _, kernel = _pool_attention_geometry(
        _config(), [[_group(0, plain)]], {0: 128}
    )
    assert spec is plain and kernel == 128
