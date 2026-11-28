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
"""Unit tests for shared TorchTPU <-> Mooncake TENT buffer bridge utilities."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import pytest
import torch

from vllm_torchtpu.distributed.kv_transfer import mooncake_tpu_utils as mc_utils


@dataclass(frozen=True)
class _FakeShard:
    fake_addr: int
    size: int
    device: int = 0


class _FakeMooncakeTpu:
    def __init__(self, shards_per_tensor: int = 1, mismatch_count: bool = False):
        self.shards_per_tensor = shards_per_tensor
        self.mismatch_count = mismatch_count
        self.prepared_tensors: list[torch.Tensor] = []
        self.unregistered: list[_FakeShard] = []
        self._next_addr = 0x1000_0000

    def prepare(self, tensors, *, provider):
        assert provider is not None
        self.prepared_tensors = list(tensors)
        count = len(tensors) - 1 if self.mismatch_count else len(tensors)
        out: list[list[_FakeShard]] = []
        for idx in range(count):
            t = tensors[idx]
            group: list[_FakeShard] = []
            for s_idx in range(self.shards_per_tensor):
                group.append(
                    _FakeShard(
                        fake_addr=self._next_addr,
                        size=t.numel() * t.element_size(),
                        device=s_idx,
                    )
                )
                self._next_addr += 0x1000_0000
            out.append(group)
        return out

    def unregister(self, shards):
        self.unregistered.extend(shards)


def test_canonicalize_deduplicates_ubp_and_aliases_storage():
    num_blocks = 4
    pool0 = torch.zeros(8, 16, dtype=torch.float32)  # 512 B -> 128 B/block
    pool1 = torch.zeros(16, 8, dtype=torch.float32)  # 512 B -> 128 B/block
    kv_caches = {
        "layers.0.attn": pool0.view(4, 2, 16),
        "layers.1.attn": pool0.view(8, 4, 4),
        "layers.2.mamba": [pool1[:, :4], pool1[:, 4:]],
    }

    pools = mc_utils.canonicalize_tpu_kv_pools(kv_caches, num_blocks)
    assert len(pools) == 2
    for pool, src in zip(pools, (pool0, pool1)):
        assert pool.page_size_bytes == 128
        assert pool.tensor.dtype == torch.int8
        assert pool.tensor.shape == (num_blocks, 128)
        assert pool.tensor.is_contiguous()
        assert pool.tensor.storage_offset() == 0
        assert (
            pool.tensor.untyped_storage().data_ptr() == src.untyped_storage().data_ptr()
        )

    pool1.view(torch.int8)[0, 3] = 77
    assert pools[1].tensor[0, 3].item() == 77


def test_canonicalize_rejects_unaligned_page_size():
    # 4 blocks x 48 bytes -> 48 % 32 != 0
    pool = torch.zeros(4, 48, dtype=torch.int8)
    with pytest.raises(ValueError, match="multiples of 32 bytes"):
        mc_utils.canonicalize_tpu_kv_pools({"attn": pool}, num_blocks=4)


def test_canonicalize_rejects_non_divisible_storage():
    pool = torch.zeros(96, dtype=torch.int8)
    with pytest.raises(ValueError, match="whole scheduler blocks"):
        mc_utils.canonicalize_tpu_kv_pools({"attn": pool}, num_blocks=5)


def test_coalesce_block_pairs():
    runs = mc_utils.coalesce_block_pairs(
        src_block_ids=[2, 3, 4, 9, 10, 11, 20],
        dst_block_ids=[5, 6, 7, 1, 2, 9, 10],
    )
    assert runs == [(2, 5, 3), (9, 1, 2), (11, 9, 1), (20, 10, 1)]
    assert mc_utils.coalesce_block_pairs([], []) == []

    with pytest.raises(ValueError, match="equal length"):
        mc_utils.coalesce_block_pairs([1, 2], [1])


def test_build_p2p_transfer_plan_coalesces_across_pools():
    local_ptrs, remote_ptrs, lengths = mc_utils.build_p2p_transfer_plan(
        local_pool_addrs=[0x1000, 0x9000],
        remote_pool_addrs=[0x2000, 0xA000],
        page_sizes=[64, 128],
        local_block_ids=[5, 6, 7, 1],
        remote_block_ids=[2, 3, 4, 9],
    )
    # 2 coalesced runs x 2 pools = 4 spans instead of 8 per-block spans.
    assert local_ptrs == [
        0x1000 + 5 * 64,
        0x1000 + 1 * 64,
        0x9000 + 5 * 128,
        0x9000 + 1 * 128,
    ]
    assert remote_ptrs == [
        0x2000 + 2 * 64,
        0x2000 + 9 * 64,
        0xA000 + 2 * 128,
        0xA000 + 9 * 128,
    ]
    assert lengths == [3 * 64, 64, 3 * 128, 128]


def test_build_store_block_buffers():
    ptrs, sizes = mc_utils.build_store_block_buffers(
        shard_addrs=[0x1000, 0x5000],
        page_sizes=[64, 128],
        block_ids=[1, 3],
    )
    assert ptrs == [
        [0x1000 + 64, 0x5000 + 128],
        [0x1000 + 3 * 64, 0x5000 + 3 * 128],
    ]
    assert sizes == [[64, 128], [64, 128]]


def test_tent_env_and_protocol_mapping(monkeypatch):
    monkeypatch.delenv("MC_TENT_CONF", raising=False)
    monkeypatch.delenv("MC_USE_TENT", raising=False)
    monkeypatch.delenv("MC_STORE_MEMCPY", raising=False)

    assert mc_utils.store_protocol_for_tent("hp_tcp") == "tcp"
    assert mc_utils.store_protocol_for_tent("tcp") == "tcp"
    assert mc_utils.store_protocol_for_tent("rdma") == "rdma"

    conf_str = mc_utils.configure_tent_tpu_env(protocol="hp_tcp", connections=16)
    assert os.environ["MC_USE_TENT"] == "1"
    assert os.environ["MC_STORE_MEMCPY"] == "0"
    parsed = json.loads(conf_str)
    assert parsed["transports"]["tpu"] == {"enable": True}
    assert parsed["transports"]["hp_tcp"] == {"enable": True, "connections": 16}

    # Operator-supplied MC_TENT_CONF is preserved.
    monkeypatch.setenv("MC_TENT_CONF", '{"custom": true}')
    assert mc_utils.configure_tent_tpu_env(protocol="tcp") == '{"custom": true}'


def test_export_and_unregister_tpu_kv_pools_and_error_cleanup():
    t0 = torch.zeros(4, 64, dtype=torch.int8)
    t1 = torch.zeros(4, 128, dtype=torch.int8)
    fake_tpu = _FakeMooncakeTpu(shards_per_tensor=1)

    shards = mc_utils.export_tpu_kv_pools(
        [t0, t1], provider=object(), mc_tpu_module=fake_tpu
    )
    assert len(shards) == 2
    assert [s.size for s in shards] == [256, 512]

    mc_utils.unregister_tpu_kv_pools(shards, mc_tpu_module=fake_tpu)
    assert fake_tpu.unregistered == shards

    # Multi-shard per pool tensor must clean up and raise.
    bad_tpu = _FakeMooncakeTpu(shards_per_tensor=2)
    with pytest.raises(RuntimeError, match="Expected 1 local TPU shard"):
        mc_utils.export_tpu_kv_pools([t0], provider=object(), mc_tpu_module=bad_tpu)
    assert len(bad_tpu.unregistered) == 2
