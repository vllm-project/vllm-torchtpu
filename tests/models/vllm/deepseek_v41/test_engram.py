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
"""Unit and TPU tests for DeepSeek-V4.1 Engram hashing, storage, and staging."""

from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm_torchtpu.models.vllm.deepseek_v41 import (
    Engram,
    NgramHashState,
    ParallelEngramEmbedding,
    TpuDPSharedEngramStorage,
    dequant_fp8_rows,
)
from vllm_torchtpu.models.vllm.deepseek_v41 import (
    engram as engram_mod,
)


def test_dequant_fp8_rows_and_unpack_bytes() -> None:
    """Verify FP8+E8M0 block dequantization and little-endian int32 byte unpacking."""
    raw = torch.zeros((2, 32), dtype=torch.uint8)
    raw[0] = torch.ones(32, dtype=torch.float8_e4m3fn).view(torch.uint8)
    raw[1] = 255
    scale = torch.tensor([[127], [255]], dtype=torch.uint8)
    deq = dequant_fp8_rows(raw, scale, block_size=32)
    assert deq.shape == (2, 32) and deq.dtype == torch.bfloat16
    assert torch.all(deq[0] == 1.0) and torch.all(deq[1] == 0.0)

    words = torch.tensor([0x04030201, 0x08070605], dtype=torch.int32)
    assert engram_mod._unpack_bytes(words, 6).tolist() == [1, 2, 3, 4, 5, 6]


def test_tp_and_dp_engram_row_gather_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify 128-lane padded TP and DP all-gather helpers for Engram rows."""
    monkeypatch.setattr(
        engram_mod, "tensor_model_parallel_all_gather", lambda x, dim: x
    )
    t_i32 = torch.ones((2, 16), dtype=torch.int32)
    assert torch.equal(engram_mod._gather_engram_rows(t_i32, 0), t_i32)
    t_bf16_narrow = torch.ones((2, 64), dtype=torch.bfloat16)
    assert engram_mod._gather_engram_rows(t_bf16_narrow, 0).shape == (2, 64)
    t_bf16_wide = torch.ones((2, 128), dtype=torch.bfloat16)
    assert engram_mod._gather_engram_rows(t_bf16_wide, 0).shape == (2, 128)

    monkeypatch.setattr(engram_mod, "get_engram_dp_group", lambda: None)
    assert engram_mod._dp_gather_engram_rows(t_bf16_narrow.unsqueeze(1), 1).shape == (
        1,
        1,
        64,
    )

    fake_dp = SimpleNamespace(
        world_size=2,
        rank_in_group=1,
        all_gather=lambda x, dim: torch.cat([x, x + 1], dim=0),
    )
    monkeypatch.setattr(engram_mod, "get_engram_dp_group", lambda: fake_dp)
    staged_n = torch.ones((4, 2, 64), dtype=torch.bfloat16)
    assert engram_mod._dp_gather_engram_rows(staged_n, 2).shape == (2, 4, 64)
    staged_w = torch.ones((4, 2, 128), dtype=torch.bfloat16)
    assert engram_mod._dp_gather_engram_rows(staged_w, 2).shape == (2, 4, 128)


def test_tpu_dp_shared_engram_storage_allocation_and_views(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Verify POSIX /dev/shm allocation on rank 0, peer mapping, and view checks."""
    monkeypatch.setattr(engram_mod.upstream, "SHM_PATH", str(tmp_path))
    monkeypatch.setattr(engram_mod.upstream, "check_shm_free_space", lambda n: None)
    monkeypatch.setattr(
        engram_mod.dist,
        "all_gather_object",
        lambda out, val, group=None: out.__setitem__(0, val),
    )
    grp0 = SimpleNamespace(
        rank_in_group=0,
        world_size=1,
        cpu_group=None,
        broadcast_object=lambda obj: obj,
    )
    st = TpuDPSharedEngramStorage(num_rows=4, dim=32, block_size=32, group=grp0)
    w, s = st.get_views(st.weight, st.weight_scale_inv)
    assert w.shape == (4, 32) and s.shape == (4, 1)
    with pytest.raises(RuntimeError, match="must not be replaced"):
        st.get_views(torch.empty_like(st.weight), st.weight_scale_inv)

    f_peer = tmp_path / "peer_shm.bin"
    f_peer.write_bytes(b"\x00" * 132)
    grp1 = SimpleNamespace(
        rank_in_group=1,
        world_size=1,
        cpu_group=None,
        broadcast_object=lambda obj: (str(f_peer), None),
    )
    st1 = TpuDPSharedEngramStorage(num_rows=4, dim=32, block_size=32, group=grp1)
    assert st1.weight.shape == (4, 32)


def test_tpu_dp_shared_engram_storage_shared_across_ranks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Writes through EDP rank 0's /dev/shm mapping are visible on rank 1."""
    monkeypatch.setattr(engram_mod.upstream, "SHM_PATH", str(tmp_path))
    monkeypatch.setattr(engram_mod.upstream, "check_shm_free_space", lambda n: None)
    world_size = 2
    barrier = threading.Barrier(world_size, timeout=30)
    board: dict[Any, Any] = {}

    def _all_gather_object(out: list[Any], val: Any, group: Any = None) -> None:
        board[("gather", group.rank)] = val
        barrier.wait()
        out[:] = [board[("gather", rank)] for rank in range(world_size)]
        barrier.wait()

    monkeypatch.setattr(engram_mod.dist, "all_gather_object", _all_gather_object)

    def _make_group(rank: int) -> SimpleNamespace:
        def _broadcast_object(obj: Any) -> Any:
            if rank == 0:
                board["broadcast"] = obj
            barrier.wait()
            return board["broadcast"]

        return SimpleNamespace(
            rank_in_group=rank,
            world_size=world_size,
            cpu_group=SimpleNamespace(rank=rank),
            broadcast_object=_broadcast_object,
        )

    storages: list[Any] = [None] * world_size
    errors: list[BaseException] = []

    def _run(rank: int) -> None:
        try:
            storages[rank] = TpuDPSharedEngramStorage(
                num_rows=4, dim=32, block_size=32, group=_make_group(rank)
            )
        except BaseException as exc:
            errors.append(exc)
            barrier.abort()

    threads = [threading.Thread(target=_run, args=(r,)) for r in range(world_size)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    assert not errors and all(storage is not None for storage in storages)
    st0, st1 = storages
    # Two separate mappings of one backing file, not aliased tensors.
    assert st0.weight.data_ptr() != st1.weight.data_ptr()
    st0.weight.view(torch.uint8)[1].fill_(0x38)
    st0.weight_scale_inv[1] = 127
    assert torch.all(st1.weight.view(torch.uint8)[1] == 0x38)
    assert int(st1.weight_scale_inv[1]) == 127
    # Rank 0 unlinks the backing file once every rank has mapped it.
    assert not list(tmp_path.iterdir())


def test_tpu_dp_shared_engram_storage_error_propagation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Verify rank-0 creation errors and peer mmap errors raise RuntimeError."""
    monkeypatch.setattr(engram_mod.upstream, "SHM_PATH", str(tmp_path))
    monkeypatch.setattr(
        engram_mod.dist,
        "all_gather_object",
        lambda out, val, group=None: out.__setitem__(0, val),
    )
    grp0 = SimpleNamespace(
        rank_in_group=0,
        world_size=1,
        cpu_group=None,
        broadcast_object=lambda obj: obj,
    )
    monkeypatch.setattr(
        engram_mod.upstream,
        "check_shm_free_space",
        lambda n: (_ for _ in ()).throw(OSError("disk full")),
    )
    with pytest.raises(RuntimeError, match="creation failed on EDP rank 0"):
        TpuDPSharedEngramStorage(num_rows=4, dim=32, block_size=32, group=grp0)

    monkeypatch.setattr(engram_mod.upstream, "check_shm_free_space", lambda n: None)
    monkeypatch.setattr(
        torch.UntypedStorage,
        "from_file",
        staticmethod(lambda *a, **k: (_ for _ in ()).throw(RuntimeError("mmap err"))),
    )
    with pytest.raises(RuntimeError, match="initialization failed"):
        TpuDPSharedEngramStorage(num_rows=4, dim=32, block_size=32, group=grp0)


def _make_ngram_state(monkeypatch: pytest.MonkeyPatch) -> NgramHashState:
    def _fake_super_init(
        self: Any, vllm_config: Any, layout: Any, swa_cache_module: Any
    ) -> None:
        torch.nn.Module.__init__(self)
        self.lookback_depth = 3
        self.block_size = 4
        self.use_slot_cache = True
        self.pad_id = 0
        self._cache = None
        self._kv_cache_ref = None
        self.swa_cache_module = swa_cache_module
        self.register_buffer(
            "token_map", torch.arange(32, dtype=torch.int32), persistent=False
        )
        self.register_buffer(
            "multipliers",
            torch.tensor([[3, 5, 7, 11], [13, 17, 19, 23]], dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "primes",
            torch.tensor([[[17, 19], [23, 29], [31, 37]]] * 2, dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "offsets",
            torch.tensor([[0, 20, 50, 80, 120, 160]] * 2, dtype=torch.int64),
            persistent=False,
        )

    monkeypatch.setattr(
        engram_mod.upstream.NgramHashState, "__init__", _fake_super_init
    )
    swa = SimpleNamespace(
        block_size=8, kv_cache=torch.zeros((2, 8, 16), dtype=torch.bfloat16)
    )
    return NgramHashState(SimpleNamespace(), SimpleNamespace(), swa)


def test_ngram_hash_state_ensure_cache_and_empty_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify NgramHashState.ensure_cache during compile/eager and empty inputs."""
    state = _make_ngram_state(monkeypatch)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    with pytest.raises(AssertionError):
        state.ensure_cache()
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
    assert state.ensure_cache() and state.block_size == 8
    # 2 blocks x 8 slots, plus the trailing dummy slot for masked writes.
    assert state._cache.numel() == 17
    # Re-ensuring the same bound KV cache keeps the cache (no extra padding).
    cache_before = state._cache
    assert state.ensure_cache() and state._cache is cache_before

    # block_size comes from the SWA module, never from kv_cache.shape: the bound
    # tensor may be the overlaid CSA page `[num_pages, page_size, 4, 128]`.
    state.swa_cache_module = SimpleNamespace(
        block_size=256, kv_cache=torch.zeros((8, 1, 4, 128), dtype=torch.uint8)
    )
    assert (
        state.ensure_cache()
        and state.block_size == 256
        and state._cache.numel() == 2049
    )
    # Slot 256 (block 1 * 256 + 0) must write and read cleanly without IndexError.
    h_p256 = state(
        torch.tensor([7], dtype=torch.int32),
        torch.tensor([4], dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([False]),
        torch.tensor([[-1, -1, -1]], dtype=torch.int32),
        torch.zeros((1, 3), dtype=torch.bool),
        torch.tensor([256], dtype=torch.int32),
        torch.tensor([[1, 2]], dtype=torch.int32),
    )
    assert h_p256.shape == (1, 2, 6) and int(state._cache[256]) == 7

    # Empty kv_cache falls back to swa_cache_module.block_size.
    state.swa_cache_module = SimpleNamespace(block_size=32, kv_cache=torch.tensor([]))
    assert not state.ensure_cache() and state.block_size == 32

    empty_ids = torch.empty((0,), dtype=torch.int32)
    z_lb = torch.zeros((1, 3), dtype=torch.int32)
    z_lbd = torch.zeros((1, 3), dtype=torch.bool)
    qsl = torch.zeros(2, dtype=torch.int32)
    assert state(
        empty_ids, empty_ids, qsl, empty_ids.bool(), z_lb, z_lbd, None, None
    ).shape == (0, 2, 6)


def test_ngram_hash_state_slot_cache_and_fallback_modes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify NgramHashState slot-cache lookup and stateless pad fallback."""
    state = _make_ngram_state(monkeypatch)
    assert state.ensure_cache()
    ids = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    pos = torch.tensor([0, 1, 2, 5], dtype=torch.int32)
    qsl = torch.tensor([0, 3, 4], dtype=torch.int32)
    dead = torch.tensor([False, False, False, False])
    lb = torch.tensor([[-1, -1, -1], [4, -1, -1]], dtype=torch.int32)
    lbd = torch.zeros((2, 3), dtype=torch.bool)
    slots = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    bt = torch.tensor([[0, 1], [0, 1]], dtype=torch.int32)

    h_slot = state(ids, pos, qsl, dead, lb, lbd, slots, bt)
    assert h_slot.shape == (4, 2, 6) and h_slot.dtype == torch.int32

    state.use_slot_cache = False
    h_noslot1 = state(ids, pos, qsl, dead, lb, lbd, None, None)
    h_noslot2 = state(ids, pos, qsl, dead, lb, lbd, None, None)
    assert torch.equal(h_noslot1, h_noslot2)
    assert torch.equal(h_slot[:3], h_noslot1[:3])


_REAL_HASH_CFG = SimpleNamespace(
    engram_layer_ids=(0, 3),
    engram_num_embeddings=(512, 512),
    engram_max_ngram_size=4,
    engram_n_heads=2,
    engram_head_dim=32,
    engram_compressed_vocab_size=32,
    engram_pad_token_id=0,
    engram_vocab_size=32,
)


def _make_real_ngram_state(monkeypatch: pytest.MonkeyPatch) -> NgramHashState:
    """``NgramHashState`` with upstream primes, offsets and int64 multipliers."""
    state = _make_ngram_state(monkeypatch)
    layout = engram_mod.upstream.EngramLayout(_REAL_HASH_CFG)
    vocab = layout.compressed_vocab_size
    # Raw ids 0..39 collapse onto 32 compressed ids, like the tokenizer map.
    state.token_map = (torch.arange(40, dtype=torch.int32) * 7 + 3) % vocab
    state.multipliers = engram_mod.upstream.compute_hash_multipliers(
        layout.layer_ids, layout.max_ngram_size, vocab
    )
    state.primes = torch.tensor(layout.primes)
    state.offsets = layout.offsets
    state.pad_id = int(state.token_map[layout.pad_token_id])
    state.lookback_depth = layout.max_ngram_size - 1
    return state


def _reference_hashes(
    state: NgramHashState,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    query_start_loc: torch.Tensor,
    dead_mask: torch.Tensor,
    lookback_ids: torch.Tensor,
    lookback_dead: torch.Tensor,
    block_table: torch.Tensor | None,
    cache: torch.Tensor | None,
) -> torch.Tensor:
    """Scalar transcription of upstream ``_hash_ids_kernel`` (vLLM v0.33.0).

    ``cache`` holds the real slots only (no trailing dummy slot).
    """
    dead_id = engram_mod.upstream.DEAD_ID
    token_map = state.token_map.tolist()
    multipliers = state.multipliers.tolist()
    primes = state.primes.tolist()
    offsets = state.offsets.tolist()
    ids, pos = input_ids.tolist(), positions.tolist()
    qsl, dead = query_start_loc.tolist(), dead_mask.tolist()
    lb, lbd = lookback_ids.tolist(), lookback_dead.tolist()
    table = block_table.tolist() if block_table is not None else []
    slots = cache.tolist() if cache is not None else None
    num_tokens, num_reqs = len(ids), len(qsl) - 1
    num_layers, max_ngram = len(multipliers), len(multipliers[0])
    num_heads = len(primes[0][0])
    depth = len(lb[0])
    block_size = state.block_size
    out = torch.zeros(
        (num_tokens, num_layers, (max_ngram - 1) * num_heads), dtype=torch.int64
    )
    for t in range(num_tokens):
        req = min(sum(t >= end for end in qsl[1:]), num_reqs - 1)
        chunk_start = pos[min(qsl[req], num_tokens - 1)]
        for layer in range(num_layers):
            rolling, blocked = 0, False
            for shift in range(max_ngram):
                past = pos[t] - shift
                col = chunk_start - 1 - past
                if past >= chunk_start:
                    src_idx = max(t - shift, 0)
                    src = dead_id if dead[src_idx] else token_map[ids[src_idx]]
                elif 0 <= col < depth and lb[req][col] >= 0:
                    src = dead_id if lbd[req][col] else token_map[lb[req][col]]
                elif slots is None:
                    src = state.pad_id
                else:
                    logical = min(max(past, 0), len(table[0]) * block_size - 1)
                    row = table[min(req, len(table) - 1)]
                    block = row[logical // block_size]
                    slot = block * block_size + logical % block_size
                    # TPU deviation: the null block 0 (or an unmapped -1) holds
                    # no history, so it reads as DEAD instead of clamped slot 0.
                    if block <= 0:
                        src = dead_id
                    else:
                        src = slots[min(max(slot, 0), len(slots) - 1)]
                blocked = blocked or past < 0 or src == dead_id
                term = (state.pad_id if blocked else src) * multipliers[layer][shift]
                # Upstream bounds the multipliers so this product never wraps.
                assert 0 <= term < 2**63
                rolling ^= term
                if shift > 0:
                    for head in range(num_heads):
                        c = (shift - 1) * num_heads + head
                        prime = primes[layer][shift - 1][head]
                        out[t, layer, c] = rolling % prime + offsets[layer][c]
    return out.to(torch.int32)


def _reference_hash_inputs() -> tuple[torch.Tensor, ...]:
    """Three requests covering every history tier, plus one padded token.

    - req 0: prefill at positions 0..4 with a dead token (n-grams hit pos < 0).
    - req 1: decode at position 10; position 9 comes from ``lookback_ids``,
      positions 8 and 7 from the slot cache (slots 28 and 27).
    - req 2: chunk at positions 6..7; position 5 from ``lookback_ids``,
      position 4 is dead in ``lookback_dead``, position 3 is dead in the cache.
    """
    ids = torch.tensor([5, 9, 13, 17, 39, 30, 2, 11, 0], dtype=torch.int32)
    pos = torch.tensor([0, 1, 2, 3, 4, 10, 6, 7, 0], dtype=torch.int32)
    qsl = torch.tensor([0, 5, 6, 8], dtype=torch.int32)
    dead = torch.tensor([False, False, True, False, False, False, False, False, False])
    lb = torch.tensor([[-1, -1, -1], [25, -1, -1], [11, 12, -1]], dtype=torch.int32)
    lbd = torch.tensor(
        [[False, False, False], [False, False, False], [False, True, False]]
    )
    slots = torch.tensor([4, 5, 6, 7, 8, 30, 18, 19, -1], dtype=torch.int32)
    bt = torch.tensor([[1, 2, 0], [5, 6, 7], [3, 4, 0]], dtype=torch.int32)
    return ids, pos, qsl, dead, lb, lbd, slots, bt


def _seed_hash_history(state: NgramHashState) -> None:
    """Slot-cache entries written by earlier steps (see ``_reference_hash_inputs``)."""
    assert state._cache is not None
    state._cache[27] = 21
    state._cache[28] = 22
    state._cache[15] = engram_mod.upstream.DEAD_ID


def test_ngram_hash_state_matches_reference_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check hash values against an independent scalar transcription."""
    state = _make_real_ngram_state(monkeypatch)
    state.swa_cache_module = SimpleNamespace(
        block_size=4, kv_cache=torch.zeros((8, 4, 16), dtype=torch.bfloat16)
    )
    assert state.ensure_cache() and state._cache is not None
    _seed_hash_history(state)
    ids, pos, qsl, dead, lb, lbd, slots, bt = _reference_hash_inputs()

    hashes = state(ids, pos, qsl, dead, lb, lbd, slots, bt)
    tm = state.token_map.tolist()
    assert state._cache[4].item() == tm[5]
    assert state._cache[6].item() == engram_mod.upstream.DEAD_ID
    expected = _reference_hashes(
        state, ids, pos, qsl, dead, lb, lbd, bt, state._cache[:-1]
    )
    assert torch.equal(hashes, expected)

    # Hand-derived values. Token 0 (position 0), layer 0, 2-gram: position -1
    # is blocked, so it hashes as pad.
    mult = state.multipliers.tolist()
    primes = state.primes.tolist()
    offsets = state.offsets.tolist()
    pad = state.pad_id
    rolling = (tm[5] * mult[0][0]) ^ (pad * mult[0][1])
    assert hashes[0, 0, :2].tolist() == [
        rolling % primes[0][0][h] + offsets[0][h] for h in range(2)
    ]
    # Token 5 (req 1, position 10), layer 1, 4-gram: batch, lookback, then cache.
    rolling = (
        (tm[30] * mult[1][0])
        ^ (tm[25] * mult[1][1])
        ^ (22 * mult[1][2])
        ^ (21 * mult[1][3])
    )
    assert hashes[5, 1, 4:].tolist() == [
        rolling % primes[1][2][h] + offsets[1][4 + h] for h in range(2)
    ]

    # Stateless (V2 runner): history the lookback window misses hashes as pad.
    # Keyword arguments use upstream's NgramHashState.forward parameter names.
    state.use_slot_cache = False
    stateless = state(
        input_ids=ids,
        positions=pos,
        query_start_loc=qsl,
        dead_mask=dead,
        lookback_token_ids=lb,
        lookback_dead_mask=lbd,
        slot_mapping=None,
        block_table=None,
    )
    expected = _reference_hashes(state, ids, pos, qsl, dead, lb, lbd, None, None)
    assert torch.equal(stateless, expected)
    assert not torch.equal(stateless[5], hashes[5])


def test_parallel_engram_embedding_modes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Verify ParallelEngramEmbedding across HBM, CPU offload, and /dev/shm."""
    monkeypatch.setattr(engram_mod.upstream, "SHM_PATH", str(tmp_path))
    monkeypatch.setattr(engram_mod.upstream, "check_shm_free_space", lambda n: None)
    monkeypatch.setattr(engram_mod, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(
        engram_mod.upstream, "get_tensor_model_parallel_rank", lambda: 0
    )
    monkeypatch.setattr(engram_mod, "get_engram_dp_size", lambda: 1)

    emb_hbm = ParallelEngramEmbedding(
        20, 32, (10, 10), block_size=32, cpu_offload=False
    )
    emb_hbm.weight.data.fill_(1.0)
    emb_hbm.weight_scale_inv.data.fill_(127)
    out_empty = torch.empty((0, 2, 32), dtype=torch.bfloat16)
    emb_hbm.lookup(torch.empty((0, 2), dtype=torch.int32), out_empty)
    out_hbm = torch.empty((2, 2, 32), dtype=torch.bfloat16)
    emb_hbm.lookup(torch.tensor([[0, 12], [-1, 99]], dtype=torch.int64), out_hbm)
    assert torch.all(out_hbm[0] == 1.0) and torch.all(out_hbm[1] == 0.0)

    emb_cpu = ParallelEngramEmbedding(20, 32, (10, 10), block_size=32, cpu_offload=True)
    emb_cpu.weight.data.fill_(1.0)
    emb_cpu.weight_scale_inv.data.fill_(127)
    raw_o = torch.empty((1, 2, 32), dtype=torch.uint8)
    sc_o = torch.empty((1, 2, 1), dtype=torch.uint8)
    emb_cpu.gather_raw(torch.tensor([[1, 11]], dtype=torch.int32), raw_o, sc_o)
    assert torch.all(sc_o == 127)
    emb_cpu.head_start = 1
    rows_p, val_p = emb_cpu._select_local_heads(
        torch.tensor([[1, 11]], dtype=torch.int32)
    )
    assert rows_p.shape == (1, 2) and not bool(val_p[0, 1])

    monkeypatch.setattr(
        engram_mod.dist,
        "all_gather_object",
        lambda out, val, group=None: out.__setitem__(0, val),
    )
    fake_grp = SimpleNamespace(
        rank_in_group=0,
        world_size=1,
        cpu_group=None,
        broadcast_object=lambda obj: obj,
    )
    monkeypatch.setattr(engram_mod, "get_engram_dp_group", lambda: fake_grp)
    with pytest.raises(ValueError, match="requires cpu_offload=True"):
        ParallelEngramEmbedding(
            20, 32, (10, 10), block_size=32, cpu_offload=False, dp_shared_memory=True
        )
    with pytest.raises(ValueError, match="size > 1"):
        ParallelEngramEmbedding(
            20, 32, (10, 10), block_size=32, cpu_offload=True, dp_shared_memory=True
        )
    monkeypatch.setattr(engram_mod, "get_engram_dp_size", lambda: 2)
    emb_shm = ParallelEngramEmbedding(
        20, 32, (10, 10), block_size=32, cpu_offload=True, dp_shared_memory=True
    )
    assert emb_shm.dp_size == 1 and emb_shm._storage()[0].shape == (20, 32)

    # 6 heads over 4 TP shards (2 per shard) would leave the last shard empty.
    monkeypatch.setattr(engram_mod, "get_engram_dp_size", lambda: 1)
    monkeypatch.setattr(engram_mod, "get_tensor_model_parallel_world_size", lambda: 4)
    with pytest.raises(AssertionError, match="without hash heads"):
        ParallelEngramEmbedding(60, 32, (10,) * 6, block_size=32)


def test_engram_module_staging_and_forward(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify Engram 1-D int32 host staging, HBM fallback, and RMS gating forward."""
    monkeypatch.setattr(engram_mod, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(
        engram_mod.upstream, "get_tensor_model_parallel_rank", lambda: 0
    )
    monkeypatch.setattr(engram_mod, "get_engram_dp_size", lambda: 1)
    cfg = SimpleNamespace(
        hidden_size=32,
        hc_mult=4,
        rms_norm_eps=1e-6,
        engram_layer_ids=(0,),
        engram_num_embeddings=(100,),
        engram_max_ngram_size=3,
        engram_n_heads=2,
        engram_head_dim=32,
        engram_compressed_vocab_size=16,
        engram_pad_token_id=0,
        engram_vocab_size=10,
    )
    layout = engram_mod.upstream.EngramLayout(cfg)

    def _fake_wkv(in_sz: int, out_sz: int, **kw: Any) -> Any:
        lin = torch.nn.Linear(in_sz, out_sz, bias=False, dtype=torch.bfloat16)
        torch.nn.init.ones_(lin.weight)
        return lin

    monkeypatch.setattr(engram_mod.upstream, "ColumnParallelLinear", _fake_wkv)

    def _make_vcfg(offload: bool) -> Any:
        return SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_batched_tokens=8),
            engram_config=SimpleNamespace(
                cpu_offload=offload, dp_shared_memory=False, use_thp=False
            ),
        )

    monkeypatch.setattr(engram_mod, "get_current_vllm_config", lambda: _make_vcfg(True))
    monkeypatch.setattr(
        engram_mod.upstream, "get_current_vllm_config", lambda: _make_vcfg(True)
    )
    with pytest.raises(NotImplementedError, match="sequence parallelism"):
        Engram(cfg, None, layout, 0, True, "model.layers.0.engram")
    eng_off = Engram(cfg, None, layout, 0, False, "model.layers.0.engram")
    eng_off.embed_tokens.weight.data.fill_(1.0)
    eng_off.embed_tokens.weight_scale_inv.data.fill_(127)
    eng_off.q_weight.data.fill_(0.5)
    eng_off.k_weight.data.fill_(0.5)
    assert eng_off.is_staged() is True
    assert eng_off._staged_raw is not None and eng_off._staged_raw.ndim == 1
    assert eng_off.staged_device_tensors() == ()
    # max_num_batched_tokens=8: a larger batch must not fall back silently.
    with pytest.raises(ValueError, match="num_tokens=9"):
        eng_off.prepare_embeddings(torch.zeros((9, 4), dtype=torch.int32))

    hashes = torch.zeros((4, 4), dtype=torch.int32)
    eng_off.prepare_embeddings(hashes)
    assert len(eng_off.staged_device_tensors()) == 2
    # Forward needing more tokens than the last prepare staged is an error.
    with pytest.raises(RuntimeError, match="num_tokens=8"):
        eng_off._ready_rows(8)
    eng_off.prepare_embeddings(hashes)
    h_in = torch.ones((4, 4, 32), dtype=torch.bfloat16)
    assert eng_off(h_in[:0], hashes[:0]).shape == (0, 4, 32)
    out1 = eng_off(h_in, hashes)
    out2 = eng_off(
        h_in, hashes, token_mask=torch.tensor([1, 0, 1, 0], dtype=torch.bool)
    )
    assert out1.shape == (4, 4, 32) and torch.equal(out2[1], h_in[1])

    eng_off.embed_tokens.dp_size = 2
    monkeypatch.setattr(engram_mod.upstream, "engram_gathered_num_tokens", lambda: 2)
    monkeypatch.setattr(
        engram_mod,
        "get_engram_dp_group",
        lambda: SimpleNamespace(
            world_size=2,
            rank_in_group=0,
            all_gather=lambda x, dim: torch.cat([x, x], dim=0),
        ),
    )
    assert eng_off._ready_rows(2).shape == (2, 8, 32)
    eng_off.embed_tokens.dp_size = 1
    eng_off.embed_tokens.tp_size = 2
    monkeypatch.setattr(
        engram_mod, "tensor_model_parallel_all_gather", lambda x, dim: x
    )
    assert eng_off.embed(hashes).shape == (4, 4, 32)

    monkeypatch.setattr(
        engram_mod, "get_current_vllm_config", lambda: _make_vcfg(False)
    )
    monkeypatch.setattr(
        engram_mod.upstream, "get_current_vllm_config", lambda: _make_vcfg(False)
    )
    eng_hbm = Engram(cfg, None, layout, 0, False, "model.layers.0.engram")
    eng_hbm.embed_tokens.weight.data.fill_(1.0)
    eng_hbm.embed_tokens.weight_scale_inv.data.fill_(127)
    eng_hbm.prepare_embeddings(hashes)
    assert eng_hbm._ready_rows(4).shape == (4, 4, 32)
    eng_hbm.embed_tokens.dp_size = 2
    monkeypatch.setattr(engram_mod.upstream, "engram_gathered_num_tokens", lambda: 2)
    assert eng_hbm._ready_rows(2).shape == (2, 8, 32)
    eng_hbm.embed_tokens.dp_size = 1

    # Multi-bucket whole-tensor staging parity (8 -> 2 -> 4 -> 2) vs HBM lookup.
    gen = torch.Generator().manual_seed(99)
    n_rows = eng_off.embed_tokens.part_num_embeddings
    w_fp8 = torch.randn((n_rows, 32), generator=gen).to(torch.float8_e4m3fn)
    s_u8 = torch.randint(124, 130, (n_rows, 1), generator=gen, dtype=torch.uint8)
    for mod in (eng_off, eng_hbm):
        mod.embed_tokens.weight.data.copy_(w_fp8)
        mod.embed_tokens.weight_scale_inv.data.copy_(s_u8)
    eng_off.embed_tokens.tp_size = eng_hbm.embed_tokens.tp_size = 1
    eng_off._active_raw = None
    uninit_rows = eng_off._ready_rows(2)
    assert uninit_rows.shape == (2, 4, 32) and torch.all(uninit_rows == 0.0)
    eng_off.prepare_embeddings(hashes[:0])
    assert eng_off._active_raw is not None and eng_off._active_raw.shape[0] == 0
    for b_sz in (8, 2, 4, 2):
        h_b = torch.randint(0, n_rows, (b_sz, 4), generator=gen, dtype=torch.int32)
        eng_off.prepare_embeddings(h_b)
        eng_hbm.prepare_embeddings(h_b)
        assert torch.equal(eng_off._ready_rows(b_sz), eng_hbm._ready_rows(b_sz))
    assert (
        eng_off._staged_by_tokens[8][0].data_ptr()
        != eng_off._staged_by_tokens[2][0].data_ptr()
    )
    # One host buffer per bucket, reused across steps (the H2D copy is sync).
    buf2 = eng_off._host_buf_by_tokens[2]
    assert buf2.data_ptr() != eng_off._host_buf_by_tokens[8].data_ptr()
    h_step0 = torch.zeros((2, 4), dtype=torch.int32)
    eng_off.prepare_embeddings(h_step0)
    assert eng_off._host_buf_by_tokens[2] is buf2
    # Invalid scales (255) over 0x7F (NaN in E4M3FN) raw bytes dequantize to 0.
    eng_off.embed_tokens.weight.data.view(torch.uint8).fill_(0x7F)
    eng_off.embed_tokens.weight_scale_inv.data.fill_(255)
    eng_off.prepare_embeddings(h_step0)
    nan_masked = eng_off._ready_rows(2)
    assert not torch.isnan(nan_masked).any() and torch.all(nan_masked == 0.0)

    # Gating forward vs an independent transcription of the upstream kernel.
    gen_fw = torch.Generator().manual_seed(7)
    for param, scale in (
        (eng_hbm.q_weight, 1.0),
        (eng_hbm.k_weight, 1.0),
        (eng_hbm.wkv.weight, 0.05),
    ):
        param.data.copy_(torch.randn(param.shape, generator=gen_fw) * scale)
    h_fw = torch.randint(0, n_rows, (4, 4), generator=gen_fw, dtype=torch.int32)
    x_fw = torch.randn((4, 4, 32), generator=gen_fw).to(torch.bfloat16)
    eng_hbm.prepare_embeddings(h_fw)
    kv_fw = eng_hbm.wkv(eng_hbm.embed(h_fw).flatten(-2))
    for mask in (None, torch.tensor([True, False, True, True])):
        torch.testing.assert_close(
            eng_hbm(x_fw, h_fw, token_mask=mask),
            _reference_engram_forward(eng_hbm, x_fw, kv_fw, mask),
            rtol=1e-2,
            atol=1e-2,
        )


def _reference_engram_forward(
    eng: Engram,
    hidden_states: torch.Tensor,
    kv: torch.Tensor,
    token_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Scalar transcription of upstream ``_fused_engram_post_wkv_kernel``."""
    dim, hc_mult = eng.dim, eng.hc_mult
    out = torch.empty_like(hidden_states)
    for t in range(hidden_states.shape[0]):
        value = kv[t, hc_mult * dim :].float()
        for hc in range(hc_mult):
            hidden = hidden_states[t, hc].float()
            key = kv[t, hc * dim : (hc + 1) * dim].float()
            q = eng.q_weight[hc].float()
            k = eng.k_weight[hc].float()
            hidden_rms = torch.rsqrt((hidden * hidden).sum() / dim + eng.eps)
            key_rms = torch.rsqrt((key * key).sum() / dim + eng.eps)
            dot = (hidden * q * k * key).sum() * hidden_rms * key_rms / dim**0.5
            gate_input = torch.sqrt(torch.clamp(dot.abs(), min=eng.clamp_value))
            gate = torch.sigmoid(-gate_input if dot < 0 else gate_input)
            if token_mask is not None and not bool(token_mask[t]):
                gate = torch.zeros_like(gate)
            out[t, hc] = (hidden + gate * value).to(out.dtype)
    return out


def test_ngram_hash_state_skips_null_block_and_padded_slots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify writes skip null block 0 and padded slots, and block 0 reads as DEAD."""
    state = _make_ngram_state(monkeypatch)
    state.swa_cache_module = SimpleNamespace(
        block_size=4, kv_cache=torch.zeros((3, 1, 4, 64), dtype=torch.bfloat16)
    )
    assert state.ensure_cache() and state._cache is not None
    # 3 blocks x 4 slots, plus the trailing dummy slot for masked writes.
    assert state._cache.numel() == 13
    state._cache[:4].fill_(777)

    # Null-block slot 0 and padded slot -1 are dropped; block 2 (slot 8) writes.
    ids = torch.tensor([5, 6, 7], dtype=torch.int32)
    pos = torch.tensor([0, 0, 0], dtype=torch.int32)
    qsl = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    dead = torch.zeros(3, dtype=torch.bool)
    lb = torch.full((3, 3), -1, dtype=torch.int32)
    lbd = torch.zeros((3, 3), dtype=torch.bool)
    state(
        ids,
        pos,
        qsl,
        dead,
        lb,
        lbd,
        torch.tensor([0, 8, -1], dtype=torch.int32),
        torch.tensor([[0], [2], [0]], dtype=torch.int32),
    )
    expected_cache = torch.zeros(12, dtype=torch.int32)
    expected_cache[:4] = 777
    expected_cache[8] = state.token_map[6]
    assert torch.equal(state._cache[:-1], expected_cache)

    # Fallback on null block 0 must read as DEAD_ID -> pad_id, matching an
    # unmapped (-1) block rather than reading the poisoned 777.
    h_null = state(
        torch.tensor([7], dtype=torch.int32),
        torch.tensor([4], dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int32),
        torch.zeros(1, dtype=torch.bool),
        torch.full((1, 3), -1, dtype=torch.int32),
        torch.zeros((1, 3), dtype=torch.bool),
        torch.tensor([9], dtype=torch.int32),
        torch.tensor([[0, 2]], dtype=torch.int32),
    )
    h_neg = state(
        torch.tensor([7], dtype=torch.int32),
        torch.tensor([4], dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int32),
        torch.zeros(1, dtype=torch.bool),
        torch.full((1, 3), -1, dtype=torch.int32),
        torch.zeros((1, 3), dtype=torch.bool),
        torch.tensor([9], dtype=torch.int32),
        torch.tensor([[-1, 2]], dtype=torch.int32),
    )
    assert torch.equal(h_null, h_neg)


@pytest.mark.tpu_test
def test_tpu_engram_dequant_and_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # pragma: no cover
    """Execute /dev/shm storage, NgramHashState, and 1-D int32 staging on TPU."""
    import torch_tpu  # noqa: F401

    dev = torch.device("tpu:0")
    raw = torch.ones((4, 64), dtype=torch.float8_e4m3fn, device=dev).view(torch.uint8)
    scale = torch.full((4, 2), 127, dtype=torch.uint8, device=dev)
    scale[3, 1] = 255
    deq = dequant_fp8_rows(raw, scale, block_size=32).cpu()
    assert deq.shape == (4, 64)
    assert deq[0].sum().item() == 64.0 and deq[3, 32:].sum().item() == 0.0

    # Verify NgramHashState on TPU (real int64 multipliers) vs the reference.
    state = _make_real_ngram_state(monkeypatch).to(dev)
    state.swa_cache_module = SimpleNamespace(
        block_size=4,
        kv_cache=torch.zeros((8, 4, 16), dtype=torch.bfloat16, device=dev),
    )
    assert state.ensure_cache() and state._cache is not None
    assert state._cache.device.type == "tpu"
    _seed_hash_history(state)
    inputs_cpu = _reference_hash_inputs()
    ids_cpu, pos_cpu, qsl_cpu, dead_cpu, lb_cpu, lbd_cpu, _, bt_cpu = inputs_cpu
    h_slot_tpu = state(*(t.to(dev) for t in inputs_cpu))
    assert h_slot_tpu.device.type == "tpu"
    h_slot_ref = _reference_hashes(
        state,
        ids_cpu,
        pos_cpu,
        qsl_cpu,
        dead_cpu,
        lb_cpu,
        lbd_cpu,
        bt_cpu,
        state._cache[:-1].cpu(),
    )
    assert torch.equal(h_slot_tpu.cpu(), h_slot_ref)

    state.use_slot_cache = False
    h_noslot_tpu = state(
        *(t.to(dev) for t in inputs_cpu[:6]),
        None,
        None,
    )
    h_host = _reference_hashes(
        state, ids_cpu, pos_cpu, qsl_cpu, dead_cpu, lb_cpu, lbd_cpu, None, None
    )
    assert torch.equal(h_noslot_tpu.cpu(), h_host)

    # Verify POSIX /dev/shm storage + 1-D int32 H2D staging + Engram.forward().
    monkeypatch.setattr(engram_mod, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(
        engram_mod.upstream, "get_tensor_model_parallel_rank", lambda: 0
    )
    # dp_shared_memory needs an Engram DP group of size > 1.
    monkeypatch.setattr(engram_mod, "get_engram_dp_size", lambda: 2)
    monkeypatch.setattr(
        engram_mod.dist,
        "all_gather_object",
        lambda out, val, group=None: out.__setitem__(0, val),
    )
    fake_grp = SimpleNamespace(
        rank_in_group=0,
        world_size=1,
        cpu_group=None,
        broadcast_object=lambda obj: obj,
    )
    monkeypatch.setattr(engram_mod, "get_engram_dp_group", lambda: fake_grp)

    cfg = SimpleNamespace(
        hidden_size=32,
        hc_mult=4,
        rms_norm_eps=1e-6,
        engram_layer_ids=(0,),
        engram_num_embeddings=(512,),
        engram_max_ngram_size=4,
        engram_n_heads=2,
        engram_head_dim=32,
        engram_compressed_vocab_size=32,
        engram_pad_token_id=0,
        engram_vocab_size=32,
    )
    layout = engram_mod.upstream.EngramLayout(cfg)

    def _dense_wkv(in_sz: int, out_sz: int, **kw: Any) -> Any:
        return torch.nn.Linear(in_sz, out_sz, bias=False, dtype=torch.bfloat16)

    monkeypatch.setattr(engram_mod.upstream, "ColumnParallelLinear", _dense_wkv)
    vcfg_shm = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8),
        engram_config=SimpleNamespace(
            cpu_offload=True, dp_shared_memory=True, use_thp=False
        ),
    )
    monkeypatch.setattr(engram_mod, "get_current_vllm_config", lambda: vcfg_shm)
    monkeypatch.setattr(
        engram_mod.upstream, "get_current_vllm_config", lambda: vcfg_shm
    )

    eng_cpu = Engram(cfg, None, layout, 0, False, "model.layers.0.engram")
    with torch.device(dev):
        eng_tpu = Engram(cfg, None, layout, 0, False, "model.layers.0.engram")

    assert isinstance(eng_tpu.embed_tokens._shared_memory, TpuDPSharedEngramStorage)
    assert eng_tpu.embed_tokens.weight.device.type == "cpu"
    assert (
        eng_tpu._staged_raw is not None
        and eng_tpu._staged_raw.ndim == 1
        and eng_tpu._staged_raw.dtype == torch.int32
        and eng_tpu._staged_raw.device.type == "tpu"
    )

    gen = torch.Generator(device="cpu").manual_seed(2026)
    n_rows = eng_cpu.embed_tokens.part_num_embeddings
    w_fp8 = torch.randn((n_rows, 32), generator=gen, dtype=torch.float32).to(
        torch.float8_e4m3fn
    )
    s_u8 = torch.randint(125, 130, (n_rows, 1), generator=gen, dtype=torch.uint8)
    s_u8[0] = 255
    wkv_w = (
        torch.randn(eng_cpu.wkv.weight.shape, generator=gen, dtype=torch.bfloat16) * 0.1
    )
    q_w = torch.randn(eng_cpu.q_weight.shape, generator=gen, dtype=torch.bfloat16)
    k_w = torch.randn(eng_cpu.k_weight.shape, generator=gen, dtype=torch.bfloat16)

    for mod, m_dev in ((eng_cpu, torch.device("cpu")), (eng_tpu, dev)):
        mod.embed_tokens.weight.data.copy_(w_fp8)
        mod.embed_tokens.weight_scale_inv.data.copy_(s_u8)
        mod.wkv.weight.data.copy_(wkv_w.to(m_dev))
        mod.q_weight.data.copy_(q_w.to(m_dev))
        mod.k_weight.data.copy_(k_w.to(m_dev))

    # Same layout as _REAL_HASH_CFG, so layer 0 hashes index this table.
    layer_hashes = h_host[:4, 0, :]
    h_in = torch.randn((4, 4, 32), generator=gen, dtype=torch.bfloat16)
    mask = torch.tensor([True, False, True, True], dtype=torch.bool)

    eng_cpu.prepare_embeddings(layer_hashes)
    out_cpu = eng_cpu(h_in, layer_hashes, token_mask=mask)

    eng_tpu.prepare_embeddings(layer_hashes)
    out_tpu = eng_tpu(h_in.to(dev), layer_hashes.to(dev), token_mask=mask.to(dev)).cpu()

    torch.testing.assert_close(out_tpu, out_cpu, rtol=1.6e-2, atol=1.6e-2)
    assert torch.equal(out_tpu[~mask], h_in[~mask])
    assert not torch.allclose(out_tpu[mask], h_in[mask], rtol=1e-3, atol=1e-3)

    # Multi-bucket (8 -> 2 -> 4 -> 2) whole-tensor H2D staging, one host buffer each.
    for b_sz in (8, 2, 4, 2):
        h_b = torch.randint(
            0, n_rows, (b_sz, layer_hashes.shape[1]), generator=gen, dtype=torch.int32
        )
        x_b = torch.randn((b_sz, 4, 32), generator=gen, dtype=torch.bfloat16)
        eng_cpu.prepare_embeddings(h_b)
        ref_b = eng_cpu(x_b, h_b)
        eng_tpu.prepare_embeddings(h_b)
        assert b_sz in eng_tpu._staged_by_tokens and b_sz in eng_tpu._host_buf_by_tokens
        assert eng_tpu._active_staged is not None
        got_b = eng_tpu(x_b.to(dev), h_b.to(dev)).cpu()
        torch.testing.assert_close(got_b, ref_b, rtol=1.6e-2, atol=1.6e-2)
