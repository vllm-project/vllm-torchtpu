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
"""TPU implementation of DeepSeek-V4.1 Engram conditional memory."""

import ctypes
import functools
import hashlib
import os
import subprocess
import weakref
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax.sharding import PartitionSpec as P
from torch import nn
from torch_tpu._internal import pallas
from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.utils import set_weight_attrs
from vllm.models.deepseek_v41.common.engram import (
    DEAD_ID,
    EngramLayout,
    build_compressed_token_map,
    compute_hash_multipliers,
)

from vllm_torchtpu import envs
from vllm_torchtpu.distributed.tp_collectives import (
    tp_all_gather,
    tp_shard_slice,
    tp_token_shard_ok,
)
from vllm_torchtpu.layers.core.quantization import e8m0_to_fp32
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    get_pallas_op_mesh,
    get_vllm_model_wrapper_context,
)

logger = init_logger(__name__)

_pallas_op_cache: dict[str, Any] = {}
_R = P()


def _engram_hash_jax(
    input_ids: jax.Array,  # i32[T]
    pos: jax.Array,  # i32[T]
    query_start_loc: jax.Array,  # i32[max_num_seqs + 1]
    swa_block_table: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    distribution: jax.Array,  # i32[3]
    hash_cache: jax.Array,  # i32[num_slots] (donated)
    token_map: jax.Array,  # i32[vocab_size]
    mult_limbs: jax.Array,  # i32[n_layers, max_ngram, 9]
    primes: jax.Array,  # i32[n_layers, max_ngram - 1, n_heads]
    offsets: jax.Array,  # i32[n_layers, (max_ngram - 1) * n_heads]
    *,
    swa_block_size: int,
    pad_id: int,
    max_ngram: int,
) -> tuple[jax.Array, jax.Array]:
    num_tokens = input_ids.shape[0]
    max_num_seqs = query_start_loc.shape[0] - 1
    pages_per_seq = swa_block_table.shape[0] // max_num_seqs
    max_pos = pages_per_seq * swa_block_size - 1
    num_slots = hash_cache.shape[0]
    num_pages = num_slots // swa_block_size

    token_idx = jnp.arange(num_tokens, dtype=jnp.int32)
    eff_max_seqs = min(max_num_seqs, max(8, ((num_tokens + 7) // 8) * 8))
    req = jnp.sum(
        (token_idx[:, None] >= query_start_loc[1:eff_max_seqs][None, :]).astype(
            jnp.int32
        ),
        axis=1,
    )
    valid = token_idx < query_start_loc[distribution[2]]
    req_start_tok = query_start_loc[req]
    compressed = token_map[jnp.clip(input_ids, 0, token_map.shape[0] - 1)]

    def _page_base(p_idx: jax.Array) -> jax.Array:
        page = swa_block_table[req * pages_per_seq + p_idx]
        if num_pages > 1:
            page = jnp.where(page > 0, ((page - 1) % (num_pages - 1)) + 1, 0)
        return page * swa_block_size

    c0 = jnp.clip(pos, 0, max_pos)
    p_idx0 = c0 // swa_block_size
    base0 = _page_base(p_idx0)
    slot0 = jnp.clip(base0 + (c0 % swa_block_size), 0, num_slots - 1)

    blocked = (pos < 0) | (compressed == DEAD_ID) | (~valid)
    tokens = [jnp.where(blocked, jnp.int32(pad_id), compressed)]

    if max_ngram > 1:
        base_prev = _page_base(jnp.maximum(p_idx0 - 1, 0))
        for shift in range(1, max_ngram):
            lookback = pos - shift
            in_batch = (token_idx - shift) >= req_start_tok
            if shift < num_tokens:
                batch_val = jnp.concatenate(
                    [
                        jnp.zeros((shift,), dtype=compressed.dtype),
                        compressed[: num_tokens - shift],
                    ],
                    axis=0,
                )
            else:
                batch_val = jnp.zeros_like(compressed)
            c = jnp.clip(lookback, 0, max_pos)
            p_idx = c // swa_block_size
            if max_ngram <= swa_block_size:
                base = jnp.where(p_idx == p_idx0, base0, base_prev)
            else:
                base = _page_base(p_idx)
            slot = jnp.clip(base + (c % swa_block_size), 0, num_slots - 1)
            src = jnp.where(in_batch, batch_val, hash_cache[slot])
            blocked = blocked | (lookback < 0) | (src == DEAD_ID) | (~valid)
            tokens.append(jnp.where(blocked, jnp.int32(pad_id), src))

    hash_cache = hash_cache.at[jnp.where(valid, slot0, num_slots)].set(
        compressed, mode="drop"
    )

    n_layers, n_orders, n_heads = primes.shape
    lo_dim = n_layers * n_orders
    if num_tokens >= 384:
        p_u32_ht = primes.reshape(lo_dim, n_heads).T.astype(jnp.uint32)[:, :, None]
        w_ht = jnp.ones_like(p_u32_ht)
        tok_2d = jnp.stack(tokens, axis=0)
        m_t = mult_limbs.transpose(2, 1, 0)[:, :, :, None]
        p_all = m_t * tok_2d[None, :, None, :]
        carry = jnp.zeros((max_ngram, n_layers, num_tokens), dtype=jnp.int32)
        total_sum_ht = jnp.zeros((n_heads, lo_dim, num_tokens), dtype=jnp.uint32)
        for k in range(9):
            v = p_all[k] + carry
            limb_k = v & jnp.int32(0x7F)
            carry = v >> 7
            roll = limb_k[0]
            xor_orders = []
            for i in range(1, max_ngram):
                roll = jnp.bitwise_xor(roll, limb_k[i])
                xor_orders.append(roll)
            lk_ht = (
                jnp.stack(xor_orders, axis=1)
                .reshape(1, lo_dim, num_tokens)
                .astype(jnp.uint32)
            )
            total_sum_ht = total_sum_ht + lk_ht * w_ht
            w_ht = (w_ht << np.uint32(7)) % p_u32_ht
        rem_ht = (total_sum_ht % p_u32_ht).astype(jnp.int32)
        rem_tlh = rem_ht.transpose(2, 1, 0).reshape(
            num_tokens, n_layers, n_orders * n_heads
        )
    else:
        p_u32_3d = primes.reshape(1, lo_dim, n_heads).astype(jnp.uint32)
        w_3d = jnp.ones_like(p_u32_3d)
        tok_2d = jnp.stack(tokens, axis=1)
        m_t = mult_limbs.transpose(2, 1, 0)[:, None, :, :]
        p_all = tok_2d[None, :, :, None] * m_t
        carry = jnp.zeros((num_tokens, max_ngram, n_layers), dtype=jnp.int32)
        total_sum_3d = jnp.zeros((num_tokens, lo_dim, n_heads), dtype=jnp.uint32)
        for k in range(9):
            v = p_all[k] + carry
            limb_k = v & jnp.int32(0x7F)
            carry = v >> 7
            roll = limb_k[:, 0, :]
            xor_orders = []
            for i in range(1, max_ngram):
                roll = jnp.bitwise_xor(roll, limb_k[:, i, :])
                xor_orders.append(roll)
            lk = (
                jnp.stack(xor_orders, axis=2)
                .reshape(num_tokens, lo_dim, 1)
                .astype(jnp.uint32)
            )
            total_sum_3d = total_sum_3d + lk * w_3d
            w_3d = (w_3d << np.uint32(7)) % p_u32_3d
        rem_3d = (total_sum_3d % p_u32_3d).astype(jnp.int32)
        rem_tlh = rem_3d.reshape(num_tokens, n_layers, n_orders * n_heads)

    out_hashes = rem_tlh + offsets[None, :, :]
    out_hashes = jnp.where(valid[:, None, None], out_hashes, jnp.int32(-1))
    return out_hashes, hash_cache


class TpuNgramHashState(nn.Module):
    """TPU n-gram hash state using 9x7-bit int32 Horner arithmetic."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        layout: EngramLayout,
        swa_cache_module: nn.Module,
    ) -> None:
        super().__init__()
        self.layout = layout
        self.swa_cache_module = swa_cache_module
        self._cache: torch.Tensor | None = None
        self._kv_cache_ref: weakref.ReferenceType[torch.Tensor] | None = None

        model_config = vllm_config.model_config
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_config.tokenizer,
            trust_remote_code=model_config.trust_remote_code,
            revision=model_config.revision,
        )
        token_map, vocab_size = build_compressed_token_map(tokenizer)
        if vocab_size != layout.compressed_vocab_size:
            raise ValueError(
                f"Compressed vocab size {vocab_size} != {layout.compressed_vocab_size}"
            )
        self.pad_id = int(token_map[layout.pad_token_id])
        multipliers = compute_hash_multipliers(
            layout.layer_ids, layout.max_ngram_size, vocab_size
        )
        mult_limbs = torch.stack(
            [(multipliers >> (7 * k)) & 0x7F for k in range(9)], dim=-1
        ).to(torch.int32)

        self.register_buffer(
            "token_map", torch.tensor(token_map, dtype=torch.int32), persistent=False
        )
        self.register_buffer("mult_limbs", mult_limbs, persistent=False)
        self.register_buffer(
            "primes", torch.tensor(layout.primes, dtype=torch.int32), persistent=False
        )
        self.register_buffer(
            "offsets", layout.offsets.to(torch.int32), persistent=False
        )
        logger.info(
            "Built TPU engram token map (%d -> %d ids) for layers %s",
            len(token_map),
            vocab_size,
            layout.layer_ids,
        )
        self._token_map_np = np.asarray(token_map, dtype=np.int64)
        self._mult_np = multipliers.cpu().numpy()
        self._primes_np = np.asarray(layout.primes, dtype=np.int64)
        self._offsets_np = layout.offsets.cpu().numpy().astype(np.int64)

    def cpu_hashes(
        self,
        token_ids: np.ndarray,
        req_idx: np.ndarray,
        pos: np.ndarray,
        *,
        span_start_by_req: np.ndarray | None = None,
        span_tokens: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute _engram_hash_jax hashes on CPU from host prompt token history."""
        tm = self._token_map_np
        blocked = np.zeros(pos.shape[0], dtype=np.bool_)
        toks = []
        for shift in range(self.layout.max_ngram_size):
            lookback = pos - shift
            blocked |= lookback < 0
            clipped_lb = np.maximum(lookback, 0)
            ids = token_ids[req_idx, clipped_lb]
            if span_start_by_req is not None and span_tokens is not None:
                in_span = lookback >= span_start_by_req[req_idx]
                if in_span.any():
                    ids = ids.copy()
                    rel = np.clip(
                        lookback[in_span] - span_start_by_req[req_idx[in_span]],
                        0,
                        span_tokens.shape[1] - 1,
                    )
                    ids[in_span] = span_tokens[req_idx[in_span], rel]
            ids = np.clip(ids, 0, tm.shape[0] - 1)
            toks.append(np.where(blocked, self.pad_id, tm[ids]))
        roll = toks[0][:, None] * self._mult_np[None, :, 0]
        out = []
        for i in range(1, self.layout.max_ngram_size):
            roll = roll ^ (toks[i][:, None] * self._mult_np[None, :, i])
            out.append(roll[:, :, None] % self._primes_np[None, :, i - 1, :])
        return (np.concatenate(out, axis=-1) + self._offsets_np[None]).astype(np.int32)

    def cpu_hashes_tail(
        self, token_ids: np.ndarray, req_idx: np.ndarray, pos: np.ndarray
    ) -> np.ndarray:
        """Precompute shift=1..max_ngram-1 cumulative XORs while shift=0 is in flight."""
        tm = self._token_map_np
        max_ngram = self.layout.max_ngram_size
        n_layers = self._mult_np.shape[0]
        cum_xor = np.zeros((pos.shape[0], n_layers, max_ngram - 1), dtype=np.int64)
        running = np.zeros((pos.shape[0], n_layers), dtype=np.int64)
        for shift in range(1, max_ngram):
            lookback = pos - shift
            ids = np.clip(
                token_ids[req_idx, np.maximum(lookback, 0)], 0, tm.shape[0] - 1
            )
            t_s = np.where(lookback < 0, self.pad_id, tm[ids])
            running = running ^ (t_s[:, None] * self._mult_np[None, :, shift])
            cum_xor[:, :, shift - 1] = running
        return cum_xor

    def cpu_hashes_finish_tok0(
        self, ids_0: np.ndarray, pos: np.ndarray, cum_xor: np.ndarray
    ) -> np.ndarray:
        """Combine resolved shift=0 token IDs with precomputed shift=1..max_ngram-1 XORs."""
        tm = self._token_map_np
        clipped_0 = np.clip(ids_0, 0, tm.shape[0] - 1)
        t_0 = np.where(pos < 0, self.pad_id, tm[clipped_0])
        term_0 = t_0[:, None, None] * self._mult_np[None, :, 0, None]
        rolled = term_0 ^ cum_xor
        hashed = (rolled[:, :, :, None] % self._primes_np[None, :, :, :]).reshape(
            pos.shape[0], self._mult_np.shape[0], -1
        ) + self._offsets_np[None, :, :]
        return hashed.astype(np.int32)

    @property
    def block_size(self) -> int:
        return self.swa_cache_module.block_size

    def ensure_cache(self) -> bool:
        kv_cache = getattr(self.swa_cache_module, "kv_cache", None)
        if kv_cache is None or kv_cache.numel() == 0:
            self._cache = None
            self._kv_cache_ref = None
            return False
        if self._kv_cache_ref is not None and self._kv_cache_ref() is kv_cache:
            return True
        self._cache = torch.full(
            (kv_cache.shape[0] * self.block_size,),
            self.pad_id,
            dtype=torch.int32,
            device=kv_cache.device,
        )
        self._kv_cache_ref = weakref.ref(kv_cache)
        return True

    def dummy_hashes(self, input_ids: torch.Tensor) -> torch.Tensor:
        return (
            (input_ids.to(torch.int32) * 0)[:, None, None]
            .expand(-1, self.primes.shape[0], self.offsets.shape[1])
            .contiguous()
        )

    @property
    def compressor_op(self):
        # Runner prebuilds ops through this attribute before tracing.
        return self._build_compressor_op()

    def _build_compressor_op(self):
        op_name = (
            f"pallas::deepseek_v41_engram_hash_b{self.block_size}"
            f"_p{self.pad_id}_n{self.layout.max_ngram_size}"
        )
        op = _pallas_op_cache.get(op_name)
        if op is None:
            num_layers = len(self.layout.layer_ids)
            n_hash_cols = (self.layout.max_ngram_size - 1) * self.layout.n_heads
            op = pallas.jax_op(
                op_name,
                functools.partial(
                    _engram_hash_jax,
                    swa_block_size=self.block_size,
                    pad_id=self.pad_id,
                    max_ngram=self.layout.max_ngram_size,
                ),
                mesh=get_pallas_op_mesh(),
                donate_argnums=(5,),
                input_partition_specs=(_R,) * 10,
            )
            op.register_fake(
                lambda input_ids, *args: (
                    input_ids.new_empty(
                        (input_ids.shape[0], num_layers, n_hash_cols), dtype=torch.int32
                    ),
                    torch.empty_like(args[4]),
                )
            )
            _pallas_op_cache[op_name] = op
        return op

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        query_start_loc: torch.Tensor,
        swa_block_table: torch.Tensor,
        distribution: torch.Tensor,
    ) -> torch.Tensor:
        assert self._cache is not None
        hashes, new_cache = self._build_compressor_op()(
            input_ids if input_ids.dtype == torch.int32 else input_ids.to(torch.int32),
            positions if positions.dtype == torch.int32 else positions.to(torch.int32),
            query_start_loc,
            swa_block_table.reshape(-1),
            distribution,
            self._cache,
            self.token_map,
            self.mult_limbs,
            self.primes,
            self.offsets,
        )
        self._cache.copy_(new_cache)
        return hashes


_CPU_ENGRAM_WEIGHTS: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
_NEXT_ENGRAM_TABLE_ID = 0
# 128-slot CPU output ring pre-sized for max_num_batched_tokens=4096.
_OUT_CPU_BUFS: list[torch.Tensor | None] = [None] * 128
_OUT_CPU_SLOT: int = 0
_DEV_ENGRAM_RING: list[torch.Tensor] = []
_DEV_ENGRAM_SLOT: int = 0
_OUT_CPU_MIN_TOKENS = 4096
_PIN_HOST_BUFS: bool = True
_ENGRAM_PTRS_CACHE: dict[tuple[int, ...], tuple[Any, ...]] = {}
_ENGRAM_PREFETCH: dict[str, Any] | None = None
_ENGRAM_BG_POOL: ThreadPoolExecutor | None = None
_ENGRAM_PREFETCH_FUTURE: Future[dict[str, Any] | None] | None = None
MADV_HUGEPAGE = 14

# OpenMP row gather from the CPU tables into the packed [value | scale] layout.
_ENGRAM_GATHER_CPP = r"""
#include <cstdint>
#include <cstring>
#include <omp.h>

extern "C" void engram_gather_omp(
    const int32_t* __restrict__ hashes, const uint8_t* const* __restrict__ w_ptrs,
    const uint8_t* const* __restrict__ s_ptrs, const int64_t* __restrict__ max_rows,
    uint8_t* __restrict__ out, int64_t num_tokens, int64_t n_tables,
    int64_t n_hash_cols, int64_t val_dim, int64_t scl_dim, int64_t packed_dim
) {
    const int64_t total_work = num_tokens * n_tables * n_hash_cols;
    if (total_work <= 0) return;
    const int max_t = omp_get_max_threads();
    const int n_threads = (max_t > 0) ? max_t : 1;
    #pragma omp parallel num_threads(n_threads)
    {
        const int tid = omp_get_thread_num();
        const int nt = omp_get_num_threads();
        int64_t work = (total_work * tid) / nt;
        const int64_t w_end = (total_work * (tid + 1)) / nt;
        int64_t item = work / n_hash_cols;
        int64_t c = work % n_hash_cols;
        int64_t idx = item % n_tables;
        while (work < w_end) {
            const int64_t rem = w_end - work;
            const int64_t c_end = (n_hash_cols - c < rem) ? n_hash_cols : (c + rem);
            const int64_t span = c_end - c;
            const uint8_t* w_ptr = w_ptrs[idx];
            const uint8_t* s_ptr = s_ptrs[idx];
            const int32_t* h_base = hashes + item * n_hash_cols;
            const bool split_2d = (packed_dim >= n_hash_cols * (val_dim + scl_dim));
            const int64_t val_section = n_hash_cols * val_dim;
            const int64_t scl_section = n_hash_cols * scl_dim;
            const int64_t pad_bytes = packed_dim - (val_section + scl_section);
            uint8_t* item_dst = out + item * packed_dim;
            uint8_t* dst = out + work * packed_dim;
            if (!w_ptr || !s_ptr || h_base[0] < 0) {
                if (split_2d) {
                    std::memset(item_dst + c * val_dim, 0, span * val_dim);
                    std::memset(item_dst + val_section + c * scl_dim, 0, span * scl_dim);
                    if (c_end == n_hash_cols && pad_bytes > 0) {
                        std::memset(item_dst + val_section + scl_section, 0, pad_bytes);
                    }
                } else {
                    std::memset(dst, 0, span * packed_dim);
                }
            } else {
                const int64_t max_row = max_rows[idx];
                const int64_t next_idx = (idx + 1 == n_tables) ? 0 : (idx + 1);
                const uint8_t* w_next = w_ptrs[next_idx];
                const uint8_t* s_next = s_ptrs[next_idx];
                const int64_t mr_next = max_rows[next_idx];
                const int32_t* h_next_base = h_base + n_hash_cols;
                for (int64_t col = c; col < c_end; ++col, dst += packed_dim) {
                    const int64_t pf_work = work + (col - c) + 4;
                    if (pf_work < w_end) {
                        if (col + 4 < n_hash_cols) {
                            int64_t h_pf = h_base[col + 4];
                            if (h_pf >= 0) {
                                if (h_pf > max_row) h_pf = max_row;
                                const uint8_t* wp = w_ptr + h_pf * val_dim;
                                for (int64_t o = 0; o < val_dim; o += 64)
                                    __builtin_prefetch(wp + o, 0, 1);
                                __builtin_prefetch(s_ptr + h_pf * scl_dim, 0, 1);
                            }
                        } else if (
                            col + 4 - n_hash_cols < n_hash_cols &&
                            w_next && s_next && h_next_base[0] >= 0
                        ) {
                            int64_t h_pf = h_next_base[col + 4 - n_hash_cols];
                            if (h_pf >= 0) {
                                if (h_pf > mr_next) h_pf = mr_next;
                                const uint8_t* wp = w_next + h_pf * val_dim;
                                for (int64_t o = 0; o < val_dim; o += 64)
                                    __builtin_prefetch(wp + o, 0, 1);
                                __builtin_prefetch(s_next + h_pf * scl_dim, 0, 1);
                            }
                        }
                    }
                    int64_t h = h_base[col];
                    h = h < 0 ? 0 : (h > max_row ? max_row : h);
                    if (split_2d) {
                        std::memcpy(item_dst + col * val_dim, w_ptr + h * val_dim, val_dim);
                        std::memcpy(
                            item_dst + val_section + col * scl_dim,
                            s_ptr + h * scl_dim,
                            scl_dim
                        );
                    } else {
                        std::memcpy(dst, w_ptr + h * val_dim, val_dim);
                        std::memcpy(dst + val_dim, s_ptr + h * scl_dim, scl_dim);
                    }
                }
                if (split_2d && c_end == n_hash_cols && pad_bytes > 0) {
                    std::memset(item_dst + val_section + scl_section, 0, pad_bytes);
                }
            }
            work += span;
            c = 0;
            ++item;
            if (++idx == n_tables) idx = 0;
        }
    }
}
"""

_ENGRAM_GATHER_FLAGS = ["-O3", "-march=native", "-fopenmp", "-shared", "-fPIC"]


@functools.cache
def _get_engram_gather_lib() -> ctypes.CDLL | None:
    """Build (once per host) and load the OpenMP Engram gather kernel."""
    key = " ".join([_ENGRAM_GATHER_CPP, *_ENGRAM_GATHER_FLAGS])
    so_path = (
        f"/tmp/libengram_gather_omp_{hashlib.sha256(key.encode()).hexdigest()[:12]}.so"
    )
    try:
        if not os.path.exists(so_path):
            tmp_so = f"{so_path}.{os.getpid()}"
            proc = subprocess.run(
                ["g++", *_ENGRAM_GATHER_FLAGS, "-x", "c++", "-", "-o", tmp_so],
                input=_ENGRAM_GATHER_CPP.encode(),
                capture_output=True,
            )
            if proc.returncode != 0:
                logger.warning(
                    "Engram C++ gather build failed:\n%s",
                    proc.stderr.decode(errors="replace"),
                )
                return None
            os.replace(tmp_so, so_path)
        lib = ctypes.CDLL(so_path)
    except OSError as e:
        logger.warning("Engram C++ gather unavailable (%s)", e)
        return None
    lib.engram_gather_omp.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int64] * 6
    lib.engram_gather_omp.restype = None
    return lib


def _engram_table_bytes(n_hash_cols: int, packed_dim: int) -> int:
    """Return per-table byte stride (6400 for 2D tile-aligned layout, or n_hash_cols * packed_dim)."""
    return packed_dim if packed_dim >= n_hash_cols * 64 else n_hash_cols * packed_dim


class _EngramTableSlice:
    """2D [max(t_local, 32), table_bytes] view for a single Engram table with logical token count."""

    __slots__ = ("buf_2d", "num_tokens", "shape", "ndim", "dtype", "device")

    def __init__(self, buf_2d: torch.Tensor, num_tokens: int) -> None:
        self.buf_2d = buf_2d
        self.num_tokens = num_tokens
        self.shape = (num_tokens, buf_2d.shape[1])
        self.ndim = 2
        self.dtype = buf_2d.dtype
        self.device = buf_2d.device


class _EngramTableSlices:
    """Indexable [:, table_idx] wrapper over a 2D [T_pad, n_tables * table_bytes] buffer."""

    __slots__ = ("_buf_2d", "_num_tokens", "_table_bytes")

    def __init__(self, buf_2d: torch.Tensor, num_tokens: int, table_bytes: int) -> None:
        t_u8 = min(max(num_tokens, 32), buf_2d.shape[0])
        self._buf_2d = buf_2d[:t_u8]
        self._num_tokens = num_tokens
        self._table_bytes = table_bytes

    def __getitem__(self, key: Any) -> _EngramTableSlice:
        idx = key[1] if isinstance(key, tuple) else key
        lo = int(idx) * self._table_bytes
        hi = lo + self._table_bytes
        return _EngramTableSlice(self._buf_2d[:, lo:hi], self._num_tokens)


def _next_host_buf(nbytes: int, row_span: int) -> torch.Tensor:
    global _OUT_CPU_SLOT, _PIN_HOST_BUFS
    slot = _OUT_CPU_SLOT
    _OUT_CPU_SLOT = (slot + 1) & 127
    buf = _OUT_CPU_BUFS[slot]
    needed = max(nbytes, _OUT_CPU_MIN_TOKENS * row_span) + 512
    if buf is None or buf.numel() < needed:
        alloc_bytes = needed
        if _PIN_HOST_BUFS:
            try:
                buf = torch.empty((alloc_bytes,), dtype=torch.uint8, pin_memory=True)
            except Exception as e:
                logger.warning("Engram: pinned host buffer unavailable (%s)", e)
                _PIN_HOST_BUFS = False
        if not _PIN_HOST_BUFS:
            buf = torch.empty((alloc_bytes,), dtype=torch.uint8)
        _OUT_CPU_BUFS[slot] = buf
    return buf


def _ensure_engram_ptrs(table_ids: list[int]) -> tuple[Any, ...]:
    key = tuple(table_ids)
    cached = _ENGRAM_PTRS_CACHE.get(key)
    if cached is None:
        n_tables = len(table_ids)
        w_ptrs = (ctypes.c_void_p * n_tables)()
        s_ptrs = (ctypes.c_void_p * n_tables)()
        max_rows = (ctypes.c_int64 * n_tables)()
        val_dim = scl_dim = 0
        pairs = [_CPU_ENGRAM_WEIGHTS.get(tid) for tid in table_ids]
        for idx, pair in enumerate(pairs):
            if pair is not None:
                w_u8, s_u8 = pair
                w_ptrs[idx], s_ptrs[idx] = w_u8.data_ptr(), s_u8.data_ptr()
                max_rows[idx] = w_u8.shape[0] - 1
                val_dim, scl_dim = w_u8.shape[1], s_u8.shape[1]
        cached = (w_ptrs, s_ptrs, max_rows, val_dim, scl_dim)
        if all(p is not None for p in pairs):
            _ENGRAM_PTRS_CACHE[key] = cached
    return cached


def _gather_into(
    lib: ctypes.CDLL,
    hashes: np.ndarray,
    table_ids: list[int],
    packed_dim: int,
    out: torch.Tensor,
) -> None:
    num_tokens, n_tables, n_hash_cols = hashes.shape
    w_ptrs, s_ptrs, max_rows, val_dim, scl_dim = _ensure_engram_ptrs(table_ids)
    lib.engram_gather_omp(
        hashes.ctypes.data,
        w_ptrs,
        s_ptrs,
        max_rows,
        out.data_ptr(),
        num_tokens,
        n_tables,
        n_hash_cols,
        val_dim,
        scl_dim,
        packed_dim,
    )


def _resolve_async_tokens_cpu(pre_async: Any) -> np.ndarray | None:
    if pre_async is None:
        return None
    copy_state = getattr(pre_async, "copy_state", None)
    sampled: torch.Tensor | None = None
    if copy_state is not None:
        copy_state.wait()
        chunks = copy_state.chunks_cpu
        if len(chunks) == 1:
            sampled = chunks[0].reshape(-1)
        elif chunks:
            sampled = torch.cat([c.reshape(-1) for c in chunks], dim=0)
    else:
        wait_fn = getattr(pre_async, "wait_for_copy", None)
        sampled = wait_fn() if callable(wait_fn) else None
        if sampled is None:
            next_tpu = getattr(pre_async, "next_tokens_tpu", None)
            if isinstance(next_tpu, torch.Tensor):
                sampled = next_tpu.to("cpu")
        if sampled is not None:
            if sampled.ndim >= 2:
                sampled = sampled[:, 0]
            sampled = sampled.reshape(-1)
    if sampled is None:
        return None
    return sampled.detach().to(torch.int64).numpy()


def _normalize_async_subst(
    async_subst: tuple[np.ndarray, np.ndarray] | list[tuple[int, int]] | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    if async_subst is None:
        return None
    if isinstance(async_subst, tuple) and len(async_subst) == 2:
        t_arr, p_arr = async_subst
        if isinstance(t_arr, np.ndarray) and isinstance(p_arr, np.ndarray):
            return (t_arr, p_arr) if t_arr.size > 0 else None
    if not async_subst:
        return None
    arr = np.asarray(async_subst, dtype=np.int32)
    return (arr[:, 0], arr[:, 1])


def _get_prefetch_state(consume: bool = False) -> dict[str, Any] | None:
    global _ENGRAM_PREFETCH, _ENGRAM_PREFETCH_FUTURE
    if _ENGRAM_PREFETCH_FUTURE is not None:
        pf = _ENGRAM_PREFETCH_FUTURE.result()
        _ENGRAM_PREFETCH_FUTURE = None
        _ENGRAM_PREFETCH = pf
    pf = _ENGRAM_PREFETCH
    if consume:
        _ENGRAM_PREFETCH = None
    return pf


def engram_prefetch_prefill(
    hash_state: TpuNgramHashState,
    table_ids: list[int],
    packed_dim: int,
    token_ids: np.ndarray,
    req_idx: np.ndarray,
    pos: np.ndarray,
    known: np.ndarray,
    padded_num_tokens: int,
    pre_async: Any = None,
    async_subst: tuple[np.ndarray, np.ndarray] | list[tuple[int, int]] | None = None,
    spec_async_subst: tuple[list[tuple[int, int, int, int, int]], int] | None = None,
    debug_tok0_out: dict[int, int] | None = None,
) -> None:
    """Gather Engram rows for scheduled tokens on CPU during input preparation."""
    global _ENGRAM_PREFETCH, _ENGRAM_PREFETCH_FUTURE, _ENGRAM_BG_POOL
    if _ENGRAM_PREFETCH_FUTURE is not None:
        try:
            _ENGRAM_PREFETCH_FUTURE.result()
        except Exception as e:
            logger.warning_once("Previous Engram prefetch future failed: %s", e)
        _ENGRAM_PREFETCH_FUTURE = None
    _ENGRAM_PREFETCH = None
    lib = _get_engram_gather_lib()
    if lib is None or not known.any():
        return
    num_tokens = int(pos.shape[0])
    n_tables = len(table_ids)
    n_hash_cols = int(hash_state.offsets.shape[1])
    row_bytes = n_tables * _engram_table_bytes(n_hash_cols, packed_dim)

    padded_tokens = int(padded_num_tokens)
    norm_subst = _normalize_async_subst(async_subst)

    tp_size = get_tensor_model_parallel_world_size()
    if tp_token_shard_ok(padded_tokens, tp_size):
        tp_rank = get_tensor_model_parallel_rank()
        t_per_rank = padded_tokens // tp_size
        t_start = tp_rank * t_per_rank
        t_end = t_start + t_per_rank
        n_real_local = max(0, min(num_tokens, t_end) - t_start)
        known_slice = known[t_start : t_start + n_real_local].copy()
        if n_real_local > 0 and not known_slice.any():
            return
        req_slice = req_idx[t_start : t_start + n_real_local].copy()
        pos_slice = pos[t_start : t_start + n_real_local].copy()
        if norm_subst is not None and n_real_local > 0:
            g_tok, g_pre = norm_subst
            in_slice = (g_tok >= t_start) & (g_tok < t_start + n_real_local)
            local_subst = (
                (g_tok[in_slice] - t_start, g_pre[in_slice]) if in_slice.any() else None
            )
        else:
            local_subst = None
        buf = _next_host_buf(t_per_rank * row_bytes, row_bytes)
        pf_meta: dict[str, Any] = {
            "key": tuple(table_ids),
            "sharded": True,
            "t_per_rank": t_per_rank,
            "num_tokens": n_real_local,
            "known": known_slice,
            "all_known": bool(known_slice.all()),
            "buf": buf,
            "row_bytes": row_bytes,
        }
    else:
        n_real_local = num_tokens
        known_slice = known.copy()
        req_slice = req_idx.copy()
        pos_slice = pos.copy()
        local_subst = norm_subst
        alloc_tokens = max(num_tokens, padded_tokens)
        buf = _next_host_buf(alloc_tokens * row_bytes, row_bytes)
        pf_meta = {
            "key": tuple(table_ids),
            "num_tokens": num_tokens,
            "known": known_slice,
            "all_known": bool(known_slice.all()),
            "buf": buf,
            "row_bytes": row_bytes,
        }

    # Resolve all PJRT / torch_tpu D2H copies and tensor data_ptrs synchronously
    # on the main worker thread across all TP ranks before any n_real_local == 0
    # early return so every TP rank drains pre_async D2H events symmetrically and
    # background threads never race with PJRT's non-thread-safe C++ event lists.
    _ensure_engram_ptrs(table_ids)
    sampled_main_np: np.ndarray | None = None
    pre_sampled_main_np: np.ndarray | None = None
    next_main_np: np.ndarray | None = None
    if pre_async is not None:
        if spec_async_subst is not None:
            pre_sampled_t = pre_async.wait_for_copy()
            next_t_fn = getattr(pre_async, "wait_for_next_tokens_cpu", None)
            next_t = next_t_fn() if callable(next_t_fn) else None
            if next_t is None:
                next_tpu: Any = getattr(pre_async, "next_tokens_tpu", None)
                if isinstance(next_tpu, torch.Tensor):
                    next_t = next_tpu.to("cpu")
            if pre_sampled_t is not None and next_t is not None:
                pre_sampled_main_np = pre_sampled_t.detach().to(torch.int64).numpy()
                if pre_sampled_main_np.ndim == 1:
                    pre_sampled_main_np = pre_sampled_main_np[:, None]
                next_main_np = next_t.detach().reshape(-1).to(torch.int64).numpy()
        elif norm_subst is not None:
            sampled_main_np = _resolve_async_tokens_cpu(pre_async)

    if n_real_local == 0 and debug_tok0_out is None:
        _ENGRAM_PREFETCH = pf_meta
        return

    if _ENGRAM_BG_POOL is None:
        _ENGRAM_BG_POOL = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="engram_prefetch"
        )

    def _populate_debug(sampled_arr: np.ndarray | None) -> None:
        if debug_tok0_out is None or sampled_arr is None or norm_subst is None:
            return
        g_tok, g_pre = norm_subst
        valid_g = (g_pre >= 0) & (g_pre < sampled_arr.shape[0])
        for ti, pi in zip(g_tok[valid_g].tolist(), g_pre[valid_g].tolist()):
            debug_tok0_out[int(ti)] = int(sampled_arr[pi])

    def _bg_hash_and_gather() -> dict[str, Any]:
        if n_real_local > 0:
            hashes_s = np.full(
                (n_real_local, n_tables, n_hash_cols), -1, dtype=np.int32
            )
            req_k = req_slice[known_slice]
            pos_k = pos_slice[known_slice]
            if local_subst is not None and pre_async is not None:
                cum_xor = hash_state.cpu_hashes_tail(token_ids, req_k, pos_k)
                ids_0_all = token_ids[req_slice, np.maximum(pos_slice, 0)].copy()
                sampled = sampled_main_np
                if sampled is not None:
                    loc_tok, loc_pre = local_subst
                    valid_p = (loc_pre >= 0) & (loc_pre < sampled.shape[0])
                    if valid_p.any():
                        ids_0_all[loc_tok[valid_p]] = sampled[loc_pre[valid_p]]
                    _populate_debug(sampled)
                hashes_s[known_slice] = hash_state.cpu_hashes_finish_tok0(
                    ids_0_all[known_slice], pos_k, cum_xor
                )
            elif spec_async_subst is not None and pre_async is not None:
                spec_reqs, stride = spec_async_subst
                if (
                    pre_sampled_main_np is not None
                    and next_main_np is not None
                    and spec_reqs
                ):
                    pre_sampled_np = pre_sampled_main_np
                    next_np = next_main_np
                    max_r = max(
                        int(token_ids.shape[0]) - 1,
                        max(r for r, _, _, _, _ in spec_reqs),
                    )
                    max_sched = max((ns for _, _, _, _, ns in spec_reqs), default=0)
                    num_rej_by_req = np.zeros(max_r + 1, dtype=np.int32)
                    span_start_by_req = np.full(
                        max_r + 1, token_ids.shape[1] + 100, dtype=np.int32
                    )
                    span_tokens = np.zeros(
                        (max_r + 1, max(2 * stride + 2, stride + max_sched + 2)),
                        dtype=token_ids.dtype,
                    )
                    for (
                        r,
                        pre_pos,
                        end_idx_no_spec,
                        n_placeholder,
                        n_sched,
                    ) in spec_reqs:
                        if 0 <= pre_pos < pre_sampled_np.shape[0]:
                            row = pre_sampled_np[pre_pos]
                            sampled_r = row[row != -1]
                            l_acc = max(1, int(sampled_r.shape[0]))
                        else:
                            sampled_r = np.zeros(1, dtype=np.int64)
                            l_acc = 1
                        n_rej = max(0, n_placeholder - l_acc)
                        start_idx = end_idx_no_spec - n_placeholder
                        src_off = pre_pos * stride
                        src = next_np[src_off : src_off + n_sched]
                        n_prev_acc = max(0, l_acc - 1)
                        if n_prev_acc > 0:
                            span_tokens[r, :n_prev_acc] = sampled_r[:n_prev_acc]
                        if src.shape[0] > 0:
                            span_tokens[r, n_prev_acc : n_prev_acc + src.shape[0]] = src
                        num_rej_by_req[r] = n_rej
                        span_start_by_req[r] = start_idx
                    pos_k_adj = pos_k - num_rej_by_req[req_k]
                    hashes_s[known_slice] = hash_state.cpu_hashes(
                        token_ids,
                        req_k,
                        pos_k_adj,
                        span_start_by_req=span_start_by_req,
                        span_tokens=span_tokens,
                    )
                else:
                    hashes_s[known_slice] = hash_state.cpu_hashes(
                        token_ids, req_k, pos_k
                    )
            else:
                hashes_s[known_slice] = hash_state.cpu_hashes(token_ids, req_k, pos_k)
                if (
                    debug_tok0_out is not None
                    and norm_subst is not None
                    and pre_async is not None
                ):
                    _populate_debug(sampled_main_np)
            _gather_into(lib, hashes_s, table_ids, packed_dim, buf)
        elif (
            debug_tok0_out is not None
            and norm_subst is not None
            and pre_async is not None
        ):
            _populate_debug(sampled_main_np)
        return pf_meta

    _ENGRAM_PREFETCH_FUTURE = _ENGRAM_BG_POOL.submit(_bg_hash_and_gather)


def take_prefetched_engram_hashes(
    num_tokens: int,
    n_tables: int,
    n_hash_cols: int,
    packed_dim: int,
    device: torch.device,
    out_1d_i32_buf: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Upload prefetched host Engram rows directly to TPU before compiled stage."""
    global _ENGRAM_PREFETCH
    pf = _get_prefetch_state(consume=False)
    if pf is None or not pf.get("all_known", False):
        return None
    _ENGRAM_PREFETCH = None
    table_bytes = _engram_table_bytes(n_hash_cols, packed_dim)
    row_bytes = n_tables * table_bytes
    target_tokens = int(pf["t_per_rank"]) if pf.get("sharded") else num_tokens
    n_real = min(int(pf["num_tokens"]), target_tokens)
    total_bytes = target_tokens * row_bytes
    rows = pf["buf"][:total_bytes].view(target_tokens, row_bytes)
    if n_real < target_tokens:
        rows[n_real:].zero_()
    if out_1d_i32_buf is not None:
        if out_1d_i32_buf.ndim == 2:
            global _DEV_ENGRAM_RING, _DEV_ENGRAM_SLOT
            ring_len = 4 if out_1d_i32_buf.shape[0] >= 2048 else 8
            if (
                len(_DEV_ENGRAM_RING) != ring_len
                or _DEV_ENGRAM_RING[0].shape != out_1d_i32_buf.shape
                or _DEV_ENGRAM_RING[0].device != device
            ):
                _DEV_ENGRAM_RING = [
                    torch.zeros_like(out_1d_i32_buf, device=device)
                    for _ in range(ring_len)
                ]
                _DEV_ENGRAM_SLOT = 0
            dev_buf = _DEV_ENGRAM_RING[_DEV_ENGRAM_SLOT]
            _DEV_ENGRAM_SLOT = (_DEV_ENGRAM_SLOT + 1) % ring_len
            pad_rows = min(max(target_tokens, 32), dev_buf.shape[0])
            pad_bytes = pad_rows * row_bytes
            if pf["buf"].numel() >= pad_bytes:
                if n_real < pad_rows:
                    pf["buf"][n_real * row_bytes : pad_bytes].zero_()
                dev_buf[:pad_rows].copy_(
                    pf["buf"][:pad_bytes].view(pad_rows, row_bytes), non_blocking=True
                )
                return dev_buf
            dev_buf[:target_tokens].copy_(rows, non_blocking=True)
            if target_tokens < pad_rows:
                dev_buf[target_tokens:pad_rows].zero_()
            return dev_buf
        static_bytes = out_1d_i32_buf.numel()
        if pf["buf"].numel() >= static_bytes:
            return pf["buf"][:static_bytes].to(device, non_blocking=True)
        out_1d_i32_buf[:total_bytes].copy_(rows.reshape(-1), non_blocking=True)
        return out_1d_i32_buf
    return rows.view(num_tokens, -1, packed_dim).to(device, non_blocking=True)


def _lookup_with_prefetch(
    lib: ctypes.CDLL,
    pf: dict[str, Any],
    engram_hashes: torch.Tensor,
    table_ids: list[int],
    packed_dim: int,
    hashes_cpu_ready: torch.Tensor | None = None,
) -> torch.Tensor | None:
    num_tokens = engram_hashes.shape[0]
    row_bytes = pf["row_bytes"]
    buf = pf["buf"]
    if buf.numel() < num_tokens * row_bytes:
        return None
    n_real = pf["num_tokens"]
    rows = buf[: num_tokens * row_bytes].view(num_tokens, row_bytes)
    rows[n_real:].zero_()
    if not pf["all_known"]:
        if hashes_cpu_ready is not None:
            hashes_dev = hashes_cpu_ready[:n_real].to(torch.int32).numpy()
        else:
            hashes_dev = (
                engram_hashes[:n_real]
                .to("cpu", non_blocking=False, dtype=torch.int32)
                .numpy()
            )
        unknown = np.flatnonzero(~pf["known"])
        if unknown.size:
            sub = np.ascontiguousarray(hashes_dev[unknown])
            tmp = torch.empty((unknown.size * row_bytes,), dtype=torch.uint8)
            _gather_into(lib, sub, table_ids, packed_dim, tmp)
            rows[torch.from_numpy(unknown)] = tmp.view(unknown.size, row_bytes)
    return buf[: num_tokens * row_bytes].to(engram_hashes.device, non_blocking=True)


_ENGRAM_D2H_STATE: tuple[torch.Tensor, torch.Tensor, Any] | None = None


@torch.library.custom_op("tpu::engram_hash_ready", mutates_args=())
def engram_hash_ready(engram_hashes: torch.Tensor) -> None:
    global _ENGRAM_D2H_STATE
    _ENGRAM_D2H_STATE = None
    if engram_hashes.device.type != "tpu":
        return None
    pf = _ENGRAM_PREFETCH
    if pf is not None and pf.get("all_known", False):
        return None
    try:
        d2h_cpu = engram_hashes.to("cpu", non_blocking=True)
        d2h_event = torch.tpu.Event()
        d2h_event.record()
        _ENGRAM_D2H_STATE = (engram_hashes, d2h_cpu, d2h_event)
    except Exception:
        _ENGRAM_D2H_STATE = None
    return None


@engram_hash_ready.register_fake
def _engram_hash_ready_fake(engram_hashes: torch.Tensor) -> None:
    return None


@torch.library.custom_op("tpu::engram_cpu_lookup_all", mutates_args=())
def engram_cpu_lookup_all(
    engram_hashes: torch.Tensor,
    table_ids: list[int],
    packed_dim: int,
) -> torch.Tensor:
    global _ENGRAM_PREFETCH, _ENGRAM_D2H_STATE
    d2h_state, _ENGRAM_D2H_STATE = _ENGRAM_D2H_STATE, None
    hashes_cpu_ready: torch.Tensor | None = None
    if (
        d2h_state is not None
        and d2h_state[0].data_ptr() == engram_hashes.data_ptr()
        and d2h_state[0].shape == engram_hashes.shape
    ):
        d2h_state[2].synchronize()
        hashes_cpu_ready = d2h_state[1]
    d2h_state = None
    lib = _get_engram_gather_lib()
    if lib is not None:
        pf, _ENGRAM_PREFETCH = _ENGRAM_PREFETCH, None
        if (
            pf is not None
            and pf["key"] == tuple(table_ids)
            and (
                pf.get("t_per_rank") == engram_hashes.shape[0]
                if pf.get("sharded")
                else pf["num_tokens"] <= engram_hashes.shape[0]
            )
        ):
            out = _lookup_with_prefetch(
                lib,
                pf,
                engram_hashes,
                table_ids,
                packed_dim,
                hashes_cpu_ready=hashes_cpu_ready,
            )
            if out is not None:
                return out
        if hashes_cpu_ready is not None:
            hashes_cpu = hashes_cpu_ready.to(torch.int32).contiguous()
        else:
            hashes_cpu = engram_hashes.to(
                "cpu", non_blocking=False, dtype=torch.int32
            ).contiguous()
        num_tokens, n_tables, n_hash_cols = hashes_cpu.shape
        row_span = n_tables * _engram_table_bytes(n_hash_cols, packed_dim)
        total_bytes = num_tokens * row_span
        out_cpu = _next_host_buf(total_bytes, row_span)[:total_bytes]
        _gather_into(lib, hashes_cpu.numpy(), table_ids, packed_dim, out_cpu)
        return out_cpu.to(engram_hashes.device, non_blocking=True)

    if hashes_cpu_ready is not None:
        hashes_cpu = hashes_cpu_ready.to(torch.int64)
    else:
        hashes_cpu = engram_hashes.to("cpu", non_blocking=False).to(torch.int64)
    num_tokens, _, n_hash_cols = hashes_cpu.shape
    table_bytes = _engram_table_bytes(n_hash_cols, packed_dim)
    split_2d = packed_dim >= n_hash_cols * 64
    packed_layers: list[torch.Tensor] = []
    for idx, tid in enumerate(table_ids):
        pair = _CPU_ENGRAM_WEIGHTS.get(tid)
        if pair is None:
            shape = (
                (num_tokens, table_bytes)
                if split_2d
                else (num_tokens, n_hash_cols, packed_dim)
            )
            packed_layers.append(torch.zeros(shape, dtype=torch.uint8))
            continue
        w_u8, s_u8 = pair
        h = hashes_cpu[:, idx, :].clamp(0, w_u8.shape[0] - 1)
        if split_2d:
            vals = w_u8[h].reshape(num_tokens, -1)
            scls = s_u8[h].reshape(num_tokens, -1)
            pad_len = table_bytes - vals.shape[1] - scls.shape[1]
            if pad_len > 0:
                pad = torch.zeros((num_tokens, pad_len), dtype=torch.uint8)
                packed_layers.append(torch.cat([vals, scls, pad], dim=-1))
            else:
                packed_layers.append(torch.cat([vals, scls], dim=-1))
        else:
            packed_layers.append(torch.cat([w_u8[h], s_u8[h]], dim=-1))
    return (
        torch.stack(packed_layers, dim=1)
        .contiguous()
        .view(-1)
        .to(engram_hashes.device, non_blocking=True)
    )


@engram_cpu_lookup_all.register_fake
def _engram_cpu_lookup_all_fake(
    engram_hashes: torch.Tensor,
    table_ids: list[int],
    packed_dim: int,
) -> torch.Tensor:
    n_tok = engram_hashes.shape[0]
    table_bytes = _engram_table_bytes(int(engram_hashes.shape[2]), packed_dim)
    return engram_hashes.new_empty(
        (n_tok * engram_hashes.shape[1] * table_bytes,),
        dtype=torch.uint8,
    )


def _engram_weight_loader(
    param: torch.nn.Parameter, loaded_weight: torch.Tensor
) -> None:
    """Attach the full Engram table in CPU Host RAM (zero-copy safetensors mmap)."""
    if loaded_weight.dtype == torch.float8_e8m0fnu:
        loaded_weight = loaded_weight.view(torch.uint8)
    param.data = loaded_weight.to("cpu")


_SCALE_EXPAND_MAT_CACHE: dict[tuple[Any, ...], torch.Tensor] = {}


def _get_scale_expand_mat(
    rows: int, active_rows: int, block_size: int, device: torch.device
) -> torch.Tensor:
    key = (str(device), rows, active_rows, block_size)
    mat = _SCALE_EXPAND_MAT_CACHE.get(key)
    if mat is None:
        cols = active_rows * block_size
        cpu_mat = torch.zeros((rows, cols), dtype=torch.bfloat16)
        for i in range(active_rows):
            cpu_mat[i, i * block_size : (i + 1) * block_size] = 1.0
        mat = cpu_mat.to(device=device)
        _SCALE_EXPAND_MAT_CACHE[key] = mat
    return mat


class TpuParallelEngramEmbedding(nn.Module):
    """CPU-offloaded Engram hash embedding table with TPU TensorCore dequant."""

    def __init__(
        self,
        num_embeddings: int,
        dim: int,
        n_hash_cols: int,
        block_size: int = 32,
    ) -> None:
        super().__init__()
        global _NEXT_ENGRAM_TABLE_ID
        self.table_id = _NEXT_ENGRAM_TABLE_ID
        _NEXT_ENGRAM_TABLE_ID += 1

        self.num_embeddings = num_embeddings
        self.dim = dim
        self.block_size = block_size
        self.scale_dim = dim // block_size
        self.packed_dim = dim + self.scale_dim
        self.n_hash_cols = n_hash_cols
        self.val_bytes = n_hash_cols * dim
        self.scl_bytes = n_hash_cols * self.scale_dim
        self.scl_pad_bytes = ((self.scl_bytes + 127) // 128) * 128
        self.table_bytes = ((self.val_bytes + 127) // 128) * 128 + self.scl_pad_bytes

        # 1-row CPU placeholders so 0 bytes of TPU HBM are used during create_model().
        self.weight = nn.Parameter(
            torch.zeros(1, self.dim, dtype=torch.float8_e4m3fn, device="cpu"),
            requires_grad=False,
        )
        self.weight_scale_inv = nn.Parameter(
            torch.zeros(1, self.scale_dim, dtype=torch.uint8, device="cpu"),
            requires_grad=False,
        )
        for param in (self.weight, self.weight_scale_inv):
            set_weight_attrs(param, {"weight_loader": _engram_weight_loader})

    def finalize_cpu_offload(self, allow_dummy: bool = False) -> None:
        """Detach CPU Engram tables from _parameters so Dynamo/XLA skips them."""
        w = self._parameters.get("weight")
        s = self._parameters.get("weight_scale_inv")
        if w is not None and s is not None and w.shape[0] > 1:
            del self.weight, self.weight_scale_inv
            w_u8, s_u8 = w.data.view(torch.uint8), s.data.view(torch.uint8)
            try:
                libc = ctypes.CDLL(None)
                for t in (w_u8, s_u8):
                    ptr, aligned = t.data_ptr(), t.data_ptr() & ~4095
                    libc.madvise(
                        ctypes.c_void_p(aligned),
                        ctypes.c_size_t(t.numel() + ptr - aligned),
                        ctypes.c_int(MADV_HUGEPAGE),
                    )
            except (OSError, AttributeError) as e:
                logger.debug("madvise(MADV_HUGEPAGE) unavailable: %s", e)
            _CPU_ENGRAM_WEIGHTS[self.table_id] = (w_u8, s_u8)
        elif allow_dummy and self.table_id not in _CPU_ENGRAM_WEIGHTS:
            for attr in ("weight", "weight_scale_inv"):
                if attr in self._parameters:
                    delattr(self, attr)
            w_u8 = torch.zeros((2, self.dim), dtype=torch.uint8, device="cpu")
            s_u8 = torch.full((2, self.scale_dim), 127, dtype=torch.uint8, device="cpu")
            _CPU_ENGRAM_WEIGHTS[self.table_id] = (w_u8, s_u8)

    def forward(
        self, packed: torch.Tensor | _EngramTableSlice
    ) -> torch.Tensor:
        """Dequantize 2D [T_pad, table_bytes] or 3D [T, n_hash_cols, packed_dim] FP8+E8M0 rows to BF16."""
        if isinstance(packed, _EngramTableSlice) or packed.ndim == 2:
            if isinstance(packed, _EngramTableSlice):
                table_2d = packed.buf_2d
                t_local = packed.num_tokens
            else:
                table_2d = packed
                t_local = table_2d.shape[0]
            val_bf16 = (
                table_2d[:, : self.val_bytes]
                .view(torch.float8_e4m3fn)
                .to(torch.bfloat16)
            )
            scl_bf16 = e8m0_to_fp32(
                table_2d[:, self.val_bytes : self.val_bytes + self.scl_pad_bytes]
            ).to(torch.bfloat16)
            if t_local < table_2d.shape[0]:
                val_bf16 = val_bf16[:t_local]
                scl_bf16 = scl_bf16[:t_local]
            expand_2d = _get_scale_expand_mat(
                self.scl_pad_bytes,
                self.scl_bytes,
                self.block_size,
                table_2d.device,
            )
            scales_bf16 = torch.matmul(scl_bf16, expand_2d)
            return (val_bf16 * scales_bf16).view(
                t_local, self.n_hash_cols, self.dim
            )

        if packed.shape[1] != self.n_hash_cols:
            packed = packed[:, : self.n_hash_cols]
        values = packed[..., : self.dim].view(torch.float8_e4m3fn)
        scales_bf16 = e8m0_to_fp32(packed[..., self.dim :]).to(torch.bfloat16)
        if packed.device.type != "cpu":
            expand_3d = _get_scale_expand_mat(
                self.scale_dim,
                self.scale_dim,
                self.block_size,
                packed.device,
            )
            scales_expanded = torch.matmul(scales_bf16, expand_3d)
            return values.to(torch.bfloat16) * scales_expanded
        return (
            values.to(torch.bfloat16).unflatten(
                -1, (self.scale_dim, self.block_size)
            )
            * scales_bf16.unsqueeze(-1)
        ).flatten(-2)


class TpuEngram(nn.Module):
    """TPU implementation of DeepSeek-V4.1 Engram conditional memory module."""

    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None,
        layout: EngramLayout,
        layer_hash_index: int,
        prefix: str,
    ) -> None:
        super().__init__()
        self.layer_hash_index = layer_hash_index
        self.dim = config.hidden_size
        self.hc_mult = config.hc_mult
        self.eps = config.rms_norm_eps
        self.clamp_value = 1e-6
        self.inv_sqrt_dim = self.dim**-0.5

        n_hash_cols = (layout.max_ngram_size - 1) * layout.n_heads
        self.embed = TpuParallelEngramEmbedding(
            layout.num_embeddings[layer_hash_index], layout.head_dim, n_hash_cols
        )
        self.wkv = ReplicatedLinear(
            n_hash_cols * layout.head_dim,
            self.dim * (self.hc_mult + 1),
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wkv",
        )
        self.q_weight = nn.Parameter(
            torch.empty(self.hc_mult, self.dim, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.k_weight = nn.Parameter(
            torch.empty(self.hc_mult, self.dim, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.register_buffer(
            "qk_weight",
            torch.empty(self.hc_mult, self.dim, dtype=torch.float32),
            persistent=False,
        )
        self._qk_weight_ready = False
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank() if tp_size > 1 else 0
        self.register_buffer(
            "tp_rank_idx",
            torch.tensor([tp_rank], dtype=torch.int64),
            persistent=False,
        )

    def process_weights_after_loading(self, allow_dummy: bool = False) -> None:
        self.embed.finalize_cpu_offload(allow_dummy=allow_dummy)
        _get_scale_expand_mat(
            self.embed.scl_pad_bytes,
            self.embed.scl_bytes,
            self.embed.block_size,
            self.q_weight.device,
        )
        _get_scale_expand_mat(
            self.embed.scale_dim,
            self.embed.scale_dim,
            self.embed.block_size,
            self.q_weight.device,
        )
        qk_cpu = (
            self.q_weight.detach().cpu().float() * self.k_weight.detach().cpu().float()
        ).contiguous()
        self.qk_weight = qk_cpu.to(device=self.q_weight.device, dtype=torch.float32)
        self._qk_weight_ready = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        hash_ids: torch.Tensor,
        token_sharded: bool = False,
    ) -> torch.Tensor:
        """hidden_states: [T or T//tp, hc_mult, dim]; hash_ids: [T or T//tp, C, P]."""
        num_tokens = hidden_states.shape[0]
        tp_size = get_tensor_model_parallel_world_size()
        if token_sharded:
            if hash_ids.shape[0] > num_tokens and tp_size > 1:
                tp_rank = get_tensor_model_parallel_rank()
                hash_ids = tp_shard_slice(hash_ids, tp_size, tp_rank, self.tp_rank_idx)
            h_in = hidden_states
            sharded = False
        else:
            sharded = hash_ids.shape[0] < num_tokens
            if not sharded and tp_token_shard_ok(num_tokens, tp_size):
                tp_rank = get_tensor_model_parallel_rank()
                hash_ids = tp_shard_slice(hash_ids, tp_size, tp_rank, self.tp_rank_idx)
                h_in = tp_shard_slice(hidden_states, tp_size, tp_rank, self.tp_rank_idx)
                sharded = True
            elif sharded:
                tp_rank = get_tensor_model_parallel_rank()
                h_in = tp_shard_slice(hidden_states, tp_size, tp_rank, self.tp_rank_idx)
            else:
                h_in = hidden_states

        emb = self.embed(hash_ids).flatten(-2)
        kv = self.wkv(emb)
        weight = (
            self.qk_weight
            if self._qk_weight_ready
            else self.q_weight.float() * self.k_weight.float()
        )
        if h_in.device.type == "tpu":
            from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v41.deepseek_v41_flash_mhc_op import (
                get_engram_gate_v41_op,
            )

            out = get_engram_gate_v41_op(self.eps)(h_in, kv, weight)
        else:
            key, value = kv.split([self.hc_mult * self.dim, self.dim], dim=-1)
            key = key.float().unflatten(-1, (self.hc_mult, self.dim))
            h = h_in.float()
            rstd = torch.rsqrt(h.square().mean(-1) + self.eps) * torch.rsqrt(
                key.square().mean(-1) + self.eps
            )
            dot = (h * weight * key).sum(-1) * rstd * self.inv_sqrt_dim
            abs_sqrt = dot.abs().clamp_min(self.clamp_value).sqrt()
            gate = torch.sigmoid(torch.where(dot < 0.0, -abs_sqrt, abs_sqrt))
            out = (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(
                hidden_states.dtype
            )
        if sharded:
            out = tp_all_gather(out, dim=0)
        return out

    def compute_kv_and_weight(
        self,
        num_tokens: int,
        hash_ids: torch.Tensor,
        token_sharded: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """Compute [T_local, (hc_mult + 1) * dim] wkv(emb) and qk_weight for fused mHC+Engram."""
        tp_size = get_tensor_model_parallel_world_size()
        if token_sharded:
            if hash_ids.shape[0] > num_tokens and tp_size > 1:
                tp_rank = get_tensor_model_parallel_rank()
                hash_ids = tp_shard_slice(hash_ids, tp_size, tp_rank, self.tp_rank_idx)
            sharded = False
        else:
            sharded = hash_ids.shape[0] < num_tokens
            if not sharded and tp_token_shard_ok(num_tokens, tp_size):
                tp_rank = get_tensor_model_parallel_rank()
                hash_ids = tp_shard_slice(hash_ids, tp_size, tp_rank, self.tp_rank_idx)
                sharded = True
        emb = self.embed(hash_ids).flatten(-2)
        kv = self.wkv(emb)
        weight = (
            self.qk_weight
            if self._qk_weight_ready
            else self.q_weight.float() * self.k_weight.float()
        )
        return kv, weight, sharded
