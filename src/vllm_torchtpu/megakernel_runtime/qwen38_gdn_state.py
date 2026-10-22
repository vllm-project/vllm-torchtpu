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
"""Qwen3.8 (qwen4_exp) GDN state adapter: unified pool <-> megakernel operands.

Paths below are relative to ``src/vllm_torchtpu/``.

Pool side (qwen4_exp, unified block pool)
-----------------------------------------
* The pool is the stock attention-shaped KV cache born at *kernel* block
  granularity: ``(num_manager_blocks * split, 128, h2p, pack, lanes)`` in the
  KV dtype (``kv_cache_materializer.py:252-302``,
  ``layers/adapter/attention.py:599-610``). For Qwen3.8 TP8 with
  ``--kv-cache-dtype fp8`` that is ``(N * 4, 128, 1, 4, 256)`` float8_e4m3fn,
  1024 bytes per token row; a bf16 KV cache gives ``(N * 4, 128, 1, 2, 256)``
  bf16, also 1024 bytes. ``split = 512 / 128 = 4`` (manager block 512 tokens,
  QSA kernel block 128: ``models/vllm/qwen4_exp/pool_layout.py:142``,
  ``models/vllm/qwen4_exp/qsa.py:71``).
* Per manager block, the GDN state occupies token rows
  ``[0, ssm_tokens)`` = ssm f32 ``[n_v, d_k, d_v] = [6, 128, 128]``
  (384 rows) and ``[ssm_tokens, ssm_tokens + conv_tokens)`` = conv f32
  ``[kernel_size - 1, conv_dim] = [3, 1280]``. ``conv_tokens`` is rounded up
  to a power of two, so it is **16 rows (384..399), not 15**; the 16th row is
  zero padding (``gdn_pool_layout.py:127-152``; conv dtype f32:
  ``models/vllm/qwen4_exp/pool_layout.py:27``, ``model.py:220``; regions:
  ``layers/core/gdn_attention.py:789-842``).
* Bytes are read through Mosaic typed views (``kernels/typed_ldst.py:33-50``):
  ``ref.bitcast`` packs ``4 / itemsize`` consecutive second-minor (``pack``)
  rows into one 32-bit word per lane, then the ref is reshaped to
  ``(rows, lanes / lane_split)``. The element stream is therefore
  token-major, then lane; the first ``rows_used`` rows reshape row-major to
  the logical state (``kernels/gdn/v3/vmem_ldst.py:108-140``). The ssm region
  is 3 whole kernel blocks read with one window DMA per request; the conv
  region is 16 rows inside kernel block 3 (``kernels/pool_adapters.py``
  ``gather_region``/``scatter_region``; same geometry as the stock plan
  ``v3_state_source``).
* MTP K=3: each request owns ``K + 1 = 4`` checkpoint blocks per layer,
  ``ckpt_indices[s, 0..3]`` (``runner/tpu_runner.py:736-752``). The initial
  state is read from ``ckpt_indices[s, slot_read_offsets[state_indices[s]]]``
  (``layers/core/gdn_attention.py:182,1067``;
  ``kernels/gdn/v3/memory_ref.py:387-398,427-428``), and the state after
  window token ``t`` is written to ``ckpt_indices[s, t]``
  (``kernels/gdn/v3/memory_ref.py:562-576``). The
  stock kernel only reads when ``seq_lens - query_lens > 0``
  (``kernels/gdn/v3/metadata.py:48``); callers pass that as ``has_initial_state``.

Library side (megakernel GDN block, verify mode, non-paged)
-----------------------------------------------------------
* Inputs per layer (``megakernel/blocks/gdn/block.py:191-201``): ``conv`` f32
  ``[NS, rup(3 + T, 8) = 8, QKV]`` with rows 0..2 = history oldest first;
  the kernel overwrites rows ``3..3+T-1`` with the new inputs and never reads
  rows ``>= 3 + T`` (``megakernel/blocks/gdn/conv.py:126-127``). ``rec`` bf16
  ``[NS * vh, KD, VD]``, request-major then v-head.
* Outputs (``block.py:203-220``): ``conv_ckpt`` f32 ``[NS, 8, QKV]`` = rows
  0..2 old history + rows ``3..3+T-1`` new inputs; ``rec_ckpt`` bf16
  ``[NS * T * vh, KD, VD]`` where row block ``b = s * T + t`` holds the state
  after token ``t`` of request ``s`` (``megakernel/blocks/gdn/recurrence.py:
  48-75``: ``recb[b * vh:(b + 1) * vh]``; ``reference.py:167,190``).

Orientation decision: no transpose
----------------------------------
Both kernels hold the state as ``[head, k, v]``: the library computes
``pred = sum(S * k[:, :, None], axis=1)`` (``reference.py:186``,
``recurrence.py:71``) and the v3 kernel ``v_updated = dot(k, S)``
contracting ``S`` dim 1 (``kernels/gdn/v3/compute_gdn.py:368-373``); the pool
region decodes to ``(num_v_heads, kq_head_dim, v_head_dim)``
(``vmem_ldst.py:188-198``, via ``load_state_region`` ``:108-140``). Both
map v-head ``h`` to k-head ``h // (vh / kh)``
(``jnp.repeat``: ``compute_gdn.py:347-352``, ``reference.py:180-181``). Conv
rows are oldest-first in both: v3's checkpoint ``w`` is ``lhs[w+1 : w+4]`` of
``[state; tokens]`` (``compute_conv1d.py:30-75``), the library's ``conv_next
= [s1, s2, x]`` (``reference.py:132``). So the conv window ending at token
``t`` is ``conv_ckpt[s, t+1 : t+4]``.

Caveats
-------
* ``rec_ckpt`` is bf16: scatter upcasts it, so the f32 pool state is
  bf16-rounded every step (integration-design §3.1 risk; not fixed here).
* All ``NL`` layers must live in one pool tensor (one raw tensor shared by
  all single-layer GDN groups). Block-major pools need the caller to fold the
  pool with ``block_major_pool.kernel_view`` and remap ids with
  ``flat_manager_ids`` first (``gdn_attention_op.py:425-431,764-767``).
* The QK pair-blocked conv layout is only accepted where it is the identity
  (Q segment of one typed row, which holds at TP8: 256 elements / 256 lanes).
* ``kernels/pool_adapters.py`` imports ``vllm.envs`` (for the HND check);
  this module has no torch/vllm import of its own.
"""

from __future__ import annotations

import dataclasses
import math

import jax
import jax.numpy as jnp

from vllm_torchtpu.kernels import pool_adapters

F32 = jnp.float32
BF16 = jnp.bfloat16


def _rup(x: int, m: int) -> int:
    return (x + m - 1) // m * m


@dataclasses.dataclass(frozen=True)
class Qwen38GdnPoolLayout:
    """Static geometry of the GDN state regions in the qwen4_exp pool.

    ``pool_shape``/``pool_dtype`` are the kernel-granularity pool as vLLM
    allocates it (5-D NHD ``(blocks, kernel_block, h2p, pack, lanes)``).
    Defaults are the Qwen3.8-Flash-Next TP8 per-device GDN geometry.
    """

    pool_shape: tuple[int, ...]
    pool_dtype: str
    manager_block_tokens: int = 512
    n_k: int = 2
    n_v: int = 6
    d_k: int = 128
    d_v: int = 128
    kernel_size: int = 4
    qk_pair_layout: bool = False
    ssm_lanes: int = 128

    def __post_init__(self):
        if len(self.pool_shape) != 5:
            raise NotImplementedError(
                f"expected a 5-D NHD pool (blocks, kb, h2p, pack, lanes), got "
                f"{self.pool_shape}"
            )
        if self.manager_block_tokens % self.kernel_block:
            raise ValueError((self.manager_block_tokens, self.kernel_block))
        if self.pool_shape[0] % self.split:
            raise ValueError(f"pool blocks {self.pool_shape[0]} % split {self.split}")
        if self.token_bytes % 4:
            raise ValueError(f"token row of {self.token_bytes} bytes is not f32-whole")
        if self.required_tokens > self.manager_block_tokens:
            raise ValueError(
                f"GDN state needs {self.required_tokens} rows > "
                f"{self.manager_block_tokens}"
            )
        # ssm: whole kernel blocks (window path) or inside one kernel block.
        if self.ssm_tokens % self.kernel_block and self.ssm_tokens > self.kernel_block:
            raise NotImplementedError(
                f"ssm region of {self.ssm_tokens} rows is neither whole kernel "
                f"blocks nor inside one ({self.kernel_block})"
            )
        if self.ssm_elems % self.ssm_lanes or self.lanes % self.ssm_lanes:
            raise ValueError((self.ssm_elems, self.lanes, self.ssm_lanes))
        kb_off = self.conv_tok0 % self.kernel_block
        if kb_off + self.conv_tokens > self.kernel_block:
            raise NotImplementedError("conv region straddles kernel blocks")
        if self.conv_elems % self.lanes or self.conv_dim % self.lanes:
            raise NotImplementedError(
                f"conv [{self.kernel_size - 1}, {self.conv_dim}] is not whole "
                f"{self.lanes}-lane rows"
            )
        if self.qk_pair_layout:
            perm = pool_adapters._conv_qk_pair_rows_perm(
                self.kernel_size - 1, self.conv_dim, self.n_v, self.d_v, self.lanes
            )
            if perm is not None:
                raise NotImplementedError(
                    "QK pair-blocked conv layout with a non-identity row "
                    "permutation is not supported"
                )

    # ---- construction -------------------------------------------------
    @classmethod
    def from_pool(cls, pool, **kwargs) -> Qwen38GdnPoolLayout:
        """Layout of a live pool array (only shape and dtype are read)."""
        return cls(
            pool_shape=tuple(int(d) for d in pool.shape),
            pool_dtype=jnp.dtype(pool.dtype).name,
            **kwargs,
        )

    # ---- pool geometry ------------------------------------------------
    @property
    def dtype(self):
        return jnp.dtype(self.pool_dtype)

    @property
    def kernel_block(self) -> int:
        return self.pool_shape[1]

    @property
    def lanes(self) -> int:
        return self.pool_shape[-1]

    @property
    def split(self) -> int:
        return self.manager_block_tokens // self.kernel_block

    @property
    def num_manager_blocks(self) -> int:
        return self.pool_shape[0] // self.split

    @property
    def token_bytes(self) -> int:
        return math.prod(self.pool_shape[2:]) * self.dtype.itemsize

    # ---- state geometry -----------------------------------------------
    @property
    def conv_dim(self) -> int:
        return 2 * self.n_k * self.d_k + self.n_v * self.d_v

    @property
    def ssm_elems(self) -> int:
        return self.n_v * self.d_k * self.d_v

    @property
    def conv_elems(self) -> int:
        return (self.kernel_size - 1) * self.conv_dim

    @property
    def ssm_tokens(self) -> int:
        # gdn_pool_layout.derive_pooled_gdn_state_layout: ceil.
        return -(-self.ssm_elems * 4 // self.token_bytes)

    @property
    def conv_tok0(self) -> int:
        return self.ssm_tokens

    @property
    def conv_tokens(self) -> int:
        # gdn_pool_layout.derive_pooled_gdn_state_layout: power of two.
        n = 1
        while n * self.token_bytes < self.conv_elems * 4:
            n *= 2
        return n

    @property
    def required_tokens(self) -> int:
        return self.ssm_tokens + self.conv_tokens

    @property
    def ssm_rows(self) -> int:
        """Typed f32 rows (of ``ssm_lanes``) the ssm gather returns."""
        return self.ssm_tokens * self.token_bytes // (4 * self.ssm_lanes)

    @property
    def conv_rows(self) -> int:
        """Typed f32 rows (of ``lanes``) the conv gather returns."""
        return self.conv_tokens * self.token_bytes // (4 * self.lanes)

    @property
    def conv_rows_used(self) -> int:
        return self.conv_elems // self.lanes

    def hist_rows(self, T: int) -> int:
        return _rup(3 + T, 8)


def _check_pool(pool, layout: Qwen38GdnPoolLayout) -> None:
    if tuple(pool.shape) != layout.pool_shape or (
        jnp.dtype(pool.dtype) != layout.dtype
    ):
        raise ValueError(
            f"pool {pool.shape}/{pool.dtype} != layout "
            f"{layout.pool_shape}/{layout.pool_dtype}"
        )
    if pool_adapters.is_seq_along_lane_pool():
        raise NotImplementedError("HND (seq-on-lane) pools are not supported")


def read_blocks_from_metadata(
    state_indices: jax.Array,
    slot_read_offsets: jax.Array,
    ckpt_indices: jax.Array,
) -> jax.Array:
    """Per-request read block ``ckpt_indices[s, slot_read_offsets[state_indices[s]]]``.

    Stack one call per GDN layer (each layer has its own metadata) to get
    the ``[NL, NS]`` ``read_blocks`` of ``gather_gdn_states``.
    """
    offs = slot_read_offsets[state_indices]
    return jnp.take_along_axis(ckpt_indices, offs[:, None].astype(jnp.int32), axis=1)[
        :, 0
    ].astype(jnp.int32)


def _ssm_rows_to_rec(ssm, layout: Qwen38GdnPoolLayout, NL: int, NS: int):
    """[NL*NS, ssm_rows, 128] f32 -> rec [NL, NS*vh, KD, VD] bf16 (no transpose)."""
    used = layout.ssm_elems // layout.ssm_lanes
    ssm = ssm[:, :used] if used != ssm.shape[1] else ssm
    return ssm.reshape(NL, NS * layout.n_v, layout.d_k, layout.d_v).astype(BF16)


def _conv_windows(conv_ckpt, T: int):
    """[NL, NS, HR, QKV] -> [NL, NS, T, 3, QKV]: window ending at token t.

    Rows ``t+1 .. t+3`` of the library history (rows 0..2 old, 3.. new).
    """
    return jnp.stack([conv_ckpt[:, :, t + 1 : t + 4] for t in range(T)], axis=2)


def gather_gdn_states(
    pool,
    read_blocks,
    *,
    layout: Qwen38GdnPoolLayout,
    T: int = 4,
    has_initial_state=None,
    interpret: bool = False,
):
    """Read NL layers' GDN states for NS requests into the library operands.

    pool: the kernel-granularity unified pool (``layout.pool_shape``).
    read_blocks: ``[NL, NS]`` int32 manager block ids (padded requests -> 0).
    has_initial_state: optional ``[NL, NS]`` or ``[NS]`` bool; False rows
      are zeroed (the stock kernel's fresh-slot masking).
    returns: ``conv [NL, NS, rup(3+T, 8), QKV]`` f32 (rows 0..2 history,
      rest zero) and ``rec [NL, NS*vh, KD, VD]`` bf16.
    """
    _check_pool(pool, layout)
    NL, NS = read_blocks.shape
    ids = read_blocks.reshape(-1).astype(jnp.int32)
    ssm = pool_adapters.gather_region(
        pool,
        ids,
        tok0=0,
        ntok=layout.ssm_tokens,
        out_dtype=F32,
        out_lanes=layout.ssm_lanes,
        split=layout.split,
        interpret=interpret,
    )
    conv = pool_adapters.gather_region(
        pool,
        ids,
        tok0=layout.conv_tok0,
        ntok=layout.conv_tokens,
        out_dtype=F32,
        split=layout.split,
        interpret=interpret,
    )
    rec = _ssm_rows_to_rec(ssm, layout, NL, NS)
    hist = conv[:, : layout.conv_rows_used].reshape(
        NL, NS, layout.kernel_size - 1, layout.conv_dim
    )
    hr = layout.hist_rows(T)
    conv_out = jnp.pad(hist, ((0, 0), (0, 0), (0, hr - hist.shape[2]), (0, 0)))
    if has_initial_state is not None:
        m = jnp.broadcast_to(jnp.asarray(has_initial_state, bool), (NL, NS))
        conv_out = jnp.where(m[:, :, None, None], conv_out, 0.0)
        rec = jnp.where(
            jnp.repeat(m, layout.n_v, axis=1)[:, :, None, None],
            rec,
            jnp.zeros((), BF16),
        )
    return conv_out, rec


def scatter_gdn_ckpts(
    pool,
    conv_ckpt,
    rec_ckpt,
    write_blocks,
    *,
    layout: Qwen38GdnPoolLayout,
    interpret: bool = False,
):
    """Write each token's checkpoint (state and conv window) into the pool.

    conv_ckpt: ``[NL, NS, HR, QKV]`` f32 library history (rows 0..2 old,
      ``3..3+T-1`` new inputs).
    rec_ckpt: ``[NL, NS*T*vh, KD, VD]`` bf16, row block ``s*T + t``.
    write_blocks: ``[NL, NS, T]`` int32; checkpoint ``t`` of request ``s``
      goes to manager block ``write_blocks[l, s, t]`` (``ckpt_indices[s, t]``;
      padded requests / unused positions -> null block 0).
    returns: the updated pool (in place when the caller donates it).
    """
    _check_pool(pool, layout)
    NL, NS, T = write_blocks.shape
    if conv_ckpt.shape[:2] != (NL, NS) or conv_ckpt.shape[2] < 3 + T:
        raise ValueError(f"conv_ckpt {conv_ckpt.shape} vs write_blocks {(NL, NS, T)}")
    if rec_ckpt.shape != (NL, NS * T * layout.n_v, layout.d_k, layout.d_v):
        raise ValueError(f"rec_ckpt {rec_ckpt.shape} vs write_blocks {(NL, NS, T)}")
    ids = write_blocks.reshape(-1).astype(jnp.int32)
    n = NL * NS * T

    used = layout.ssm_elems // layout.ssm_lanes
    ssm = rec_ckpt.astype(F32).reshape(n, used, layout.ssm_lanes)
    if used < layout.ssm_rows:
        ssm = jnp.pad(ssm, ((0, 0), (0, layout.ssm_rows - used), (0, 0)))
    pool = pool_adapters.scatter_region(
        pool,
        ssm,
        ids,
        tok0=0,
        ntok=layout.ssm_tokens,
        split=layout.split,
        interpret=interpret,
    )

    win = _conv_windows(conv_ckpt.astype(F32), T).reshape(
        n, layout.conv_rows_used, layout.lanes
    )
    # Rows past the live state are zeroed, as the stock store_state_region does.
    win = jnp.pad(win, ((0, 0), (0, layout.conv_rows - layout.conv_rows_used), (0, 0)))
    return pool_adapters.scatter_region(
        pool,
        win,
        ids,
        tok0=layout.conv_tok0,
        ntok=layout.conv_tokens,
        split=layout.split,
        interpret=interpret,
    )
