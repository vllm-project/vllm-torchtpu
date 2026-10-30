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
"""f32 pooled GDN conv state: megakernel split pool I/O vs the stock v3 kernel.

The serving framework's stock GDN v3 kernel (`kernels.gdn.v3.wrapper.
fused_conv1d_gdn`) owns the unified-pool state layout; the megakernel's split
gather / scatter (`megakernel.pool`, `PoolLayout(conv_dtype=f32)`) must read and
write the same bytes. These tests check that contract with the stock kernel as
the oracle instead of a hand-written layout:

* stock write -> megakernel read: the stock kernel runs on a unified pool and
  writes new conv / recurrent states into it (prefill, then decode); the
  megakernel gather reads them back. They must equal, bit for bit, the new
  states of a stock run on dense (non-pooled) state tensors with the same
  inputs.
* megakernel write -> stock read: the megakernel scatter writes conv /
  recurrent checkpoints (T=1 and T=4) into the pool; the stock kernel then
  reads them as its initial states. Its output and new states must equal a
  stock run on dense states holding the values the scatter was given.
* MTP round trip: the stock kernel's speculative verify window
  (`num_spec_tokens=3`) reads a megakernel T=4 checkpoint and rewrites all four
  checkpoints in place; the megakernel gather reads them back.
* stale rows: the stock kernel moves only the live conv granules (u32 rows
  0..35 of the conv page for this geometry), so rows 36..127 keep whatever the
  block held before. Every case also runs with those rows filled with
  0xFFFFFFFF (f32 NaN) and must give the same bytes and a finite output.

Geometry: one Qwen3.5 TP=8 rank (2 K heads, 8 V heads, KD = VD = 128, so
QKV = 1536), fp8 pages `(2, 64, 4, 128)` (64 KiB), and per slot 4 bf16
recurrent pages plus 1 f32 conv page (a 640-token block). The stock kernel sees
the pool the way the framework's pooled caller passes it in the HND layout
(`_as_logical_block_pool`): `(num_pages // 5, 10, 64, 4, 128)`, one logical
block per megakernel checkpoint slot.

The stock kernel runs in Pallas TPU interpret mode on CPU (`_stock_interpret`).
JAX's interpreter resolves only `NDIndexer` ref transforms, but the kernel
addresses state through `granule_view` (`ref.bitcast(uint32).reshape(...)`), so
`_stock_interpret` resolves bitcast / reshape ref views by byte index. A
bitcast to a wider dtype packs consecutive second-minor rows into one word,
the TPU convention main's `tests/kernels/gdn/v3/gdn_pool_state_utils.py`
documents; `test_pool_bytes_match_stock_state_utils` checks the pool bytes
against that module where it is available.
"""

from __future__ import annotations

import contextlib
import functools
import sys
from collections.abc import Iterator
from typing import Any
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.megakernel import pool
from vllm_torchtpu.megakernel.exec_spec import ExecSpec, Tiles
from vllm_torchtpu.megakernel.ops import F32
from vllm_torchtpu.megakernel.pool import CacheExec, PoolLayout

wrapper = pytest.importorskip(
    "vllm_torchtpu.kernels.gdn.v3.wrapper",
    reason="needs the stock GDN v3 kernel (kernels/gdn/v3)",
)
try:
    from jax._src.state import indexing
    from jax._src.state import types as state_types
    from jax.sharding import AbstractDevice, AbstractMesh, use_abstract_mesh

    _MI = sys.modules[pltpu.reset_tpu_interpret_mode_state.__module__]
    for _name in (
        "get",
        "store",
        "swap",
        "fail_on_exception",
        "interpret_utils",
        "TPU_MEMORY_SPACE_NAMES",
        "_get_shared_memory",
        "_local_core_id_or_zero_if_hbm",
    ):
        getattr(_MI, _name)
except (ImportError, AttributeError) as e:  # pragma: no cover - JAX internals
    pytest.skip(
        f"JAX Pallas TPU interpret internals unavailable: {e}", allow_module_level=True
    )

from vllm_torchtpu import envs  # noqa: E402

# One TP=8 rank of Qwen3.5's GDN layers.
N_KQ, N_V, D = 2, 8, 128
DIM = 2 * N_KQ * D + N_V * D  # 1536
KERNEL_SIZE = 4
TAPS = KERNEL_SIZE - 1
LAYOUT = PoolLayout(num_pools=1, slots_per_pool=1, kh=N_KQ, vh=N_V, conv_dtype=F32)
PPS = LAYOUT.pages_per_slot  # 5
PAGE_SHAPE = LAYOUT.kv_page_shape  # (2, 64, 4, 128)
NUM_SLOTS = 12  # slot 0 is the null slot; the last slot's pages hold KV
NUM_PAGES = NUM_SLOTS * PPS
KV_PAGE = (NUM_SLOTS - 1) * PPS
LIVE_ROWS = LAYOUT.conv_rows(1)  # 36
SEQ_LEN = 11  # longer than the step, so decode sequences have an initial state
PREFILL_TOKENS = 4  # >= TAPS, so every tap of the new conv state is a token
FP8 = jnp.float8_e4m3fn
_TINY_TILES = Tiles(bank=512 << 10, nbd=10, nbe=6, ebank=256 << 10, vmem_mib=32)

assert LAYOUT.qkv_per_rank(1) == DIM
assert LIVE_ROWS == 36 and LAYOUT.conv_page_rows == 128
assert PAGE_SHAPE == (2, 64, 4, 128) and PPS == 5


@pytest.fixture(autouse=True)
def _use_cpu_default_device() -> Iterator[None]:
    with jax.default_device(jax.devices("cpu")[0]):
        yield


# ---------------------------------------------------------------------------
# Stock kernel in Pallas TPU interpret mode
# ---------------------------------------------------------------------------
def _byte_index(shape: tuple[int, ...], dtype: Any, transforms: Any) -> Any:
    """Byte offsets into a C-contiguous `shape` `dtype` buffer of a ref view.

    Returns `(idx, view_dtype)`, where `idx[..., j]` is byte `j` of each
    element of the view `transforms` selects.
    """
    dt = np.dtype(dtype)
    nbytes = int(np.prod(shape)) * dt.itemsize
    idx = np.arange(nbytes, dtype=np.int64).reshape(*shape, dt.itemsize)
    to_slice = _MI.interpret_utils._transform_slice_or_index
    for t in transforms:
        if isinstance(t, indexing.NDIndexer):
            idx = idx[tuple(to_slice(i) for i in t.indices)]
        elif isinstance(t, state_types.BitcastTransform):
            a, b = idx.shape[-1], np.dtype(t.dtype).itemsize
            *lead, rows, lanes = idx.shape[:-1]
            if b > a:  # rows k*r .. k*r+k-1 become bytes groups 0..k-1 of row r
                k = b // a
                idx = idx.reshape(*lead, rows // k, k, lanes, a)
                idx = np.moveaxis(idx, -3, -2).reshape(*lead, rows // k, lanes, b)
            elif b < a:
                k = a // b
                idx = idx.reshape(*lead, rows, lanes, k, b)
                idx = np.moveaxis(idx, -2, -3).reshape(*lead, rows * k, lanes, b)
            dt = np.dtype(t.dtype)
        elif isinstance(t, state_types.ReshapeTransform):
            idx = idx.reshape(*t.shape, idx.shape[-1])
        else:
            raise NotImplementedError(f"unsupported ref transform: {t}")
    return idx, dt


def _view(device_id, local_core_id, memory_space, buffer_id, transforms) -> Any:
    """`(flat uint8 buffer, byte index, view dtype)` of a bitcast / reshape view,
    or None for plain `NDIndexer` views."""
    if all(isinstance(t, indexing.NDIndexer) for t in transforms):
        return None
    space = _MI.TPU_MEMORY_SPACE_NAMES[int(memory_space)]
    core = _MI._local_core_id_or_zero_if_hbm(int(local_core_id), space)
    key = (space, int(buffer_id), int(device_id), core)
    buf = _MI._get_shared_memory().mem[key]
    content = buf.content
    assert tuple(buf.shape) == tuple(buf.logical_shape), key
    assert content.flags.c_contiguous, key
    idx, dt = _byte_index(content.shape, content.dtype, transforms)
    return content.reshape(-1).view(np.uint8), idx, dt


def _read(flat: np.ndarray, idx: np.ndarray, dt: Any) -> np.ndarray:
    return np.ascontiguousarray(flat[idx]).view(dt).reshape(idx.shape[:-1])


def _write(flat: np.ndarray, idx: np.ndarray, val: Any) -> None:
    val = np.ascontiguousarray(val)
    assert val.dtype.itemsize == idx.shape[-1], (val.dtype, idx.shape)
    flat[idx] = val.view(np.uint8).reshape(idx.shape)


_ORIG_GET, _ORIG_STORE, _ORIG_SWAP = _MI.get, _MI.store, _MI.swap


def _view_get(token, dev, core, space, buf, transforms, *args, **kwargs):
    transforms = jax.tree.map(int, transforms)
    view = _view(dev, core, space, buf, transforms)
    if view is None:
        return _ORIG_GET.__wrapped__(
            token, dev, core, space, buf, transforms, *args, **kwargs
        )
    return token, _read(*view)


def _view_store(token, dev, core, space, buf, transforms, val, *args, **kwargs):
    transforms = jax.tree.map(int, transforms)
    view = _view(dev, core, space, buf, transforms)
    if view is None:
        return _ORIG_STORE.__wrapped__(
            token, dev, core, space, buf, transforms, val, *args, **kwargs
        )
    flat, idx, _ = view
    _write(flat, idx, np.asarray(val))
    return token


def _view_swap(token, dev, core, space, buf, transforms, val, mask, **kwargs):
    transforms = jax.tree.map(int, transforms)
    view = _view(dev, core, space, buf, transforms)
    if view is None:
        return _ORIG_SWAP.__wrapped__(
            token, dev, core, space, buf, transforms, val, mask, **kwargs
        )
    flat, idx, dt = view
    old = _read(flat, idx, dt)
    new = np.asarray(val).astype(dt)
    if mask is not None:
        new = np.where(np.asarray(mask), new, old)
    _write(flat, idx, new)
    return token, old


_VIEW_FNS = {
    name: _MI.fail_on_exception(fn)
    for name, fn in (("get", _view_get), ("store", _view_store), ("swap", _view_swap))
}


@contextlib.contextmanager
def _stock_interpret() -> Iterator[None]:
    """Run the stock v3 kernel in Pallas TPU interpret mode on CPU (TPU7x)."""
    mesh = AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
    pltpu.reset_tpu_interpret_mode_state()
    try:
        with contextlib.ExitStack() as stack:
            stack.enter_context(use_abstract_mesh(mesh))
            stack.enter_context(
                mock.patch.object(envs, "TPU_ENABLE_GDN_DYNAMIC_TILING", False)
            )
            for name, fn in _VIEW_FNS.items():
                stack.enter_context(mock.patch.object(_MI, name, fn))
            stack.enter_context(
                pltpu.force_tpu_interpret_mode(
                    pltpu.InterpretParams(
                        dma_execution_mode="eager",
                        detect_races=False,
                        skip_floating_point_ops=False,
                    )
                )
            )
            yield
    finally:
        pltpu.reset_tpu_interpret_mode_state()


@functools.cache
def _weights() -> dict[str, jax.Array]:
    k = jax.random.split(jax.random.key(7), 4)
    return dict(
        conv_weight=jax.random.normal(k[0], (DIM, 1, KERNEL_SIZE)) * 0.5,
        conv_bias=jax.random.normal(k[1], (DIM,)) * 0.1,
        a_log=jax.random.normal(k[2], (N_V,)) * 0.5,
        dt_bias=jax.random.normal(k[3], (N_V,)) * 0.5,
    )


def _tokens(seed: int, num_tokens: int) -> dict[str, jax.Array]:
    """Full-f32 token inputs, so conv taps the stock kernel writes from them are
    not bf16-representable and any bf16 round trip would show."""
    k = jax.random.split(jax.random.key(seed), 3)
    return dict(
        qkv=jax.random.normal(k[0], (num_tokens, DIM), jnp.float32),
        b=jax.random.normal(k[1], (num_tokens, N_V), jnp.float32),
        a=jax.random.normal(k[2], (num_tokens, N_V), jnp.float32),
    )


def _stock(
    states: jax.Array | tuple[jax.Array, jax.Array],
    tokens: dict[str, jax.Array],
    state_indices: Any,
    *,
    prefill: bool = False,
    read_offsets: Any = None,
) -> tuple[Any, np.ndarray]:
    """One stock `fused_conv1d_gdn` call on a unified pool (logical blocks) or
    on dense `(conv, rec)` states; equal-length sequences.

    `prefill`: cold-start prefill (no initial state). Otherwise every sequence
    decodes from an initial state: one token from `state_indices` `(n,)`, or,
    with `read_offsets`, a speculative verify window of `T` tokens whose
    initial state is checkpoint `state_indices[s, read_offsets[s]]` of an
    `(n, T)` checkpoint table and whose new checkpoints go to
    `state_indices[s, t]`.
    """
    table = np.asarray(state_indices)
    spec = read_offsets is not None
    n = table.shape[0] if spec else table.size
    q = tokens["qkv"].shape[0] // n
    assert q * n == tokens["qkv"].shape[0]
    assert prefill or q == (table.shape[1] if spec else 1)
    kwargs: dict[str, Any] = {}
    if spec:
        kwargs.update(
            read_offsets=jnp.asarray(read_offsets, jnp.int32),
            ckpt_indices=jnp.asarray(table, jnp.int32),
            num_spec_tokens=q - 1,
        )
        table = table[:, 0]
    sidx = jnp.asarray(table.ravel(), jnp.int32)
    if isinstance(states, jax.Array):
        kwargs.update(
            recurrent_dtype=jnp.dtype(jnp.bfloat16), conv_dtype=jnp.dtype(jnp.float32)
        )
    with _stock_interpret():
        new_states, out = wrapper.fused_conv1d_gdn(
            states=states,
            **tokens,
            **_weights(),
            query_start_loc=jnp.arange(n + 1, dtype=jnp.int32) * q,
            state_indices=sidx,
            distribution=jnp.asarray([0 if prefill else n, n, n], jnp.int32),
            seq_lens=jnp.full((n,), q if prefill else SEQ_LEN, jnp.int32),
            n_kq=N_KQ,
            n_v=N_V,
            d_k=D,
            d_v=D,
            kernel_size=KERNEL_SIZE,
            # A decode tile reads metadata for all its sequence rows; the
            # framework pads its sequence arrays, these tests do not, so keep a
            # tile within the real sequences (verify windows have q > 1 tokens
            # per sequence, so the default `min(4, num_tokens)` would not).
            decode_tile_size=min(4, n),
            # CPU interpret runs chunked prefill one token per tile.
            mixed_tile_size=1,
            **kwargs,
        )
        jax.block_until_ready((new_states, out))
    return new_states, np.asarray(out)


def _dense_states(n: int) -> tuple[jax.Array, jax.Array]:
    return (
        jnp.zeros((n, TAPS, DIM), jnp.float32),
        jnp.zeros((n, N_V, D, D), jnp.bfloat16),
    )


# ---------------------------------------------------------------------------
# Unified pool and megakernel split pool I/O
# ---------------------------------------------------------------------------
def _garbage_pool(seed: int) -> np.ndarray:
    """Raw pool bytes `(NUM_PAGES, *PAGE_SHAPE)` uint8, random everywhere."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (NUM_PAGES, *PAGE_SHAPE), dtype=np.uint8)


def _fill_stale_rows(raw_u8: np.ndarray, slots: Any, fill: int) -> np.ndarray:
    """Set u32 rows `LIVE_ROWS..127` of each slot's conv page to `fill` bytes."""
    raw_u8 = raw_u8.copy()
    for c in np.asarray(slots).ravel():
        page = raw_u8[int(c) * PPS + LAYOUT.conv_page]
        # (2, 64, 4, 128) -> (u32 row, byte, lane); row = h * 64 + d.
        page.reshape(LAYOUT.conv_page_rows, 4, 128)[LIVE_ROWS:] = fill
    return raw_u8


def _stale_rows(raw_u8: np.ndarray, slots: Any) -> np.ndarray:
    return np.stack(
        [
            raw_u8[int(c) * PPS + LAYOUT.conv_page].reshape(-1, 4, 128)[LIVE_ROWS:]
            for c in np.asarray(slots).ravel()
        ]
    )


def _to_pool(raw_u8: np.ndarray) -> jax.Array:
    return jnp.asarray(raw_u8.view(FP8))


def _to_u8(raw: jax.Array) -> np.ndarray:
    return np.asarray(raw).view(np.uint8).reshape(NUM_PAGES, *PAGE_SHAPE).copy()


def _logical_blocks(raw: jax.Array) -> jax.Array:
    """The pool as the framework's pooled caller hands it to the stock kernel."""
    return raw.reshape(NUM_SLOTS, PAGE_SHAPE[0] * PPS, *PAGE_SHAPE[1:])


def _raw(blocks: jax.Array) -> jax.Array:
    return blocks.reshape(NUM_PAGES, *PAGE_SHAPE)


def _mk_exec(num_seqs: int, T: int) -> ExecSpec:
    return ExecSpec(
        R=1,
        B=num_seqs * T,
        T=T,
        ctx=256,
        stack_e=True,
        interpret=True,
        tiles=_TINY_TILES,
        cache=CacheExec(paged_kv=True, num_pages=NUM_PAGES),
    )


def _mk_read(raw: jax.Array, slots: Any, T: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Megakernel split gather of `slots`: conv `(n, TAPS, DIM)` f32, rec
    `(n, N_V, D, D)` bf16. `T=1` takes the decode path (packed conv rows);
    `T=4` the verify-window path (direct conv window rows)."""
    slots = np.asarray(slots).ravel()
    n = slots.size
    ex = _mk_exec(n, T)
    ckpt = jnp.asarray(np.repeat(slots[:, None], T, axis=1)[None], jnp.int32)
    roff = jnp.zeros((1, n), jnp.int32)
    conv, rec = pool.gather_gdn_states(LAYOUT, ex, (raw,), ckpt, roff)
    conv = np.asarray(conv)[0]
    if T == 1:  # rows j * n + s hold tap j of sequence s
        taps = conv[: TAPS * n].reshape(TAPS, n, DIM).swapaxes(0, 1)
        pad = conv[TAPS * n :]
    else:  # one (rup(TAPS + T, 8), DIM) window per sequence
        taps, pad = conv[:, :TAPS], conv[:, TAPS:]
    assert not pad.any()
    return taps, np.asarray(rec)[0].reshape(n, N_V, D, D)


def _mk_write(
    raw: jax.Array, slots: np.ndarray, conv: jax.Array, rec: jax.Array
) -> jax.Array:
    """Megakernel split scatter of `(NS, T)` checkpoints into `slots`."""
    NS, T = slots.shape
    ex = _mk_exec(NS, T)
    pos = jnp.asarray([10 + t for _ in range(NS) for t in range(T)], jnp.int32)
    page_table = jnp.asarray([[KV_PAGE + s] for s in range(NS)], jnp.int32)
    kv = jnp.zeros((LAYOUT.num_pools, ex.MR, LAYOUT.hd), FP8)
    ckpt = jnp.asarray(slots[None], jnp.int32)
    (out,) = pool.scatter_pool_states(
        LAYOUT, ex, (raw,), pos, page_table, kv, kv, conv, rec, ckpt
    )
    return out


def _assert_bytes_equal(got: Any, want: Any, what: str) -> None:
    got, want = np.asarray(got), np.asarray(want)
    assert got.shape == want.shape and got.dtype == want.dtype, (what, got, want)
    np.testing.assert_array_equal(got.view(np.uint8), want.view(np.uint8), err_msg=what)


def _assert_finite(x: Any, what: str) -> None:
    assert np.isfinite(np.asarray(x, np.float32)).all(), what


# ---------------------------------------------------------------------------
# Stock write -> megakernel read
# ---------------------------------------------------------------------------
_STOCK_SLOTS = (3, 4)


@functools.cache
def _dense_reference() -> dict[str, Any]:
    """Stock prefill then decode on dense states (the oracle for the pool)."""
    n = len(_STOCK_SLOTS)
    ident = np.arange(n)
    states, out_pre = _stock(
        _dense_states(n), _tokens(1, n * PREFILL_TOKENS), ident, prefill=True
    )
    pre = tuple(np.asarray(s) for s in states)
    states, out_dec = _stock(states, _tokens(2, n), ident)
    dec = tuple(np.asarray(s) for s in states)
    return dict(pre=pre, out_pre=out_pre, dec=dec, out_dec=out_dec)


def _stock_prefill_pool(fill: int) -> tuple[jax.Array, np.ndarray]:
    n = len(_STOCK_SLOTS)
    raw_u8 = _fill_stale_rows(_garbage_pool(0), _STOCK_SLOTS, fill)
    blocks, out = _stock(
        _logical_blocks(_to_pool(raw_u8)),
        _tokens(1, n * PREFILL_TOKENS),
        _STOCK_SLOTS,
        prefill=True,
    )
    return _raw(blocks), out


@pytest.mark.parametrize("fill", [0x00, 0xFF], ids=["clean", "stale_nan"])
def test_stock_write_megakernel_read(fill: int):
    ref = _dense_reference()
    n = len(_STOCK_SLOTS)

    raw, out = _stock_prefill_pool(fill)
    _assert_bytes_equal(out, ref["out_pre"], "stock prefill output")
    _assert_finite(out, "stock prefill output")
    # The stock kernel leaves the stale rows alone, so the megakernel reads a
    # page whose rows past the live window hold `fill`.
    assert (_stale_rows(_to_u8(raw), _STOCK_SLOTS) == fill).all()
    for T in (1, 4):
        conv, rec = _mk_read(raw, _STOCK_SLOTS, T)
        _assert_bytes_equal(conv, ref["pre"][0], f"conv after stock prefill, T={T}")
        _assert_bytes_equal(rec, ref["pre"][1], f"rec after stock prefill, T={T}")

    blocks, out = _stock(_logical_blocks(raw), _tokens(2, n), _STOCK_SLOTS)
    _assert_bytes_equal(out, ref["out_dec"], "stock decode output")
    _assert_finite(out, "stock decode output")
    raw = _raw(blocks)
    for T in (1, 4):
        conv, rec = _mk_read(raw, _STOCK_SLOTS, T)
        _assert_bytes_equal(conv, ref["dec"][0], f"conv after stock decode, T={T}")
        _assert_bytes_equal(rec, ref["dec"][1], f"rec after stock decode, T={T}")


def test_pool_bytes_match_stock_state_utils():
    """The interpret-mode pool bytes agree with main's host view of the stock
    layout (`gdn_pool_state_utils`, checked against the kernel on TPU)."""
    utils = pytest.importorskip(
        "kernels.gdn.v3.gdn_pool_state_utils",
        reason="needs tests/kernels/gdn/v3/gdn_pool_state_utils.py",
    )
    ref = _dense_reference()
    raw, _ = _stock_prefill_pool(0x00)
    idx = jnp.asarray(_STOCK_SLOTS, jnp.int32)
    rec, conv = utils.read_states(
        raw,
        idx,
        split=PPS,
        recurrent_shape=(N_V, D, D),
        conv_shape=(TAPS, DIM),
        recurrent_dtype=jnp.bfloat16,
    )
    _assert_bytes_equal(conv, ref["pre"][0], "stock-written conv via state utils")
    _assert_bytes_equal(rec, ref["pre"][1], "stock-written rec via state utils")

    placed = utils.write_states(
        _to_pool(_garbage_pool(1)),
        jnp.asarray(ref["dec"][1]),
        jnp.asarray(ref["dec"][0]),
        idx,
        split=PPS,
        recurrent_dtype=jnp.bfloat16,
    )
    conv, rec = _mk_read(placed, _STOCK_SLOTS)
    _assert_bytes_equal(conv, ref["dec"][0], "state-utils-placed conv via megakernel")
    _assert_bytes_equal(rec, ref["dec"][1], "state-utils-placed rec via megakernel")


# ---------------------------------------------------------------------------
# Megakernel write -> stock read
# ---------------------------------------------------------------------------
def _mk_checkpoints(
    slots: np.ndarray, fill: int, seed: int
) -> tuple[jax.Array, np.ndarray, np.ndarray]:
    """Megakernel-scatter random f32 conv / bf16 recurrent checkpoints into the
    `(NS, T)` `slots` of a garbage pool with stale conv rows set to `fill`.

    Returns `(raw pool, conv, rec)`: checkpoint `(s, t)` (row `s * T + t`) holds
    taps `conv[s * T + t]` `(TAPS, DIM)` and recurrent state `rec[s * T + t]`.
    """
    NS, T = slots.shape
    B = NS * T
    k = jax.random.split(jax.random.key(seed), 2)
    if T == 1:  # decode layout: row j * B + s is tap j of sequence s
        conv = jax.random.normal(k[0], (1, 8 * ((TAPS * B + 7) // 8), DIM))
        rows = np.asarray(conv)[0]
        exp_conv = np.stack([rows[[j * B + s for j in range(TAPS)]] for s in range(NS)])
    else:  # verify window per sequence; checkpoint t keeps rows 1 + t .. 3 + t
        conv = jax.random.normal(k[0], (1, NS, 8 * ((TAPS + T + 7) // 8), DIM))
        win = np.asarray(conv)[0]
        exp_conv = np.stack(
            [win[s, 1 + t : 1 + t + TAPS] for s in range(NS) for t in range(T)]
        )
    rec = (jax.random.normal(k[1], (1, B * N_V, D, D)) * 0.1).astype(jnp.bfloat16)
    raw_u8 = _fill_stale_rows(_garbage_pool(seed), slots, fill)
    raw = _mk_write(_to_pool(raw_u8), slots, conv, rec)
    # The scatter rewrites whole conv pages; restore the stale rows after it.
    raw = _to_pool(_fill_stale_rows(_to_u8(raw), slots, fill))
    return raw, exp_conv, np.asarray(rec)[0].reshape(B, N_V, D, D)


@pytest.mark.parametrize("fill", [0x00, 0xFF], ids=["clean", "stale_nan"])
@pytest.mark.parametrize("T", [1, 4])
def test_megakernel_write_stock_read(T: int, fill: int):
    """Megakernel checkpoints (decode T=1, verify T=4) are initial states for
    stock decode steps, one sequence per checkpoint."""
    NS = 2
    slots = 2 + np.arange(NS * T).reshape(NS, T)
    raw, exp_conv, exp_rec = _mk_checkpoints(slots, fill, seed=30 + T)
    tokens = _tokens(40 + T, NS * T)

    blocks, out = _stock(_logical_blocks(raw), tokens, slots)
    ref_states, ref_out = _stock(
        (jnp.asarray(exp_conv), jnp.asarray(exp_rec)), tokens, np.arange(NS * T)
    )
    _assert_bytes_equal(out, ref_out, "stock output from megakernel checkpoints")
    _assert_finite(out, "stock output from megakernel checkpoints")
    new_conv, new_rec = _mk_read(_raw(blocks), slots)
    _assert_bytes_equal(new_conv, ref_states[0], "stock new conv")
    _assert_bytes_equal(new_rec, ref_states[1], "stock new rec")


@pytest.mark.parametrize("fill", [0x00, 0xFF], ids=["clean", "stale_nan"])
def test_megakernel_checkpoints_through_stock_verify(fill: int):
    """MTP round trip on one checkpoint table: megakernel T=4 scatter -> stock
    verify window (`num_spec_tokens=3`) reads checkpoint `read_offsets[s]` and
    rewrites all four checkpoints in place -> megakernel gather reads them."""
    NS, T = 2, 4
    slots = 2 + np.arange(NS * T).reshape(NS, T)
    read_offsets = np.array([2, 1])
    raw, exp_conv, exp_rec = _mk_checkpoints(slots, fill, seed=50)
    tokens = _tokens(51, NS * T)

    blocks, out = _stock(_logical_blocks(raw), tokens, slots, read_offsets=read_offsets)
    ref_states, ref_out = _stock(
        (jnp.asarray(exp_conv), jnp.asarray(exp_rec)),
        tokens,
        np.arange(NS * T).reshape(NS, T),
        read_offsets=read_offsets,
    )
    _assert_bytes_equal(out, ref_out, "stock verify output")
    _assert_finite(out, "stock verify output")
    for T_read in (1, T):
        new_conv, new_rec = _mk_read(_raw(blocks), slots, T_read)
        _assert_bytes_equal(new_conv, ref_states[0], f"verify checkpoints, T={T_read}")
        _assert_bytes_equal(new_rec, ref_states[1], f"verify checkpoints, T={T_read}")
