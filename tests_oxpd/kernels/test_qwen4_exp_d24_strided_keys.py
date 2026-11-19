# SPDX-License-Identifier: Apache-2.0
"""q38 L2 D24: VLLM_TPU_QSA_INDEXER_STRIDED_KEYS reads the QSA index keys (BF16
values stored as bytes in the 8-bit pool carrier) as 32-bit words with a
sublane-strided load and a word -> BF16 vreg bitcast, instead of the BF16 view
whose reshape and lane slice repack sublanes.

The real StreamIndex scoring kernel runs in TPU interpret mode on CPU (the
shared-stream indexer test's fixtures); a JAX top-k stands in for the
SparseCore selector, so flag off and flag on are compared on the winners' score
bits and the selected group ids, bit for bit. The pool is the 8-bit carrier
(the served layout), so the strided reader engages; a spy checks it did. One
case runs the O1s request-shard selection inputs (request_shard_step's rows of
one TP rank). The TPU7x lowering carries the -sk kernel name only with the flag.

Interpret mode checks the reader against the BF16 view under interpret
semantics; the TPU bit equality is the 4-chip kernel bench's inline check.

q38 L2-SPLITVER: the CPU TPU interpreter of the pinned jax (0.11.2) reads only
through index transforms: interpret/utils.py `to_range` (:411-416) assumes every
ref transform is an NDIndexer, so ANY ref-level bitcast on a load raises
"'BitcastTransform' object has no attribute 'indices'" (interpret_pallas_call.py
:636 `get`). Both carrier readers bitcast the ref (the BF16 view
`ref.bitcast(bfloat16)`, the strided reader `ref.bitcast(uint32).reshape(...)`),
so every interpret case failed before comparing anything. The fixture
`ref_loads` teaches the interpreter's `get` to read the index prefix, then apply
the remaining ref transforms (bitcast / reshape / index) in numpy with the same
semantics as jax's own discharge rule (state/discharge.py `transform_array`,
state/utils.py `bitcast`); `test_ref_load_shim_matches_jax_discharge` checks that
equivalence on the reader's exact transform chains. Test-only: no src change.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src.mesh import AbstractDevice, AbstractMesh
from jax._src.pallas.mosaic.interpret import interpret_pallas_call as _interp
from jax._src.state import indexing as _indexing
from jax._src.state import types as _state_types
from jax.experimental.pallas import tpu as pltpu
from test_qwen4_exp_shared_stream_indexer import (  # noqa: F401
    BLOCK_TOPK,
    RATIO,
    capture_scores,
    inputs,
    interpret,
)

from vllm_torchtpu.kernels.qwen4_exp import indexer_components, pooled_qsa

FLAG = "VLLM_TPU_QSA_INDEXER_STRIDED_KEYS"


def _np_bitcast(x, dtype):
    """numpy twin of jax._src.state.utils.bitcast (the TPU ref-bitcast rule: the
    second-minor dimension absorbs the bit-width change, little-endian packing)."""
    dtype = np.dtype(dtype)
    xb, yb = x.dtype.itemsize * 8, dtype.itemsize * 8
    shape = list(x.shape)
    if xb != yb:
        assert len(shape) >= 2 and shape[-2] * xb % yb == 0, (x.shape, x.dtype, dtype)
    shape[-2] = shape[-2] * xb // yb
    if xb < yb:
        r = yb // xb
        x = x.reshape(*x.shape[:-2], x.shape[-2] // r, r, x.shape[-1]).swapaxes(-1, -2)
        return np.ascontiguousarray(x).view(dtype)[..., 0]
    if xb > yb:
        r = xb // yb
        y = np.ascontiguousarray(x).view(dtype).reshape(*x.shape, r)
        return y.swapaxes(-1, -2).reshape(shape)
    return np.ascontiguousarray(x).view(dtype)


def _np_index(x, indexer):
    idx = []
    for i in indexer.indices:
        if isinstance(i, _indexing.Slice):
            start, size, stride = int(i.start), int(i.size), int(i.stride)
            idx.append(slice(start, start + size * stride, stride))
        else:
            idx.append(int(i))
    return x[tuple(idx)]


def _np_transform(x, transforms):
    for t in transforms:
        if isinstance(t, _indexing.NDIndexer):
            x = _np_index(x, t)
        elif isinstance(t, _state_types.BitcastTransform):
            x = _np_bitcast(x, t.dtype)
        elif isinstance(t, _state_types.ReshapeTransform):
            x = x.reshape(t.shape)
        else:
            raise NotImplementedError(f"ref transform {type(t).__name__} on an interpret-mode load")
    return x


def _describe(transforms):
    """(bitcast dtypes, max index stride) of one load's transform chain."""
    casts, stride = [], 1
    for t in transforms:
        if isinstance(t, _state_types.BitcastTransform):
            casts.append(str(np.dtype(t.dtype)))
        elif isinstance(t, _indexing.NDIndexer):
            for i in t.indices:
                if isinstance(i, _indexing.Slice):
                    stride = max(stride, int(i.stride))
    return tuple(casts), stride


def _loads_through_transforms(read, seen):
    """``read`` (the interpreter's get, or its undecorated body) reading the index prefix of a
    load, then the remaining ref transforms applied in numpy."""

    def get(token, device_id, local_core_id, memory_space, buffer_id, transforms, *args, **kwargs):
        cut = next((i for i, t in enumerate(transforms) if not isinstance(t, _indexing.NDIndexer)), None)
        if cut is None:
            return read(token, device_id, local_core_id, memory_space, buffer_id, transforms, *args, **kwargs)
        head, rest = tuple(transforms[:cut]), jax.tree.map(int, tuple(transforms[cut:]))
        token, value = read(token, device_id, local_core_id, memory_space, buffer_id, head, *args, **kwargs)
        seen.append(_describe(rest))
        return token, np.asarray(_np_transform(np.asarray(value), rest))

    return get


@pytest.fixture
def ref_loads(monkeypatch):
    """Interpret-mode loads through ref-level bitcast / reshape transforms (see the module
    docstring). Returns the list of (bitcast dtypes, max stride) of every such load."""
    seen = []
    original = _interp.get
    get = _loads_through_transforms(original, seen)
    # the interpreter's DMA path calls get.__wrapped__ (interpret_pallas_call.py:962): keep that
    # attribute, with the same transform handling around the undecorated body
    get.__wrapped__ = _loads_through_transforms(original.__wrapped__, seen)
    monkeypatch.setattr(_interp, "get", get)
    return seen


def carrier(pool_bf16):
    """BF16 pool [P, T, 1, 2, W] -> 8-bit carrier [P, T, 1, 4, W]: carrier row
    2j + b holds byte b of BF16 row j (the same bytes, the served layout)."""
    raw = jax.lax.bitcast_convert_type(pool_bf16, jnp.uint8)  # [P, T, 1, 2, W, 2]
    raw = jnp.moveaxis(raw, -1, -2)  # [P, T, 1, 2, 2, W]
    raw = raw.reshape(*pool_bf16.shape[:3], 4, pool_bf16.shape[-1])
    return jax.lax.bitcast_convert_type(raw, jnp.float8_e4m3fn)


@pytest.fixture
def spy(monkeypatch):
    """Record (strided, carrier dtype) of every key read the kernel traces."""
    seen = []
    original = indexer_components.BF16KeyReader.read

    def read(self, ref, rows, width):
        seen.append((self.strided, str(ref.dtype)))
        return original(self, ref, rows, width)

    monkeypatch.setattr(indexer_components.BF16KeyReader, "read", read)
    return seen


def select(args):
    return np.asarray(pooled_qsa.select_blocks(**args, ratio=RATIO, block_topk=BLOCK_TOPK))


def run_both(monkeypatch, args, spy, loads):
    calls = capture_scores(monkeypatch)
    monkeypatch.setattr(pooled_qsa, "decode_query_chunk_size", lambda pages: 1 << 20)
    monkeypatch.setenv(FLAG, "0")
    spy.clear()
    loads.clear()
    jax.clear_caches()
    expected = select(args)
    off, off_loads = list(spy), list(loads)
    monkeypatch.setenv(FLAG, "1")
    spy.clear()
    loads.clear()
    jax.clear_caches()
    actual = select(args)
    on, on_loads = list(spy), list(loads)
    assert off and all(not s for s, _ in off), off
    assert on and all(s for s, _ in on), on
    assert all(d != "bfloat16" for _, d in off + on), "the carrier path must engage"
    # The interpreter really executed each reader: flag off loads through the BF16 view
    # (bitcast to bfloat16), flag on through the word-strided read (bitcast to uint32,
    # a sublane stride > 1 at the tests' 256-lane keys), never the other way round.
    assert off_loads and all(c == ("bfloat16",) for c, _ in off_loads), sorted(set(off_loads))
    assert on_loads and all(c == ("uint32",) for c, _ in on_loads), sorted(set(on_loads))
    assert any(s > 1 for _, s in on_loads), sorted(set(on_loads))
    print(f"D24_REF_LOADS off={len(off_loads)} {sorted(set(off_loads))} on={len(on_loads)} {sorted(set(on_loads))}")
    return expected, actual, calls


# (q_lens per request row, context per request row, padded tokens): 16 / 32 / 48 rows of
# MTP verify seats, idle rows, decode rows; contexts over several KV blocks.
CASES = {
    "rows16": ([4, 4, 1, 3, 4, 0, 0, 0] + [0] * 8, [3001, 2048, 1203, 4000, 517] + [0] * 11, 16),
    "rows32": ([4] * 7 + [1, 3] + [0] * 23, [3001, 2048, 1203, 4000, 517, 999, 3500, 7, 2047] + [0] * 23, 32),
    "rows48": ([4] * 11 + [2, 2] + [0] * 35,
               [3001, 2048, 1203, 4000, 517, 999, 3500, 2047, 4093 - 3, 64, 1022, 3998, 5] + [0] * 35, 48),
}


@pytest.mark.parametrize("data", ["integer", "normal"])
@pytest.mark.parametrize("query_block,kv_block", [(4, 256), (4, 1024)])
@pytest.mark.parametrize("case", sorted(CASES))
def test_strided_keys_same_scores_and_selection(monkeypatch, interpret, spy, ref_loads, case, query_block,
                                                kv_block, data):
    q_lens, context, tokens = CASES[case]
    args = inputs(len(case) + kv_block, q_lens, context, tokens, data)
    args["pool"] = carrier(args["pool"])
    monkeypatch.setenv("VLLM_TPU_QSA_DECODE_QUERY_BLOCK", str(query_block))
    monkeypatch.setenv("VLLM_TPU_QSA_DECODE_KV_BLOCK", str(kv_block))
    expected, actual, calls = run_both(monkeypatch, args, spy, ref_loads)
    (_, bits_off), (_, bits_on) = calls
    np.testing.assert_array_equal(np.asarray(bits_on), np.asarray(bits_off))
    np.testing.assert_array_equal(actual, expected)
    assert (expected >= 0).any()


def test_strided_keys_request_shard_rows(monkeypatch, interpret, spy, ref_loads):
    """O1s: one TP rank's request-shard rows (request_shard_step) select the
    same groups with the flag (8 ranks, 32-row step, 4 rows per rank)."""
    monkeypatch.setenv("VLLM_TPU_QSA_DECODE_REQUEST_SHARD", "1")
    monkeypatch.setenv("VLLM_TPU_QSA_DECODE_QUERY_BLOCK", "4")
    monkeypatch.setenv("VLLM_TPU_QSA_DECODE_KV_BLOCK", "256")
    q_lens, context, tokens = CASES["rows32"]
    full = inputs(77, q_lens, context, tokens, "normal")
    full["pool"] = carrier(full["pool"])
    size = 8
    width = tokens // size
    for index in (0, 3):
        base, sub_starts, sub_context = pooled_qsa.request_shard_step(
            full["starts"], full["context"], jnp.asarray([index], jnp.int32), width, size, 4)
        own = lambda x: jax.lax.dynamic_slice_in_dim(x, base, width)  # noqa: E731
        args = dict(full, query=own(full["query"]), lengths=own(full["lengths"]), req=own(full["req"]),
                    starts=sub_starts, context=sub_context)
        expected, actual, calls = run_both(monkeypatch, args, spy, ref_loads)
        (_, bits_off), (_, bits_on) = calls
        np.testing.assert_array_equal(np.asarray(bits_on), np.asarray(bits_off))
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("rows,width", [(8, 256), (16, 256), (8, 128)])
def test_ref_load_shim_matches_jax_discharge(rows, width):
    """The ref_loads shim applies the readers' exact transform chains like jax's own
    discharge rule (state/discharge.py transform_array), bit for bit."""
    from jax._src.state import discharge
    from jax.experimental import pallas as pl

    rng = np.random.default_rng(rows + width)
    x = rng.integers(0, 256, size=(2, rows, 4, width), dtype=np.uint8).view(jnp.float8_e4m3fn)
    tiles = width // 128
    nd = _indexing.NDIndexer.from_indices_shape
    full = lambda shape: nd(tuple(slice(None) for _ in shape), shape)  # noqa: E731
    chains = {
        # BF16 view: ref.at[b].bitcast(bfloat16)[...]
        "bf16_view": (nd((1,), x.shape), _state_types.BitcastTransform(jnp.dtype(jnp.bfloat16)),
                      full((rows, 2, width))),
        # word-strided read: ref.at[b].bitcast(uint32).reshape(-1, 128)[pl.ds(t, rows, tiles)]
        "strided": (nd((1,), x.shape), _state_types.BitcastTransform(jnp.dtype(jnp.uint32)),
                    _state_types.ReshapeTransform((rows * tiles, 128)),
                    nd((pl.ds(tiles - 1, rows, tiles), slice(None)), (rows * tiles, 128))),
        "word_rows": (nd((0,), x.shape), _state_types.BitcastTransform(jnp.dtype(jnp.uint32)),
                      _state_types.ReshapeTransform((rows * tiles, 128)),
                      nd((pl.ds(0, rows, tiles), slice(None)), (rows * tiles, 128))),
    }
    for name, transforms in chains.items():
        concrete = jax.tree.map(int, transforms)
        mine = _np_transform(x, concrete)
        ref_bits = np.asarray(discharge.transform_array(jnp.asarray(x), concrete))
        assert mine.dtype == ref_bits.dtype and mine.shape == ref_bits.shape, (name, mine.shape, ref_bits.shape)
        np.testing.assert_array_equal(mine.view(np.uint8), ref_bits.view(np.uint8), err_msg=name)
    # the value-level word -> BF16 bitcast the strided reader applies after its load
    words = _np_transform(x, jax.tree.map(int, chains["word_rows"]))
    np.testing.assert_array_equal(_np_bitcast(words, jnp.bfloat16).view(np.uint16),
                                  np.asarray(jax.jit(lambda w: pltpu.bitcast(w, jnp.bfloat16))(jnp.asarray(words))).view(np.uint16))


def _lower_select(rows):
    S = jax.ShapeDtypeStruct
    args = (S((rows, 4, 128), jnp.bfloat16), S((512, 1664, 1, 4, 256), jnp.float8_e4m3fn), S((rows,), jnp.int32),
            S((64, 158), jnp.int32), S((rows,), jnp.int32), S((65,), jnp.int32), S((64,), jnp.int32))
    fn = jax.jit(functools.partial(pooled_qsa.select_blocks, ratio=4, block_topk=512, max_query_len=4))
    device = AbstractDevice(device_kind="TPU7x", num_cores=1, platform="tpu")
    with jax.sharding.use_abstract_mesh(AbstractMesh((), (), abstract_device=device)):
        return fn.trace(*args).lower(lowering_platforms=("tpu",)).as_text()


@pytest.mark.parametrize("flag", ["0", "1"])
def test_strided_keys_kernel_name(monkeypatch, flag):
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)
    for k, v in (("VLLM_TPU_QSA_DECODE_QUERY_BLOCK", "4"), ("VLLM_TPU_QSA_DECODE_KV_BLOCK", "3328"),
                 ("VLLM_TPU_QSA_DECODE_SINGLE_KV_BLOCK", "1024"), ("VLLM_TPU_QSA_DECODE_POOL_PAGE_TABLE", "1"),
                 ("VLLM_TPU_QSA_DECODE_POOL_PAGE_DMA", "1"), ("VLLM_TPU_QSA_INDEXER_MIXED_ONLY", "1"), (FLAG, flag)):
        monkeypatch.setenv(k, v)
    jax.clear_caches()
    text = _lower_select(64)
    assert 'kernel_name = "StreamIdxTC' in text
    assert ("-sk\"" in text) == (flag == "1")


def _lower_select_shape(rows, slots, max_query_len):
    S = jax.ShapeDtypeStruct
    args = (S((rows, 4, 128), jnp.bfloat16), S((512, 1664, 1, 4, 256), jnp.float8_e4m3fn), S((rows,), jnp.int32),
            S((slots, 158), jnp.int32), S((rows,), jnp.int32), S((slots + 1,), jnp.int32), S((slots,), jnp.int32))
    fn = jax.jit(functools.partial(pooled_qsa.select_blocks, ratio=4, block_topk=512, max_query_len=max_query_len))
    device = AbstractDevice(device_kind="TPU7x", num_cores=1, platform="tpu")
    with jax.sharding.use_abstract_mesh(AbstractMesh((), (), abstract_device=device)):
        return fn.trace(*args).lower(lowering_platforms=("tpu",)).as_text()


def _streamidx_kernels(text):
    import re

    return re.findall(r'kernel_name = "(StreamIdxTC-[^"]*)"', text)


@pytest.mark.parametrize("flag", ["0", "1"])
def test_strided_keys_scoped_to_decode(monkeypatch, flag):
    """q38 L2 D24 scope: the strided read is proven bitwise on device for the decode call only
    (KB-D24 T95j), so the prefill indexer call keeps the BF16-view reader whatever the flag: its
    StreamIdx kernel never carries -sk, while the decode call carries -sk exactly with the flag."""
    from vllm_torchtpu import envs

    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)
    for k, v in (("VLLM_TPU_QSA_DECODE_QUERY_BLOCK", "4"), ("VLLM_TPU_QSA_DECODE_KV_BLOCK", "3328"),
                 ("VLLM_TPU_QSA_DECODE_SINGLE_KV_BLOCK", "1024"), ("VLLM_TPU_QSA_DECODE_POOL_PAGE_TABLE", "1"),
                 ("VLLM_TPU_QSA_DECODE_POOL_PAGE_DMA", "1"), ("VLLM_TPU_QSA_INDEXER_MIXED_ONLY", "1"),
                 ("VLLM_TPU_QSA_INDEXER_SC_BOUNDED_COPY", "1"), ("VLLM_TPU_QSA_PREFILL_INDEXER_BKV", "1024"),
                 ("VLLM_TPU_QSA_PREFILL_INDEXER_CHUNK", "640"), ("VLLM_TPU_QSA_PREFILL_SC_BOUNDED_COPY", "1"),
                 (FLAG, flag)):
        if k in vars(envs):  # order-proof: drop a frozen module attribute (envs resolves the rest lazily)
            monkeypatch.delattr(envs, k)
        monkeypatch.setenv(k, v)
    assert bool(envs.VLLM_TPU_QSA_INDEXER_STRIDED_KEYS) == (flag == "1")
    jax.clear_caches()
    prefill = _streamidx_kernels(_lower_select_shape(6400, 8, 1))
    assert prefill, "no StreamIdx kernel in the prefill program"
    # The chunked prefill kernel name carries "-c<N>" after the layout tags, so match "sk" as a
    # dash-separated name token, not as a '-sk"' suffix (that suffix test passed with -sk present).
    assert not any("sk" in name.split("-") for name in prefill), prefill
    jax.clear_caches()
    decode = _streamidx_kernels(_lower_select_shape(64, 64, 4))
    assert decode, "no StreamIdx kernel in the decode program"
    assert all(("sk" in name.split("-")) == (flag == "1") for name in decode), decode
    assert indexer_components.qsa_indexing(strided_keys=False).keys == indexer_components.BF16KeyReader()
    assert indexer_components.qsa_indexing(strided_keys=True).keys == indexer_components.BF16KeyReader(strided=True)


def test_flag_is_registered_default_off():
    from vllm_torchtpu import envs

    assert FLAG in envs.environment_variables
    assert envs.environment_variables[FLAG]() is False
    assert indexer_components.qsa_indexing().keys == indexer_components.BF16KeyReader()
