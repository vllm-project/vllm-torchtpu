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
"""Per-tile candidate selection (`msa_dcp._tile_candidates`, the DCP prefill
path of `msa_dcp.msa_write_and_candidates_dcp` at `TILE_SELECT_MIN_TOKENS`,
which writes the all-gather's slots in place) against the class path it
replaces (`msa_dcp._class_candidates`: head-major
fold, `local_candidates` per tile class), in interpret mode: the candidates
of every real token must match bit for bit, ties and forced blocks
included; padding tokens get none."""

import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.models.minimax_m3.packing import pack, rounded
from vllm_torchtpu.models.minimax_m3.kernels import (
    msa,
    msa_dcp,
    msa_fp4,
    msa_write,
)

PAGE = 128
HEAD_DIM = 128
MAX_SEQS = 4
PAGES = 96  # physical pages per rank
TOPK = 16
# The served write kernel, captured before the fixture replaces it.
_KERNEL_WRITE_ROWS = msa_write.write_rows
NUM_TOKENS = 448
MAX_LEN = 4500


def _xla_write_rows(cache, rows, pages, slots, *, max_runs, bits=None, interpret=False):
    del max_runs, bits, interpret
    packing = cache.shape[2]
    return cache.at[pages, slots // packing, slots % packing].set(rows, mode="drop")


@pytest.fixture(autouse=True)
def _scatter_writes(monkeypatch):
    monkeypatch.setattr(msa_write, "write_rows", _xla_write_rows)


@pytest.fixture
def _retrace():
    """Drop jit caches around a test that patches module code, so the
    patched kernel is traced (and the next test sees the real one)."""
    jax.clear_caches()
    yield
    jax.clear_caches()


def _caches(rng, dtype, heads):
    if dtype == "fp4":
        kv = msa_fp4.pack_rows(
            jnp.zeros((PAGES, PAGE, 2 * HEAD_DIM * heads), jnp.uint8),
            scales=jnp.zeros((PAGES, 2 * HEAD_DIM * heads), jnp.float32),
        )
        # Few distinct scales and nibbles: many equal block maxima.
        index = msa_fp4.pack_rows(
            jnp.asarray(rng.integers(0, 16, (PAGES, PAGE, HEAD_DIM)), jnp.uint8),
            scales=jnp.asarray(rng.choice([0.5, 1.0], (PAGES, HEAD_DIM)), jnp.float32),
        )
        return kv, index
    kv = pack(np.zeros((PAGES, PAGE, 2 * HEAD_DIM * heads), np.float32), dtype)
    index = pack(rounded(rng.integers(-2, 3, (PAGES, PAGE, HEAD_DIM)), dtype), dtype)
    return kv, index


def _inputs(rng, q_lens, ctx_lens, dcp, heads, pad):
    num_seqs = len(q_lens)
    num_real = sum(q_lens)
    # One padded size and table width for every batch: one compile per
    # (DCP, dtype, path) in interpret mode.
    num_tokens = NUM_TOKENS
    assert num_real + pad <= num_tokens
    seq_lens = np.zeros(MAX_SEQS, np.int32)
    seq_lens[:num_seqs] = np.add(ctx_lens, q_lens)
    cu_q = np.full(MAX_SEQS + 1, num_real, np.int32)
    cu_q[: num_seqs + 1] = np.concatenate([[0], np.cumsum(q_lens)])
    logical_pages = -(-MAX_LEN // (PAGE * dcp))
    assert seq_lens.max() <= MAX_LEN
    assert num_seqs * logical_pages < PAGES
    tables = np.zeros((MAX_SEQS, logical_pages), np.int32)
    tables[:num_seqs] = rng.permutation(np.arange(1, PAGES))[
        : num_seqs * logical_pages
    ].reshape(num_seqs, logical_pages)

    def bf16(*shape):
        # Integer-valued queries: exact dots, frequent equal scores.
        return jnp.asarray(rng.integers(-2, 3, shape)).astype(jnp.bfloat16)

    tensors = (
        bf16(num_tokens, heads, HEAD_DIM),
        bf16(num_tokens, heads, HEAD_DIM),
        bf16(num_tokens, heads, HEAD_DIM),
        bf16(num_tokens, HEAD_DIM),
    )
    meta = (
        jnp.asarray(seq_lens),
        jnp.asarray(tables.reshape(-1)),
        jnp.asarray(cu_q),
        jnp.asarray([0, 0, num_seqs], jnp.int32),
    )
    return num_real, tensors, meta


def _candidates(monkeypatch, tile, caches, tensors, meta, rank, dcp, **config):
    monkeypatch.setattr(msa_dcp, "TILE_SELECT_ENABLED", tile)
    monkeypatch.setattr(msa_dcp, "TILE_SELECT_MIN_TOKENS", 0)
    _, _, cand = msa_dcp.msa_write_and_candidates_dcp(
        *tensors, *caches, *meta, rank=rank, dcp=dcp, interpret=True, **config
    )
    return np.asarray(cand)


def _check(monkeypatch, q_lens, ctx_lens, dcp, dtype, *, ranks=None, seed=0, **config):
    heads = max(1, dcp // 2)
    if dtype == "fp4":
        # The step's own fp4 writes are the same on both paths (covered by
        # test_msa_write); score the cache as given.
        monkeypatch.setattr(msa, "write_fp4", lambda kv, index, *args: (kv, index))
    rng = np.random.default_rng(seed)
    caches = _caches(rng, dtype, heads)
    num_real, tensors, meta = _inputs(rng, q_lens, ctx_lens, dcp, heads, pad=5)
    config = dict(topk=TOPK, init_blocks=1, local_blocks=1) | config
    for rank in ranks if ranks is not None else range(dcp):
        want = _candidates(monkeypatch, False, caches, tensors, meta, rank, dcp, **config)
        got = _candidates(monkeypatch, True, caches, tensors, meta, rank, dcp, **config)
        topk = config["topk"]
        assert got.shape == want.shape == (2 * topk, NUM_TOKENS * heads)
        assert got.dtype == want.dtype == np.int32
        real = num_real * heads  # folded columns t * G + h of the real tokens
        np.testing.assert_array_equal(got[:, :real], want[:, :real])
        # Every column's list is sorted, as the network merge needs:
        # score descending, ties to the lower block, misses last.
        sc, bl = got[:topk, :real].view(np.float32), got[topk:, :real]
        ahead = (sc[:-1] > sc[1:]) | ((sc[:-1] == sc[1:]) & (bl[:-1] < bl[1:]))
        assert np.all(ahead | ((sc[1:] == -np.inf) & (bl[1:] == -1)))
        # Padding tokens: no candidate.
        pad = got[:, real:]
        assert np.all(pad[:topk].view(np.float32) == -np.inf)
        assert np.all(pad[topk:] == -1)


# (q_lens, ctx_lens): a prefill chunk over long context next to decode rows
# (both tile classes; `test_tile_select_chunking_control` runs several
# selection chunks); fresh
# prefills of fewer blocks than topk * dcp (-inf ties); decode / MTP rows
# only (small class only; the large class has no live tile).
_MIXED = ([300, 1, 8, 130], [3000, 2500, 500, 0])
_FRESH = ([260, 129, 40], [0, 0, 0])
_DECODE = ([1, 8, 4, 1], [3000, 128, 127, 4000])
# Requests 1-7 rows apart around a long chunk: their slot columns share
# 128-lane groups with the chunk's tiles.
_ADJACENT = ([5, 300, 3, 134], [2000, 1500, 100, 0])


@pytest.mark.parametrize("dcp", [2, 4, 8])
@pytest.mark.parametrize(
    "batch", [_MIXED, _FRESH, _DECODE, _ADJACENT], ids=["mixed", "fresh", "decode", "adjacent"]
)
def test_tile_select_matches_rows_bf16(monkeypatch, dcp, batch):
    _check(monkeypatch, *batch, dcp, jnp.bfloat16, ranks=[0, dcp - 1], seed=dcp)


@pytest.mark.parametrize("dcp", [2, 4, 8])
def test_tile_select_matches_rows_fp4(monkeypatch, dcp):
    _check(monkeypatch, *_MIXED, dcp, "fp4", ranks=[dcp // 2], seed=10 + dcp)


def test_tile_select_matches_rows_fp8_forced(monkeypatch):
    """fp8 cache, more forced init / local blocks and a small topk."""
    _check(
        monkeypatch, *_MIXED, 4, jnp.float8_e4m3fn, ranks=[1], seed=7,
        topk=4, init_blocks=3, local_blocks=2,
    )


def test_tile_select_chunking_control(monkeypatch, _retrace):
    """No-fault control: 8-page selection steps give the same candidates."""
    monkeypatch.setattr(msa_dcp, "_select_chunk", lambda num_pages: 8)
    _check(monkeypatch, *_MIXED, 2, jnp.bfloat16, ranks=[1], seed=3)


def _faulty_kernel(old, new):
    src = inspect.getsource(msa_dcp._tile_select_kernel)
    assert old in src
    namespace = dict(vars(msa_dcp))
    exec(compile(src.replace(old, new), "<fault>", "exec"), namespace)
    return namespace["_tile_select_kernel"]


@pytest.mark.parametrize(
    "fault",
    [
        ("ties", ("gt = v > m", "gt = v >= m")),
        ("visibility", ("// block_size + 1", "// block_size + 2")),
        ("head fold", (">> heads_shift", ">> (heads_shift + 1)")),
        ("forced init", ("blocks < init_blocks", "blocks <= init_blocks")),
    ],
    ids=lambda f: f[0],
)
def test_tile_select_faults_are_caught(monkeypatch, _retrace, fault):
    _, (old, new) = fault
    monkeypatch.setattr(msa_dcp, "_tile_select_kernel", _faulty_kernel(old, new))
    # DCP 2, both ranks: > 8 pages per rank (equal scores in one sublane
    # slot) and block 1 on rank 1.
    with pytest.raises(AssertionError):
        _check(monkeypatch, *_MIXED, 2, jnp.bfloat16, ranks=[0, 1], seed=4)


def _faulty_place(old, new):
    src = inspect.getsource(msa_dcp._place)
    assert old in src
    namespace = dict(vars(msa_dcp))
    exec(compile(src.replace(old, new), "<fault>", "exec"), namespace)
    return namespace["_place"]


@pytest.mark.parametrize(
    "fault",
    [
        ("one column too many", ("lane < off + n", "lane <= off + n")),
        ("window shifted", ("(off - src0 + window)", "(off - src0 + window + 1)")),
    ],
    ids=lambda f: f[0],
)
def test_tile_select_placement_faults_are_caught(monkeypatch, _retrace, fault):
    """The slots are written in place at unaligned columns; a placement that
    writes one column too many or one lane off fails the comparison."""
    _, (old, new) = fault
    monkeypatch.setattr(msa_dcp, "_place", _faulty_place(old, new))
    with pytest.raises(AssertionError):
        _check(monkeypatch, *_ADJACENT, 2, jnp.bfloat16, ranks=[0], seed=5)


def test_tile_select_small_placement_fault_is_caught(monkeypatch, _retrace):
    monkeypatch.setattr(
        msa_dcp,
        "_tile_select_kernel",
        _faulty_kernel("src - group, lo,", "src - group + 1, lo,"),
    )
    with pytest.raises(AssertionError):
        _check(monkeypatch, *_ADJACENT, 2, jnp.bfloat16, ranks=[0], seed=5)


def test_tile_select_threshold(monkeypatch):
    assert not msa_dcp.tile_select(8192, 4)  # off by default
    monkeypatch.setattr(msa_dcp, "TILE_SELECT_ENABLED", True)
    assert not msa_dcp.tile_select(256)
    assert msa_dcp.tile_select(1024) and msa_dcp.tile_select(8192, 4)
    assert not msa_dcp.tile_select(8192, 3)


# (dcp, G, tokens, index cache): the served prefill buckets of DCP 2, 4 and
# 8 (4-bit caches with folded scales, as served, and fp8).
_COMPILE_SHAPES = [
    pytest.param(2, 1, 1024, "fp4", id="dcp2-1024-fp4"),
    pytest.param(4, 2, 2048, "fp4", id="dcp4-2048-fp4"),
    pytest.param(4, 2, 8192, "fp4", id="dcp4-8192-fp4"),
    pytest.param(4, 2, 8192, "fp8", id="dcp4-8192-fp8"),
    pytest.param(8, 4, 4096, "fp4", id="dcp8-4096-fp4"),
]


def _bf16_scatters(text):
    """Result types of the `stablehlo.scatter` ops in `text` that hold bf16."""
    lines = text.splitlines()
    found = []
    for i, line in enumerate(lines):
        if "stablehlo.scatter" not in line:
            continue
        # A scatter with an update region ends at its `}) : (...) -> ...` line.
        end = next((j for j in range(i, len(lines)) if "->" in lines[j]), i)
        if "bf16" in lines[end].split("->")[-1]:
            found.append(lines[end].strip())
    return found


@pytest.mark.parametrize("dcp,heads,num_tokens,cache", _COMPILE_SHAPES)
def test_tile_select_compiles_for_tpu7x(monkeypatch, dcp, heads, num_tokens, cache):
    """The DCP score op of a prefill step (writes, both tile classes and
    their per-tile selection) compiles for tpu7x without a device, rank as
    an input, with no head-fold scatter left."""
    from jax.experimental import topologies
    from jax.sharding import AbstractDevice, AbstractMesh, SingleDeviceSharding
    from jax.sharding import use_abstract_mesh

    try:
        topology = topologies.get_topology_desc("tpu7x:2x2x1", "tpu")
    except Exception as e:  # no libtpu
        pytest.skip(f"no deviceless tpu7x topology: {e}")
    sharding = SingleDeviceSharding(topology.devices[0])
    monkeypatch.setattr(msa_write, "write_rows", _KERNEL_WRITE_ROWS)

    def spec(shape, dt):
        return jax.ShapeDtypeStruct(shape, dt, sharding=sharding)

    max_seqs, pages_per_seq, i32, bf16 = 64, 2048 // dcp, jnp.int32, jnp.bfloat16
    if cache == "fp4":
        kv = spec((4096, PAGE // 8 + 1, 4, 2 * HEAD_DIM * heads), jnp.uint8)
        index = spec((4096, PAGE // 8 + 1, 4, HEAD_DIM), jnp.uint8)
    else:
        kv = spec((4096, PAGE // 4, 4, 2 * HEAD_DIM * heads), jnp.float8_e4m3fn)
        index = spec((4096, PAGE // 4, 4, HEAD_DIM), jnp.float8_e4m3fn)
    args = (
        spec((num_tokens, heads, HEAD_DIM), bf16),
        spec((num_tokens, heads, HEAD_DIM), bf16),
        spec((num_tokens, heads, HEAD_DIM), bf16),
        spec((num_tokens, HEAD_DIM), bf16),
        kv,
        index,
        spec((max_seqs,), i32),
        spec((max_seqs * pages_per_seq,), i32),
        spec((max_seqs + 1,), i32),
        spec((3,), i32),
        spec((1,), i32),
    )

    def score(*args):
        *rest, rank = args
        return msa_dcp.msa_write_and_candidates_dcp(
            *rest, topk=TOPK, init_blocks=1, local_blocks=1, rank=rank, dcp=dcp
        )

    monkeypatch.setattr(msa_dcp, "TILE_SELECT_ENABLED", True)
    assert msa_dcp.tile_select(num_tokens)
    tpu7x = AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
    with use_abstract_mesh(tpu7x):
        lowered = jax.jit(score).lower(*args)
        text = lowered.as_text()
        assert "minimax_m3_msa_tile_select" in text
        # The int32 step-metadata scatter stays; the bf16 head fold must not.
        assert not _bf16_scatters(text)
        assert lowered.out_info[2].shape == (2 * TOPK, num_tokens * heads)
        assert lowered.compile() is not None


def test_tile_select_toggle():
    from types import SimpleNamespace

    from vllm_torchtpu.models.minimax_m3 import attention

    assert not msa_dcp.TILE_SELECT_ENABLED
    assert not attention.tile_select_enabled(SimpleNamespace(additional_config={}))
    assert not attention.tile_select_enabled(SimpleNamespace(additional_config=None))
    on = SimpleNamespace(additional_config={attention.TILE_SELECT_KEY: True})
    assert attention.tile_select_enabled(on)
    with pytest.raises(ValueError):
        attention.tile_select_enabled(
            SimpleNamespace(additional_config={attention.TILE_SELECT_KEY: "yes"})
        )
