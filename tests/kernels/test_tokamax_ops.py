# SPDX-License-Identifier: Apache-2.0
"""`kernels/tokamax_ops.py`: tokamax's CSA gather, lightning indexer and
SparseCore top-k in place of the in-tree kernels.

The routing tests fake both sides and run on CPU. The parity tests feed the
same inputs to tokamax and to the in-tree kernel on TPU; the in-tree kernels
are held against references in their own test files.
"""

import importlib
import sys
import types
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels import tokamax_ops

_CSA_API = "tokamax._src.ops.experimental.tpu.csa_gather.api"
_INDEXER = "tokamax._src.ops.experimental.lightning_indexer.pallas_mosaic_tpu"
_TOPK = "tokamax._src.ops.experimental.tpu.topk.pallas_mosaic_tpu"

_IN_TREE = {
    "csa_gather": "vllm_torchtpu.kernels.deepseek_v4.core_attention.csa_gather",
    "streamindex_topk": (
        "vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.streamindex_topk"
    ),
    "sparsecore_topk": "vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk",
}



def _install(monkeypatch, dotted: str) -> types.ModuleType:
    """Installs an empty fake `dotted` module, parents included."""
    parts = dotted.split(".")
    for i in range(1, len(parts) + 1):
        name = ".".join(parts[:i])
        if not isinstance(sys.modules.get(name), types.ModuleType) or i == len(parts):
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    for i in range(1, len(parts)):
        monkeypatch.setattr(
            sys.modules[".".join(parts[:i])],
            parts[i],
            sys.modules[".".join(parts[: i + 1])],
            raising=False,
        )
    return sys.modules[dotted]


@pytest.fixture
def in_tree(monkeypatch):
    """Replaces the three in-tree kernels; returns {name: [kwargs per call]}."""
    calls = {name: [] for name in _IN_TREE}
    for name, module in _IN_TREE.items():

        def fake(*args, _name=name, **kwargs):
            calls[_name].append((args, kwargs))
            return f"in-tree {_name}"

        monkeypatch.setattr(importlib.import_module(module), name, fake)
    return calls


@pytest.fixture
def logger(monkeypatch):
    logger = mock.Mock()
    monkeypatch.setattr(tokamax_ops, "logger", logger)
    return logger


@pytest.fixture
def fake_tokamax(monkeypatch):
    """Fakes the three tokamax operators; records their calls."""
    calls = {}

    def csa_gather(*args, **kwargs):
        calls["csa_gather"] = (args, kwargs)
        return "tokamax"

    _install(monkeypatch, _CSA_API).csa_gather = csa_gather

    class Op:
        def __init__(self, name, config=None):
            self.name, self.config = name, config

        def __call__(self, *args, **kwargs):
            calls[self.name] = (self.config, args, kwargs)
            return "tokamax"

    indexer = _install(monkeypatch, _INDEXER)
    indexer.Config = lambda **kw: kw
    indexer.KVLayout = {"HEAD_ALONG_SUBLANE": "HAS", "SEQ_ALONG_LANE": "SAL"}
    indexer.PallasTpuLightningIndexer = lambda config: Op("indexer", config)

    topk = _install(monkeypatch, _TOPK)
    topk.Config = lambda **kw: kw
    topk.PallasTpuTopK = lambda config: Op("topk", config)
    return calls


def _indexer_args():
    return tuple(f"arg{i}" for i in range(6)) + ((0, 1, 2),)


# --- routing (CPU) --------------------------------------------------------------


def test_csa_gather_runs_tokamax(fake_tokamax, in_tree, logger):
    assert tokamax_ops.csa_gather("nope", "rope", "idx", 7, top_k=256) == "tokamax"
    args, kwargs = fake_tokamax["csa_gather"]
    assert args == ("nope", "rope", "idx", 7)
    assert kwargs == {"top_k": 256, "implementation": "mosaic_tpu"}
    assert not in_tree["csa_gather"] and not logger.warning_once.called


def test_indexer_maps_the_in_tree_tuning_onto_tokamax_config(
    fake_tokamax, in_tree, logger
):
    out = tokamax_ops.streamindex_topk(
        *_indexer_args(),
        k=512,
        compression_ratio=4,
        num_kv_pages_per_block=(16, 8, 8),
        num_queries_per_block=(1, 128, 128),
        decode_req_batch_size=4,
        chunk_tokens=128,
        buffer_count=None,
    )
    assert out == "tokamax" and not in_tree["streamindex_topk"]
    config, args, kwargs = fake_tokamax["indexer"]
    # Set values carry over; a None means "the kernel's default" and is dropped.
    assert config == {
        "num_kv_pages_per_block": (16, 8, 8),
        "num_queries_per_block": (1, 128, 128),
        "decode_req_batch_size": 4,
        "chunk_tokens": 128,
    }
    np.testing.assert_array_equal(args[-1], [0, 1, 2])
    assert kwargs["kv_layout"] == "HAS"
    assert kwargs["k"] == 512 and kwargs["compression_ratio"] == 4


def test_topk_runs_tokamax_with_the_scheduling_group(fake_tokamax, in_tree, logger):
    out = tokamax_ops.sparsecore_topk(
        "scores", 2048, "lens", scheduling_group_id=3, return_scores=True
    )
    assert out == "tokamax" and not in_tree["sparsecore_topk"]
    config, args, kwargs = fake_tokamax["topk"]
    assert config == {"scheduling_group_id": 3}
    assert args == ("scores", 2048, "lens") and kwargs == {"return_scores": True}


@pytest.mark.parametrize(
    "call, name, reason",
    [
        (
            lambda: tokamax_ops.streamindex_topk(
                *_indexer_args(), k=8, compression_ratio=1, cp_size=2
            ),
            "streamindex_topk",
            "cp_size=2",
        ),
        (
            lambda: tokamax_ops.sparsecore_topk("s", 16, write_empty_rows=False),
            "sparsecore_topk",
            "write_empty_rows=False",
        ),
    ],
    ids=["indexer-dcp", "topk-no-empty-rows"],
)
def test_arguments_outside_tokamax_keep_the_in_tree_kernel(
    fake_tokamax, in_tree, logger, call, name, reason
):
    assert call() == f"in-tree {name}"
    assert reason in logger.warning_once.call_args.args[-1]


def test_kill_switch_keeps_every_in_tree_kernel(
    fake_tokamax, in_tree, logger, monkeypatch
):
    monkeypatch.setenv("VLLM_TPU_USE_TOKAMAX", "0")
    assert tokamax_ops.csa_gather("n", "r", "i") == "in-tree csa_gather"
    assert tokamax_ops.sparsecore_topk("s", 16) == "in-tree sparsecore_topk"
    assert (
        tokamax_ops.streamindex_topk(*_indexer_args(), k=8, compression_ratio=1)
        == "in-tree streamindex_topk"
    )
    assert not fake_tokamax


def test_missing_tokamax_falls_back(in_tree, logger, monkeypatch):
    monkeypatch.setitem(sys.modules, _CSA_API, None)
    # A package that already imported `api` would hand it back as an attribute.
    parent, _, child = _CSA_API.rpartition(".")
    if parent in sys.modules:
        monkeypatch.delattr(sys.modules[parent], child, raising=False)
    assert tokamax_ops.csa_gather("n", "r", "i") == "in-tree csa_gather"
    # ModuleNotFoundError when tokamax is absent altogether.
    assert "Error" in logger.warning_once.call_args.args[-1]


def test_not_implemented_in_tokamax_falls_back(fake_tokamax, in_tree, logger):
    def refuse(*_a, **_k):
        raise NotImplementedError("needs TPU v7")

    sys.modules[_CSA_API].csa_gather = refuse
    assert tokamax_ops.csa_gather("n", "r", "i") == "in-tree csa_gather"
    assert "needs TPU v7" in logger.warning_once.call_args.args[-1]


def test_fallback_message_names_the_kernel(in_tree, logger, monkeypatch):
    monkeypatch.setenv("VLLM_TPU_USE_TOKAMAX", "0")
    tokamax_ops.csa_gather("n", "r", "i")
    fmt, *args = logger.warning_once.call_args.args
    assert fmt % tuple(args) == (
        "csa_gather: falling back to the in-tree kernel because of "
        "VLLM_TPU_USE_TOKAMAX=0."
    )


# --- parity (TPU) ---------------------------------------------------------------


@pytest.fixture
def tpu():
    """Skips off TPU. A fixture rather than a skipif: the conftest refuses test
    modules that initialize a JAX backend while being collected."""
    if jax.default_backend() != "tpu":
        pytest.skip("Pallas TPU kernels need a TPU.")


@pytest.fixture
def no_fallback(monkeypatch):
    """Fails the test if tokamax_ops quietly ran the in-tree kernel instead."""
    logger = mock.Mock()
    monkeypatch.setattr(tokamax_ops, "logger", logger)
    yield
    assert not logger.warning_once.called, logger.warning_once.call_args


@pytest.mark.tpu_test
def test_csa_gather_matches_in_tree(tpu, no_fallback):
    from vllm_torchtpu.kernels.deepseek_v4.core_attention.csa_gather import csa_gather

    pages, page_size, top_k, n = 64, 64, 128, 512
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    words = lambda key, shape: jax.random.randint(  # noqa: E731
        key, shape, -(2**31), 2**31 - 1, jnp.int32
    )
    nope = words(k1, (pages, page_size, 128))
    rope = words(k2, (pages, page_size // 4, 128))
    indices = jax.random.randint(k3, (n,), 0, pages * page_size, jnp.int32)

    got = jax.jit(tokamax_ops.csa_gather, static_argnames="top_k")(
        nope, rope, indices, top_k=top_k
    )
    want = csa_gather(nope, rope, indices, top_k=top_k)
    for g, w in zip(got, want):
        np.testing.assert_array_equal(np.asarray(g), np.asarray(w))


def _sorted_rows(x):
    return np.sort(np.asarray(x), axis=-1)


@pytest.mark.tpu_test
@pytest.mark.parametrize("b, n, k", [(16, 8192, 2048), (64, 4096, 512)])
def test_sparsecore_topk_matches_in_tree(tpu, no_fallback, b, n, k):
    from vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk import sparsecore_topk

    rng = np.random.default_rng(0)
    # Distinct scores, so the exact top-k is unique and unsorted outputs compare
    # as sets.
    scores = jnp.asarray(
        np.stack([rng.permutation(n) for _ in range(b)]).astype(np.float32)
    )
    row_lengths = jnp.asarray(rng.integers(k // 2, n + 1, size=b), jnp.int32)

    got = jax.jit(tokamax_ops.sparsecore_topk, static_argnums=1)(
        scores, k, row_lengths
    )
    want = sparsecore_topk(scores, k, row_lengths)
    np.testing.assert_array_equal(_sorted_rows(got), _sorted_rows(want))


def _packed_index_cache(rng, num_pages, page_size, d):
    """fp8 index keys with a UE8M0 scale byte, HEAD_ALONG_SUBLANE-packed."""
    fp8_max = float(jnp.finfo(jnp.float8_e4m3fn).max)
    width = ((d + d // 128 + 127) // 128) * 128
    cache = np.zeros((num_pages, page_size // 4, 4, width), np.uint8)
    for p in range(num_pages):
        x = jnp.asarray(rng.standard_normal((page_size, d), np.float32))
        amax = jnp.clip(jnp.max(jnp.abs(x), axis=-1, keepdims=True), 1e-4, None)
        scale = jnp.exp2(jnp.ceil(jnp.log2(amax / fp8_max)))
        quant = (x / scale).astype(jnp.float8_e4m3fn)
        scale = scale.astype(jnp.float8_e8m0fnu)
        record = np.concatenate(
            [
                np.asarray(jax.lax.bitcast_convert_type(quant, jnp.uint8)),
                np.asarray(jax.lax.bitcast_convert_type(scale, jnp.uint8)),
            ],
            axis=-1,
        )
        record = np.pad(record, ((0, 0), (0, width - record.shape[-1])))
        cache[p] = record.reshape(page_size // 4, 4, width)
    return jnp.asarray(cache)


@pytest.mark.tpu_test
def test_streamindex_topk_matches_in_tree(tpu, no_fallback):
    from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.streamindex_topk import (
        streamindex_topk,
    )

    rng = np.random.default_rng(0)
    page_size, h, d, k = 128, 4, 128, 256
    q_lens, kv_lens = (1, 1, 40, 88), (300, 512, 512, 512)
    pages_per_seq = 4
    num_tokens = sum(q_lens)
    kwargs = dict(
        q=jnp.asarray(rng.standard_normal((num_tokens, h, d), np.float32)).astype(
            jnp.float8_e4m3fn
        ),
        indexer_weights=jnp.asarray(rng.uniform(-1.5, 1.5, (num_tokens, h)), jnp.float32),
        cache_kv=_packed_index_cache(rng, len(q_lens) * pages_per_seq, page_size, d),
        seq_lens=jnp.asarray(kv_lens, jnp.int32),
        page_indices=jnp.asarray(rng.permutation(len(q_lens) * pages_per_seq), jnp.int32),
        cu_q_lens=jnp.asarray(np.concatenate([[0], np.cumsum(q_lens)]), jnp.int32),
        distribution=jnp.asarray([2, 2, 4], jnp.int32),
        k=k,
        compression_ratio=1,
        num_kv_pages_per_block=(2, 2, 2),
        num_queries_per_block=(1, 8, 8),
    )

    got = tokamax_ops.streamindex_topk(**kwargs)
    want = streamindex_topk(**kwargs)
    np.testing.assert_array_equal(_sorted_rows(got), _sorted_rows(want))
