# SPDX-License-Identifier: Apache-2.0
"""Token-sharded prefill indexer (VLLM_TPU_QSA_INDEXER_TOKEN_SHARD).

Under prefill sequence parallelism every rank indexes the whole gathered
step. With the token shard each rank indexes one contiguous 1/size of the
rows (``token_shard_step``); the concatenated shards must select exactly the
replicated indexer's blocks. Real StreamIndex kernel in TPU interpret mode on
CPU, a JAX top-k in place of the SparseCore selector.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import test_qwen4_exp_mtp_draft_rows as mtp_rows
from test_qwen4_exp_indexer_shard import (  # noqa: F401 (fixtures)
    assert_same_selection,
    reference_kernel,
    select,
)
from test_qwen4_exp_mtp_draft_rows import rowwise_attention  # noqa: F401
from test_qwen4_exp_shared_stream_indexer import (  # noqa: F401 (fixtures)
    inputs,
    interpret,
)

from vllm_torchtpu.kernels.qwen4_exp import pooled_qsa


def token_sharded(args, size):
    tokens = len(args["query"])
    rows = tokens // size
    shards = []
    for s in range(size):
        base, starts, context = pooled_qsa.token_shard_step(
            args["starts"], args["context"], jnp.asarray([s], jnp.int32), rows
        )
        assert int(base) == s * rows
        part = dict(args, starts=starts, context=context)
        for name in ("query", "lengths", "req"):
            part[name] = args[name][s * rows : (s + 1) * rows]
        shards.append(select(part))
    return np.concatenate(shards)


@pytest.mark.parametrize(
    "q_lens,context,tokens,size",
    [
        # Requests crossing shard borders, an idle row, a partial group tail
        # and a request shorter than the budget; the bucket's tail is empty.
        ([40, 0, 20, 4], [700, 0, 1500, 9], 64, 8),
        # Leading one-token decodes (the decode prefix lands in shard 0) and
        # a shard boundary leaving a single row of a request in a shard.
        ([1, 1, 25, 37], [900, 33, 1203, 0], 64, 8),
        # A full bucket, every shard live, one long request.
        ([128], [2999], 128, 8),
        ([60, 68], [1500, 260], 128, 4),
    ],
)
def test_token_shards_match_replicated(
    reference_kernel,  # noqa: F811
    q_lens,
    context,
    tokens,
    size,
):
    args = inputs(5, q_lens, context, tokens, "integer")
    expected = select(args)
    actual = token_sharded(args, size)
    assert_same_selection(actual, expected, args["lengths"])


def test_shard_step_layout():
    starts = jnp.asarray([0, 1, 2, 27, 64], jnp.int32)
    context = jnp.asarray([900, 33, 1203, 0], jnp.int32)
    base, sub, ctx = pooled_qsa.token_shard_step(
        starts, context, jnp.asarray([3], jnp.int32), 8
    )
    assert int(base) == 24
    np.testing.assert_array_equal(np.asarray(sub), [0, 0, 0, 3, 8])
    np.testing.assert_array_equal(np.asarray(ctx), [901, 34, 1225, 0])


@pytest.mark.parametrize("size", [2, 4, 8])
def test_token_shard_steps_compose_to_the_pooled_step(
    rowwise_attention,  # noqa: F811
    size,
):
    """select on every token shard -> gather -> attend equals the replicated
    step. The stand-in indexer shifts its blocks when the shard's ragged
    batch (starts, context) disagrees with the rows' lengths and requests."""
    head, pool, tail = mtp_rows._inputs(False, mtp_rows.STARTS, mtp_rows.CONTEXT)
    options = dict(mtp_rows.OPTIONS, token_topk=256)
    want_pool, want_out, want_selected = jax.jit(
        lambda *args: pooled_qsa.pooled_qsa_step(*args, **options)
    )(*head, pool, *tail)

    results = [
        jax.jit(
            lambda *args: pooled_qsa.pooled_qsa_token_select_step(
                *args, shard_size=size, **options
            )
        )(*head, pool, *tail, jnp.asarray([s], jnp.int32))
        for s in range(size)
    ]
    new_pool, q, gate, lengths, req, starts, _ = results[0]
    np.testing.assert_array_equal(
        np.asarray(new_pool).view(np.uint8), np.asarray(want_pool).view(np.uint8)
    )
    gathered = jnp.concatenate([result[-1] for result in results])
    _, out, selected = jax.jit(
        lambda *args: pooled_qsa.pooled_qsa_shard_attend_step(
            *args, kv_heads=2, compress_ratio=4, token_topk=256, merged=True
        )
    )(q, gate, lengths, req, starts, tail[3], tail[1], new_pool, gathered)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(want_out))
    np.testing.assert_array_equal(np.asarray(selected), np.asarray(want_selected))


@pytest.mark.parametrize(
    "min_tokens,max_num_seqs,active",
    [(1024, 32, True), (1024, 128, False), (4096, 256, True)],
)
def test_mode_is_decided_once_for_the_smallest_sp_bucket(
    monkeypatch, min_tokens, max_num_seqs, active
):
    from vllm_torchtpu import envs
    from vllm_torchtpu.models.vllm.qwen4_exp.qsa import Qwen4ExpQSA

    for name, value in (
        ("VLLM_TPU_QSA_INDEXER_TOKEN_SHARD", True),
        ("QWEN4_EXP_PREFILL_SP", True),
        ("QWEN4_EXP_PREFILL_SP_MIN_TOKENS", min_tokens),
    ):
        monkeypatch.setattr(envs, name, value, raising=False)
    assert Qwen4ExpQSA._token_shard_mode(8, True, max_num_seqs) is active
    assert Qwen4ExpQSA._token_shard_mode(1, True, max_num_seqs) is False
    assert Qwen4ExpQSA._token_shard_mode(8, False, max_num_seqs) is False
    monkeypatch.setattr(envs, "QWEN4_EXP_PREFILL_SP", False, raising=False)
    assert Qwen4ExpQSA._token_shard_mode(8, True, max_num_seqs) is False


@pytest.mark.parametrize(
    "tokens,max_num_seqs,compressed_dcp,expected",
    [
        (256, 16, False, True),
        (512, 24, False, True),
        (3072, 32, False, True),
        (256, 32, False, False),  # 32 rows per rank, not more than 32 slots
        (128, 8, False, False),  # below the 256-row minimum
        (264, 16, False, True),
        (260, 16, False, False),  # does not divide by TP
        (512, 16, True, False),  # compressed DCP: DECODE_TOKEN_SHARD's job
    ],
)
def test_replicated_token_shard_eligibility(
    monkeypatch, tokens, max_num_seqs, compressed_dcp, expected
):
    from vllm_torchtpu import envs
    from vllm_torchtpu.models.vllm.qwen4_exp import sequence_parallel as sp

    monkeypatch.delenv("LIBTPU_INIT_ARGS", raising=False)
    monkeypatch.setattr(
        envs, "VLLM_TPU_QSA_PREFILL_REPLICATED_TOKEN_SHARD", True, raising=False
    )
    monkeypatch.setattr(
        envs, "VLLM_TPU_QSA_COMPRESSED_DCP", compressed_dcp, raising=False
    )
    assert sp.replicated_token_shard_eligible(tokens, 8, max_num_seqs) is expected
    assert sp.replicated_token_shard_eligible(tokens, 1, max_num_seqs) is False
    monkeypatch.setattr(
        envs, "VLLM_TPU_QSA_PREFILL_REPLICATED_TOKEN_SHARD", False, raising=False
    )
    assert sp.replicated_token_shard_eligible(tokens, 8, max_num_seqs) is False


_SC_AG = "--xla_tpu_enable_sparse_core_collective_offload_all_gather"


@pytest.mark.parametrize(
    "libtpu_args,expected",
    [
        ("", True),
        (f"{_SC_AG}=false", True),
        (f"{_SC_AG}=true", False),
        # The last setting wins, as in libtpu (the recipe appends overrides).
        (f"{_SC_AG}=true {_SC_AG}=false", True),
        (f"{_SC_AG}=false {_SC_AG}=true", False),
        # The c32 recipe's offload flags around it.
        (
            "--xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=8388608 "
            f"{_SC_AG}=true "
            "--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true",
            False,
        ),
    ],
)
def test_replicated_token_shard_off_under_sparse_core_all_gather_offload(
    monkeypatch, libtpu_args, expected
):
    """The selection all-gather of Qwen4ExpBackboneRP halts a SparseCore with
    the all-gather offload on (c32): the router and the QSA layers both drop
    the graph there."""
    from vllm_torchtpu import envs
    from vllm_torchtpu.models.vllm.qwen4_exp import sequence_parallel as sp
    from vllm_torchtpu.models.vllm.qwen4_exp.qsa import Qwen4ExpQSA

    monkeypatch.setenv("LIBTPU_INIT_ARGS", libtpu_args)
    monkeypatch.setattr(
        envs, "VLLM_TPU_QSA_PREFILL_REPLICATED_TOKEN_SHARD", True, raising=False
    )
    monkeypatch.setattr(envs, "VLLM_TPU_QSA_COMPRESSED_DCP", False, raising=False)
    assert sp.replicated_token_shard_enabled() is expected
    assert sp.replicated_token_shard_eligible(512, 8, 24) is expected
    assert Qwen4ExpQSA._replicated_token_shard_mode(8, True, "model.layers.3") is expected
    assert Qwen4ExpQSA._replicated_token_shard_mode(8, True, "mtp.layers.0") is False
