# SPDX-License-Identifier: Apache-2.0
"""tokamax's operators for kernels vllm-torchtpu also ships in-tree.

Each function takes the in-tree kernel's arguments and runs the tokamax
operator that was ported from it, falling back to the in-tree kernel -- with
one warning per reason -- when tokamax is not importable, when
`VLLM_TPU_USE_TOKAMAX=0`, or when the operator or these arguments are outside
what tokamax covers. They are called inside a JAX trace, so the choice is made
once at trace time and costs nothing per step.

Batched RPA is routed separately, at the torch level, by
`PallasAttentionBackendImpl._build_rpa_kernel`.
"""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


def _run_tokamax(name: str, unsupported: str | None, run: Callable[[], Any]) -> Any:
    """Returns `run()`, tokamax's result, or None if the in-tree kernel must run."""
    if not envs.VLLM_TPU_USE_TOKAMAX:
        unsupported = "VLLM_TPU_USE_TOKAMAX=0"
    if unsupported is None:
        try:
            return run()
        except (ImportError, NotImplementedError) as e:
            unsupported = f"{type(e).__name__}: {e}"
    logger.warning_once(
        "%s: falling back to the in-tree kernel because of %s.", name, unsupported
    )
    return None


def csa_gather(
    nope_cache: jax.Array,
    rope_cache: jax.Array,
    indices: jax.Array,
    num_valid_indices: jax.Array | None = None,
    *,
    top_k: int = 1024,
) -> tuple[jax.Array, jax.Array]:
    """DeepSeek-V4 CSA gather; see `core_attention.csa_gather.csa_gather`."""

    def run():
        from tokamax._src.ops.experimental.tpu.csa_gather import api

        return api.csa_gather(
            nope_cache,
            rope_cache,
            indices,
            num_valid_indices,
            top_k=top_k,
            implementation="mosaic_tpu",
        )

    out = _run_tokamax("csa_gather", None, run)
    if out is not None:
        return out
    from vllm_torchtpu.kernels.deepseek_v4.core_attention.csa_gather import (
        csa_gather as in_tree_csa_gather,
    )

    return in_tree_csa_gather(
        nope_cache, rope_cache, indices, num_valid_indices, top_k=top_k
    )


def streamindex_topk(
    q: jax.Array,
    indexer_weights: jax.Array,
    cache_kv: jax.Array,
    seq_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    k: int,
    compression_ratio: int,
    cp_size: int = 1,
    cp_rank: jax.Array | int = 0,
    interleave_size: int = 1,
    return_scores: bool = False,
    kv_layout: Any = None,
    **tuning: Any,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Lightning indexer score + top-k; see `streamindex_topk.streamindex_topk`.

    `tuning` takes the in-tree kernel's block-size and buffering keywords
    (`num_kv_pages_per_block`, `num_queries_per_block`, `buffer_count`,
    `vmem_limit_bytes`, `decode_req_batch_size`, `enable_early_exit`,
    `chunk_tokens`), which are exactly the fields of tokamax's `Config`.
    """
    from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import config as in_tree_config

    layout = in_tree_config.KVLayout.parse(kv_layout)
    kwargs = dict(
        k=k,
        compression_ratio=compression_ratio,
        cp_size=cp_size,
        cp_rank=cp_rank,
        interleave_size=interleave_size,
        return_scores=return_scores,
    )

    def run():
        from tokamax._src.ops.experimental.lightning_indexer import pallas_mosaic_tpu

        # Only what the caller set: None means "the kernel's default" in the
        # in-tree signature, and tokamax's Config spells those defaults out.
        config = pallas_mosaic_tpu.Config(
            **{key: value for key, value in tuning.items() if value is not None}
        )
        op = pallas_mosaic_tpu.PallasTpuLightningIndexer(config=config)
        return op(
            q,
            indexer_weights,
            cache_kv,
            seq_lens,
            page_indices,
            cu_q_lens,
            # The in-tree kernel also takes a Python (decode, prefill, mixed)
            # tuple; tokamax's signature wants an array.
            jnp.asarray(distribution, jnp.int32),
            kv_layout=pallas_mosaic_tpu.KVLayout[layout.name],
            **kwargs,
        )

    # DCP and PCP keep the in-tree kernel, which owns the cross-rank merge.
    unsupported = None if cp_size == 1 else f"cp_size={cp_size}"
    out = _run_tokamax("streamindex_topk", unsupported, run)
    if out is not None:
        return out
    from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.streamindex_topk import (
        streamindex_topk as in_tree_streamindex_topk,
    )

    return in_tree_streamindex_topk(
        q,
        indexer_weights,
        cache_kv,
        seq_lens,
        page_indices,
        cu_q_lens,
        distribution,
        kv_layout=layout,
        **kwargs,
        **tuning,
    )


def sparsecore_topk(
    scores: jax.Array,
    k: int,
    row_lengths: jax.Array | None = None,
    *,
    write_empty_rows: bool = True,
    scheduling_group_id: int | None = None,
    return_scores: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Exact SparseCore top-k; see `sparsecore_topk.sparsecore_topk`."""

    def run():
        from tokamax._src.ops.experimental.tpu.topk import pallas_mosaic_tpu

        op = pallas_mosaic_tpu.PallasTpuTopK(
            config=pallas_mosaic_tpu.Config(scheduling_group_id=scheduling_group_id)
        )
        return op(scores, k, row_lengths, return_scores=return_scores)

    # tokamax always writes the rows past a row's length, the only behaviour
    # the in-tree callers ask for.
    unsupported = None if write_empty_rows else "write_empty_rows=False"
    out = _run_tokamax("sparsecore_topk", unsupported, run)
    if out is not None:
        return out
    from vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk import (
        sparsecore_topk as in_tree_sparsecore_topk,
    )

    return in_tree_sparsecore_topk(
        scores,
        k,
        row_lengths,
        write_empty_rows=write_empty_rows,
        scheduling_group_id=scheduling_group_id,
        return_scores=return_scores,
    )
