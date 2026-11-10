# SPDX-License-Identifier: Apache-2.0
"""Per-shape tile tables for the Qwen4-exp serving kernels (VLLM_TPU_SHAPE_TUNED).

One tile config for every shape leaves time on the table: the best decode
attention tiles at 8 rows differ from those at 128 rows, and an 8K prefill
chunk wants a longer query block than a 1K chunk. Each table maps a kernel's
*static* shape bucket (known at trace time: decode rows, prefill chunk tokens)
to the tiles measured fastest on TPU v7x at the served AgentX geometry (TP8
rank Q3/KV1, 512-token pages, MTP K=3, main-KV DCP-2) by
``tests/shapetune`` sweeps over 60K-200K contexts. Context length is dynamic,
so it never keys a table; the sweeps check that a bucket's winner holds
across contexts.

With VLLM_TPU_SHAPE_TUNED unset every lookup returns ``None`` and kernels use
their env-var tiles unchanged. Rows are (upper bound inclusive, value); the
first row whose bound covers the shape wins; past the last row -> ``None``.
"""

from vllm_torchtpu import envs


def _lookup(table, size):
    if not envs.VLLM_TPU_SHAPE_TUNED:
        return None
    for bound, value in table:
        if size <= bound:
            return value
    return None


# QSA stock decode attention (qsa_rpa.rpa_attention, compact schedule):
# decode rows per call -> (BKV tokens, lanes, max queries per RPA call).
# Layout-only for lanes / max queries; BKV changes the online-softmax block
# split (see shapetune.md for the measured bitwise status).
RPA_DECODE_TILES: tuple = ()


def rpa_decode_tiles(rows: int):
    return _lookup(RPA_DECODE_TILES, rows)


# QSA prefill attention, DCP block-sparse path (qsa_rpa._prefill_attention):
# chunk tokens -> (query_block, query_tile). Bit-identical across query blocks.
DCP_PREFILL_TILES: tuple = ()


def dcp_prefill_tiles(tokens: int):
    return _lookup(DCP_PREFILL_TILES, tokens)


# QSA indexer, decode/verify StreamIndex (pooled_qsa.select_blocks shared
# stream): query rows -> (VLLM_TPU_QSA_DECODE_KV_BLOCK, _QUERY_BLOCK).
INDEXER_DECODE_TILES: tuple = ()


def indexer_decode_tiles(rows: int):
    return _lookup(INDEXER_DECODE_TILES, rows)


# QSA indexer, ragged prefill StreamIndex: per-rank query rows (chunk / SP) ->
# (PREFILL_INDEXER_BKV keys, BQ, chunk tokens; chunk == rows means one call).
INDEXER_PREFILL_TILES: tuple = ()


def indexer_prefill_tiles(rows: int):
    return _lookup(INDEXER_PREFILL_TILES, rows)
