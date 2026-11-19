# SPDX-License-Identifier: Apache-2.0
"""Worker-side identity log for A/B microbenches (q38mm DM lane): one line per newly added request with a hash of
the prompt token ids the worker placed in ``input_batch.token_ids_cpu``. Default off; ``Q38MM_DM_TOKHASH=1`` turns
it on. It reads only host buffers and never changes them, so it is harness-only for the compile key.

Line: ``Q38MM_TOKHASH rid=<request id> n=<prompt tokens> sha1=<first 16 hex of sha1(int32 bytes)>``
"""

from __future__ import annotations

import hashlib
import os

ENV = "Q38MM_DM_TOKHASH"
_ON = os.environ.get(ENV, "0") == "1"

try:
    from vllm.logger import init_logger

    logger = init_logger("vllm." + __name__)
except Exception:  # noqa: BLE001
    import logging

    logger = logging.getLogger(__name__)


def row_hash(token_ids_cpu, index: int, n: int) -> str:
    return hashlib.sha1(token_ids_cpu[index, :n].astype("int32", copy=False).tobytes()).hexdigest()[:16]


def log_new_requests(input_batch, scheduler_output) -> int:
    if not _ON:
        return 0
    k = 0
    for nrd in scheduler_output.scheduled_new_reqs:
        idx = input_batch.req_id_to_index.get(nrd.req_id)
        if idx is None:
            continue
        n = int(input_batch.num_prompt_tokens[idx])
        logger.info("Q38MM_TOKHASH rid=%s n=%d sha1=%s", nrd.req_id, n, row_hash(input_batch.token_ids_cpu, idx, n))
        k += 1
    return k
