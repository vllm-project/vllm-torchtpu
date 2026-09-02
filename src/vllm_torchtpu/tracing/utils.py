# Copyright 2026 Google LLC
"""Helper utilities for request ID and KV cache profiler tracing."""
import logging
from typing import Any

logger = logging.getLogger(__name__)


def trim_request_id_suffix(request_id: str) -> str:
    parts = request_id.split("-")
    if len(parts) >= 6 and parts[0] == "cmpl":
        request_id = "-".join(parts[:6])
    return request_id


def extract_request_ids_for_tracing(req_ids: list[Any],
                                    start_index: int = 0,
                                    num_reqs: int = -1) -> dict[str, str]:
    req_id_kwargs = {}
    try:
        active_ids = req_ids[start_index:start_index +
                             num_reqs] if num_reqs != -1 else req_ids
        trimmed_req_ids = [
            trim_request_id_suffix(str(rid)) for rid in active_ids
            if rid is not None
        ]
        for i, rid in enumerate(trimmed_req_ids):
            req_id_kwargs[f"request_id{i+1}"] = rid
    except Exception as e:
        logger.warning(f"Failed to extract request IDs for tracing: {e}")

    return req_id_kwargs


def extract_kv_lens_for_tracing(num_computed_tokens: Any,
                                start_index: int = 0,
                                num_reqs: int = -1) -> dict[str, int]:
    kv_len_kwargs = {}
    try:
        active_lens = (num_computed_tokens[start_index:start_index + num_reqs]
                       if num_reqs != -1 else num_computed_tokens)
        active_lens = [int(kv_len) for kv_len in active_lens]
        for i, kv_len in enumerate(active_lens):
            kv_len_kwargs[f"kv_len{i+1}"] = kv_len
        if active_lens:
            kv_len_kwargs["min_kv_len"] = min(active_lens)
            kv_len_kwargs["max_kv_len"] = max(active_lens)
            kv_len_kwargs["avg_kv_len"] = sum(active_lens) // len(active_lens)
    except Exception as e:
        logger.warning(f"Failed to extract KV lens for tracing: {e}")

    return kv_len_kwargs
