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
