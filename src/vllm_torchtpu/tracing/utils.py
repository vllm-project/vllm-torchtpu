# Copyright 2026 Google LLC
"""Helper utilities for request ID and KV cache profiler tracing."""

from typing import Any


def trim_request_id_suffix(request_id: str) -> str:
    parts = request_id.split("-")
    if len(parts) >= 6 and parts[0] == "cmpl":
        request_id = "-".join(parts[:6])
    return request_id


def _summarize_lens(name: str, lens: list[int]) -> dict[str, int]:
    return {
        f"min_{name}": min(lens),
        f"max_{name}": max(lens),
        f"avg_{name}": sum(lens) // len(lens),
    }


def extract_kv_lens_for_tracing(
    num_computed_tokens: Any, start_index: int = 0, num_reqs: int = -1
) -> dict[str, int]:
    """Min/max/avg KV context length (tokens cached before this step)."""
    end_index = start_index + num_reqs if num_reqs != -1 else None
    active_lens = [int(kv_len) for kv_len in num_computed_tokens[start_index:end_index]]
    return _summarize_lens("kv_len", active_lens) if active_lens else {}


def extract_input_lens_for_tracing(
    num_prompt_tokens: Any, start_index: int = 0, num_reqs: int = -1
) -> dict[str, int]:
    """Min/max/avg input sequence length (prompt tokens)."""
    end_index = start_index + num_reqs if num_reqs != -1 else None
    active_lens = [int(isl) for isl in num_prompt_tokens[start_index:end_index]]
    return _summarize_lens("isl", active_lens) if active_lens else {}


# One entry per request in the `reqs` arg, space separated.
REQUEST_DETAILS_FORMAT = "req_id/isl/kv_len/new_tokens"
# At most this many entries per event (~15 KB); the summary covers the rest.
MAX_REQUEST_DETAILS = 256
# The trace encodes args as `name#key=value,...#`, so these cannot appear in a
# value; the space and slash would make an entry ambiguous.
_UNSAFE_REQ_ID_CHARS = str.maketrans({c: "_" for c in ",#= /"})


def extract_request_details_for_tracing(
    req_ids: list[Any],
    num_prompt_tokens: Any,
    num_computed_tokens: Any,
    num_scheduled_tokens: dict[str, int],
    start_index: int = 0,
    num_reqs: int = -1,
    max_entries: int = MAX_REQUEST_DETAILS,
) -> dict[str, Any]:
    """Each request's id, input length, KV length and new tokens, as one arg.

    One arg instead of one per request keeps the event readable. Past
    `max_entries` requests the list stops, `reqs` ends with `+N more` and
    `reqs_omitted` says how many were left out.
    """
    end_index = start_index + num_reqs if num_reqs != -1 else None
    entries = []
    num_omitted = 0
    for req_id, isl, kv_len in zip(
        req_ids[start_index:end_index],
        num_prompt_tokens[start_index:end_index],
        num_computed_tokens[start_index:end_index],
    ):
        if req_id is None:
            continue
        if len(entries) == max_entries:
            num_omitted += 1
            continue
        name = trim_request_id_suffix(str(req_id)).translate(_UNSAFE_REQ_ID_CHARS)
        new_tokens = num_scheduled_tokens.get(req_id, 0)
        entries.append(f"{name}/{int(isl)}/{int(kv_len)}/{new_tokens}")
    if not entries:
        return {}
    details: dict[str, Any] = {"reqs_format": REQUEST_DETAILS_FORMAT}
    if num_omitted:
        entries.append(f"+{num_omitted} more")
        details["reqs_omitted"] = num_omitted
    details["reqs"] = " ".join(entries)
    return details
