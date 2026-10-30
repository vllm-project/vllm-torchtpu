# SPDX-License-Identifier: Apache-2.0
"""Leave padding buckets out (TPU_TOKEN_BUCKET_DROP / TPU_REQ_BUCKET_DROP)."""

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


def drop_buckets(buckets: list[int], drop: list[int], kind: str) -> list[int]:
    """``buckets`` without ``drop``, always keeping the largest so every size
    still pads up to a kept bucket."""
    if not drop:
        return buckets
    largest = max(buckets)
    kept = [b for b in buckets if b not in drop or b == largest]
    logger.info(
        "Dropped %s buckets %s (kept %s)", kind, sorted(set(buckets) - set(kept)), kept
    )
    return kept
