# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Logger for llm_tpu_core with zero external dependencies."""

from __future__ import annotations

import logging


def init_logger(name: str) -> logging.Logger:
    """Return a standard logger for the given name under llm_tpu_core."""
    logger = logging.getLogger(f"llm_tpu_core.{name}")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
