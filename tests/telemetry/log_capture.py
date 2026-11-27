# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Log capture for the telemetry tests."""

from __future__ import annotations

import contextlib
import logging


@contextlib.contextmanager
def captured_records(logger: logging.Logger):
    """Collect INFO records emitted directly on ``logger``.

    vLLM's logging configuration sets ``propagate=False`` on the ``vllm``
    logger, so records from its children never reach the root logger where
    pytest's ``caplog`` listens, making ``caplog`` depend on test import order.
    """
    records: list[logging.LogRecord] = []
    handler = logging.Handler(logging.INFO)
    handler.emit = records.append
    old_level = logger.level
    old_disable = logging.root.manager.disable
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logging.disable(logging.NOTSET)
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)
        logging.disable(old_disable)
