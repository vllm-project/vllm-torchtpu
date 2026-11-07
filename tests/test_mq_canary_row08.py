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
"""Merge-queue canary row 8: a TPU test that fails in the merge queue.

The PR's own build runs CPU checks only, so this fails only in the queue.
"""

import pytest

pytestmark = pytest.mark.tpu_test


def test_mq_canary_row08_fails_on_tpu() -> None:
    pytest.fail("row 8: the merge queue must drop this PR")
