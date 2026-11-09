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
"""Merge-queue canary row 7: passes on the PR head, fails in the merge group.

After this PR's own build, mq-canary gains tests/mq_canary_row07.flag, so the
merge-group build (new base + this PR) fails its CPU tests and TPU never runs.
"""

import pathlib

_FLAG = pathlib.Path(__file__).with_name("mq_canary_row07.flag")


def test_mq_canary_row07_flag_absent() -> None:
    assert not _FLAG.exists(), "row 7 flag is present: the merge group must fail"
