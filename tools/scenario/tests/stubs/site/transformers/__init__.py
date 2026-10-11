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
"""transformers' Auto classes, as far as the remote-code warm uses them.

STUB_WARM_FAIL=1 makes the tokenizer's import fail, as a half-written
module would.
"""

import os

from stub_record import record


class AutoConfig:
    @staticmethod
    def from_pretrained(path: str, trust_remote_code: bool = False) -> None:
        record("transformers AutoConfig", [path], trust_remote_code=trust_remote_code)


class AutoTokenizer:
    @staticmethod
    def from_pretrained(path: str, trust_remote_code: bool = False) -> None:
        record(
            "transformers AutoTokenizer", [path], trust_remote_code=trust_remote_code
        )
        if os.environ.get("STUB_WARM_FAIL") == "1":
            raise AttributeError(
                "module 'tokenization_kimi' has no attribute 'TikTokenTokenizer'"
            )
