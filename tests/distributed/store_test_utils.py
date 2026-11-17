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
"""An in-memory stand-in for the start-up rendezvous store."""

import threading

TIMEOUT = 20


class FakeStore:
    """``set``, ``get`` and ``wait`` of a ``torch.distributed.Store``, shared
    by threads. ``wait`` gives up after TIMEOUT seconds, whatever timeout it
    is passed, so a key nobody sets fails the test instead of hanging it."""

    def __init__(self):
        self.cv = threading.Condition()
        self.values: dict[str, bytes] = {}

    def set(self, key: str, value: str) -> None:
        # torch's Store declares text values.
        if not isinstance(value, str):
            raise TypeError(f"a store value is text, got {type(value).__name__}")
        with self.cv:
            self.values[key] = value.encode()
            self.cv.notify_all()

    def get(self, key: str) -> bytes:
        with self.cv:
            return self.values[key]

    def wait(self, keys: list[str], timeout) -> None:
        with self.cv:
            if not self.cv.wait_for(
                lambda: all(key in self.values for key in keys), timeout=TIMEOUT
            ):
                raise TimeoutError(f"{keys} were never all set")
