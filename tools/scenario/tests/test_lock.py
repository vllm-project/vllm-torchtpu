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
"""uv.lock must resolve on any machine, so it may name only the public index."""

import pathlib
import re

LOCK = pathlib.Path(__file__).resolve().parents[1] / "uv.lock"
PUBLIC = ("https://pypi.org/simple", "https://files.pythonhosted.org/")


def test_lock_names_only_the_public_index() -> None:
    text = LOCK.read_text()
    registries = set(re.findall(r'registry = "([^"]+)"', text))
    assert registries == {PUBLIC[0]}
    urls = re.findall(r'url = "([^"]+)"', text)
    assert urls
    assert [url for url in urls if not url.startswith(PUBLIC)] == []
