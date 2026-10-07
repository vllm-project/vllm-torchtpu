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
"""The accelerator table."""

import pytest
from scenario.accel import SliceError, known_slices, parse_slice


@pytest.mark.parametrize(
    ("text", "hosts", "chips_per_host", "chips", "devices"),
    [
        ("tpu7x/1x1x1", 1, 1, 1, 2),
        ("tpu7x/2x2x1", 1, 4, 4, 8),
        ("tpu7x/2x2x2", 2, 4, 8, 16),
        ("tpu7x/2x2x4", 4, 4, 16, 32),
        ("tpu7x/2x4x4", 8, 4, 32, 64),
        ("tpu7x/4x4x4", 16, 4, 64, 128),
    ],
)
def test_counts_are_chips(
    text: str, hosts: int, chips_per_host: int, chips: int, devices: int
) -> None:
    shape = parse_slice(text)
    assert (shape.hosts, shape.chips_per_host) == (hosts, chips_per_host)
    assert shape.chips == chips
    assert shape.devices == devices
    assert shape.devices_per_host == chips_per_host * 2
    assert str(shape) == text


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("tpu7x/3x3x3", "tpu7x has no topology '3x3x3'; known: 1x1x1, 2x2x1"),
        ("tpu8/2x2x2", "unknown accelerator 'tpu8'; known: tpu7x"),
        ("tpu7x-8", "a slice is <accelerator>/<topology>"),
        ("tpu7x/", "a slice is <accelerator>/<topology>"),
        ("/2x2x2", "a slice is <accelerator>/<topology>"),
    ],
)
def test_rejects(text: str, message: str) -> None:
    with pytest.raises(SliceError, match="^" + message.replace("(", r"\(")):
        parse_slice(text)


def test_known_slices_parse() -> None:
    assert "tpu7x/2x2x2" in known_slices()
    for text in known_slices():
        assert str(parse_slice(text)) == text
