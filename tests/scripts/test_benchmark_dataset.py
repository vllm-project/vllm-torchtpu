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
"""Unit tests for scripts/vllm/benchmarking/benchmark_dataset.py."""

import importlib
import pathlib
import sys
import types
from unittest import mock

import pytest
from PIL import Image

_BENCH_DIR = (
    pathlib.Path(__file__).resolve().parents[2] / "scripts" / "vllm" / "benchmarking"
)

with mock.patch.object(sys, "path", [str(_BENCH_DIR), *sys.path]):
    benchmark_dataset = importlib.import_module("benchmark_dataset")


@pytest.mark.parametrize(
    "subset,row",
    [
        ("vision", {"image": Image.new("RGB", (4, 4))}),
        (
            "standard (10 options)",
            {
                "question": "q",
                "image_1": Image.new("RGB", (4, 4)),
                "image_2": Image.new("RGB", (4, 4)),
            },
        ),
    ],
)
def test_mmmu_pro_requests_carry_every_row_image(monkeypatch, subset, row):
    row = {"options": "['a', 'b']", "answer": "B", **row}
    num_images = sum(key.startswith("image") for key in row)
    datasets = types.SimpleNamespace(load_dataset=lambda *args, **kwargs: [row])
    monkeypatch.setitem(sys.modules, "datasets", datasets)

    (request,) = benchmark_dataset.MMMUProDataset(subset=subset).sample(
        tokenizer=None, num_requests=1
    )

    content = request.messages[0]["content"]
    assert [part["type"] for part in content] == ["image_url"] * num_images + ["text"]
