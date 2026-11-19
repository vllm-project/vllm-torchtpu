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
"""Unit tests for .buildkite/scripts/tpu_generations.py."""

from __future__ import annotations

import importlib
import pathlib
import re
import sys
import unittest
from typing import Any
from unittest import mock

_SCRIPTS_DIR = pathlib.Path(__file__).resolve().parents[2] / ".buildkite" / "scripts"


class TestTpuGenerations(unittest.TestCase):
    gens: Any

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        if "tpu_generations" not in sys.modules:
            cls.addClassCleanup(sys.modules.pop, "tpu_generations", None)
        with mock.patch.object(sys, "path", [str(_SCRIPTS_DIR), *sys.path]):
            cls.gens = importlib.import_module("tpu_generations")

    def test_stamp_prefixes_keys_labels_and_own_dependencies(self) -> None:
        gens = self.gens
        v6e = gens.TPU_V6E

        def stamped(steps: list[Any]) -> list[Any]:
            gens.stamp_tpu_generation(steps, v6e)
            return steps

        # Every key the pipeline defines takes the key prefix, a group's and
        # its steps' alike, and every label and group name the label prefix. A
        # depends_on naming one of those keys follows it, in each form
        # Buildkite reads; one naming another upload's step (the image build)
        # does not. A step that is not an object (`wait`) is passed over.
        self.assertEqual(
            stamped(
                [
                    "wait",
                    {
                        "group": "Unit Tests on TPU",
                        "key": "unit",
                        "steps": [
                            {
                                "label": "General",
                                "key": "general",
                                "depends_on": "build_docker",
                            },
                            {"label": "No key", "depends_on": ["general", "unit"]},
                        ],
                    },
                    {
                        "label": "Report",
                        "key": "report",
                        "depends_on": [
                            "build_docker",
                            {"step": "general", "allow_failure": True},
                            {"step": "build_docker_omni"},
                        ],
                    },
                    {"label": "Alone", "depends_on": "report"},
                ]
            ),
            [
                "wait",
                {
                    "group": "v6e Unit Tests on TPU",
                    "key": "v6e_unit",
                    "steps": [
                        {
                            "label": "v6e General",
                            "key": "v6e_general",
                            "depends_on": "build_docker",
                        },
                        {
                            "label": "v6e No key",
                            "depends_on": ["v6e_general", "v6e_unit"],
                        },
                    ],
                },
                {
                    "label": "v6e Report",
                    "key": "v6e_report",
                    "depends_on": [
                        "build_docker",
                        {"step": "v6e_general", "allow_failure": True},
                        {"step": "build_docker_omni"},
                    ],
                },
                {"label": "v6e Alone", "depends_on": "v6e_report"},
            ],
        )

        # A step in a form not read here is refused, not passed over: it would
        # go up under another generation's key or label.
        for step in (
            {"label": "x", "id": "general"},
            {"label": "x", "identifier": "general"},
            {"name": "x", "key": "general"},
            {"label": "x", "key": 7},
            {"label": ["x"]},
            {"label": "x", "depends_on": [7]},
            {"label": "x", "depends_on": {"allow_failure": True}},
        ):
            with self.subTest(step=step), self.assertRaises(ValueError):
                gens.stamp_tpu_generation([step], v6e)

    def test_shapes_agree_with_the_nightly_model_uploads(self) -> None:
        # upload_models_and_features.sh keeps its own copy of each generation's
        # shapes (see the TODO in tpu_generations.py). Until it reads them from
        # there, the two must agree.
        script = (_SCRIPTS_DIR / "upload_models_and_features.sh").read_text(
            encoding="utf-8"
        )
        single, multi = (
            re.findall(rf'^\s*export MODEL_SHAPE_{kind}="(.+)"$', script, re.M)
            for kind in ("SINGLE", "MULTI")
        )
        generations = {
            gen for listed in self.gens.TPU_GENERATION_FILES.values() for gen in listed
        }
        self.assertEqual(
            set(zip(single, multi, strict=True)),
            {(gen.shape_single, gen.shape_multi) for gen in generations},
        )


if __name__ == "__main__":
    unittest.main()
