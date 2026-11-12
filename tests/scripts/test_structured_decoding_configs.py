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
"""Unit tests for the qwen3-0.6b-structured-* benchmark configs.

Sources each config the way run_benchmarks.sh does and checks the request
body and bench flags it produces. Runs without a TPU.
"""

import json
import pathlib
import subprocess

import pytest

pytestmark = pytest.mark.cpu_test

_BENCH_DIR = (
    pathlib.Path(__file__).resolve().parents[2] / "scripts" / "vllm" / "benchmarking"
)


class TestConfigs:
    @pytest.mark.parametrize(
        "config,schema_key",
        [
            ("qwen3-0.6b-structured-flat-json", "properties"),
            ("qwen3-0.6b-structured-complex-ast", "$defs"),
            ("qwen3-0.6b-structured-character-dataset", "properties"),
        ],
    )
    def test_config_sends_a_structured_outputs_grammar(self, config, schema_key):
        """Source the config the way run_benchmarks.sh does and check the
        request body it produces is a real structured_outputs grammar."""
        body, _, _ = self._source(config)
        grammar = json.loads(body)["structured_outputs"]["json"]
        assert grammar["type"] == "object"
        assert schema_key in grammar

    @pytest.mark.parametrize(
        "config",
        [
            "qwen3-0.6b-structured-flat-json",
            "qwen3-0.6b-structured-complex-ast",
            "qwen3-0.6b-structured-character-dataset",
        ],
    )
    def test_config_saves_generated_text_and_keeps_caller_args(self, config):
        _, bench_args, _ = self._source(config, extra_bench_args="--trust-remote-code")
        assert bench_args.split() == ["--trust-remote-code", "--save-detailed"]

    @pytest.mark.parametrize(
        "config",
        [
            "qwen3-0.6b-structured-flat-json",
            "qwen3-0.6b-structured-complex-ast",
            "qwen3-0.6b-structured-character-dataset",
        ],
    )
    def test_config_disables_free_whitespace_and_keeps_caller_args(self, config):
        """xgrammar reads disable_any_whitespace from the server config only.
        run_benchmarks.sh word-splits EXTRA_SERVE_ARGS, so the JSON value
        must stay a single word."""
        _, _, serve_args = self._source(config, extra_serve_args="--seed 0")
        words = serve_args.split()
        assert words[:2] == ["--seed", "0"]
        flag = words.index("--structured-outputs-config")
        assert json.loads(words[flag + 1]) == {
            "backend": "xgrammar",
            "disable_any_whitespace": True,
        }

    @pytest.mark.parametrize(
        "config,array_key",
        [
            ("qwen3-0.6b-structured-flat-json", "users"),
            ("qwen3-0.6b-structured-complex-ast", "expressions"),
        ],
    )
    def test_grammar_cannot_close_before_the_output_length(self, config, array_key):
        """With --ignore-eos, tokens after the JSON closes are unconstrained.
        Each item is at least 17 tokens compact, so 10 items outlast OSL 128."""
        body, _, _ = self._source(config)
        grammar = json.loads(body)["structured_outputs"]["json"]
        assert grammar["required"] == [array_key]
        assert grammar["properties"][array_key]["minItems"] >= 10

    @staticmethod
    def _source(config, extra_bench_args="", extra_serve_args=""):
        script = (
            f'SCRIPT_DIR="{_BENCH_DIR}"; EXTRA_BENCH_ARGS="{extra_bench_args}"; '
            f'EXTRA_SERVE_ARGS="{extra_serve_args}"; '
            f'source "$SCRIPT_DIR/configs/{config}.sh"; '
            'printf "%s\\n%s\\n%s" "$EXTRA_BENCH_ARGS" "$EXTRA_SERVE_ARGS" '
            '"$BENCHMARK_EXTRA_BODY"'
        )
        out = subprocess.run(
            ["bash", "-c", script], check=True, capture_output=True, text=True
        ).stdout
        bench_args, serve_args, body = out.split("\n", 2)
        return body, bench_args, serve_args

    def test_complex_ast_grammar_is_recursive(self):
        body = json.loads(
            (_BENCH_DIR / "structured_schemas" / "complex_ast.json").read_text()
        )
        defs = body["structured_outputs"]["json"]["$defs"]
        assert {"$ref": "#/$defs/Node"} in defs["Expression"]["properties"].values()
        assert {"$ref": "#/$defs/Expression"} in defs["Node"]["anyOf"]
        items = body["structured_outputs"]["json"]["properties"]["expressions"]["items"]
        assert items["properties"]["ast"] == {"$ref": "#/$defs/Expression"}

    def test_character_dataset_and_schema(self):
        schema_path = _BENCH_DIR / "structured_schemas" / "character_schema.json"
        assert schema_path.exists()
        schema = json.loads(schema_path.read_text())["structured_outputs"]["json"]
        assert set(schema["required"]) == {
            "name",
            "house",
            "blood status",
            "occupation",
            "wand",
            "alive",
            "patronus",
            "bogart",
        }

        dataset_path = _BENCH_DIR / "datasets" / "character_benchmark.json"
        assert dataset_path.exists()
        dataset = json.loads(dataset_path.read_text())
        assert len(dataset) == 50
        for i, item in enumerate(dataset):
            assert item["id"] == str(i)
            assert len(item["conversations"]) == 2
            assert item["conversations"][0]["from"] == "human"
            assert "Harry Potter" in item["conversations"][0]["value"]
