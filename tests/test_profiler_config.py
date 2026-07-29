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
"""TpuProfilerConfig and the patch that points VllmConfig at it."""

import logging
import os
import pickle

import pytest
from pydantic import TypeAdapter, ValidationError
from vllm.config import VllmConfig
from vllm.config.profiler import ProfilerConfig

from vllm_torchtpu.profiler_config import \
    PHASED_DECODE_ONLY_KV_LEN_THRESHOLD_DISABLED as DISABLED
from vllm_torchtpu.profiler_config import (TORCH_PHASED_PROFILER_KIND,
                                           TORCH_PROFILER_KINDS,
                                           TpuProfilerConfig)

PHASED = TORCH_PHASED_PROFILER_KIND


class TestPhasedKind:
    """The kind vLLM does not define."""

    def test_accepted_with_dir(self):
        config = TpuProfilerConfig(profiler=PHASED,
                                   torch_profiler_dir="/tmp/x")

        assert config.profiler == PHASED
        assert config.torch_profiler_dir == "/tmp/x"

    def test_rejected_without_dir(self):
        with pytest.raises(ValueError, match="torch_profiler_dir must be set"):
            TpuProfilerConfig(profiler=PHASED)

    def test_dir_is_expanded_like_torch(self):
        """Delegating to the base validator carries its ~-expansion along."""
        config = TpuProfilerConfig(profiler=PHASED,
                                   torch_profiler_dir="~/relative")

        assert config.torch_profiler_dir == os.path.abspath(
            os.path.expanduser("~/relative"))

    def test_uri_dir_is_left_alone(self):
        config = TpuProfilerConfig(profiler=PHASED,
                                   torch_profiler_dir="gs://bucket/traces")

        assert config.torch_profiler_dir == "gs://bucket/traces"

    def test_no_frontend_overhead_warning(self, caplog):
        """AsyncLLM starts its frontend profiler only for profiler == 'torch',
        so warning a phased run about frontend overhead would mislead."""
        with caplog.at_level(logging.WARNING):
            TpuProfilerConfig(profiler=PHASED,
                              torch_profiler_dir="/tmp/x",
                              max_iterations=5)

        assert "high overhead" not in caplog.text

    def test_ignore_frontend_is_not_clobbered(self):
        """The delegation flips ignore_frontend internally; it must not leak."""
        assert TpuProfilerConfig(
            profiler=PHASED,
            torch_profiler_dir="/tmp/x").ignore_frontend is False
        assert TpuProfilerConfig(profiler=PHASED,
                                 torch_profiler_dir="/tmp/x",
                                 ignore_frontend=True).ignore_frontend is True


class TestPhasedDecodeOnlyKvLenThreshold:
    """The knob that used to live in additional_config.

    It gates capture of the decode_only phase until the KV cache is warm.
    It has no upstream counterpart, which is why it needed additional_config
    before this class existed.
    """

    def test_defaults_to_disabled(self):
        assert TpuProfilerConfig(
        ).phased_decode_only_kv_len_threshold == DISABLED

    def test_accepts_a_threshold(self):
        config = TpuProfilerConfig(profiler=PHASED,
                                   torch_profiler_dir="/tmp/x",
                                   phased_decode_only_kv_len_threshold=128)

        assert config.phased_decode_only_kv_len_threshold == 128

    def test_rejects_values_below_disabled(self):
        with pytest.raises(ValidationError):
            TpuProfilerConfig(phased_decode_only_kv_len_threshold=-2)

    def test_stays_out_of_the_compile_cache_key(self):
        """ProfilerConfig.compute_hash() hashes an empty factors list, so
        inheriting it keeps this knob from forcing an XLA recompile. This is
        what made the additional_config carve-out patch unnecessary."""
        assert TpuProfilerConfig(
            profiler=PHASED,
            torch_profiler_dir="/tmp/a",
            phased_decode_only_kv_len_threshold=128).compute_hash(
            ) == TpuProfilerConfig(
                profiler=PHASED,
                torch_profiler_dir="/tmp/b",
                phased_decode_only_kv_len_threshold=999).compute_hash()


class TestUpstreamBehaviourPreserved:
    """Everything vLLM already validated must still validate the same way."""

    def test_torch_with_dir(self):
        config = TpuProfilerConfig(profiler="torch",
                                   torch_profiler_dir="/tmp/x")

        assert config.profiler == "torch"

    def test_torch_without_dir_rejected(self):
        with pytest.raises(ValueError, match="torch_profiler_dir must be set"):
            TpuProfilerConfig(profiler="torch")

    def test_cuda_with_dir_rejected(self):
        with pytest.raises(ValueError, match="only applicable"):
            TpuProfilerConfig(profiler="cuda", torch_profiler_dir="/tmp/x")

    def test_cuda_alone(self):
        assert TpuProfilerConfig(profiler="cuda").profiler == "cuda"

    def test_dir_without_profiler_rejected(self):
        with pytest.raises(ValueError, match="only applicable"):
            TpuProfilerConfig(torch_profiler_dir="/tmp/x")

    def test_unknown_kind_rejected(self):
        with pytest.raises(ValidationError):
            TpuProfilerConfig(profiler="nonsense", torch_profiler_dir="/tmp/x")

    def test_no_profiler_at_all(self):
        assert TpuProfilerConfig().profiler is None


class TestSubstitution:
    """How the subclass reaches vLLM."""

    def test_vllm_profiler_config_is_left_alone(self):
        """The whole point of subclassing: vLLM's class is not mutated."""
        annotation = ProfilerConfig.__pydantic_fields__["profiler"].annotation

        assert PHASED not in str(annotation)

    def test_vllm_config_field_points_at_the_subclass(self):
        """Applied by the vllm.general_plugins entry point, which vLLM loads
        before it builds the --profiler-config argument."""
        from vllm_torchtpu.platforms import register_tpu_config

        register_tpu_config()

        assert VllmConfig.__pydantic_fields__[
            "profiler_config"].annotation is TpuProfilerConfig

    def test_accepts_a_plain_profiler_config(self):
        """Regression: repointing VllmConfig's field made it reject vLLM's own
        class, while two defaults are still built from it -- EngineArgs' field
        factory and the argparse default.
        Without this coercion every run that does not profile fails with
        "Input should be ... an instance of TpuProfilerConfig"."""
        adapter = TypeAdapter(TpuProfilerConfig)

        upgraded = adapter.validate_python(ProfilerConfig())

        assert type(upgraded) is TpuProfilerConfig
        assert upgraded.profiler is None

    def test_accepts_a_configured_plain_profiler_config(self):
        adapter = TypeAdapter(TpuProfilerConfig)

        upgraded = adapter.validate_python(
            ProfilerConfig(profiler="torch", torch_profiler_dir="/tmp/x"))

        assert upgraded.profiler == "torch"
        assert upgraded.torch_profiler_dir == "/tmp/x"

    def test_coercion_does_not_weaken_validation(self):
        """The before-validator must not become a bypass."""
        adapter = TypeAdapter(TpuProfilerConfig)

        with pytest.raises(ValidationError, match="only applicable"):
            adapter.validate_python(
                ProfilerConfig(profiler="cuda", torch_profiler_dir="/tmp/x"))

    def test_registered_as_a_general_plugin(self):
        """The entry point is what applies the patch. It lives in
        pyproject.toml, so a rename here fails silently -- nothing would call
        it and torch_phased would be rejected at the CLI."""
        from importlib.metadata import entry_points

        registered = {
            ep.value
            for ep in entry_points(group="vllm.general_plugins")
        }

        assert ("vllm_torchtpu.platforms:register_tpu_config" in registered)

    def test_is_a_profiler_config(self):
        """Anything type-checking against vLLM's class keeps working."""
        assert isinstance(TpuProfilerConfig(), ProfilerConfig)

    def test_survives_pickling(self):
        """VllmConfig is pickled out to spawned worker processes."""
        config = TpuProfilerConfig(profiler=PHASED,
                                   torch_profiler_dir="/tmp/x")

        restored = pickle.loads(pickle.dumps(config))

        assert type(restored) is TpuProfilerConfig
        assert restored.profiler == PHASED

    def test_offered_on_the_cli(self):
        from vllm.engine.arg_utils import get_kwargs

        assert PHASED in get_kwargs(TpuProfilerConfig)["profiler"]["choices"]

    def test_cli_parses_the_phased_kind(self):
        """End-to-end through vLLM's real serve parser. The argument's
        converter is bound at add_cli_args() time, so this is what catches it
        being left pointed at vLLM's class."""
        from vllm.entrypoints.openai.cli_args import make_arg_parser
        from vllm.utils.argparse_utils import FlexibleArgumentParser

        args = make_arg_parser(FlexibleArgumentParser()).parse_args([
            "--model", "Qwen/Qwen3-0.6B", "--profiler-config.profiler", PHASED,
            "--profiler-config.torch_profiler_dir", "/tmp/x"
        ])

        assert type(args.profiler_config) is TpuProfilerConfig
        assert args.profiler_config.profiler == PHASED


def test_torch_profiler_kinds_constant():
    assert TORCH_PROFILER_KINDS == ("torch", PHASED)
