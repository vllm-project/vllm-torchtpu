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
"""The runtime patch that widens ProfilerConfig to accept 'torch_phased'.

``ProfilerConfig.profiler`` is a closed pydantic ``Literal["torch", "cuda"]``
whose validator rejects ``torch_profiler_dir`` for anything but 'torch'. This
patch widens it at runtime so ``profiler_config.profiler=torch_phased`` works
without an upstream vLLM change.
"""

import logging
import os

import pytest
import vllm.config.profiler as profiler_module
from pydantic import ValidationError
from vllm.config.profiler import ProfilerConfig

from vllm_torchtpu import _patch_vllm_profiler_kind_torch_phased
from vllm_torchtpu.runner.utils import (TORCH_PHASED_PROFILER_KIND,
                                        TORCH_PROFILER_KINDS)

_PATCH_STAMP = "_tpu_torch_phased_profiler_kind_patch"


@pytest.fixture
def patch_applied(monkeypatch):
    """Apply the patch on a snapshot of ProfilerConfig, then fully restore it.

    Unlike a patch that just wraps one method, this one mutates several
    pydantic-internal caches in place (the field annotation in three places,
    the validator, and the compiled core schema), so each is individually
    snapshotted via monkeypatch to keep the patch's effect scoped to this
    test file -- other test files must see the pristine, unpatched class.
    """
    field = ProfilerConfig.__pydantic_fields__["profiler"]
    dc_field = ProfilerConfig.__dataclass_fields__["profiler"]
    decorator = ProfilerConfig.__pydantic_decorators__.model_validators[
        "_validate_profiler_config"]

    monkeypatch.setattr(profiler_module, "ProfilerKind",
                        profiler_module.ProfilerKind)
    monkeypatch.setattr(field, "annotation", field.annotation)
    monkeypatch.setattr(dc_field, "type", dc_field.type)
    monkeypatch.setitem(ProfilerConfig.__annotations__, "profiler",
                        ProfilerConfig.__annotations__["profiler"])
    monkeypatch.setattr(ProfilerConfig, "_validate_profiler_config",
                        ProfilerConfig._validate_profiler_config)
    monkeypatch.setattr(decorator, "func", decorator.func)
    monkeypatch.setattr(ProfilerConfig, "__pydantic_core_schema__",
                        ProfilerConfig.__pydantic_core_schema__)
    monkeypatch.setattr(ProfilerConfig, "__pydantic_validator__",
                        ProfilerConfig.__pydantic_validator__)
    monkeypatch.setattr(ProfilerConfig, "__pydantic_serializer__",
                        ProfilerConfig.__pydantic_serializer__)
    # apply_tpu_patches() may already have run in this process (it's
    # idempotent by design), so clear the stamp to force a re-application on
    # top of the snapshot just taken above.
    monkeypatch.delattr(ProfilerConfig, _PATCH_STAMP, raising=False)

    _patch_vllm_profiler_kind_torch_phased()

    yield

    # The patch stamps this class attribute with a plain assignment, which
    # monkeypatch never saw and so cannot undo on its own.
    if _PATCH_STAMP in ProfilerConfig.__dict__:
        delattr(ProfilerConfig, _PATCH_STAMP)


def test_torch_phased_with_dir_is_accepted(patch_applied):
    config = ProfilerConfig(profiler=TORCH_PHASED_PROFILER_KIND,
                            torch_profiler_dir="/tmp/x")

    assert config.profiler == TORCH_PHASED_PROFILER_KIND
    assert config.torch_profiler_dir == "/tmp/x"


def test_torch_phased_without_dir_is_rejected(patch_applied):
    with pytest.raises(ValueError, match="torch_profiler_dir must be set"):
        ProfilerConfig(profiler=TORCH_PHASED_PROFILER_KIND)


def test_torch_phased_dir_is_expanded_like_torch(patch_applied):
    """The delegate-to-original-validator path handles ~-expansion too."""
    config = ProfilerConfig(profiler=TORCH_PHASED_PROFILER_KIND,
                            torch_profiler_dir="~/relative")

    assert config.torch_profiler_dir == os.path.abspath(
        os.path.expanduser("~/relative"))


def test_torch_phased_does_not_warn_about_frontend_overhead(
        patch_applied, caplog):
    """AsyncLLM only starts its frontend profiler for profiler == 'torch', so
    warning a phased run about frontend overhead would be misleading."""
    with caplog.at_level(logging.WARNING):
        ProfilerConfig(profiler=TORCH_PHASED_PROFILER_KIND,
                       torch_profiler_dir="/tmp/x",
                       max_iterations=5)

    assert "high overhead" not in caplog.text


def test_torch_phased_preserves_ignore_frontend(patch_applied):
    """The delegation flips ignore_frontend internally; it must not leak."""
    assert ProfilerConfig(profiler=TORCH_PHASED_PROFILER_KIND,
                          torch_profiler_dir="/tmp/x").ignore_frontend is False
    assert ProfilerConfig(profiler=TORCH_PHASED_PROFILER_KIND,
                          torch_profiler_dir="/tmp/x",
                          ignore_frontend=True).ignore_frontend is True


def test_torch_phased_uri_dir_is_left_alone(patch_applied):
    config = ProfilerConfig(profiler=TORCH_PHASED_PROFILER_KIND,
                            torch_profiler_dir="gs://bucket/traces")

    assert config.torch_profiler_dir == "gs://bucket/traces"


def test_torch_with_dir_still_works(patch_applied):
    config = ProfilerConfig(profiler="torch", torch_profiler_dir="/tmp/x")

    assert config.profiler == "torch"


def test_torch_without_dir_still_rejected(patch_applied):
    with pytest.raises(ValueError, match="torch_profiler_dir must be set"):
        ProfilerConfig(profiler="torch")


def test_cuda_with_dir_still_rejected(patch_applied):
    with pytest.raises(ValueError, match="only applicable"):
        ProfilerConfig(profiler="cuda", torch_profiler_dir="/tmp/x")


def test_cuda_alone_still_works(patch_applied):
    config = ProfilerConfig(profiler="cuda")

    assert config.profiler == "cuda"


def test_dir_without_profiler_still_rejected(patch_applied):
    with pytest.raises(ValueError, match="only applicable"):
        ProfilerConfig(torch_profiler_dir="/tmp/x")


def test_unknown_kind_still_rejected(patch_applied):
    with pytest.raises(ValidationError):
        ProfilerConfig(profiler="nonsense", torch_profiler_dir="/tmp/x")


def test_no_profiler_at_all_still_works(patch_applied):
    config = ProfilerConfig()

    assert config.profiler is None


def test_patch_is_idempotent(patch_applied):
    before = ProfilerConfig._validate_profiler_config

    _patch_vllm_profiler_kind_torch_phased()

    assert ProfilerConfig._validate_profiler_config is before


def test_torch_profiler_kinds_constant():
    assert TORCH_PROFILER_KINDS == ("torch", TORCH_PHASED_PROFILER_KIND)


def test_patch_is_applied_by_plugin_registration():
    """The patch must land before any ProfilerConfig is built.

    Importing vllm_torchtpu is enough to activate the platform plugin, which
    calls register_tpu_platform(). Were the patch deferred to
    apply_tpu_patches() (invoked from check_and_update_config), the CLI would
    already have constructed ProfilerConfig and rejected 'torch_phased'.
    """
    assert ProfilerConfig(
        profiler=TORCH_PHASED_PROFILER_KIND,
        torch_profiler_dir="/tmp/x").profiler == (TORCH_PHASED_PROFILER_KIND)


def test_torch_phased_is_offered_on_the_cli():
    from vllm.engine.arg_utils import get_kwargs

    assert TORCH_PHASED_PROFILER_KIND in get_kwargs(
        ProfilerConfig)["profiler"]["choices"]
