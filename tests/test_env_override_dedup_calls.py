# SPDX-License-Identifier: Apache-2.0
"""A model plan's libtpu flag reaches the environment spawned processes inherit."""

import pytest

from vllm_torchtpu import env_override

_FLAG = "--xla_tpu_enable_deduplicated_calls"


@pytest.mark.parametrize("existing", ["", "--xla_other=1"])
def test_plan_flag_is_added(monkeypatch, existing):
    monkeypatch.setenv("LIBTPU_INIT_ARGS", existing)
    env_override.add_libtpu_flag(f"{_FLAG}=false")
    args = env_override.os.environ["LIBTPU_INIT_ARGS"].split()
    assert f"{_FLAG}=false" in args
    assert set(existing.split()) <= set(args)


def test_explicit_flag_wins(monkeypatch):
    monkeypatch.setenv("LIBTPU_INIT_ARGS", f"{_FLAG}=true")
    env_override.add_libtpu_flag(f"{_FLAG}=false")
    args = env_override.os.environ["LIBTPU_INIT_ARGS"].split()
    assert [a for a in args if a.startswith(_FLAG)] == [f"{_FLAG}=true"]


def test_flag_is_added_once(monkeypatch):
    monkeypatch.setenv("LIBTPU_INIT_ARGS", "")
    env_override.add_libtpu_flag(f"{_FLAG}=false")
    env_override.add_libtpu_flag(f"{_FLAG}=false")
    assert (
        env_override.os.environ["LIBTPU_INIT_ARGS"].split().count(f"{_FLAG}=false") == 1
    )
