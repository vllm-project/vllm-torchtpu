import pytest

from vllm_torchtpu.tracing.options import (
    parse_profile_options,
    resolve_profile_dir_and_opts,
)

pytestmark = pytest.mark.cpu_test


def test_resolve_profile_dir_only():
    profile_dir, standard, advanced = resolve_profile_dir_and_opts(
        "/tmp/base", "my_run_123"
    )
    assert profile_dir == "/tmp/base/my_run_123"
    assert standard == {
        "host_tracer_level": 2,
        "device_tracer_level": 1,
        "python_tracer_level": 1,
    }
    assert advanced == {
        "tpu_trace_mode": "TRACE_COMPUTE",
        "tpu_num_sparse_cores_to_trace": 1,
        "tpu_num_sparse_core_tiles_to_trace": 1,
    }


def test_resolve_with_options():
    profile_dir, standard, advanced = resolve_profile_dir_and_opts(
        "/tmp/base", "host_tracer_level:3;e2e_true:true;random_str:hello;"
    )
    assert profile_dir == "/tmp/base"
    assert standard == {
        "host_tracer_level": 3,
        "device_tracer_level": 1,
        "python_tracer_level": 1,
    }
    assert advanced == {
        "e2e_true": True,
        "random_str": "hello",
        "tpu_trace_mode": "TRACE_COMPUTE",
        "tpu_num_sparse_cores_to_trace": 1,
        "tpu_num_sparse_core_tiles_to_trace": 1,
    }


def test_parse_profile_options_invalid():
    with pytest.raises(ValueError, match="Invalid profile option format"):
        parse_profile_options("host_tracer_level-3;")


def test_parse_profile_options_invalid_chars():
    with pytest.raises(ValueError, match="Invalid characters in option value"):
        parse_profile_options("host_tracer_level:3$%#;")


def test_parse_profile_options_none_or_empty():
    s, a = parse_profile_options(None)
    assert s == {}
    assert a == {}

    s, a = parse_profile_options("   ")
    assert s == {}
    assert a == {}


def test_resolve_with_profiler_kwargs():
    profile_dir, standard, advanced = resolve_profile_dir_and_opts(
        "/tmp/base",
        None,
        profiler_kwargs={
            "host_tracer_level": 3,
            "e2e_enable_fw_throttle": True,
            "tpu_trace_mode": "TRACE_COMPUTE_AND_SYNC",
        },
    )
    assert profile_dir == "/tmp/base"
    assert standard == {
        "host_tracer_level": 3,
        "device_tracer_level": 1,
        "python_tracer_level": 1,
    }
    assert advanced == {
        "e2e_enable_fw_throttle": True,
        "tpu_trace_mode": "TRACE_COMPUTE_AND_SYNC",
        "tpu_num_sparse_cores_to_trace": 1,
        "tpu_num_sparse_core_tiles_to_trace": 1,
    }


def test_resolve_with_profile_prefix_and_profiler_kwargs():
    profile_dir, standard, advanced = resolve_profile_dir_and_opts(
        "/tmp/base",
        "custom_run",
        profiler_kwargs={"host_tracer_level": 3, "e2e_true": "true"},
    )
    assert profile_dir == "/tmp/base/custom_run"
    assert standard["host_tracer_level"] == 3
    assert advanced["e2e_true"] is True
