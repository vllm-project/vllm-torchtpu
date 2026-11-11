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
"""Parsing rules for the `envs.py` builders.

These knobs used to be read inline in `distributed/utils.py`, where each site
invented its own rules: four boolean vocabularies and three different answers
to "what if the value does not parse". There is now one answer to each, and
these tests pin it rather than any individual knob.

The house rule: a value that does not parse raises, because a misspelled flag
is a launch-script bug and defaulting hides it.
"""

import types

import pytest

from vllm_torchtpu.envs import (
    env_bool,
    env_float,
    env_int,
    env_nonnegative_float,
    env_nonnegative_int,
    env_nonnegative_int_or_auto,
    env_str,
)

VAR = "TPU_TEST_ONLY_KNOB"


@pytest.fixture(autouse=True)
def _clear(monkeypatch):
    monkeypatch.delenv(VAR, raising=False)


# ---------------------------------------------------------------------------
# Unset and empty both mean "not configured".
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "builder,default",
    [
        (env_bool, True),
        (env_int, 7),
        (env_nonnegative_int, 7),
        (env_float, 1.5),
        (env_nonnegative_float, 1.5),
    ],
)
def test_unset_reads_as_the_default(builder, default):
    assert builder(VAR, default)() == default


@pytest.mark.parametrize(
    "builder,default",
    [
        (env_bool, True),
        (env_int, 7),
        (env_nonnegative_int, 7),
        (env_float, 1.5),
        (env_nonnegative_float, 1.5),
    ],
)
@pytest.mark.parametrize("value", ["", "   "])
def test_empty_reads_as_the_default(monkeypatch, builder, default, value):
    """`FOO=` is how a shell unsets a value in practice. The old inline int
    parsing crashed on it with `int('')`."""
    monkeypatch.setenv(VAR, value)
    assert builder(VAR, default)() == default


# ---------------------------------------------------------------------------
# env_bool: one vocabulary, replacing the four that existed.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value", ["1", "true", "True", "TRUE", "yes", "YES", "on", " on "]
)
def test_bool_true_spellings(monkeypatch, value):
    monkeypatch.setenv(VAR, value)
    assert env_bool(VAR, default=False)() is True


@pytest.mark.parametrize(
    "value", ["0", "false", "False", "FALSE", "no", "NO", "off", " off "]
)
def test_bool_false_spellings(monkeypatch, value):
    monkeypatch.setenv(VAR, value)
    assert env_bool(VAR, default=True)() is False


@pytest.mark.parametrize("value", ["banana", "2", "-1", "t", "enable"])
def test_bool_rejects_anything_else(monkeypatch, value):
    monkeypatch.setenv(VAR, value)
    with pytest.raises(ValueError, match="Invalid boolean value"):
        env_bool(VAR, default=False)()


def test_bool_error_names_the_variable_and_the_options(monkeypatch):
    """The message is read by someone whose launch script just failed, so it
    has to say which variable and what would have worked."""
    monkeypatch.setenv(VAR, "enabled")
    with pytest.raises(ValueError) as exc:
        env_bool(VAR)()

    message = str(exc.value)
    assert VAR in message
    assert "'enabled'" in message
    assert "yes" in message and "off" in message


# ---------------------------------------------------------------------------
# Numbers.
# ---------------------------------------------------------------------------


def test_int_parses_and_accepts_negatives(monkeypatch):
    monkeypatch.setenv(VAR, "-5")
    assert env_int(VAR, 0)() == -5


@pytest.mark.parametrize("value", ["2.5", "banana", "1_000_000_000_000x"])
def test_int_rejects_non_integers(monkeypatch, value):
    monkeypatch.setenv(VAR, value)
    with pytest.raises(ValueError, match=VAR):
        env_int(VAR, 0)()


@pytest.mark.parametrize("builder", [env_nonnegative_int, env_nonnegative_float])
def test_nonnegative_accepts_zero(monkeypatch, builder):
    """0 is meaningful for these knobs: auto-size for the int ones, "check
    once, do not wait" for TPU_PRECOMPILE_PEER_WAIT_S."""
    monkeypatch.setenv(VAR, "0")
    assert builder(VAR, 4)() == 0


@pytest.mark.parametrize("builder", [env_nonnegative_int, env_nonnegative_float])
def test_nonnegative_rejects_negatives(monkeypatch, builder):
    """These feed ThreadPoolExecutor(max_workers=...), slot counts and waits.
    The old helper clamped a negative to the default and logged a warning,
    which left the process running a pool size nobody asked for."""
    monkeypatch.setenv(VAR, "-1")
    with pytest.raises(ValueError, match="must be >= 0"):
        builder(VAR, 0)()


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_nonnegative_float_rejects_non_finite(monkeypatch, value):
    """As a timeout, nan compares false against everything and inf never
    expires; -inf is caught as negative."""
    monkeypatch.setenv(VAR, value)
    with pytest.raises(ValueError, match=VAR):
        env_nonnegative_float(VAR, 1.0)()


def test_float_parses_ints_too(monkeypatch):
    monkeypatch.setenv(VAR, "30")
    assert env_float(VAR, 1.0)() == 30.0


def test_float_rejects_non_numbers(monkeypatch):
    monkeypatch.setenv(VAR, "soon")
    with pytest.raises(ValueError, match=VAR):
        env_float(VAR, 1.0)()


@pytest.mark.parametrize("value", [None, "", "  ", "auto", "AUTO", " Auto "])
def test_int_or_auto_reads_none_for_unset_empty_or_auto(monkeypatch, value):
    if value is not None:
        monkeypatch.setenv(VAR, value)
    assert env_nonnegative_int_or_auto(VAR)() is None


@pytest.mark.parametrize("value,expected", [("12", 12), ("0", 0)])
def test_int_or_auto_parses_integers(monkeypatch, value, expected):
    monkeypatch.setenv(VAR, value)
    assert env_nonnegative_int_or_auto(VAR)() == expected


@pytest.mark.parametrize("value", ["automatic", "-1"])
def test_int_or_auto_rejects_other_words_and_negatives(monkeypatch, value):
    monkeypatch.setenv(VAR, value)
    with pytest.raises(ValueError, match=VAR):
        env_nonnegative_int_or_auto(VAR)()


# ---------------------------------------------------------------------------
# Strings.
# ---------------------------------------------------------------------------


def test_str_strips_surrounding_whitespace(monkeypatch):
    """A trailing space in a launch script used to travel into a port string
    and a socket path."""
    monkeypatch.setenv(VAR, "  9100  ")
    assert env_str(VAR, "0")() == "9100"


def test_str_keeps_an_explicit_empty_value_distinct_from_unset(monkeypatch):
    monkeypatch.setenv(VAR, "")
    assert env_str(VAR, "fallback")() == ""


# ---------------------------------------------------------------------------
# The knobs migrated out of distributed/utils.py resolve through the
# registry, with the defaults the inline reads used to hard-code.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,expected",
    [
        ("TPU_KV_RESHARD_TRANSPORT", "raiden"),
        ("TPU_KV_TRANSFER_PORT", "9100"),
        ("TPU_P2P_WAIT_PULL_TIMEOUT", 120),
        ("TPU_RAIDEN_STAGE3_STATUS_PROBE_S", 1.0),
        ("TPU_RAIDEN_STAGE3_DEFERRED_SUBMIT", True),
        ("TPU_RAIDEN_STAGE3_REGISTRATION_WAIT_S", 30.0),
        ("TPU_RAIDEN_TEST_REGISTRATION_DELAY_S", 0.0),
        ("TPU_RAIDEN_POOL_STAGING_LEASES", 8),
        ("TPU_RAIDEN_INLINE_LOAD", False),
        ("KDA_MANUAL_STATE_DMA", None),
        ("KDA_MANUAL_H0_DMA", None),
        ("KDA_MANUAL_HT_DMA", None),
        ("KDA_OVERLAP_H0_DMA", True),
        ("KDA_OVERLAP_HT_DMA", True),
        ("KDA_PACK_HEAD_INV", True),
        ("KDA_PACKED_METADATA", True),
        ("KDA_FWD_MB", None),
        ("SPEC_WARMUP", True),
        ("RAIDEN_DISABLE_SINGLETON_WORKER", True),
        ("RAIDEN_SHM_KEY", ""),
        ("VLLM_TPU_OFFLOAD_WAIT_TIMEOUT_S", 30.0),
        ("VLLM_TPU_OFFLOAD_SAVE_RETRIES", 1),
        ("TPU_SHARDED_LOAD_SYNC_EVERY", 512),
        ("VLLM_TPU_DEBUG_PCP_LAYOUT", False),
        ("TPU_LOCAL_RANK_OFFSET", 0),
        ("DEBUG_TPU_LOCAL_RANK_OFFSET", 0),
        ("TORCH_TPU_BASE_PORT", 8070),
        ("TORCH_TPU_MP_RENDEZVOUS_PORT", None),
        ("TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT", ""),
        ("SC_ALLREDUCE_ALLGATHER_OFFLOAD_MIN_BYTES", None),
    ],
)
def test_migrated_knob_defaults(monkeypatch, name, expected):
    from vllm_torchtpu import envs

    monkeypatch.delenv(name, raising=False)
    assert envs.environment_variables[name]() == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, "raiden"),
        ("raiden", "raiden"),
        ("RAIDEN", "RAIDEN"),
    ],
)
def test_tpu_kv_reshard_transport_accepts_only_raiden(monkeypatch, raw, expected):
    """Unset defaults to 'raiden' and 'raiden' is accepted case-insensitively;
    the variable is kept only for launch-config compatibility."""
    from vllm_torchtpu import envs

    monkeypatch.delitem(vars(envs), "TPU_KV_RESHARD_TRANSPORT", raising=False)
    if raw is None:
        monkeypatch.delenv("TPU_KV_RESHARD_TRANSPORT", raising=False)
    else:
        monkeypatch.setenv("TPU_KV_RESHARD_TRANSPORT", raw)
    actual = envs.TPU_KV_RESHARD_TRANSPORT
    assert actual == expected


@pytest.mark.parametrize("raw", ["zmq", "custom", "radien", ""])
def test_tpu_kv_reshard_transport_rejects_non_raiden(monkeypatch, raw):
    """The removed 'zmq' transport, custom names, typos, and an explicit empty
    string fail on first read instead of being passed through."""
    from vllm_torchtpu import envs

    monkeypatch.delitem(vars(envs), "TPU_KV_RESHARD_TRANSPORT", raising=False)
    monkeypatch.setenv("TPU_KV_RESHARD_TRANSPORT", raw)
    with pytest.raises(ValueError, match=r"Valid options: \['raiden'\]"):
        _ = envs.TPU_KV_RESHARD_TRANSPORT


def _premapped_buffer_block():
    """Compile only env_override.py's TPU_PREMAPPED_BUFFER_SIZE block.

    Reloading the whole module would re-run libtpu flag setup, the GCE
    metadata probe and the import-time patch registry, so the block is
    executed on its own against a fake ``os``.
    """
    import ast
    import inspect

    from vllm_torchtpu import env_override

    tree = ast.parse(inspect.getsource(env_override))
    blocks = [
        node
        for node in tree.body
        if isinstance(node, ast.If) and "TPU_PREMAPPED_BUFFER_SIZE" in ast.unparse(node)
    ]
    assert len(blocks) == 1
    module = ast.Module(body=blocks, type_ignores=[])
    return compile(module, env_override.__file__, "exec")


def _run_premapped_buffer_block(env):
    fake_os = types.SimpleNamespace(environ=env, getenv=env.get)
    exec(_premapped_buffer_block(), {"os": fake_os})  # noqa: S102
    return env


@pytest.mark.parametrize("raw", ["1", "true", "TRUE", "yes", "on", " On "])
def test_env_override_premaps_host_buffer_for_raiden_cache_manager(raw):
    """Every env_bool truthy spelling of TPU_USE_RAIDEN_KV_CACHE_MANAGER sizes
    libtpu's premapped host buffer pool to 16 GiB."""
    env = _run_premapped_buffer_block({"TPU_USE_RAIDEN_KV_CACHE_MANAGER": raw})
    assert env["TPU_PREMAPPED_BUFFER_SIZE"] == str(16 << 30)


@pytest.mark.parametrize("raw", [None, "0", "false", ""])
def test_env_override_skips_premapped_buffer_without_raiden_cache_manager(raw):
    """TPU_KV_RESHARD_TRANSPORT=raiden alone (its default) no longer enables the
    premapped pool, so single-host engines keep torch_tpu's default."""
    env = {"TPU_KV_RESHARD_TRANSPORT": "raiden"}
    if raw is not None:
        env["TPU_USE_RAIDEN_KV_CACHE_MANAGER"] = raw
    env = _run_premapped_buffer_block(env)
    assert "TPU_PREMAPPED_BUFFER_SIZE" not in env


def test_env_override_keeps_a_preset_premapped_buffer_size():
    env = _run_premapped_buffer_block(
        {"TPU_USE_RAIDEN_KV_CACHE_MANAGER": "1", "TPU_PREMAPPED_BUFFER_SIZE": "123"}
    )
    assert env["TPU_PREMAPPED_BUFFER_SIZE"] == "123"


def test_spec_warmup_off_spellings(monkeypatch):
    """SPEC_WARMUP accepts the usual on/off spellings; anything else raises
    rather than silently leaving warmup on."""
    from vllm_torchtpu import envs

    for value in ("0", "false", "off"):
        monkeypatch.setenv("SPEC_WARMUP", value)
        assert envs.SPEC_WARMUP is False
    monkeypatch.setenv("SPEC_WARMUP", "1")
    assert envs.SPEC_WARMUP is True
    monkeypatch.setenv("SPEC_WARMUP", "nope")
    with pytest.raises(ValueError, match="SPEC_WARMUP"):
        _ = envs.SPEC_WARMUP
