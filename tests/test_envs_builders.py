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
import pytest

from vllm_torchtpu.envs import (env_bool, env_float, env_int,
                                env_nonnegative_int,
                                env_nonnegative_int_or_auto, env_str)

VAR = "TPU_TEST_ONLY_KNOB"


@pytest.fixture(autouse=True)
def _clear(monkeypatch):
    monkeypatch.delenv(VAR, raising=False)


# ---------------------------------------------------------------------------
# Unset and empty both mean "not configured".
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("builder,default", [
    (env_bool, True),
    (env_int, 7),
    (env_nonnegative_int, 7),
    (env_float, 1.5),
])
def test_unset_reads_as_the_default(builder, default):
    assert builder(VAR, default)() == default


@pytest.mark.parametrize("builder,default", [
    (env_bool, True),
    (env_int, 7),
    (env_nonnegative_int, 7),
    (env_float, 1.5),
])
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
    "value", ["1", "true", "True", "TRUE", "yes", "YES", "on", " on "])
def test_bool_true_spellings(monkeypatch, value):
    monkeypatch.setenv(VAR, value)
    assert env_bool(VAR, default=False)() is True


@pytest.mark.parametrize(
    "value", ["0", "false", "False", "FALSE", "no", "NO", "off", " off "])
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


def test_nonnegative_int_accepts_zero(monkeypatch):
    """0 is meaningful for these knobs: every one of them spells auto-size."""
    monkeypatch.setenv(VAR, "0")
    assert env_nonnegative_int(VAR, 4)() == 0


def test_nonnegative_int_rejects_negatives(monkeypatch):
    """These feed ThreadPoolExecutor(max_workers=...) and slot counts. The
    old helper clamped a negative to the default and logged a warning, which
    left the process running a pool size nobody asked for."""
    monkeypatch.setenv(VAR, "-1")
    with pytest.raises(ValueError, match="must be >= 0"):
        env_nonnegative_int(VAR, 0)()


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


@pytest.mark.parametrize("name,expected", [
    ("TPU_KV_TRANSFER_PORT", "9100"),
    ("TPU_SIDE_CHANNEL_PORT", "9600"),
    ("TPU_NODE_ID", 0),
    ("TPU_KV_TRANSFER_CHANNEL_NUMBER", 0),
    ("TPU_P2P_WAIT_PULL_TIMEOUT", 120),
    ("TPU_RAIDEN_STAGE3_STATUS_PROBE_S", 1.0),
    ("TPU_RAIDEN_STAGE3_DEFERRED_SUBMIT", True),
    ("TPU_RAIDEN_STAGE3_REGISTRATION_WAIT_S", 30.0),
    ("TPU_RAIDEN_TEST_REGISTRATION_DELAY_S", 0.0),
    ("TPU_KV_STAGE_WAIT_TIMEOUT_SECS", 30.0),
    ("TPU_KV_SHM_POOL_GB", 128.0),
    ("TPU_KV_TRANSFER_NAMESPACE", ""),
    ("TPU_IPC_SOCKET_DIR", "/tmp"),
    ("TPU_KV_WARMUP_ENABLED", True),
    ("TPU_KV_LATENCY_LOG_INTERVAL", 30.0),
    ("TPU_KV_PIN_SHM", False),
    ("TPU_USE_RAIDEN_CONNECTOR", False),
    ("TPU_RAIDEN_TRANSFER_NUM_SLOTS", 0),
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
    ("VLLM_TORCHTPU_IPC_KEY", ""),
    ("TPU_SHARDED_LOAD_SYNC_EVERY", 512),
    ("VLLM_TPU_DEBUG_PCP_LAYOUT", False),
    ("TPU_LOCAL_RANK_OFFSET", 0),
    ("DEBUG_TPU_LOCAL_RANK_OFFSET", 0),
    ("TORCH_TPU_BASE_PORT", 8070),
    ("TORCH_TPU_MP_RENDEZVOUS_PORT", None),
    ("TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT", ""),
    ("SC_ALLREDUCE_ALLGATHER_OFFLOAD_MIN_BYTES", None),
])
def test_migrated_knob_defaults(monkeypatch, name, expected):
    from vllm_torchtpu import envs

    monkeypatch.delenv(name, raising=False)
    assert envs.environment_variables[name]() == expected


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
