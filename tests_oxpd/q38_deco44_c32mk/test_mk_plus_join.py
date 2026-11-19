# SPDX-License-Identifier: Apache-2.0
"""q38 INT-4: the imported (MM 37b960ea9) Qwen3.8 megakernel driver accepts '+'-joined VLLM_TPU_MEGAKERNEL options.

The served recipe splits DECODE_EXTRA_SERVER_ENV on ',' (infx-qwen38-flash-agentx-disagg.jsonnet extraServerEnv), so a served
option list must be joined with '+'; the imported driver's parsers split on ',' only. These tests run the driver's OWN
lines (from `raw = str(envs.VLLM_TPU_MEGAKERNEL ...` through the marked '# q38 INT-4 end'), taken from the source of
Qwen38SegmentDriver.maybe_create, with a stub envs, then parse the result with the driver's own helpers. Without the
normalization line the '+' string parses whole / wholeunion / ... as OFF and these tests fail. CPU only, no TPU."""
import inspect
import textwrap
import types

from vllm_torchtpu.megakernel_runtime import qwen38_driver as d

PLUS = "1+whole=1+wholeshard=1+wholeunion=1+wholeloop=1+wholerows=64+wholecap=262144"
COMMA = PLUS.replace("+", ",")


def _raw_from_driver(value):
    src = textwrap.dedent(inspect.getsource(d.Qwen38SegmentDriver.maybe_create.__func__)).split("\n")
    start = next(i for i, ln in enumerate(src) if "raw = str(envs.VLLM_TPU_MEGAKERNEL" in ln)
    end = next(i for i, ln in enumerate(src) if ln.strip() == "# q38 INT-4 end")
    ns = {"envs": types.SimpleNamespace(VLLM_TPU_MEGAKERNEL=value)}
    exec(textwrap.dedent("\n".join(src[start : end + 1])), ns)  # noqa: S102 - the driver's own lines
    return ns["raw"]


def _opts(raw):
    return {
        "whole": d._parse_flag(raw, "whole", False),
        "wholeshard": d._parse_flag(raw, "wholeshard", False),
        "wholeunion": d._parse_flag(raw, "wholeunion", False),
        "wholeloop": d._parse_flag(raw, "wholeloop", False),
        "wholerows": d._parse_int(raw, "wholerows"),
        "wholecap": d._parse_int(raw, "wholecap"),
        "buckets": d._parse_buckets(raw),
    }


WANT = {"whole": True, "wholeshard": True, "wholeunion": True, "wholeloop": True, "wholerows": 64, "wholecap": 262144,
        "buckets": d.DEFAULT_BUCKETS}


def test_plus_joined_options_parse_like_the_comma_form():
    assert _opts(_raw_from_driver(PLUS)) == WANT


def test_comma_form_unchanged():
    assert _opts(_raw_from_driver(COMMA)) == WANT


def test_bare_enable_keeps_whole_off():
    o = _opts(_raw_from_driver("1"))
    assert o["whole"] is False and o["wholeunion"] is False and o["wholecap"] is None


def test_plus_buckets_token_parses():
    raw = _raw_from_driver("1+buckets=16:32:48:64+whole=1")
    assert d._parse_buckets(raw) == (16, 32, 48, 64) and d._parse_flag(raw, "whole", False) is True
