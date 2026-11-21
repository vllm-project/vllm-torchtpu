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
"""The ``tpu_scheduler`` additional_config section."""

import pathlib
from types import SimpleNamespace

import pytest
import yaml

from vllm_torchtpu import additional_config_sections
from vllm_torchtpu.core import scheduler_options
from vllm_torchtpu.models import qwen3_5
from vllm_torchtpu.runner.utils import HASH_IGNORED_ADDITIONAL_CONFIG_KEYS


def test_defaults_are_mains_behaviour():
    assert scheduler_options.parse(None) == scheduler_options.SchedulerOptions()
    opts = scheduler_options.parse({})
    assert opts.promote_warmest is True
    # The decode levers are opt-in.
    assert opts.kv_prefetch is False
    assert opts.decode_inflight_throttle is False
    assert scheduler_options.get(SimpleNamespace(additional_config=None)) == opts


def test_a_config_without_additional_config_fails_loudly():
    """VllmConfig always has additional_config; a config object without it is
    an error, not a run on the defaults."""
    with pytest.raises(AttributeError, match="additional_config"):
        scheduler_options.get(SimpleNamespace())


def test_parse_reads_the_options():
    cfg = SimpleNamespace(
        additional_config={
            "tpu_scheduler": {
                "promote_warmest": False,
                "kv_prefetch": True,
                "decode_inflight_throttle": True,
            }
        }
    )
    opts = scheduler_options.get(cfg)
    assert opts.promote_warmest is False
    assert opts.kv_prefetch is True
    assert opts.decode_inflight_throttle is True


@pytest.mark.parametrize(
    "raw, match",
    [
        ({"promote": False}, "unknown keys"),
        ({"promote_warmest": 0}, "must be a bool"),
        ({"promote_warmest": "false"}, "must be a bool"),
        ({"kv_prefetch": 1}, "must be a bool"),
        ({"decode_inflight_throttle": "true"}, "must be a bool"),
        ([], "must be a JSON object"),
    ],
)
def test_parse_is_strict(raw, match):
    with pytest.raises(ValueError, match=match):
        scheduler_options.parse(raw)


def test_section_is_registered_and_kept_out_of_the_compile_cache_key():
    assert scheduler_options.SECTION in additional_config_sections.KNOWN_SECTIONS
    assert scheduler_options.SECTION in HASH_IGNORED_ADDITIONAL_CONFIG_KEYS


def _decode_legs():
    folder = pathlib.Path(qwen3_5.__file__).parent / "validation"
    for path in sorted(folder.glob("p[0-9][0-9]-*.yaml")):
        with path.open() as f:
            entry = yaml.safe_load(f)
        yield path.stem, entry["point"], entry["legs"]["decode"]


def _decode_context_parallel_size(serve_args):
    flag = "--decode-context-parallel-size"
    return int(serve_args[serve_args.index(flag) + 1]) if flag in serve_args else 1


def test_the_decode_legs_turn_on_the_levers_the_record_jobs_ran():
    seen = []
    for name, point, leg in _decode_legs():
        opts = scheduler_options.get(
            SimpleNamespace(additional_config=leg["additional_config"])
        )
        dcp = _decode_context_parallel_size(leg["serve_args"])
        # Every decode leg prefetched, the DCP 4 legs throttled, and
        # warmest-first promotion ran at #1 only.
        assert opts.kv_prefetch, name
        assert opts.decode_inflight_throttle is (dcp > 1), name
        assert opts.promote_warmest is (point == 1), name
        seen.append(name)
    assert len(seen) == 5
