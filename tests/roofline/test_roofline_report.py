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
"""CPU unit tests for roofline HTML report generation."""

from __future__ import annotations

import json
import re
from typing import Any

import pytest

from vllm_torchtpu.roofline import (
    CHIPS,
    Dtypes,
    TrafficScenario,
    build_solve_payload,
    solve_serving_plan,
)
from vllm_torchtpu.roofline.geometry import from_hf_config
from vllm_torchtpu.roofline.report import (
    MAX_EMBEDDED_CANDIDATES,
    render_html,
)


@pytest.fixture
def sample_payload() -> dict[str, Any]:
    cfg = {
        "model_type": "llama",
        "torch_dtype": "bfloat16",
        "num_hidden_layers": 8,
        "hidden_size": 1024,
        "intermediate_size": 2048,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "vocab_size": 32000,
    }
    geom = from_hf_config(cfg)
    chip = CHIPS["tpu7x"]
    dt = Dtypes(kv=1.0)
    scenario = TrafficScenario(
        name="chat-8k1k",
        isl=2048,
        osl=256,
        prefix_hit_rate=0.0,
        accept_len=1.0,
        think_time_s=0.0,
        concurrencies=(1, 2),
        kv_working_set_turns=1.0,
    )
    plan = solve_serving_plan(
        geom,
        dt,
        chip,
        scenario,
        max_chips=4,
        mode="mixed",
    )
    return build_solve_payload(plan, chip=chip, dtypes=dt, model="test-llama", top_k=3)


def test_report_html_is_deterministic(sample_payload: dict[str, Any]) -> None:
    """render_html produces byte-identical output across consecutive calls."""
    html1 = render_html(sample_payload)
    html2 = render_html(sample_payload)
    assert html1 == html2
    assert len(html1) > 500


def test_report_has_no_external_fetch(sample_payload: dict[str, Any]) -> None:
    """HTML report is fully offline: zero external URLs, stylesheets, or scripts."""
    html = render_html(sample_payload)
    assert "<link" not in html
    assert not re.search(r"<script[^>]+src=", html, re.IGNORECASE)
    assert "https://" not in html
    # Check that any http:// URL is solely the standard SVG XML namespace
    found_urls = re.findall(r"http://[a-zA-Z0-9./_-]+", html)
    assert set(found_urls).issubset({"http://www.w3.org/2000/svg"})


def test_report_escapes_untrusted_strings(sample_payload: dict[str, Any]) -> None:
    """Model names and warning text with HTML or script tags are properly escaped."""
    payload = dict(sample_payload)
    payload["model"] = '<script>alert("xss")</script>'
    payload["warnings"] = ["<b>Unsafe warning</b>", "Normal warning"]
    html = render_html(payload)
    assert '<script>alert("xss")</script>' not in html
    assert (
        "&lt;script&gt;alert(&quot;xss&quot;)&lt;/script&gt;" in html
        or "&lt;script&gt;" in html
    )
    assert "<b>Unsafe warning</b>" not in html
    assert "&lt;b&gt;Unsafe warning&lt;/b&gt;" in html
    # JSON script tag must not contain literal </script>
    match = re.search(
        r'<script type="application/json"[^>]*>(.*?)</script>', html, re.DOTALL
    )
    assert match is not None
    json_blob = match.group(1)
    assert "</" not in json_blob or r"<\/" in json_blob


def test_report_sections_present(sample_payload: dict[str, Any]) -> None:
    """All 7 required functional sections and anchors are present."""
    html = render_html(sample_payload)
    assert 'id="executive-summary"' in html
    assert 'id="search-funnel"' in html
    assert 'id="frontier-chart"' in html
    assert 'id="limits-chart"' in html
    assert 'id="memory-chart"' in html
    assert 'id="top-layouts-table"' in html
    assert 'id="roofline-data"' in html
    assert "<svg" in html


def test_report_without_feasible_layout(sample_payload: dict[str, Any]) -> None:
    """When no layout is feasible, renders warning banner and empty state gracefully."""
    payload = dict(sample_payload)
    payload["recommended"] = None
    payload["top_k"] = []
    payload["warnings"] = ["SLO_UNMET: no feasible topology met latency constraints"]
    html = render_html(payload)
    assert "SLO_UNMET" in html
    assert "No feasible layout met constraints" in html or "SLO_UNMET" in html
    assert "<svg" in html


def test_report_caps_embedded_candidates(sample_payload: dict[str, Any]) -> None:
    """Payloads with large candidate counts downsample embedded JSON to
    MAX_EMBEDDED_CANDIDATES.
    """
    payload = dict(sample_payload)
    base_cand = payload["candidates"][0] if payload["candidates"] else {}
    # Fake 3000 candidates
    payload["candidates"] = [dict(base_cand) for _ in range(3000)]
    html = render_html(payload)
    match = re.search(
        r'<script type="application/json"[^>]*>(.*?)</script>', html, re.DOTALL
    )
    assert match is not None
    raw_json = match.group(1).replace(r"<\/", "</")
    embedded = json.loads(raw_json)
    assert len(embedded["candidates"]) <= MAX_EMBEDDED_CANDIDATES


def test_report_slo_unmet_recommended_renders_warning_banner(
    sample_payload: dict[str, Any],
) -> None:
    html_ok = render_html(sample_payload)
    assert "max_throughput" in html_ok
    assert "<th>Role</th>" in html_ok
    assert "<th>KV Cap / Hit</th>" in html_ok

    payload = dict(sample_payload)
    assert sample_payload["recommended"] is not None
    rec_unmet = dict(sample_payload["recommended"])
    rec_unmet["slo_met"] = False
    payload["recommended"] = rec_unmet
    payload["top_k"] = []
    payload["warnings"] = []
    html_unmet = render_html(payload)
    assert "INFEASIBLE" in html_unmet
    assert '<div class="alert-box alert-warning">' in html_unmet
    assert "recommended layout is infeasible" in html_unmet
