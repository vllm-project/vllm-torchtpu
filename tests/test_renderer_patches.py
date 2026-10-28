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
"""Chat-template content-format detection with namespace aliases."""

import pytest

from vllm_torchtpu.renderer_patches import patch_chat_template_namespace_aliases


@pytest.fixture
def hf(monkeypatch):
    from vllm.renderers import hf

    monkeypatch.setattr(hf, "_is_var_access", hf._is_var_access)
    monkeypatch.setattr(
        hf, "_iter_nodes_assign_var_or_elems", hf._iter_nodes_assign_var_or_elems
    )
    hf._detect_content_format.cache_clear()
    patch_chat_template_namespace_aliases()
    yield hf
    hf._detect_content_format.cache_clear()


@pytest.mark.parametrize(
    "chat_template",
    [
        """
        {% set ns = namespace(rest=messages) %}
        {% for message in ns.rest %}
          {% for item in message['content'] %}{{ item['type'] }}{% endfor %}
        {% endfor %}
        """,
        """
        {% set ns = namespace(rest=[]) %}
        {% set ns.rest = messages %}
        {% for message in ns.rest %}
          {% for item in message['content'] %}{{ item['type'] }}{% endfor %}
        {% endfor %}
        """,
        """
        {% set ns = namespace(rest=messages) %}
        {% set ns.rest = messages[1:] %}
        {% for message in messages %}
          {% for part in message.content %}{{ part.text }}{% endfor %}
        {% endfor %}
        """,
    ],
)
def test_namespace_aliases_detect_openai_content(hf, chat_template):
    assert hf._detect_content_format(chat_template, default="string") == "openai"


def test_namespace_alias_of_string_content_stays_string(hf):
    chat_template = """
    {% set ns = namespace(rest=messages) %}
    {% for message in ns.rest %}{{ message['content'] }}{% endfor %}
    """
    assert hf._detect_content_format(chat_template, default="openai") == "string"
