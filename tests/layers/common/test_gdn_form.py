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
"""The pooled GDN entry follows TPU_GDN_TABLES_IN_KERNEL, read through envs."""
import pytest

from vllm_torchtpu import envs
from vllm_torchtpu.layers.common import gdn_attention

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("value,form", [(None, "two_calls"), ("", "two_calls"),
                                        ("0", "two_calls"),
                                        ("false", "two_calls"),
                                        ("1", "tables"), ("true", "tables")])
def test_gdn_form_follows_the_flag(monkeypatch, value, form):
    """Off unless the variable enables it, as envs.py declares."""
    if value is None:
        monkeypatch.delenv("TPU_GDN_TABLES_IN_KERNEL", raising=False)
    else:
        monkeypatch.setenv("TPU_GDN_TABLES_IN_KERNEL", value)
    assert envs.TPU_GDN_TABLES_IN_KERNEL is (form == "tables")
    assert gdn_attention._gdn_form() == form


@pytest.mark.parametrize("tables", [True, False])
def test_gdn_form_follows_envs(monkeypatch, tables):
    """The envs value selects the entry, also when tests set it on envs
    with the variable unset."""
    monkeypatch.delenv("TPU_GDN_TABLES_IN_KERNEL", raising=False)
    monkeypatch.setattr(envs, "TPU_GDN_TABLES_IN_KERNEL", tables)
    assert gdn_attention._gdn_form() == ("tables" if tables else "two_calls")
