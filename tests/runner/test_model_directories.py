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
"""Every package under vllm_torchtpu.models registers itself."""


def test_model_directories_register_themselves(tmp_path, monkeypatch) -> None:
    import vllm_torchtpu.models as models
    from vllm_torchtpu.models.vllm import register_model_directories

    calls = []
    pkg = tmp_path / "_test_model_dir"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(
        "CALLS = []\ndef register():\n    CALLS.append('registered')\n"
    )
    (tmp_path / "_test_readme_only").mkdir()  # no __init__: not a package
    monkeypatch.setattr(models, "__path__", [*models.__path__, str(tmp_path)])

    register_model_directories()

    import vllm_torchtpu.models._test_model_dir as registered

    calls.extend(registered.CALLS)
    assert calls == ["registered"]
