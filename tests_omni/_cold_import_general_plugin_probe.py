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
"""Cold-import probe for the vllm_omni.general_plugins entry point."""


def main() -> None:
    import vllm_torchtpu.omni as torchtpu_omni

    torchtpu_omni.get_num_chips = lambda: 8
    import vllm_omni.diffusion.stage_diffusion_proc  # noqa: F401
    from vllm_omni.plugins import load_omni_general_plugins

    load_omni_general_plugins()

    from vllm_omni.diffusion.diffusion_engine import DiffusionEngine

    from vllm_torchtpu.omni.patches import _WARMUP_SYNC_PATCHED

    assert getattr(DiffusionEngine.run_startup_warmup, _WARMUP_SYNC_PATCHED, False)


if __name__ == "__main__":
    main()
