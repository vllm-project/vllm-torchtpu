# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project
"""Idempotent installer to wire CUJ-2 telemetry into any vllm-torchtpu container tree."""

from pathlib import Path


def install_into_tree(pkg_root: Path | None = None) -> None:
    """Wire `VLLM_TPU_AGENTIC_TELEMETRY` and `install_agentic_telemetry_patches` in-place."""
    if pkg_root is None:
        pkg_root = Path(__file__).resolve().parent.parent

    envs_path = pkg_root / "envs.py"
    if envs_path.exists():
        envs_text = envs_path.read_text(encoding="utf-8")
        if '"VLLM_TPU_AGENTIC_TELEMETRY"' not in envs_text:
            anchor = "environment_variables: dict[str, Callable[[], Any]] = {\n"
            addition = (
                '    "VLLM_TPU_AGENTIC_TELEMETRY": env_bool("VLLM_TPU_AGENTIC_TELEMETRY", default=False),\n'
                '    "VLLM_TPU_AGENTIC_SPAN_LOG": env_str("VLLM_TPU_AGENTIC_SPAN_LOG", default=""),\n'
            )
            if anchor in envs_text:
                envs_path.write_text(
                    envs_text.replace(anchor, anchor + addition, 1),
                    encoding="utf-8",
                )

    reg_path = pkg_root / "patch_registry.py"
    if reg_path.exists():
        reg_text = reg_path.read_text(encoding="utf-8")
        target_str = "vllm_torchtpu.telemetry.vllm_hooks:install_agentic_telemetry_patches"
        if target_str not in reg_text:
            anchor = '        "vllm_torchtpu.platforms.tpu_platform:_patch_api_server_kernel_reload_endpoint",\n        _ACTIVE,\n    ),\n'
            addition = (
                "    Patch(\n"
                f'        "{target_str}",\n'
                "        _ENGINE,\n"
                "    ),\n"
            )
            if anchor in reg_text:
                reg_path.write_text(
                    reg_text.replace(anchor, anchor + addition, 1),
                    encoding="utf-8",
                )


if __name__ == "__main__":
    install_into_tree()
    print("CUJ-2 agentic telemetry wired into vllm_torchtpu.")
