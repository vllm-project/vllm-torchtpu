# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project
"""Idempotent installer to wire CUJ-2 telemetry into any vllm-torchtpu container tree."""

from pathlib import Path

PINNED_OTEL_PACKAGES: tuple[str, ...] = (
    "opentelemetry-api==1.27.0",
    "opentelemetry-sdk==1.27.0",
    "opentelemetry-exporter-otlp-proto-http==1.27.0",
    "opentelemetry-exporter-otlp-proto-grpc==1.27.0",
)


def install_into_tree(pkg_root: Path | None = None) -> None:
    """Wire `VLLM_TPU_AGENTIC_TELEMETRY` and `install_agentic_telemetry_patches` in-place.

    Raises `RuntimeError` if target files or expected anchors are missing.
    """
    if pkg_root is None:
        pkg_root = Path(__file__).resolve().parent.parent

    envs_path = pkg_root / "envs.py"
    if not envs_path.exists():
        raise RuntimeError(f"Missing target file: {envs_path}")
    envs_text = envs_path.read_text(encoding="utf-8")
    if '"VLLM_TPU_AGENTIC_TELEMETRY"' not in envs_text:
        anchor = "environment_variables: dict[str, Callable[[], Any]] = {\n"
        if anchor not in envs_text:
            raise RuntimeError(
                f"Failed to locate environment_variables anchor in {envs_path}"
            )
        addition = (
            '    "VLLM_TPU_AGENTIC_TELEMETRY": env_bool("VLLM_TPU_AGENTIC_TELEMETRY", default=False),\n'
            '    "VLLM_TPU_AGENTIC_SPAN_LOG": env_str("VLLM_TPU_AGENTIC_SPAN_LOG", default=""),\n'
        )
        envs_path.write_text(
            envs_text.replace(anchor, anchor + addition, 1),
            encoding="utf-8",
        )

    reg_path = pkg_root / "patch_registry.py"
    if not reg_path.exists():
        raise RuntimeError(f"Missing target file: {reg_path}")
    reg_text = reg_path.read_text(encoding="utf-8")
    target_str = (
        "vllm_torchtpu.telemetry.vllm_hooks:install_agentic_telemetry_patches"
    )
    if target_str not in reg_text:
        anchor = '        "vllm_torchtpu.platforms.tpu_platform:_patch_api_server_kernel_reload_endpoint",\n        _ACTIVE,\n    ),\n'
        if anchor not in reg_text:
            raise RuntimeError(
                f"Failed to locate PATCHES anchor in {reg_path}"
            )
        addition = (
            "    Patch(\n"
            f'        "{target_str}",\n'
            "        _ENGINE,\n"
            "    ),\n"
        )
        reg_path.write_text(
            reg_text.replace(anchor, anchor + addition, 1),
            encoding="utf-8",
        )


if __name__ == "__main__":
    install_into_tree()
    print("CUJ-2 agentic telemetry wired into vllm_torchtpu.")
