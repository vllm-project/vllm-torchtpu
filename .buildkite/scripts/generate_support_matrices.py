#!/usr/bin/env python3
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

"""Generates support matrices for models and features across TPU hardware."""

import csv
import os
import re
import shutil
import subprocess
import sys


def get_buildkite_metadata(key: str, default: str = "") -> str:
    """Retrieves a metadata value from buildkite-agent if available."""
    try:
        result = subprocess.run(
            ["buildkite-agent", "meta-data", "get", key, "--default", default],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.rstrip("\r\n")
    except (subprocess.SubprocessError, FileNotFoundError):
        return default


def set_buildkite_metadata(key: str, value: str) -> None:
    """Sets a metadata value using buildkite-agent."""
    try:
        subprocess.run(
            ["buildkite-agent", "meta-data", "set", key, value],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        print(f"Warning: Could not set metadata '{key}': {e}", file=sys.stderr)


def upload_buildkite_artifact(file_path: str) -> None:
    """Uploads an artifact using buildkite-agent."""
    try:
        subprocess.run(
            ["buildkite-agent", "artifact", "upload", file_path],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        print(f"Warning: Could not upload artifact '{file_path}': {e}", file=sys.stderr)


def get_tpu_generation(key: str) -> str:
    """Maps quantization dtype to supported TPU generations."""
    mapping = {
        "INT8 W8A8": "v6",
        "INT4 W4A16": "v6",
        "FP8 W8A8": "v7",
        "FP8 W8A16": "v7",
        "FP4 W4A16": "v7",
        "NVFP4 W4A16": "v7",
    }
    return mapping.get(key, "N/A")


def get_quantization_method(key: str) -> str:
    """Maps quantization dtype to quantization method."""
    mapping = {
        "INT8 W8A8": "compressed-tensor",
        "INT4 W4A16": "awq",
        "FP8 W8A8": "compressed-tensor",
        "FP8 W8A16": "compressed-tensor",
        "FP4 W4A16": "mxfp4",
        "NVFP4 W4A16": "modelopt_fp4",
    }
    return mapping.get(key, "N/A")


def format_status(raw: str) -> str:
    """Formats custom raw status strings into emoji status labels."""
    normalized = raw.strip().lower()
    if normalized == "beta":
        return "⚠️ Beta"
    if normalized == "experimental":
        return "🧪 Experimental"
    if normalized == "planned":
        return "📝 Planned"
    if normalized == "unplanned":
        return "⛔️ Unplanned"
    return raw


def main() -> None:
    tpu_version = os.environ.get("TPU_VERSION", "tpu6e")
    if tpu_version.startswith("v7") or "7x" in tpu_version or "tpu7" in tpu_version:
        tpu_dir = "v7x"
        tpu_metadata_prefix = "v7"
    else:
        tpu_dir = "v6e"
        tpu_metadata_prefix = "v6"

    os.makedirs(tpu_dir, exist_ok=True)
    print(f"Output directory set to: {tpu_dir} (Prefix: '{tpu_metadata_prefix}')")

    any_failed = False

    # 1. Process Models
    raw_models = get_buildkite_metadata("model-list", "")
    model_list = [m.strip() for m in raw_models.splitlines() if m.strip()]
    model_stages = ["Type", "UnitTest", "Accuracy/Correctness", "Benchmark"]

    if model_list:
        model_csv_path = os.path.join(tpu_dir, "model_support_matrix.csv")
        model_rows = []

        valid_outcomes = {"✅ Passing", "⚪ N/A", "❓ Untested", "not enough HBM"}

        for model in model_list:
            category = get_buildkite_metadata(
                f"{tpu_metadata_prefix}{model}_category", default="text-only"
            )

            row: dict[str, str] = {"Model": model}

            if category == "multimodal":
                row["Type"] = "Multimodal"
            elif category == "embedding":
                row["Type"] = "Embedding"
            elif category == "diffusion":
                row["Type"] = "Diffusion"
            else:
                row["Type"] = "Text"

            for stage in ["UnitTest", "Accuracy/Correctness", "Benchmark"]:
                result = get_buildkite_metadata(
                    f"{tpu_metadata_prefix}{model}:{stage}", default="❓ Untested"
                )
                row[stage] = result
                if result not in valid_outcomes:
                    any_failed = True

            model_rows.append(row)

        # Sort by Type, then Model
        model_rows.sort(key=lambda r: (r["Type"], r["Model"]))

        with open(model_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["Model"] + model_stages)
            writer.writeheader()
            writer.writerows(model_rows)

        print(f"--- Uploading Model Matrix: {model_csv_path} ---")
        with open(model_csv_path, encoding="utf-8") as f:
            print(f.read())
        upload_buildkite_artifact(model_csv_path)

    # 2. Process Default Features
    default_features_file = ".buildkite/features/default_features.txt"
    default_features: list[tuple[str, str]] = []
    if os.path.isfile(default_features_file):
        with open(default_features_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                match = re.match(r"^(.+) \((.+)\)$", line)
                if match:
                    feat_name, cat = match.group(1).strip(), match.group(2).strip()
                    default_features.append((feat_name, cat))
                    set_buildkite_metadata(
                        f"{tpu_metadata_prefix}{feat_name}_category", cat
                    )
                else:
                    default_features.append((line, "feature support matrix"))

    # 3. Process Features
    raw_features = get_buildkite_metadata("feature-list", "")
    metadata_features = [f.strip() for f in raw_features.splitlines() if f.strip()]

    feature_groups: dict[
        str, list[tuple[str, str]]
    ] = {}  # category -> list of (feature, mode)
    for feat, cat in default_features:
        feature_groups.setdefault(cat, []).append((feat, "DEFAULT"))
    for feat in metadata_features:
        cat = get_buildkite_metadata(
            f"{tpu_metadata_prefix}{feat}_category", default="feature support matrix"
        )
        feature_groups.setdefault(cat, []).append((feat, "METADATA"))

    valid_feature_outcomes = {
        "✅ Passing",
        "⚪ N/A",
        "❓ Untested",
        "⚠️ Beta",
        "🧪 Experimental",
        "📝 Planned",
        "⛔️ Unplanned",
    }

    feature_csv_files = []
    for category, features in feature_groups.items():
        category_filename = category.replace(" ", "_")
        category_csv = os.path.join(tpu_dir, f"{category_filename}.csv")

        if category == "quantization support matrix":
            headers = [
                "Quantization dtype",
                "Quantization methods",
                "Recommended TPU Generations",
                "CorrectnessTest",
                "PerformanceTest",
            ]
            stages = [
                "QuantizationMethods",
                "RecommendedTPUGenerations",
                "CorrectnessTest",
                "PerformanceTest",
            ]
        elif category == "kernel support matrix microbenchmarks":
            headers = ["kernels", "CorrectnessTest", "PerformanceTest"]
            stages = ["CorrectnessTest", "PerformanceTest"]
        elif category == "parallelism support matrix":
            headers = [
                "Feature",
                "Single-Host CorrectnessTest",
                "Single-Host PerformanceTest",
                "Multi-Host CorrectnessTest",
                "Multi-Host PerformanceTest",
            ]
            stages = [
                "Single-Host CorrectnessTest",
                "Single-Host PerformanceTest",
                "Multi-Host CorrectnessTest",
                "Multi-Host PerformanceTest",
            ]
        else:
            headers = ["Feature", "CorrectnessTest", "PerformanceTest"]
            stages = ["CorrectnessTest", "PerformanceTest"]

        rows = []
        for feature, mode in features:
            row = {headers[0]: feature}
            for stage in stages:
                if stage == "RecommendedTPUGenerations":
                    row[stage] = get_tpu_generation(feature)
                elif stage == "QuantizationMethods":
                    row[stage] = get_quantization_method(feature)
                elif mode == "DEFAULT":
                    row[stage] = "✅ Passing"
                else:
                    raw_res = get_buildkite_metadata(
                        f"{tpu_metadata_prefix}{feature}:{stage}", default="❓ Untested"
                    )
                    formatted_res = format_status(raw_res)
                    row[stage] = formatted_res
                    if formatted_res not in valid_feature_outcomes:
                        any_failed = True

            rows.append(row)

        rows.sort(key=lambda r: r[headers[0]])

        with open(category_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)

        feature_csv_files.append((category_csv, category))

    # Upload feature matrices (skip raw kernel microbenchmarks, pivot below)
    for csv_file, cat in feature_csv_files:
        if "kernel_support_matrix_microbenchmarks.csv" in csv_file:
            print(f"Skipping direct upload for {csv_file} (will be pivoted later).")
            continue
        print(f"--- Uploading Feature Matrix: {csv_file} ---")
        with open(csv_file, encoding="utf-8") as f:
            print(f.read())
        upload_buildkite_artifact(csv_file)

    # 4. Pivot Logic (Microbenchmarks)
    kernel_input_csv = os.path.join(
        tpu_dir, "kernel_support_matrix_microbenchmarks.csv"
    )
    kernel_output_csv = os.path.join(
        tpu_dir, "kernel_support_matrix-microbenchmarks.csv"
    )
    if os.path.isfile(kernel_input_csv):
        quant_order = ["w16a16", "w8a8", "w8a16", "w4a4", "w4a8", "w4a16"]
        pivot_headers = [
            "Kernel",
            "W16 A16 (Corr)",
            "W16 A16 (Perf)",
            "W8 A8 (Corr)",
            "W8 A8 (Perf)",
            "W8 A16 (Corr)",
            "W8 A16 (Perf)",
            "W4 A4 (Corr)",
            "W4 A4 (Perf)",
            "W4 A8 (Corr)",
            "W4 A8 (Perf)",
            "W4 A16 (Corr)",
            "W4 A16 (Perf)",
        ]

        matrix: dict[tuple[str, str], tuple[str, str]] = {}
        kernel_names: list[str] = []

        with open(kernel_input_csv, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                feat = r.get("kernels", "")
                corr = r.get("CorrectnessTest", "❓ Untested")
                perf = r.get("PerformanceTest", "❓ Untested")

                m = re.search(r"-(w\d+a\d+)$", feat)
                if m:
                    quant_type = m.group(1)
                    base_kernel = feat[: m.start()]
                else:
                    base_kernel = feat
                    quant_type = "w16a16"

                matrix[(base_kernel, quant_type)] = (corr, perf)
                if base_kernel not in kernel_names:
                    kernel_names.append(base_kernel)

        pivot_rows = []
        for k in kernel_names:
            out_name = k
            if out_name == "generic ragged paged attention v3":
                out_name = "generic ragged paged<br>attention v3*"
            elif out_name == "mla":
                out_name = "mla*"
            elif out_name == "ragged paged attention v3 head_dim 64":
                out_name = "ragged paged attention v3<br>head_dim 64*"

            row_data = [out_name]
            for q in quant_order:
                corr, perf = matrix.get((k, q), ("❓ Untested", "❓ Untested"))
                row_data.extend([corr, perf])
            pivot_rows.append(row_data)

        with open(kernel_output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(pivot_headers)
            writer.writerows(pivot_rows)

        print(f"--- Uploading Pivoted Kernel Matrix: {kernel_output_csv} ---")
        with open(kernel_output_csv, encoding="utf-8") as f:
            print(f.read())
        upload_buildkite_artifact(kernel_output_csv)

    # 5. Record overall failure status for notify_test_results.sh
    failure_str = "true" if any_failed else "false"
    set_buildkite_metadata(f"{tpu_metadata_prefix}_CI_TESTS_FAILED", failure_str)
    print(f"Support matrix generation finished. ANY_FAILED={failure_str}")

    # Cleanup
    shutil.rmtree(tpu_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
