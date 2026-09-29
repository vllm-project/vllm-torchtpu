# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Topology-aware rank ordering logic for Context Parallelism on TPU.

Strictly zero dependencies on vLLM.
"""

from __future__ import annotations

import numpy as np


def build_topology_mesh_groups(
    mesh: np.ndarray, pcp_size: int, tp_size: int
) -> dict[str, list[list[int]]]:
    """Extract PCP rings and TP lanes from a 3D topology mesh.

    Keyed by group name:
    - 'pcp': rings along the pcp axis
    - 'tp': lanes along the tp axis
    """
    return {
        "pcp": [
            [int(r) for r in ring]
            for ring in np.swapaxes(mesh, 1, 2).reshape(-1, pcp_size)
        ],
        "tp": [
            [int(r) for r in lane] for lane in np.asarray(mesh).reshape(-1, tp_size)
        ],
    }


def validate_mesh_permutation(
    group_ranks_by_name: dict[str, list[list[int]]],
    expected_ranks: list[int],
    mesh_shape: tuple[int, ...],
) -> None:
    """Validate that every axis group partitions the expected ranks cleanly."""
    for axis, groups in group_ranks_by_name.items():
        flat = sorted(r for group in groups for r in group)
        if flat != expected_ranks:
            raise RuntimeError(
                f"topology_aware_mesh({mesh_shape}) produced {axis} groups "
                f"{groups} whose ranks {flat} are not a permutation of "
                f"{expected_ranks}."
            )
