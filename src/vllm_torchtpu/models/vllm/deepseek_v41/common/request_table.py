# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-token lookups into per-request tables."""

import torch


def per_request(values: torch.Tensor, requests: torch.Tensor) -> torch.Tensor:
    """``values[requests]`` for a 1-D table with one entry per request.

    XLA expands a gather from a table this small into a chain of selects over
    scalars it keeps in HBM: ~30 us per lookup on v7, ~16 lookups per decode
    step. A one-hot sum is one vector compare and a lane reduction (~1 us) and
    exact: every row adds one table entry to zeros.

    Args:
      values: [requests] table.
      requests: request index per token, each in [0, values.numel()).

    Returns:
      ``values[requests]``, with ``requests``' shape and ``values``' dtype.
    """
    entries = torch.arange(values.shape[0], device=values.device)
    hits = requests.unsqueeze(-1) == entries
    return torch.where(hits, values, 0).sum(-1, dtype=values.dtype)
