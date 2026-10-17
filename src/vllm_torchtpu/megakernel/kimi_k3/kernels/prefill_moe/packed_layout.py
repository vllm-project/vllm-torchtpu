# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Move whole packed token rows through the twelve ring streams."""


class _PackedView:
    """Present packed rows as a logical matrix to the ring schedule.

    Every column slice moves both packing halves. This is a Ref view adapter,
    not an array reshape or an HBM conversion.
    """

    def __init__(self, ref):
        self.ref = ref
        self.shape = (*ref.shape[:-2], ref.shape[-1])

    @property
    def at(self):
        return self

    def __getitem__(self, index):
        return self.ref.at[(*index[:-1], slice(None), index[-1])]


def row_regions(rows, half):
    """Stripe whole packed rows across the 12 directional streams."""
    sizes = [rows // 12 + (s < rows % 12) for s in range(12)]
    begin = 0
    result = []
    for size in sizes:
        result.append((begin, size, 0, half))
        begin += size
    return tuple(result)
