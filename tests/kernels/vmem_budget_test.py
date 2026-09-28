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
"""The shared Pallas VMEM budget must never over-claim the chip.

These tests deliberately avoid touching a real accelerator: the physical
capacity probe is mocked so the per-generation arithmetic can be checked
from any host, including CPU-only CI.
"""

import os
from unittest import mock

from absl.testing import absltest, parameterized

from vllm_torchtpu.kernels import vmem_budget

# Capacities measured on real hardware.
V6E_CAPACITY = 134217728  # 128.00 MiB
V7X_CAPACITY = 67043328  # 63.94 MiB


class VmemBudgetTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        self.enterContext(mock.patch.dict(os.environ, {}, clear=False))
        os.environ.pop("TPU_KERNEL_VMEM_LIMIT_BYTES", None)

    @parameterized.named_parameters(
        # v6e has more VMEM than the historic constant claimed, so the clamp
        # holds it at 100 MiB and behaviour is bit-for-bit unchanged.
        ("v6e", V6E_CAPACITY, vmem_budget.LEGACY_VMEM_LIMIT_BYTES),
        # v7x has less, so the old 100 MiB constant was a 36 MiB lie. The
        # budget now tracks the hardware exactly.
        ("v7x", V7X_CAPACITY, V7X_CAPACITY),
    )
    def test_compiler_limit_tracks_hardware(self, capacity, expected):
        with mock.patch.object(
            vmem_budget, "physical_vmem_bytes", return_value=capacity
        ):
            self.assertEqual(vmem_budget.compiler_vmem_limit_bytes(), expected)

    @parameterized.named_parameters(
        ("v6e", V6E_CAPACITY),
        ("v7x", V7X_CAPACITY),
    )
    def test_compiler_limit_never_exceeds_capacity(self, capacity):
        # The invariant Trial F violated: a program that fits the chip must
        # never be rejected, and one that does not must not be promised room.
        with mock.patch.object(
            vmem_budget, "physical_vmem_bytes", return_value=capacity
        ):
            self.assertLessEqual(vmem_budget.compiler_vmem_limit_bytes(), capacity)

    @parameterized.named_parameters(
        ("v6e", V6E_CAPACITY),
        ("v7x", V7X_CAPACITY),
    )
    def test_planning_budget_leaves_headroom(self, capacity):
        # Analytic vmem_need estimates cannot see VREG spills, so the
        # planning budget must sit strictly below the compiler ceiling
        # whenever the ceiling is the hardware itself.
        with mock.patch.object(
            vmem_budget, "physical_vmem_bytes", return_value=capacity
        ):
            planning = vmem_budget.planning_vmem_budget_bytes()
            self.assertLess(planning, capacity)
            self.assertLessEqual(planning, vmem_budget.compiler_vmem_limit_bytes())

    def test_env_override_wins(self):
        os.environ["TPU_KERNEL_VMEM_LIMIT_BYTES"] = str(7 * vmem_budget.MIB)
        with mock.patch.object(
            vmem_budget, "physical_vmem_bytes", return_value=V7X_CAPACITY
        ):
            self.assertEqual(
                vmem_budget.compiler_vmem_limit_bytes(), 7 * vmem_budget.MIB
            )
            self.assertEqual(
                vmem_budget.planning_vmem_budget_bytes(), 7 * vmem_budget.MIB
            )

    def test_resolve_prefers_explicit_caller_value(self):
        self.assertEqual(vmem_budget.resolve_compiler_limit(1234), 1234)
        self.assertEqual(
            vmem_budget.resolve_compiler_limit(None),
            vmem_budget.compiler_vmem_limit_bytes(),
        )

    def test_probe_falls_back_off_tpu(self):
        # get_tpu_info raises on CPU hosts; the budget must still resolve.
        vmem_budget.physical_vmem_bytes.cache_clear()
        try:
            self.assertGreater(vmem_budget.physical_vmem_bytes(), 0)
        finally:
            vmem_budget.physical_vmem_bytes.cache_clear()


if __name__ == "__main__":
    absltest.main()
