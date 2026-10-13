# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp (Qwen3.8 Flash Next) on TPU.

Kept import-free: plugin registration imports ``config`` at startup, and the
registry imports ``model`` lazily by its module path.
"""
