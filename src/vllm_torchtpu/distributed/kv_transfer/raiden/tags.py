# SPDX-License-Identifier: Apache-2.0
"""Pool-tag vocabulary shared with the Raiden controller.

These strings are wire-visible: they name pools in manifests and byte-span
registrations on both sides of a disagg pair, and the live-pair acceptance
gate builds its expected tag list from them. The values are frozen; changing
one is a coupled two-repo protocol change, not a refactor.
"""
TAG_FA = "fa"
TAG_GDN_CONV = "gdn.conv"
TAG_GDN_SSM = "gdn.ssm"
TAG_DSA_IDX = "dsa.idx"
