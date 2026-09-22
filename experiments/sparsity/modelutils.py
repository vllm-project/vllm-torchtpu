"""
modelutils.py
-------------
PyTorch model inspection and layer traversal utilities for extracting linear projection weights.
"""

import torch
import torch.nn as nn

DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def find_layers(module, layers=None, name=""):
    """
    Recursively scans a PyTorch nn.Module hierarchy and returns a flat dictionary
    mapping dot-separated layer names to their corresponding target submodules (e.g. nn.Linear).
    Supports subclasses via isinstance.
    """
    if layers is None:
        layers = [nn.Linear]
    if any(isinstance(module, layer_cls) for layer_cls in layers):
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_layers(child,
                        layers=layers,
                        name=name + "." + name1 if name != "" else name1))
    return res
