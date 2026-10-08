import math
import time
from typing import Optional, Tuple

import torch
import torch.nn as nn
import transformers

from .quant import quantize

DEBUG = False

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class SparseGPT:

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out, blocksize=1024):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(
                self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def _clone_weight_matrix(self) -> torch.Tensor:
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()
        if hasattr(self, "quantizer"):
            if not self.quantizer.ready():
                self.quantizer.find_params(W, weight=True)
        return W

    def _factorize_hessian(
        self,
        W: torch.Tensor,
        percdamp: float = .01,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Consumes ``self.H``, zeroes dead columns in ``W``, and returns ``(L, Hinv)``.

        - ``L`` is the lower-triangular Cholesky factor of the dampened Hessian
          ``H + damp * I = L @ L^T``.
        - ``Hinv`` is the upper-triangular Cholesky factor of ``(H + damp * I)^-1``
          used by the blockwise Optimal Brain Surgeon (OBS) solver.
        """
        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        # Factorize dampened Hessian: H = L @ L^T
        L = torch.linalg.cholesky(H)
        # Fast multi-threaded inversion: solve L @ Linv = I, then H^-1 = Linv^T @ Linv
        Linv = torch.linalg.solve_triangular(L,
                                             torch.eye(self.columns,
                                                       device=self.dev,
                                                       dtype=L.dtype),
                                             upper=False)
        Hinv = Linv.t().matmul(Linv)
        # Obtain upper Cholesky factor for OBS block updates
        Hinv = torch.linalg.cholesky(Hinv, upper=True)
        return L, Hinv

    def _obs_prune_blocks(
        self,
        W: torch.Tensor,
        Hinv: torch.Tensor,
        sparsity: float,
        prunen: int = 0,
        prunem: int = 0,
        blocksize: int = 128,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs the SparseGPT blockwise OBS pruning loop on ``W`` in place."""
        Losses = torch.zeros(self.rows, device=self.dev)

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            if prunen == 0:
                if mask is not None:
                    mask1 = mask[:, i1:i2]
                else:
                    tmp = W1**2 / (torch.diag(Hinv1).reshape((1, -1)))**2
                    thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() *
                                                              sparsity)]
                    mask1 = tmp <= thresh
            else:
                mask1 = torch.zeros_like(W1) == 1

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if prunen != 0 and i % prunem == 0:
                    tmp = W1[:, i:(i + prunem)]**2 / (torch.diag(Hinv1)[i:(
                        i + prunem)].reshape((1, -1)))**2
                    mask1.scatter_(
                        1, i + torch.topk(
                            tmp, prunem - prunen, dim=1, largest=False)[1],
                        True)

                q = w.clone()
                q[mask1[:, i]] = 0

                if hasattr(self, "quantizer"):
                    q = quantize(q.unsqueeze(1), self.quantizer.scale,
                                 self.quantizer.zero,
                                 self.quantizer.maxq).flatten()

                Q1[:, i] = q
                Losses1[:, i] = (w - q)**2 / d**2

                err1 = (w - q) / d
                # In-place rank-1 update: W1[:, i:] -= err1 @ Hinv1[i, i:]
                W1[:, i:].addr_(err1, Hinv1[i, i:], alpha=-1.0)
                Err1[:, i] = err1

            W[:, i1:i2] = Q1
            Losses += torch.sum(Losses1, 1) / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            if DEBUG:
                self.layer.weight.data[:, :i2] = W[:, :i2]
                self.layer.weight.data[:, i2:] = W[:, i2:]
                print(torch.sum((self.layer(self.inp1) - self.out1)**2))
                print(torch.sum(Losses))

        return W, Losses

    def _write_weight_matrix(self, W: torch.Tensor) -> None:
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype)

    def fasterprune(self,
                    sparsity,
                    prunen=0,
                    prunem=0,
                    blocksize=128,
                    percdamp=.01):
        W = self._clone_weight_matrix()
        tick = time.time()

        _, Hinv = self._factorize_hessian(W, percdamp=percdamp)
        W, Losses = self._obs_prune_blocks(
            W,
            Hinv,
            sparsity=sparsity,
            prunen=prunen,
            prunem=prunem,
            blocksize=blocksize,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print("time %.2f" % (time.time() - tick))
        print("error", torch.sum(Losses).item())

        self._write_weight_matrix(W)
        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1)**2))

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
