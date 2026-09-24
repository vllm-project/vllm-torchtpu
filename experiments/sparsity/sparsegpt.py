import math
import time

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

    def fasterprune(self,
                    sparsity,
                    prunen=0,
                    prunem=0,
                    blocksize=128,
                    percdamp=.01):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        if hasattr(self, 'quantizer'):
            if not self.quantizer.ready():
                self.quantizer.find_params(W, weight=True)

        tick = time.time()

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros(self.rows, device=self.dev)

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

        mask = None

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

                if hasattr(self, 'quantizer'):
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

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print('time %.2f' % (time.time() - tick))
        print('error', torch.sum(Losses).item())

        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype)
        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1)**2))

    def lrc_prune(
        self,
        sparsity=0.5,
        prunen=2,
        prunem=4,
        blocksize=128,
        percdamp=0.01,
        T=3,
        k=128,
    ):
        W_orig = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W_orig = W_orig.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W_orig = W_orig.t()
        W_orig = W_orig.float()

        if hasattr(self, 'quantizer'):
            if not self.quantizer.ready():
                self.quantizer.find_params(W_orig, weight=True)

        tick = time.time()

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W_orig[:, dead] = 0

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp

        # Retained Cholesky factor L: H = L @ L^T (lower triangular)
        L = torch.linalg.cholesky(H)
        self.L = L

        # Fast multi-threaded inversion: solve L @ Linv = I, then H^-1 = Linv^T @ Linv
        Linv = torch.linalg.solve_triangular(L,
                                             torch.eye(self.columns,
                                                       device=self.dev,
                                                       dtype=L.dtype),
                                             upper=False)
        Hinv = Linv.t().matmul(Linv)
        # Obtain upper Cholesky factor for OBS block updates
        Hinv = torch.linalg.cholesky(Hinv, upper=True)

        k_eff = min(k, self.rows, self.columns)
        U = torch.zeros((self.rows, k_eff), device=self.dev)
        V = torch.zeros((k_eff, self.columns), device=self.dev)

        error_sparsegpt = None
        error_lrc = None

        for t in range(T):
            # 1. Fix UV; prune W_eff = W_orig - UV with SparseGPT (OBS in L-metric) -> S
            W = (W_orig - U.matmul(V)).clone()

            for i1 in range(0, self.columns, blocksize):
                i2 = min(i1 + blocksize, self.columns)
                count = i2 - i1

                W1 = W[:, i1:i2].clone()
                Q1 = torch.zeros_like(W1)
                Err1 = torch.zeros_like(W1)
                Hinv1 = Hinv[i1:i2, i1:i2]

                if prunen == 0:
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

                    if hasattr(self, 'quantizer'):
                        q = quantize(q.unsqueeze(1), self.quantizer.scale,
                                     self.quantizer.zero,
                                     self.quantizer.maxq).flatten()

                    Q1[:, i] = q
                    err1 = (w - q) / d
                    # In-place rank-1 update: W1[:, i:] -= err1 @ Hinv1[i, i:]
                    W1[:, i:].addr_(err1, Hinv1[i, i:], alpha=-1.0)
                    Err1[:, i] = err1

                W[:, i1:i2] = Q1
                W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            S = W

            # Whitened residual in L-metric: R_tilde = (W_orig - S) @ L
            R_tilde = (W_orig - S).matmul(L)
            if t == 0:
                error_sparsegpt = torch.norm(R_tilde, p='fro').item()

            # 2. Fix S; rank-k projection via Gram-matrix eigh on smaller dimension
            if k_eff > 0:
                if self.rows >= self.columns:
                    # Tall or square matrix: Gram matrix G = R_tilde^T @ R_tilde (columns x columns)
                    G = R_tilde.t().matmul(R_tilde)
                    eig_vals, eig_vecs = torch.linalg.eigh(G)
                    idx = torch.argsort(eig_vals, descending=True)[:k_eff]
                    Vh_k = eig_vecs[:, idx].t()
                    U = R_tilde.matmul(eig_vecs[:, idx])
                    # Solve triangular system: V @ L = Vh_k <=> V = Vh_k @ L^-1
                    V = torch.linalg.solve_triangular(L,
                                                      Vh_k,
                                                      upper=False,
                                                      left=False)
                else:
                    # Wide matrix: Gram matrix G = R_tilde @ R_tilde^T (rows x rows)
                    G = R_tilde.matmul(R_tilde.t())
                    eig_vals, eig_vecs = torch.linalg.eigh(G)
                    idx = torch.argsort(eig_vals, descending=True)[:k_eff]
                    Sigma_k = torch.sqrt(torch.clamp(eig_vals[idx], min=0.0))
                    U_k = eig_vecs[:, idx]
                    inv_sigma = torch.where(Sigma_k > 1e-7, 1.0 / Sigma_k,
                                            torch.zeros_like(Sigma_k))
                    Vh_k = inv_sigma.unsqueeze(1) * (U_k.t().matmul(R_tilde))
                    U = U_k * Sigma_k.unsqueeze(0)
                    # Solve triangular system: V @ L = Vh_k <=> V = Vh_k @ L^-1
                    V = torch.linalg.solve_triangular(L,
                                                      Vh_k,
                                                      upper=False,
                                                      left=False)

                R_lrc = (W_orig - S - U.matmul(V)).matmul(L)
                error_lrc = torch.norm(R_lrc, p='fro').item()
            else:
                error_lrc = error_sparsegpt

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        print('time %.2f' % (time.time() - tick))
        print('error SparseGPT: %.4f | LRC (T=%d, k=%d): %.4f' %
              (error_sparsegpt, T, k, error_lrc))

        self.S = S
        self.U = U
        self.V = V
        self.error_sparsegpt = error_sparsegpt
        self.error_lrc = error_lrc

        # Materialize W_hat = S + U @ V back into layer weight tensor
        W_final = S + U.matmul(V)
        if isinstance(self.layer, transformers.Conv1D):
            W_final = W_final.t()
        self.layer.weight.data = W_final.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype)

        return U, V

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        if hasattr(self, 'L'):
            self.L = None
        if hasattr(self, 'U'):
            self.U = None
        if hasattr(self, 'V'):
            self.V = None
        if hasattr(self, 'S'):
            self.S = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
