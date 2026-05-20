"""Mamba-2 / SSD block for the PyTorch backend.

Same architecture as the MLX implementation. Implements the Structured State
Space Duality (SSD) scan with chunked processing: O(T * chunk_size) overall.

Drop-in replacement for the transformer Block: both accept [B, T, d_model]
and return [B, T, d_model].
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _ssd_chunk_torch(
    X: torch.Tensor,
    A_log: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    h: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single chunk of the SSD algorithm.

    Args:
        X: [B, L, H, P]  — input projected to SSM heads
        A_log: [B, L, H]  — log decay rates (negative or zero)
        B: [B, L, N]      — SSM input projection
        C: [B, L, N]      — SSM output projection
        h: [B, H, P, N]   — running state from previous chunk

    Returns:
        y: [B, L, H, P]
        h_new: [B, H, P, N]
    """
    B_sz, L, H, P = X.shape

    cumlog = torch.cumsum(A_log, dim=1)  # [B, L, H]
    cumlog_start = torch.cat(
        [torch.zeros(B_sz, 1, H, device=X.device, dtype=X.dtype), cumlog[:, :-1]], dim=1
    )  # [B, L, H]

    # Clamp before exp: valid entries (t >= s) have log_M <= 0 by construction.
    # Without the clamp, upper-triangle entries (t < s) are positive and can
    # overflow to inf; inf * 0 from the causal mask then produces NaN.
    log_M = (cumlog[:, :, None, :] - cumlog_start[:, None, :, :]).clamp(max=0.0)
    M = torch.exp(log_M)
    causal_mask = torch.tril(torch.ones(L, L, device=X.device, dtype=X.dtype))
    M = M * causal_mask[None, :, :, None]  # [B, L_t, L_s, H]

    # State contribution: gamma[b,t,h] * (h[b,h,p,:] · C[b,t,:])
    # Use einsum to avoid materialising [B, L, H, P, N].
    gamma = torch.exp(cumlog)  # [B, L, H]
    hC = torch.einsum("bhpn,bln->blhp", h, C)  # [B, L, H, P]
    y_h = gamma[:, :, :, None] * hC  # [B, L, H, P]

    # Intra-chunk: BC[b,t,s] = B[b,s,:] · C[b,t,:]  — only [B, L, L], small.
    BC = torch.einsum("bsn,btn->bts", B, C)  # [B, L_t, L_s] (result[b,t,s])
    MB = M * BC.unsqueeze(-1)  # [B, L_t, L_s, H]
    y_intra = torch.einsum("btsh,bshp->bthp", MB, X)  # [B, L, H, P]

    y = y_h + y_intra

    # State update — decay_to_end always <= 0, clamp for numerical safety.
    gamma_full = torch.exp(cumlog[:, -1, :])  # [B, H]
    decay = torch.exp(
        (cumlog[:, -1:, :] - cumlog_start).clamp(max=0.0)
    )  # [B, L, H]
    dX = decay[:, :, :, None] * X  # [B, L, H, P]
    dBX = torch.einsum("bshp,bsn->bhpn", dX, B)  # [B, H, P, N]
    h_new = gamma_full[:, :, None, None] * h + dBX

    return y, h_new


class Mamba2Block(nn.Module):
    """Mamba-2 SSM block with chunked SSD scan (PyTorch).

    Drop-in replacement for the transformer Block. Takes [B, T, d_model]
    and returns [B, T, d_model] with a pre-norm + residual pattern.
    """

    def __init__(
        self,
        d_model: int,
        state_size: int = 64,
        expand: int = 2,
        d_conv: int = 4,
        head_dim: int = 64,
        chunk_size: int = 64,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand
        self.n_heads = self.d_inner // head_dim
        self.head_dim = head_dim
        self.state_size = state_size
        self.d_conv = d_conv
        self.chunk_size = chunk_size

        if self.d_inner % head_dim:
            raise ValueError(
                f"d_inner ({self.d_inner}) must be divisible by head_dim ({head_dim})"
            )

        self.norm = nn.RMSNorm(d_model)
        # in_proj splits into: x_ssm (d_inner), z (d_inner), B_ssm (state_size),
        # C_ssm (state_size), log_dt (n_heads)
        self.in_proj = nn.Linear(
            d_model,
            2 * self.d_inner + 2 * state_size + self.n_heads,
            bias=False,
        )
        # Depthwise conv parameters
        self.conv_weight = nn.Parameter(torch.randn(d_conv, self.d_inner) * 0.02)
        self.conv_bias = nn.Parameter(torch.zeros(self.d_inner))
        # SSM learnable parameters
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, self.n_heads + 1, dtype=torch.float32))
        )
        self.D = nn.Parameter(torch.ones(self.n_heads, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.zeros(self.n_heads, dtype=torch.float32))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def _causal_conv(self, x: torch.Tensor) -> torch.Tensor:
        """Apply causal depthwise conv with left-padding."""
        T = x.shape[1]
        x_pad = F.pad(x, (0, 0, self.d_conv - 1, 0))
        y = sum(
            x_pad[:, k : k + T, :] * self.conv_weight[k]
            for k in range(self.d_conv)
        )
        return y + self.conv_bias

    def _ssd_scan(
        self,
        X: torch.Tensor,
        A_log_dt: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        """Chunked SSD scan over the full sequence."""
        B_sz, T, H, P = X.shape
        N = B.shape[-1]
        L = self.chunk_size
        device = X.device
        dtype = X.dtype

        pad = (-T) % L
        if pad:
            X = F.pad(X, (0, 0, 0, 0, 0, pad))
            A_log_dt = F.pad(A_log_dt, (0, 0, 0, pad))
            B = F.pad(B, (0, 0, 0, pad))
            C = F.pad(C, (0, 0, 0, pad))

        Tp = T + pad
        n_chunks = Tp // L
        Xc = X.reshape(B_sz, n_chunks, L, H, P)
        Ac = A_log_dt.reshape(B_sz, n_chunks, L, H)
        Bc = B.reshape(B_sz, n_chunks, L, N)
        Cc = C.reshape(B_sz, n_chunks, L, N)

        h = torch.zeros(B_sz, H, P, N, device=device, dtype=dtype)
        chunks: list[torch.Tensor] = []
        for c in range(n_chunks):
            y_chunk, h = _ssd_chunk_torch(Xc[:, c], Ac[:, c], Bc[:, c], Cc[:, c], h)
            chunks.append(y_chunk)

        Y = torch.cat(chunks, dim=1)
        return Y[:, :T]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        residual = x
        x = self.norm(x)

        proj = self.in_proj(x)
        splits = [
            self.d_inner,
            2 * self.d_inner,
            2 * self.d_inner + self.state_size,
            2 * self.d_inner + 2 * self.state_size,
        ]
        x_ssm = proj[..., : splits[0]]
        z = proj[..., splits[0] : splits[1]]
        B_ssm = proj[..., splits[1] : splits[2]]
        C_ssm = proj[..., splits[2] : splits[3]]
        log_dt = proj[..., splits[3] :]

        x_ssm = F.silu(self._causal_conv(x_ssm))
        dt = F.softplus(log_dt + self.dt_bias)
        A_log_dt = -torch.exp(self.A_log)[None, None, :] * dt  # [B, T, H]

        x_heads = x_ssm.reshape(B, T, self.n_heads, self.head_dim)
        y = self._ssd_scan(x_heads, A_log_dt, B_ssm, C_ssm)

        D_skip = self.D[None, None, :, None] * x_heads
        y = y + D_skip
        y = y.reshape(B, T, self.d_inner) * F.silu(z)
        y = self.out_proj(y)
        return residual + y
