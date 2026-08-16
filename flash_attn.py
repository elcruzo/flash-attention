"""FlashAttention-2 forward: tiled online softmax. Never allocates an (N, N) score matrix.

Dao et al. FA-2 algorithm (not F.scaled_dot_product_attention):
  Tile Q into Br rows, K/V into Bc columns.
  For each Q-tile i, iterate K/V tiles j:
      S_ij = Q_i K_j^T / sqrt(d)
      m_new = max(m, rowmax(S))
      P = exp(S - m_new)
      l_new = e^{m-m_new} * l + rowsum(P)
      O = e^{m-m_new} * O + P V
  O = O / l

Backward is omitted (see README).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def naive_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, causal: bool = False) -> np.ndarray:
    """Full (N, N) attention — test oracle only."""
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    *prefix, nq, d = q.shape
    nk = k.shape[-2]
    scale = 1.0 / math.sqrt(d)
    # (..., Nq, d) @ (..., d, Nk)
    s = np.matmul(q, np.swapaxes(k, -1, -2)) * scale
    if causal:
        q_idx = np.arange(nq)[:, None]
        k_idx = np.arange(nk)[None, :]
        s = np.where(k_idx <= q_idx, s, -np.inf)
    s = s - np.max(s, axis=-1, keepdims=True)
    p = np.exp(s)
    p = np.where(np.isfinite(p), p, 0.0)
    denom = np.sum(p, axis=-1, keepdims=True)
    denom = np.where(denom == 0, 1.0, denom)
    p = p / denom
    return np.matmul(p, v)


def _online_softmax_from_tiles(row_tiles: list[np.ndarray]) -> np.ndarray:
    """Online softmax over a sequence of 1-D score tiles. Equals full softmax."""
    m, d = -np.inf, 0.0
    for tile in row_tiles:
        block_m = float(np.max(tile)) if tile.size else -np.inf
        m_new = block_m if block_m > m else m
        d = d * np.exp(m - m_new) + float(np.sum(np.exp(tile - m_new)))
        m = m_new
    parts = [np.exp(t - m) / d for t in row_tiles]
    return np.concatenate(parts)


def flash_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    causal: bool = False,
    br: int = 16,
    bc: int = 16,
    stats: dict[str, Any] | None = None,
) -> np.ndarray:
    """FA-2 forward. Score scratch is only ever (Br, Bc) — never (N, N)."""
    if br < 1 or bc < 1:
        raise ValueError("br and bc must be >= 1")
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    if q.shape[-1] != k.shape[-1] or k.shape[-1] != v.shape[-1]:
        raise ValueError("head dim mismatch")
    if k.shape[-2] != v.shape[-2]:
        raise ValueError("K/V sequence length mismatch")

    *prefix, nq, d = q.shape
    nk = k.shape[-2]
    scale = 1.0 / math.sqrt(d)
    batch = int(np.prod(prefix)) if prefix else 1
    q3 = q.reshape(batch, nq, d)
    k3 = k.reshape(batch, nk, d)
    v3 = v.reshape(batch, nk, d)
    o3 = np.zeros((batch, nq, d), dtype=np.float64)

    peak_score_elems = 0
    allocated_full_n2 = False
    score_shapes: list[tuple[int, int]] = []

    for b in range(batch):
        for i0 in range(0, nq, br):
            i1 = min(i0 + br, nq)
            qi = q3[b, i0:i1]  # (Br, d)
            oi = np.zeros((i1 - i0, d), dtype=np.float64)
            mi = np.full((i1 - i0,), -np.inf, dtype=np.float64)
            li = np.zeros((i1 - i0,), dtype=np.float64)
            for j0 in range(0, nk, bc):
                j1 = min(j0 + bc, nk)
                kj = k3[b, j0:j1]
                vj = v3[b, j0:j1]
                # S_ij is the only score buffer: shape (actual_Br, actual_Bc)
                sij = (qi @ kj.T) * scale
                score_shapes.append((int(sij.shape[0]), int(sij.shape[1])))
                peak_score_elems = max(peak_score_elems, int(sij.size))
                if sij.shape[0] * sij.shape[1] > br * bc:
                    allocated_full_n2 = True
                if causal:
                    q_idx = np.arange(i0, i1)[:, None]
                    k_idx = np.arange(j0, j1)[None, :]
                    sij = np.where(k_idx <= q_idx, sij, -np.inf)
                row_max = np.max(sij, axis=1)
                if not np.isfinite(row_max).any():
                    continue
                m_new = np.maximum(mi, row_max)
                p = np.exp(sij - m_new[:, None])
                p = np.where(np.isfinite(p), p, 0.0)
                alpha = np.exp(mi - m_new)
                alpha = np.where(np.isfinite(alpha), alpha, 0.0)
                li = alpha * li + p.sum(axis=1)
                oi = alpha[:, None] * oi + p @ vj
                mi = m_new
            denom = np.where(li > 0.0, li, 1.0)
            o3[b, i0:i1] = oi / denom[:, None]

    if stats is not None:
        stats["peak_score_elems"] = peak_score_elems
        stats["allocated_full_n2"] = allocated_full_n2
        stats["br"] = br
        stats["bc"] = bc
        stats["score_shapes"] = score_shapes
    return o3.reshape(q.shape)


# Re-export for the numerical online-softmax test.
online_softmax_from_tiles = _online_softmax_from_tiles
