"""FlashAttention-2: tiled online-softmax forward and backward.

Dao et al. FA-2 algorithm (not F.scaled_dot_product_attention):
  Forward — tile Q into Br rows, K/V into Bc columns; keep only (Br, Bc) scores.
  Backward — recompute P tiles from saved logsumexp L; never store the full P.

FA-3 (Shah et al.): Hopper TMA / WGMMA / warp specialization — documented in
README only; this module is the synchronous FA-2 math on NumPy.
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


def naive_attention_backward(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    do: np.ndarray,
    causal: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full-matrix attention backward — test oracle only."""
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    do = np.asarray(do, dtype=np.float64)
    *prefix, nq, d = q.shape
    nk = k.shape[-2]
    scale = 1.0 / math.sqrt(d)
    s = np.matmul(q, np.swapaxes(k, -1, -2)) * scale
    if causal:
        q_idx = np.arange(nq)[:, None]
        k_idx = np.arange(nk)[None, :]
        s = np.where(k_idx <= q_idx, s, -np.inf)
    m = np.max(s, axis=-1, keepdims=True)
    p = np.exp(s - m)
    p = np.where(np.isfinite(p), p, 0.0)
    denom = np.sum(p, axis=-1, keepdims=True)
    denom = np.where(denom == 0, 1.0, denom)
    p = p / denom

    dv = np.matmul(np.swapaxes(p, -1, -2), do)
    dp = np.matmul(do, np.swapaxes(v, -1, -2))
    # dsoftmax: ds = p * (dp - rowsum(dp * p))
    dsum = np.sum(dp * p, axis=-1, keepdims=True)
    ds = p * (dp - dsum)
    dq = np.matmul(ds, k) * scale
    dk = np.matmul(np.swapaxes(ds, -1, -2), q) * scale
    return dq, dk, dv


def _online_softmax_from_tiles(
    row_tiles: list[np.ndarray],
    *,
    poison_d: float | None = None,
) -> np.ndarray:
    """Online softmax over a sequence of 1-D score tiles. Equals full softmax.

    Uses running (m, d). If poison_d is set, replace d before the final normalize —
    output must then diverge from safe softmax (proves d is actually used).
    """
    m, d = -np.inf, 0.0
    for tile in row_tiles:
        block_m = float(np.max(tile)) if tile.size else -np.inf
        m_new = block_m if block_m > m else m
        d = d * np.exp(m - m_new) + float(np.sum(np.exp(tile - m_new)))
        m = m_new
    if poison_d is not None:
        d = poison_d
    parts = [np.exp(t - m) / d for t in row_tiles]
    return np.concatenate(parts)


def _reshape_batch(q: np.ndarray, k: np.ndarray, v: np.ndarray):
    if q.shape[-1] != k.shape[-1] or k.shape[-1] != v.shape[-1]:
        raise ValueError("head dim mismatch")
    if k.shape[-2] != v.shape[-2]:
        raise ValueError("K/V sequence length mismatch")
    *prefix, nq, d = q.shape
    nk = k.shape[-2]
    batch = int(np.prod(prefix)) if prefix else 1
    q3 = q.reshape(batch, nq, d)
    k3 = k.reshape(batch, nk, d)
    v3 = v.reshape(batch, nk, d)
    return prefix, batch, nq, nk, d, q3, k3, v3


def flash_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    causal: bool = False,
    br: int = 16,
    bc: int = 16,
    stats: dict[str, Any] | None = None,
    return_lse: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """FA-2 forward. Score scratch is only ever (Br, Bc) — never (N, N).

    If return_lse is True, also returns row-wise logsumexp L = m + log(l)
    needed by the tiled backward (Algorithm 2).
    """
    if br < 1 or bc < 1:
        raise ValueError("br and bc must be >= 1")
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    prefix, batch, nq, nk, d, q3, k3, v3 = _reshape_batch(q, k, v)
    scale = 1.0 / math.sqrt(d)
    o3 = np.zeros((batch, nq, d), dtype=np.float64)
    lse3 = np.full((batch, nq), -np.inf, dtype=np.float64)

    peak_score_elems = 0
    allocated_full_n2 = False
    score_shapes: list[tuple[int, int]] = []

    for b in range(batch):
        for i0 in range(0, nq, br):
            i1 = min(i0 + br, nq)
            qi = q3[b, i0:i1]
            oi = np.zeros((i1 - i0, d), dtype=np.float64)
            mi = np.full((i1 - i0,), -np.inf, dtype=np.float64)
            li = np.zeros((i1 - i0,), dtype=np.float64)
            for j0 in range(0, nk, bc):
                j1 = min(j0 + bc, nk)
                kj = k3[b, j0:j1]
                vj = v3[b, j0:j1]
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
            # L = m + log(l); rows with no valid scores stay -inf
            safe_l = np.where(li > 0.0, li, 1.0)
            lse3[b, i0:i1] = np.where(li > 0.0, mi + np.log(safe_l), -np.inf)

    if stats is not None:
        stats["peak_score_elems"] = peak_score_elems
        stats["allocated_full_n2"] = allocated_full_n2
        stats["br"] = br
        stats["bc"] = bc
        stats["score_shapes"] = score_shapes

    out = o3.reshape(q.shape)
    if return_lse:
        lse_shape = q.shape[:-1]
        return out, lse3.reshape(lse_shape)
    return out


def flash_attention_backward(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    o: np.ndarray,
    do: np.ndarray,
    lse: np.ndarray,
    causal: bool = False,
    br: int = 16,
    bc: int = 16,
    stats: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """FA-2 backward (Algorithm 2): recompute P tiles from L; score tile ≤ (Br, Bc).

    Outer loop over K/V tiles, inner over Q tiles (FA-2 column-parallel schedule).
    """
    if br < 1 or bc < 1:
        raise ValueError("br and bc must be >= 1")
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    o = np.asarray(o, dtype=np.float64)
    do = np.asarray(do, dtype=np.float64)
    lse = np.asarray(lse, dtype=np.float64)
    prefix, batch, nq, nk, d, q3, k3, v3 = _reshape_batch(q, k, v)
    if o.shape != q.shape or do.shape != q.shape:
        raise ValueError("O / dO shape must match Q")
    if lse.shape != q.shape[:-1]:
        raise ValueError("LSE shape must match Q without head dim")
    scale = 1.0 / math.sqrt(d)
    o3 = o.reshape(batch, nq, d)
    do3 = do.reshape(batch, nq, d)
    lse3 = lse.reshape(batch, nq)

    dq3 = np.zeros_like(q3)
    dk3 = np.zeros_like(k3)
    dv3 = np.zeros_like(v3)

    peak_score_elems = 0
    allocated_full_n2 = False
    score_shapes: list[tuple[int, int]] = []

    for b in range(batch):
        # D_i = rowsum(dO ◦ O) — FA-2 Algorithm 2 step 4
        d_row = np.sum(do3[b] * o3[b], axis=1)  # (Nq,)
        for j0 in range(0, nk, bc):
            j1 = min(j0 + bc, nk)
            kj = k3[b, j0:j1]
            vj = v3[b, j0:j1]
            dkj = np.zeros((j1 - j0, d), dtype=np.float64)
            dvj = np.zeros((j1 - j0, d), dtype=np.float64)
            for i0 in range(0, nq, br):
                i1 = min(i0 + br, nq)
                qi = q3[b, i0:i1]
                doi = do3[b, i0:i1]
                li = lse3[b, i0:i1]
                di = d_row[i0:i1]
                sij = (qi @ kj.T) * scale
                score_shapes.append((int(sij.shape[0]), int(sij.shape[1])))
                peak_score_elems = max(peak_score_elems, int(sij.size))
                if sij.shape[0] * sij.shape[1] > br * bc:
                    allocated_full_n2 = True
                if causal:
                    q_idx = np.arange(i0, i1)[:, None]
                    k_idx = np.arange(j0, j1)[None, :]
                    sij = np.where(k_idx <= q_idx, sij, -np.inf)
                # P = exp(S - L); masked / empty rows → 0
                p = np.exp(sij - li[:, None])
                p = np.where(np.isfinite(p), p, 0.0)
                dvj = dvj + p.T @ doi
                dp = doi @ vj.T
                ds = p * (dp - di[:, None])
                dq3[b, i0:i1] = dq3[b, i0:i1] + (ds @ kj) * scale
                dkj = dkj + (ds.T @ qi) * scale
            dk3[b, j0:j1] = dkj
            dv3[b, j0:j1] = dvj

    if stats is not None:
        stats["peak_score_elems"] = peak_score_elems
        stats["allocated_full_n2"] = allocated_full_n2
        stats["br"] = br
        stats["bc"] = bc
        stats["score_shapes"] = score_shapes

    return dq3.reshape(q.shape), dk3.reshape(k.shape), dv3.reshape(v.shape)


# Re-export for the numerical online-softmax test.
online_softmax_from_tiles = _online_softmax_from_tiles
