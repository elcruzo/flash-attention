"""FlashAttention-2 tests — fail if the math is wrong."""

from __future__ import annotations

import math

import numpy as np
import pytest

from flash_attn import (
    flash_attention,
    flash_attention_backward,
    naive_attention,
    naive_attention_backward,
    online_softmax_from_tiles,
)


@pytest.mark.parametrize("causal", [False, True])
def test_forward_matches_naive_seq128(causal: bool):
    rng = np.random.default_rng(0)
    n, d = 128, 32
    q = rng.standard_normal((n, d))
    k = rng.standard_normal((n, d))
    v = rng.standard_normal((n, d))
    got = flash_attention(q, k, v, causal=causal, br=16, bc=16)
    ref = naive_attention(q, k, v, causal=causal)
    assert np.allclose(got, ref, atol=1e-4), float(np.max(np.abs(got - ref)))


def test_tiles_never_allocate_nxn():
    rng = np.random.default_rng(1)
    n, d = 128, 32
    q = rng.standard_normal((n, d))
    k = rng.standard_normal((n, d))
    v = rng.standard_normal((n, d))
    stats: dict = {}
    flash_attention(q, k, v, causal=True, br=16, bc=16, stats=stats)
    assert stats["peak_score_elems"] <= 16 * 16
    assert stats["allocated_full_n2"] is False
    assert stats["peak_score_elems"] < n * n
    n_q_tiles = (n + 15) // 16
    n_k_tiles = (n + 15) // 16
    assert len(stats["score_shapes"]) == n_q_tiles * n_k_tiles
    assert all(r <= 16 and c <= 16 for r, c in stats["score_shapes"])
    assert all(r * c <= 16 * 16 for r, c in stats["score_shapes"])
    assert (n, n) not in stats["score_shapes"]


def test_tile_count_non_multiple_edge():
    """ceil(N/br)*ceil(N/bc) tiles; edge tiles smaller than (br, bc)."""
    rng = np.random.default_rng(11)
    n, d, br, bc = 37, 8, 16, 16
    q = rng.standard_normal((n, d))
    k = rng.standard_normal((n, d))
    v = rng.standard_normal((n, d))
    stats: dict = {}
    flash_attention(q, k, v, causal=False, br=br, bc=bc, stats=stats)
    want = math.ceil(n / br) * math.ceil(n / bc)
    assert len(stats["score_shapes"]) == want
    assert all(r <= br and c <= bc for r, c in stats["score_shapes"])
    # At least one edge tile is smaller than a full (br, bc) block
    assert any(r < br or c < bc for r, c in stats["score_shapes"])
    assert stats["peak_score_elems"] <= br * bc


def test_online_softmax_equals_full():
    rng = np.random.default_rng(2)
    row = rng.standard_normal(64)
    tiles = [row[i : i + 8] for i in range(0, 64, 8)]
    online = online_softmax_from_tiles(tiles)
    m = np.max(row)
    full = np.exp(row - m) / np.sum(np.exp(row - m))
    assert np.allclose(online, full, atol=1e-10)


def test_batched_heads_shape():
    rng = np.random.default_rng(3)
    q = rng.standard_normal((2, 4, 32, 16))
    k = rng.standard_normal((2, 4, 32, 16))
    v = rng.standard_normal((2, 4, 32, 16))
    got = flash_attention(q, k, v, causal=True, br=8, bc=8)
    ref = naive_attention(q, k, v, causal=True)
    assert got.shape == q.shape
    assert np.allclose(got, ref, atol=1e-4)


@pytest.mark.parametrize("causal", [False, True])
def test_backward_matches_naive(causal: bool):
    rng = np.random.default_rng(4)
    n, d = 48, 16
    q = rng.standard_normal((n, d))
    k = rng.standard_normal((n, d))
    v = rng.standard_normal((n, d))
    do = rng.standard_normal((n, d))
    o, lse = flash_attention(q, k, v, causal=causal, br=16, bc=8, return_lse=True)
    assert np.allclose(o, naive_attention(q, k, v, causal=causal), atol=1e-5)

    bstats: dict = {}
    dq, dk, dv = flash_attention_backward(
        q, k, v, o, do, lse, causal=causal, br=16, bc=8, stats=bstats
    )
    rq, rk, rv = naive_attention_backward(q, k, v, do, causal=causal)
    assert np.allclose(dq, rq, atol=1e-4), float(np.max(np.abs(dq - rq)))
    assert np.allclose(dk, rk, atol=1e-4), float(np.max(np.abs(dk - rk)))
    assert np.allclose(dv, rv, atol=1e-4), float(np.max(np.abs(dv - rv)))

    want = math.ceil(n / 16) * math.ceil(n / 8)
    assert len(bstats["score_shapes"]) == want
    assert all(r <= 16 and c <= 8 for r, c in bstats["score_shapes"])
    assert bstats["allocated_full_n2"] is False
    assert bstats["peak_score_elems"] <= 16 * 8


@pytest.mark.parametrize("causal", [False, True])
def test_gradcheck_finite_diff(causal: bool):
    """Central finite differences on loss = sum(O) vs tiled backward."""
    rng = np.random.default_rng(5)
    n, d = 12, 4
    q = rng.standard_normal((n, d))
    k = rng.standard_normal((n, d))
    v = rng.standard_normal((n, d))
    o, lse = flash_attention(q, k, v, causal=causal, br=4, bc=4, return_lse=True)
    do = np.ones_like(o)
    dq, dk, dv = flash_attention_backward(q, k, v, o, do, lse, causal=causal, br=4, bc=4)

    eps = 1e-5

    def loss(qq, kk, vv):
        return float(np.sum(flash_attention(qq, kk, vv, causal=causal, br=4, bc=4)))

    for name, x, gx in (("q", q, dq), ("k", k, dk), ("v", v, dv)):
        for idx in np.ndindex(x.shape):
            xp = x.copy()
            xm = x.copy()
            xp[idx] += eps
            xm[idx] -= eps
            if name == "q":
                num = (loss(xp, k, v) - loss(xm, k, v)) / (2 * eps)
            elif name == "k":
                num = (loss(q, xp, v) - loss(q, xm, v)) / (2 * eps)
            else:
                num = (loss(q, k, xp) - loss(q, k, xm)) / (2 * eps)
            assert abs(num - gx[idx]) < 2e-3, (name, idx, num, gx[idx])


def test_backward_batched():
    rng = np.random.default_rng(6)
    q = rng.standard_normal((2, 2, 24, 8))
    k = rng.standard_normal((2, 2, 24, 8))
    v = rng.standard_normal((2, 2, 24, 8))
    do = rng.standard_normal((2, 2, 24, 8))
    o, lse = flash_attention(q, k, v, causal=True, br=8, bc=8, return_lse=True)
    dq, dk, dv = flash_attention_backward(q, k, v, o, do, lse, causal=True, br=8, bc=8)
    rq, rk, rv = naive_attention_backward(q, k, v, do, causal=True)
    assert np.allclose(dq, rq, atol=1e-4)
    assert np.allclose(dk, rk, atol=1e-4)
    assert np.allclose(dv, rv, atol=1e-4)
