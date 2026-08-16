"""FlashAttention-2 tests — fail if the math is wrong."""

from __future__ import annotations

import numpy as np
import pytest

from flash_attn import flash_attention, naive_attention, online_softmax_from_tiles


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
    assert all(r * c <= 16 * 16 for r, c in stats["score_shapes"])
    assert (n, n) not in stats["score_shapes"]


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
