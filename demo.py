"""CPU demo: FA-2 vs naive attention."""

from __future__ import annotations

import numpy as np

from flash_attn import flash_attention, naive_attention

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    q, k, v = (rng.standard_normal((64, 32)) for _ in range(3))
    stats: dict = {}
    o = flash_attention(q, k, v, causal=True, br=16, bc=16, stats=stats)
    r = naive_attention(q, k, v, causal=True)
    print("max err", float(np.max(np.abs(o - r))))
    print("peak score elems", stats["peak_score_elems"], "vs N^2", 64 * 64)
