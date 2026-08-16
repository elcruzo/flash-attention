"""CPU demo: FA-2 forward + backward vs naive attention."""

from __future__ import annotations

import numpy as np

from flash_attn import (
    flash_attention,
    flash_attention_backward,
    naive_attention,
    naive_attention_backward,
)

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    q, k, v = (rng.standard_normal((64, 32)) for _ in range(3))
    do = rng.standard_normal((64, 32))
    stats: dict = {}
    o, lse = flash_attention(q, k, v, causal=True, br=16, bc=16, stats=stats, return_lse=True)
    r = naive_attention(q, k, v, causal=True)
    print("fwd max err", float(np.max(np.abs(o - r))))
    print("peak score elems", stats["peak_score_elems"], "vs N^2", 64 * 64)
    print("tile count", len(stats["score_shapes"]))

    bstats: dict = {}
    dq, dk, dv = flash_attention_backward(
        q, k, v, o, do, lse, causal=True, br=16, bc=16, stats=bstats
    )
    rq, rk, rv = naive_attention_backward(q, k, v, do, causal=True)
    print(
        "bwd max err",
        float(max(np.max(np.abs(dq - rq)), np.max(np.abs(dk - rk)), np.max(np.abs(dv - rv)))),
    )
    print("bwd peak score elems", bstats["peak_score_elems"], "tiles", len(bstats["score_shapes"]))
