# FlashAttention-2 (tiled online softmax)

Reference implementation of the **FA-2 algorithm** in NumPy. Tests never call `F.scaled_dot_product_attention`.

## Papers

- Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention* (2022) — IO-aware tiling, online softmax.
- Dao, *FlashAttention-2* (2023) — better work partitioning; this folder implements **this** recurrence.
- Shah et al., *FlashAttention-3* (2024) — Hopper asynchrony (TMA, WGMMA, warp specialization, FP8). **Documented only.** No Hopper / async hardware is required; `flash_attn.cu` is the synchronous FA-2 kernel.

## Algorithm

Tile Q into `Br` rows and K/V into `Bc` columns. For each Q-tile, stream K/V tiles:

```
S_ij = Q_i K_j^T / sqrt(d)          # shape (Br, Bc) only
m'   = max(m, rowmax(S_ij))
P    = exp(S_ij - m')
l'   = exp(m - m') * l + rowsum(P)
O    = exp(m - m') * O + P V_j
```

Causal: mask `S_ij` where `k_idx > q_idx` before the row-max. Final `O /= l`.

The reference **loops tiles** and never allocates an `(N, N)` score matrix. Peak extra score storage is `Br * Bc`.

## Backward

**Omitted.** A correct FA-2 backward recomputes tiles from saved `m, l` (and optionally O) instead of storing `P`. Only the forward is implemented and tested.

## Run

```bash
python demo.py
python -m pytest test_flash_attn.py -q
```

`flash_attn.cu` is not compiled by the tests (`nvcc` optional).
