# FlashAttention-2 (tiled online softmax)

Reference implementation of the **FA-2 algorithm** in NumPy — forward **and** backward. Tests never call `F.scaled_dot_product_attention`.

## Papers

- Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention* (2022) — IO-aware tiling, online softmax.
- Dao, *FlashAttention-2* (2023) — better work partitioning; this folder implements **this** recurrence (Algorithms 1–2).
- Shah et al., *FlashAttention-3* (2024) — Hopper asynchrony (TMA, WGMMA, warp specialization, FP8). **Documented only.** No CUDA FA-3 kernel runs on Mac; `flash_attn.cu` is the synchronous FA-2 forward sketch.

## Forward (Algorithm 1)

Tile Q into `Br` rows and K/V into `Bc` columns. For each Q-tile, stream K/V tiles:

```
S_ij = Q_i K_j^T / sqrt(d)          # shape (Br, Bc) only
m'   = max(m, rowmax(S_ij))
P̃    = exp(S_ij - m')
l'   = exp(m - m') * l + rowsum(P̃)
Õ    = exp(m - m') * Õ + P̃ V_j
```

Causal: mask `S_ij` where `k_idx > q_idx` before the row-max. Final `O = Õ / l`, and save `L = m + log(l)` (logsumexp) for backward.

Peak extra score storage is `Br * Bc` — never an `(N, N)` score matrix.

## Backward (Algorithm 2)

Recompute attention tiles from saved `L` (do not store `P`):

```
D_i  = rowsum(dO ◦ O)
# outer loop over K/V tiles j, inner over Q tiles i:
S_ij = Q_i K_j^T / sqrt(d)          # (Br, Bc) only
P_ij = exp(S_ij - L_i)
dV_j += P_ij^T dO_i
dP   = dO_i V_j^T
dS   = P_ij ◦ (dP - D_i)
dQ_i += dS K_j / sqrt(d)
dK_j += dS^T Q_i / sqrt(d)
```

Gradients are checked against a full-matrix naive oracle and central finite differences.

## FA-3 (Hopper only — not claimed here)

FA-3 keeps the same math but changes the **schedule**: producer/consumer warp specialization, TMA async copies, overlapping softmax with asynchronous WGMMA, plus FP8 block quantization. That requires NVIDIA Hopper (H100). This repo documents it; it does not ship a Hopper FA-3 CUDA kernel.

## Papers on disk

- [`papers/dao-flashattention-2022.pdf`](papers/dao-flashattention-2022.pdf) — Dao et al. FlashAttention (2022) ([arXiv:2205.14135](https://arxiv.org/abs/2205.14135))
- [`papers/dao-flashattention-2-2023.pdf`](papers/dao-flashattention-2-2023.pdf) — Dao. FlashAttention-2 (2023) ([arXiv:2307.08691](https://arxiv.org/abs/2307.08691))
- [`papers/shah-flashattention-3-2024.pdf`](papers/shah-flashattention-3-2024.pdf) — Shah et al. FlashAttention-3 (2024) ([arXiv:2407.08608](https://arxiv.org/abs/2407.08608))

## Run

```bash
python demo.py
python -m pytest test_flash_attn.py -q
```

`flash_attn.cu` is not compiled by the tests (`nvcc` optional).
