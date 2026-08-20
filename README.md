# FlashAttention-2 (tiled online softmax)

Reference implementation of the **FA-2 algorithm** in NumPy — forward **and** backward. Tests compare tiled output to a full-matrix NumPy oracle.

## Papers

- Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention* (2022) — IO-aware tiling, online softmax.
- Dao, *FlashAttention-2* (2023) — better work partitioning; this folder implements **this** recurrence (Algorithms 1–2).
- Shah et al., *FlashAttention-3* (2024) — Hopper asynchrony (TMA, WGMMA, warp specialization, FP8). Documented here; runnable code is FA-2 (`flash_attn.py` + the synchronous `flash_attn.cu` sketch).

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

Peak extra score storage is `Br * Bc`.

## Backward (Algorithm 2)

Recompute attention tiles from saved `L`:

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

## FA-3 (Hopper schedule)

FA-3 keeps the same math and changes the **schedule**: producer/consumer warp specialization, TMA async copies, overlapping softmax with asynchronous WGMMA, plus FP8 block quantization. That schedule targets NVIDIA Hopper (H100). This folder documents it; the code here is FA-2.

## Papers on disk

- [`papers/dao-flashattention-2022.pdf`](papers/dao-flashattention-2022.pdf) — Dao et al. FlashAttention (2022) ([arXiv:2205.14135](https://arxiv.org/abs/2205.14135))
- [`papers/dao-flashattention-2-2023.pdf`](papers/dao-flashattention-2-2023.pdf) — Dao. FlashAttention-2 (2023) ([arXiv:2307.08691](https://arxiv.org/abs/2307.08691))
- [`papers/shah-flashattention-3-2024.pdf`](papers/shah-flashattention-3-2024.pdf) — Shah et al. FlashAttention-3 (2024) ([arXiv:2407.08608](https://arxiv.org/abs/2407.08608))

## Compared to flash-attn

**What you learn here:**
- FA-2 tiled online softmax forward + recomputed backward (Algorithms 1–2)
- Peak score scratch is `Br×Bc` ($N{=}64 \to 256$ vs $N^2{=}4096$ in `main.py`)
- Exact match to naive attention (no approximation)

| | This repo | flash-attn / FA-2 CUDA |
|---|---|---|
| Runtime | NumPy CPU reference | CUDA A100/H100 kernels |
| Goal | Correct tiling + grads | IO-aware wall-clock speedup |
| FA-3 | Documented only | Hopper TMA/WGMMA |

### Numbers (2026-08-16, Darwin 25.5.0 arm64 / Apple M5)

| Metric | This repo | Baseline | Source |
|---|---|---|---|
| Fwd max \|err\| vs naive | $5.6{\times}10^{-16}$ (N=64) | 0 (exact) | `python main.py` |
| Peak score elems | 256 vs $N^2{=}4096$ | linear in tiles | same |
| Speedup vs naive | ~0.6× on CPU NumPy (N=256) | 3–10× vs PyTorch attn (A100) | Dao FA-2 2023; timed here |

```bash
python main.py
```

## Run

```bash
python main.py
python -m pytest test_flash_attn.py -q
```

Tests run `flash_attn.py`. `flash_attn.cu` is optional `nvcc` source for the same FA-2 recurrence.
