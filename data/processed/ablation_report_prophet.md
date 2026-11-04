# RIMS++ Multi-Perspective Simulation Benchmark Report (PROPHET ARRIVAL)

- **Arrival Mode:** `prophet`
- **Evaluated Cases per Mode:** 1,000
- **Real Historical Mean Cycle Time:** 15.5 days
- **Evaluation Suite:** Meneghello et al. (Information Systems 2025) / Chapela-Campa et al. (BPM 2023)

## Comparative Benchmark Leaderboard

| Simulation Mode | CFLD | $L^{traces}$ | NGD 2-Gram | AED (h) | CED (h) | RED (h) | CAR (h) | CTD (WD) | Cycle MAE | Run Time |
|---|---|---|---|---|---|---|---|---|---|---|
| **REAL HISTORICAL** | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `hybrid_residual` ★ (Best Fit) | 1.0000 | 5.17 | 1.0000 | 12.69 | 11.81 | 180.21 | 12.50 | 11.98 days | 11.98 days | 13.0s |
| `pure_physics` | 0.0379 | 0.19 | 0.0121 | 12.70 | 10.86 | 193.99 | 12.50 | 14.64 days | 6.90 days | 20.9s |
