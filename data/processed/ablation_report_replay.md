# RIMS++ Simulation Ablation Benchmark Report (REPLAY ARRIVAL)

- **Arrival Mode:** `replay`
- **Evaluated Cases per Mode:** 500
- **Real Historical Mean Cycle Time:** 11.3 days

| Simulation Mode | Mean Cycle Time | Median Cycle Time | Cycle Time MAE | Wasserstein Dist | RED Dist (EMD) | Run Time |
|---|---|---|---|---|---|---|
| **REAL HISTORICAL** | 11.3 days | 8.3 days | --- | --- | --- | --- |
| `hybrid_residual` ★ (Best) | 7.9 days | 2.7 days | 3.5 days | 5.4 days | 71.25h | 26.6s |
| `pure_physics` | 5.0 days | 1.2 days | 6.3 days | 6.3 days | 90.38h | 18.2s |
