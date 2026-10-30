# RIMS++ Simulation Ablation Benchmark Report (REPLAY ARRIVAL)

- **Arrival Mode:** `replay`
- **Evaluated Cases per Mode:** 500
- **Real Historical Mean Cycle Time:** 11.3 days

| Simulation Mode | Mean Cycle Time | Median Cycle Time | Cycle Time MAE | Wasserstein Dist | RED Dist (EMD) | Run Time |
|---|---|---|---|---|---|---|
| **REAL HISTORICAL** | 11.3 days | 8.3 days | --- | --- | --- | --- |
| `hybrid_residual` ★ (Best) | 16.2 days | 6.9 days | 4.9 days | 7.1 days | 105.25h | 37.9s |
| `pure_physics` | 3.3 days | 1.2 days | 8.0 days | 8.1 days | 143.83h | 28.0s |
