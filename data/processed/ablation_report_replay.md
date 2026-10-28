# RIMS++ Simulation Ablation Benchmark Report (REPLAY ARRIVAL)

- **Arrival Mode:** `replay`
- **Evaluated Cases per Mode:** 500
- **Real Historical Mean Cycle Time:** 11.3 days

| Simulation Mode | Mean Cycle Time | Median Cycle Time | Cycle Time MAE | Wasserstein Dist | RED Dist (EMD) | Run Time |
|---|---|---|---|---|---|---|
| **REAL HISTORICAL** | 11.3 days | 8.3 days | --- | --- | --- | --- |
| `hybrid_residual` ★ (Best) | 8.9 days | 3.4 days | 2.5 days | 6.1 days | 85.87h | 44.2s |
| `pure_physics` | 5.1 days | 1.9 days | 6.3 days | 6.3 days | 103.40h | 24.3s |
