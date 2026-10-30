# RIMS++ Simulation Ablation Benchmark Report (PROPHET ARRIVAL)

- **Arrival Mode:** `prophet`
- **Evaluated Cases per Mode:** ALL
- **Real Historical Mean Cycle Time:** 11.3 days

| Simulation Mode | Mean Cycle Time | Median Cycle Time | Cycle Time MAE | Wasserstein Dist | RED Dist (EMD) | Run Time |
|---|---|---|---|---|---|---|
| **REAL HISTORICAL** | 11.3 days | 8.3 days | --- | --- | --- | --- |
| `hybrid_residual` ★ (Best) | 10.8 days | 3.0 days | 13.8 hours | 4.8 days | 135.47h | 238.0s |
| `pure_physics` | 8.5 days | 4.3 hours | 2.8 days | 5.6 days | 88.36h | 127.0s |
