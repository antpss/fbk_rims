# RIMS+ Simulation Ablation Benchmark Report (REPLAY ARRIVAL)

- **Arrival Mode:** `replay`
- **Evaluated Cases per Mode:** 9,000
- **Real Historical Mean Cycle Time:** 11.3 days

| Simulation Mode | Mean Cycle Time | Median Cycle Time | Cycle Time MAE | Wasserstein Dist | Run Time |
|---|---|---|---|---|---|
| **REAL HISTORICAL** | 11.3 days | 8.3 days | --- | --- | --- |
| `pure_physics` ★ (Best) | 13.3 days | 1.4 days | 2.0 days | 8.0 days | 1028.1s |
| `hybrid_residual` | 17.8 days | 3.9 days | 6.5 days | 9.0 days | 1501.3s |
