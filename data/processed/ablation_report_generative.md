# RIMS++ Simulation Ablation Benchmark Report (GENERATIVE ARRIVAL)

- **Arrival Mode:** `generative`
- **Evaluated Cases per Mode:** 9,000
- **Real Historical Mean Cycle Time:** 11.3 days

| Simulation Mode | Mean Cycle Time | Median Cycle Time | Cycle Time MAE | Wasserstein Dist | Run Time |
|---|---|---|---|---|---|
| **REAL HISTORICAL** | 11.3 days | 8.3 days | --- | --- | --- |
| `hybrid_residual` ★ (Best) | 13.3 days | 3.7 days | 1.9 days | 6.8 days | 1214.7s |
| `pure_physics` | 9.3 days | 1.1 days | 2.0 days | 7.1 days | 714.4s |
