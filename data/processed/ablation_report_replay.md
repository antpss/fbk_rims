# RIMS++ Simulation Ablation Benchmark Report (REPLAY ARRIVAL)

- **Arrival Mode:** `replay`
- **Evaluated Cases per Mode:** 100
- **Real Historical Mean Cycle Time:** 11.3 days
- **Ranking Criterion:** Lowest Wasserstein Distance (overall cycle time distribution similarity)

| Simulation Mode | Mean Cycle Time | Median Cycle Time | Cycle Time MAE | Wasserstein Dist | RED Dist (EMD) | Run Time |
|---|---|---|---|---|---|---|
| **REAL HISTORICAL** | 11.3 days | 8.3 days | --- | --- | --- | --- |
| `pure_physics` ★ (Best Fit) | 13.6 days | 3.5 hours | 2.3 days | 6.13 days | 171.49h | 1.4s |
| `hybrid_residual` | 15.6 days | 2.4 days | 4.3 days | 6.19 days | 177.65h | 3.0s |
