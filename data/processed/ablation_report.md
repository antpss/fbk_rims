# RIMS++ Simulation Ablation Benchmark Report (PROPHET ARRIVAL)

- **Arrival Mode:** `prophet`
- **Evaluated Cases per Mode:** ALL
- **Real Historical Mean Cycle Time:** 11.3 days
- **Ranking Criterion:** Lowest Wasserstein Distance (overall cycle time distribution similarity)

| Simulation Mode | Mean Cycle Time | Median Cycle Time | Cycle Time MAE | Wasserstein Dist | RED Dist (EMD) | Run Time |
|---|---|---|---|---|---|---|
| **REAL HISTORICAL** | 11.3 days | 8.3 days | --- | --- | --- | --- |
| `hybrid_residual` ★ (Best Fit) | 13.5 days | 3.3 days | 2.1 days | 4.93 days | 175.62h | 180.4s |
| `pure_physics` | 12.0 days | 19.0 hours | 15.5 hours | 5.04 days | 148.24h | 101.3s |
