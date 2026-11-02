# RIMS++ Simulation Ablation Benchmark Report (PROPHET ARRIVAL)

- **Arrival Mode:** `prophet`
- **Evaluated Cases per Mode:** 20
- **Real Historical Mean Cycle Time:** 11.3 days
- **Ranking Criterion:** Lowest Wasserstein Distance (overall cycle time distribution similarity)

| Simulation Mode | Mean Cycle Time | Median Cycle Time | Cycle Time MAE | Wasserstein Dist | RED Dist (EMD) | Run Time |
|---|---|---|---|---|---|---|
| **REAL HISTORICAL** | 11.3 days | 8.3 days | --- | --- | --- | --- |
| `hybrid_residual` ★ (Best Fit) | 4.4 days | 43.5s | 7.0 days | 7.02 days | 92.03h | 0.3s |
| `pure_physics` | 5.4 days | 43.5s | 5.9 days | 7.08 days | 77.35h | 0.2s |
