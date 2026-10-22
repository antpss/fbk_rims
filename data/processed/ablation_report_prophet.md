# RIMS+ Simulation Ablation Benchmark Report (PROPHET ARRIVAL)

- **Arrival Mode:** `prophet`
- **Evaluated Cases per Mode:** 1,000
- **Real Historical Mean Cycle Time:** 11.3 days

| Simulation Mode | Mean Cycle Time | Median Cycle Time | Cycle Time MAE | Wasserstein Dist | RED Dist (EMD) | Run Time |
|---|---|---|---|---|---|---|
| **REAL HISTORICAL** | 11.3 days | 8.3 days | --- | --- | --- | --- |
| `hybrid_residual` ★ (Best) | 10.2 days | 4.9 days | 1.2 days | 3.6 days | 191.47h | 169.1s |
| `pure_physics` | 9.5 days | 3.1 days | 1.9 days | 4.7 days | 266.15h | 204.3s |
