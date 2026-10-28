# RIMS++ Simulation Ablation Benchmark Report (PROPHET ARRIVAL)

- **Arrival Mode:** `prophet`
- **Evaluated Cases per Mode:** ALL
- **Real Historical Mean Cycle Time:** 11.3 days

| Simulation Mode | Mean Cycle Time | Median Cycle Time | Cycle Time MAE | Wasserstein Dist | RED Dist (EMD) | Run Time |
|---|---|---|---|---|---|---|
| **REAL HISTORICAL** | 11.3 days | 8.3 days | --- | --- | --- | --- |
| `hybrid_residual` ★ (Best) | 16.6 days | 4.8 days | 5.3 days | 7.9 days | 881.06h | 2075.9s |
| `pure_physics` | 13.9 days | 2.2 days | 2.6 days | 8.7 days | 819.93h | 1333.6s |
