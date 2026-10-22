# RIMS+ Simulation Benchmark Report

- **Simulation Mode:** `hybrid_residual`
- **Arrival Mode:** `prophet`
- **Simulated Cases:** 3
- **Total Work Events:** 9
- **Workers in Multi-Skilled Pool:** 59 human resources
- **Calendar Engine:** Working Hours 8:00 - 20:00 (Mon-Fri)
- **Simulated Log Path (CSV):** `/home/antonio/fbk_rims/data/processed/simulated_log.csv`
- **Simulated Log Path (XES):** `/home/antonio/fbk_rims/data/processed/simulated_log.xes`

## Overall Cycle Time Benchmark

| Metric | Real Historical | Simulated RIMS+ | Difference |
|---|---|---|---|
| **Cases Evaluated** | 9658 | 3 | --- |
| **Completed Work Tasks** | 75,598 | 9 | --- |
| **Avg Tasks / Case** | 7.83 | 3.00 | -4.83 |
| **Mean Case Cycle Time** | 11.3 days | 4.8 days | 6.6 days |
| **Median Case Cycle Time** | 8.3 days | 4.0 days | --- |
| **Cycle Time Wasserstein Dist** | --- | --- | 7.0 days |
| **Relative Event Dist (RED)** | --- | --- | **146.75h EMD** |

## Activity Execution Breakdown

| Activity | Simulated Count | Sim Mean Duration | Sim Mean Wait | Sim Mean Queue Wait |
|---|---|---|---|---|
| `W_Assess fraud` | 1 | 7.9s | 1.0s | 0.0s |
| `W_Call after incomplete files` | 0 | 0.0s | 0.0s | 0.0s |
| `W_Call after offers` | 3 | 1.6 mins | 3.3 days | 0.0s |
| `W_Complete application` | 2 | 4.2 mins | 1.0s | 0.0s |
| `W_Handle leads` | 0 | 0.0s | 0.0s | 0.0s |
| `W_Validate application` | 3 | 13.1 mins | 1.4 days | 0.0s |
