# RIMS++ Simulation Benchmark Report

- **Simulation Mode:** `hybrid_residual`
- **Arrival Mode:** `replay`
- **Simulated Cases:** 500
- **Total Work Events:** 1,671
- **Workers in Multi-Skilled Pool:** 58 human resources
- **Calendar Engine:** Working Hours 8:00 - 20:00 (Mon-Fri)
- **Simulated Log Path (CSV):** `/home/antonio/fbk_rims/data/generated/csv/simulated_log.csv`
- **Simulated Log Path (XES):** `/home/antonio/fbk_rims/data/generated/xes/simulated_log.xes`

## Overall Cycle Time Benchmark

| Metric | Real Historical | Simulated RIMS++ | Difference |
|---|---|---|---|
| **Cases Evaluated** | 9658 | 500 | --- |
| **Completed Work Tasks** | 75,598 | 1,671 | --- |
| **Avg Tasks / Case** | 7.83 | 3.34 | -4.49 |
| **Mean Case Cycle Time** | 11.3 days | 8.9 days | 2.5 days |
| **Median Case Cycle Time** | 8.3 days | 3.4 days | --- |
| **Cycle Time Wasserstein Dist** | --- | --- | 6.1 days |
| **Relative Event Dist (RED)** | --- | --- | **85.87h EMD** |

## Activity Execution Breakdown

| Activity | Simulated Count | Sim Mean Duration | Sim Mean Wait | Sim Mean Queue Wait |
|---|---|---|---|---|
| `W_Assess fraud` | 229 | 1.3 mins | 9.7 hours | 1.5 mins |
| `W_Call after incomplete files` | 457 | 3.6 mins | 19.6 hours | 0.0s |
| `W_Call after offers` | 271 | 1.7 mins | 3.3 days | 0.0s |
| `W_Complete application` | 271 | 4.5 mins | 8.2 hours | 1.8s |
| `W_Handle leads` | 94 | 1.8 hours | 7.2 hours | 2.5s |
| `W_Validate application` | 349 | 8.7 hours | 8.6 days | 0.0s |
