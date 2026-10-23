# RIMS+ Simulation Benchmark Report

- **Simulation Mode:** `hybrid_residual`
- **Arrival Mode:** `replay`
- **Simulated Cases:** 5
- **Total Work Events:** 9
- **Workers in Multi-Skilled Pool:** 59 human resources
- **Calendar Engine:** Working Hours 8:00 - 20:00 (Mon-Fri)
- **Simulated Log Path (CSV):** `/home/antonio/fbk_rims/data/generated/csv/simulated_log.csv`
- **Simulated Log Path (XES):** `/home/antonio/fbk_rims/data/generated/xes/simulated_log.xes`

## Overall Cycle Time Benchmark

| Metric | Real Historical | Simulated RIMS+ | Difference |
|---|---|---|---|
| **Cases Evaluated** | 9658 | 5 | --- |
| **Completed Work Tasks** | 75,598 | 9 | --- |
| **Avg Tasks / Case** | 7.83 | 1.80 | -6.03 |
| **Mean Case Cycle Time** | 11.3 days | 6.9 days | 4.4 days |
| **Median Case Cycle Time** | 8.3 days | 8.0s | --- |
| **Cycle Time Wasserstein Dist** | --- | --- | 5.5 days |
| **Relative Event Dist (RED)** | --- | --- | **128.02h EMD** |

## Activity Execution Breakdown

| Activity | Simulated Count | Sim Mean Duration | Sim Mean Wait | Sim Mean Queue Wait |
|---|---|---|---|---|
| `W_Assess fraud` | 3 | 8.5s | 2.1 days | 0.0s |
| `W_Call after incomplete files` | 0 | 0.0s | 0.0s | 0.0s |
| `W_Call after offers` | 2 | 1.7 mins | 15.2 days | 0.0s |
| `W_Complete application` | 2 | 2.4 mins | 2.2 days | 0.0s |
| `W_Handle leads` | 0 | 0.0s | 0.0s | 0.0s |
| `W_Validate application` | 2 | 13.1 mins | 2.1 days | 0.0s |
