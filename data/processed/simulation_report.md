# RIMS+ Simulation Benchmark Report

- **Simulation Mode:** `hybrid_residual`
- **Arrival Mode:** `replay`
- **Simulated Cases:** 10
- **Total Work Events:** 33
- **Workers in Multi-Skilled Pool:** 59 human resources
- **Calendar Engine:** Working Hours 8:00 - 20:00 (Mon-Fri)
- **Simulated Log Path (CSV):** `/home/antonio/fbk_rims/data/generated/csv/simulated_log.csv`
- **Simulated Log Path (XES):** `/home/antonio/fbk_rims/data/generated/xes/simulated_log.xes`

## Overall Cycle Time Benchmark

| Metric | Real Historical | Simulated RIMS+ | Difference |
|---|---|---|---|
| **Cases Evaluated** | 9658 | 10 | --- |
| **Completed Work Tasks** | 75,598 | 33 | --- |
| **Avg Tasks / Case** | 7.83 | 3.30 | -4.53 |
| **Mean Case Cycle Time** | 11.3 days | 8.5 days | 2.9 days |
| **Median Case Cycle Time** | 8.3 days | 8.5s | --- |
| **Cycle Time Wasserstein Dist** | --- | --- | 5.3 days |
| **Relative Event Dist (RED)** | --- | --- | **112.57h EMD** |

## Activity Execution Breakdown

| Activity | Simulated Count | Sim Mean Duration | Sim Mean Wait | Sim Mean Queue Wait |
|---|---|---|---|---|
| `W_Assess fraud` | 6 | 8.8s | 2.0 days | 0.0s |
| `W_Call after offers` | 9 | 1.6 mins | 7.6 days | 0.0s |
| `W_Complete application` | 4 | 3.6 mins | 1.6 days | 0.0s |
| `W_Handle leads` | 7 | 1.3 mins | 6.9 hours | 0.0s |
| `W_Validate application` | 7 | 11.7 mins | 2.4 days | 0.0s |
