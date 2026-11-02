# RIMS++ Simulation Benchmark Report

- **Simulation Mode:** `hybrid_residual`
- **Arrival Mode:** `replay`
- **Simulated Cases:** 100
- **Total Work Events:** 836
- **Workers in Multi-Skilled Pool:** 58 human resources
- **Calendar Engine:** Working Hours 8:00 - 20:00 (Mon-Fri)
- **Simulated Log Path (CSV):** `/home/antonio/fbk_rims/data/generated/csv/simulated_log.csv`
- **Simulated Log Path (XES):** `/home/antonio/fbk_rims/data/generated/xes/simulated_log.xes`

## Overall Cycle Time Benchmark

| Metric | Real Historical | Simulated RIMS++ | Difference |
|---|---|---|---|
| **Cases Evaluated** | 9658 | 100 | --- |
| **Completed Work Tasks** | 75,598 | 836 | --- |
| **Avg Tasks / Case** | 7.83 | 8.36 | +0.53 |
| **Mean Case Cycle Time** | 11.3 days | 15.6 days | 4.3 days |
| **Median Case Cycle Time** | 8.3 days | 2.4 days | --- |
| **Cycle Time Wasserstein Dist** | --- | --- | 6.2 days |
| **Relative Event Dist (RED)** | --- | --- | **177.65h EMD** |

## Activity Execution Breakdown

| Activity | Simulated Count | Sim Mean Duration | Sim Mean Wait | Sim Mean Queue Wait |
|---|---|---|---|---|
| `W_Assess fraud` | 46 | 50.8s | 23.1 hours | 2.6 mins |
| `W_Call after incomplete files` | 74 | 2.1 mins | 18.1 hours | 0.0s |
| `W_Call after offers` | 314 | 1.4 mins | 3.5 days | 0.0s |
| `W_Complete application` | 304 | 2.2 mins | 23.5 hours | 0.0s |
| `W_Handle leads` | 26 | 1.8 hours | 14.9 hours | 0.0s |
| `W_Validate application` | 72 | 32.1 mins | 2.0 days | 0.0s |
