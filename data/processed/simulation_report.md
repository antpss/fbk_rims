# RIMS++ Simulation Benchmark Report

- **Simulation Mode:** `hybrid_residual`
- **Arrival Mode:** `replay`
- **Simulated Cases:** 500
- **Total Work Events:** 1,570
- **Workers in Multi-Skilled Pool:** 58 human resources
- **Calendar Engine:** Working Hours 8:00 - 20:00 (Mon-Fri)
- **Simulated Log Path (CSV):** `/home/antonio/fbk_rims/data/generated/csv/simulated_log.csv`
- **Simulated Log Path (XES):** `/home/antonio/fbk_rims/data/generated/xes/simulated_log.xes`

## Overall Cycle Time Benchmark

| Metric | Real Historical | Simulated RIMS++ | Difference |
|---|---|---|---|
| **Cases Evaluated** | 9658 | 500 | --- |
| **Completed Work Tasks** | 75,598 | 1,570 | --- |
| **Avg Tasks / Case** | 7.83 | 3.14 | -4.69 |
| **Mean Case Cycle Time** | 11.3 days | 7.9 days | 3.5 days |
| **Median Case Cycle Time** | 8.3 days | 2.7 days | --- |
| **Cycle Time Wasserstein Dist** | --- | --- | 5.4 days |
| **Relative Event Dist (RED)** | --- | --- | **71.25h EMD** |

## Activity Execution Breakdown

| Activity | Simulated Count | Sim Mean Duration | Sim Mean Wait | Sim Mean Queue Wait |
|---|---|---|---|---|
| `W_Assess fraud` | 242 | 1.2 mins | 9.5 hours | 1.3 mins |
| `W_Call after incomplete files` | 373 | 6.0 mins | 17.6 hours | 0.0s |
| `W_Call after offers` | 258 | 1.6 mins | 3.2 days | 0.0s |
| `W_Complete application` | 258 | 4.5 mins | 8.0 hours | 0.0s |
| `W_Handle leads` | 88 | 1.6 hours | 8.4 hours | 0.0s |
| `W_Validate application` | 351 | 6.8 hours | 7.7 days | 0.0s |
