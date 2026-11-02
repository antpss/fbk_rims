# RIMS++ Simulation Benchmark Report

- **Simulation Mode:** `hybrid_residual`
- **Arrival Mode:** `prophet`
- **Simulated Cases:** 9,658
- **Total Work Events:** 69,397
- **Workers in Multi-Skilled Pool:** 58 human resources
- **Calendar Engine:** Working Hours 8:00 - 20:00 (Mon-Fri)
- **Simulated Log Path (CSV):** `/home/antonio/fbk_rims/data/generated/csv/simulated_log.csv`
- **Simulated Log Path (XES):** `/home/antonio/fbk_rims/data/generated/xes/simulated_log.xes`

## Overall Cycle Time Benchmark

| Metric | Real Historical | Simulated RIMS++ | Difference |
|---|---|---|---|
| **Cases Evaluated** | 9658 | 9,658 | --- |
| **Completed Work Tasks** | 75,598 | 69,397 | --- |
| **Avg Tasks / Case** | 7.83 | 7.19 | -0.64 |
| **Mean Case Cycle Time** | 11.3 days | 13.5 days | 2.1 days |
| **Median Case Cycle Time** | 8.3 days | 3.3 days | --- |
| **Cycle Time Wasserstein Dist** | --- | --- | 4.9 days |
| **Relative Event Dist (RED)** | --- | --- | **175.62h EMD** |

## Activity Execution Breakdown

| Activity | Simulated Count | Sim Mean Duration | Sim Mean Wait | Sim Mean Queue Wait |
|---|---|---|---|---|
| `W_Assess fraud` | 4,403 | 3.7 mins | 9.2 hours | 6.8 mins |
| `W_Call after incomplete files` | 8,102 | 2.4 mins | 18.7 hours | 4.0 mins |
| `W_Call after offers` | 24,978 | 2.4 mins | 3.4 days | 7.1 mins |
| `W_Complete application` | 22,368 | 4.8 mins | 21.9 hours | 5.0 mins |
| `W_Handle leads` | 2,570 | 2.0 hours | 6.8 hours | 3.3 mins |
| `W_Validate application` | 6,976 | 2.5 hours | 2.7 days | 9.7 mins |
