# RIMS++ Simulation Benchmark Report

- **Simulation Mode:** `hybrid_residual`
- **Arrival Mode:** `prophet`
- **Simulated Cases:** 9,658
- **Total Work Events:** 58,414
- **Workers in Multi-Skilled Pool:** 58 human resources
- **Calendar Engine:** Working Hours 8:00 - 20:00 (Mon-Fri)
- **Simulated Log Path (CSV):** `/home/antonio/fbk_rims/data/generated/csv/simulated_log.csv`
- **Simulated Log Path (XES):** `/home/antonio/fbk_rims/data/generated/xes/simulated_log.xes`

## Overall Cycle Time Benchmark

| Metric | Real Historical | Simulated RIMS++ | Difference |
|---|---|---|---|
| **Cases Evaluated** | 9658 | 9,658 | --- |
| **Completed Work Tasks** | 75,598 | 58,414 | --- |
| **Avg Tasks / Case** | 7.83 | 6.05 | -1.78 |
| **Mean Case Cycle Time** | 11.3 days | 10.8 days | 13.8 hours |
| **Median Case Cycle Time** | 8.3 days | 3.0 days | --- |
| **Cycle Time Wasserstein Dist** | --- | --- | 4.8 days |
| **Relative Event Dist (RED)** | --- | --- | **135.47h EMD** |

## Activity Execution Breakdown

| Activity | Simulated Count | Sim Mean Duration | Sim Mean Wait | Sim Mean Queue Wait |
|---|---|---|---|---|
| `W_Assess fraud` | 4,304 | 2.8 mins | 9.1 hours | 5.9 mins |
| `W_Call after incomplete files` | 8,395 | 2.3 mins | 18.2 hours | 3.6 mins |
| `W_Call after offers` | 19,763 | 2.5 mins | 3.3 days | 6.6 mins |
| `W_Complete application` | 16,235 | 4.3 mins | 20.1 hours | 4.3 mins |
| `W_Handle leads` | 2,589 | 1.9 hours | 6.7 hours | 3.4 mins |
| `W_Validate application` | 7,128 | 2.6 hours | 2.7 days | 10.3 mins |
