# RIMS++ Simulation Benchmark Report

- **Simulation Mode:** `hybrid_residual`
- **Arrival Mode:** `generative`
- **Simulated Cases:** 9,000
- **Total Work Events:** 47,193
- **Workers in Multi-Skilled Pool:** 59 human resources
- **Calendar Engine:** Working Hours 8:00 - 20:00 (Mon-Fri)
- **Simulated Log Path (CSV):** `/home/antonio/fbk_rims/data/processed/simulated_log.csv`
- **Simulated Log Path (XES):** `/home/antonio/fbk_rims/data/processed/simulated_log.xes`

## Overall Cycle Time Benchmark

| Metric | Real Historical | Simulated RIMS++ | Difference |
|---|---|---|---|
| **Cases Evaluated** | 9658 | 9,000 | --- |
| **Total Work Events** | 141567 | 47,193 | --- |
| **Mean Case Cycle Time** | 11.3 days | 13.3 days | 1.9 days |
| **Median Case Cycle Time** | 8.3 days | 3.7 days | --- |
| **Cycle Time Wasserstein Dist** | --- | --- | 6.8 days |

## Activity Execution Breakdown

| Activity | Simulated Count | Sim Mean Duration | Sim Mean Wait | Sim Mean Queue Wait |
|---|---|---|---|---|
| `W_Assess fraud` | 4,175 | 10.5s | 11.0 mins | 3.4s |
| `W_Call after incomplete files` | 7,567 | 5.3 mins | 1.9 days | 1.2 mins |
| `W_Call after offers` | 19,321 | 2.8 mins | 3.9 days | 2.2 mins |
| `W_Complete application` | 6,478 | 10.6 mins | 23.0 hours | 27.5s |
| `W_Handle leads` | 3,194 | 5.6 mins | 5.0 mins | 2.0s |
| `W_Validate application` | 6,458 | 36.8 mins | 3.6 days | 4.1 mins |
