# RIMS+ Simulation Benchmark Report

- **Simulation Mode:** `hybrid_residual`
- **Arrival Mode:** `replay`
- **Simulated Cases:** 9,000
- **Total Work Events:** 56,436
- **Workers in Multi-Skilled Pool:** 59 human resources
- **Calendar Engine:** Working Hours 8:00 - 20:00 (Mon-Fri)
- **Simulated Log Path (CSV):** `/home/antonio/fbk_rims/data/processed/simulated_log.csv`
- **Simulated Log Path (XES):** `/home/antonio/fbk_rims/data/processed/simulated_log.xes`

## Overall Cycle Time Benchmark

| Metric | Real Historical | Simulated RIMS+ | Difference |
|---|---|---|---|
| **Cases Evaluated** | 9658 | 9,000 | --- |
| **Total Work Events** | 141567 | 56,436 | --- |
| **Mean Case Cycle Time** | 11.3 days | 17.8 days | 6.5 days |
| **Median Case Cycle Time** | 8.3 days | 3.9 days | --- |
| **Cycle Time Wasserstein Dist** | --- | --- | 9.0 days |

## Activity Execution Breakdown

| Activity | Simulated Count | Sim Mean Duration | Sim Mean Wait | Sim Mean Queue Wait |
|---|---|---|---|---|
| `W_Assess fraud` | 4,062 | 7.8s | 7.1 hours | 1.6 mins |
| `W_Call after incomplete files` | 7,511 | 4.8 mins | 1.8 days | 49.7s |
| `W_Call after offers` | 25,926 | 2.9 mins | 3.9 days | 1.8 mins |
| `W_Complete application` | 9,702 | 5.6 mins | 2.5 days | 1.2 mins |
| `W_Handle leads` | 2,718 | 4.6 mins | 5.2 hours | 1.4 mins |
| `W_Validate application` | 6,517 | 26.0 mins | 3.5 days | 2.7 mins |
