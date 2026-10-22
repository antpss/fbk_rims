# RIMS+ Simulation Benchmark Report

- **Simulation Mode:** `pure_physics`
- **Arrival Mode:** `prophet`
- **Simulated Cases:** 9,000
- **Total Work Events:** 46,552
- **Workers in Multi-Skilled Pool:** 59 human resources
- **Calendar Engine:** Working Hours 8:00 - 20:00 (Mon-Fri)
- **Simulated Log Path (CSV):** `/home/antonio/fbk_rims/data/processed/simulated_log_pure_physics.csv`
- **Simulated Log Path (XES):** `/home/antonio/fbk_rims/data/processed/simulated_log.xes`

## Overall Cycle Time Benchmark

| Metric | Real Historical | Simulated RIMS+ | Difference |
|---|---|---|---|
| **Cases Evaluated** | 9658 | 9,000 | --- |
| **Total Work Events** | 141567 | 46,552 | --- |
| **Mean Case Cycle Time** | 11.3 days | 9.5 days | 1.9 days |
| **Median Case Cycle Time** | 8.3 days | 24.0 hours | --- |
| **Cycle Time Wasserstein Dist** | --- | --- | 7.4 days |

## Activity Execution Breakdown

| Activity | Simulated Count | Sim Mean Duration | Sim Mean Wait | Sim Mean Queue Wait |
|---|---|---|---|---|
| `W_Assess fraud` | 4,239 | 21.6s | 3.5 mins | 3.5 mins |
| `W_Call after incomplete files` | 7,544 | 3.5 mins | 1.7 days | 1.5 mins |
| `W_Call after offers` | 18,972 | 2.0 mins | 2.6 days | 2.1 mins |
| `W_Complete application` | 6,337 | 3.8 mins | 22.5 hours | 2.4 mins |
| `W_Handle leads` | 3,118 | 4.4 mins | 2.4 mins | 2.4 mins |
| `W_Validate application` | 6,342 | 30.3 mins | 2.7 days | 3.5 mins |
