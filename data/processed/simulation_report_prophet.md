# RIMS+ Simulation Benchmark Report

- **Simulation Mode:** `hybrid_residual`
- **Arrival Mode:** `prophet`
- **Simulated Cases:** 9,000
- **Total Work Events:** 46,152
- **Workers in Multi-Skilled Pool:** 59 human resources
- **Calendar Engine:** Working Hours 8:00 - 20:00 (Mon-Fri)
- **Simulated Log Path (CSV):** `/home/antonio/fbk_rims/data/processed/simulated_log.csv`
- **Simulated Log Path (XES):** `/home/antonio/fbk_rims/data/processed/simulated_log.xes`

## Overall Cycle Time Benchmark

| Metric | Real Historical | Simulated RIMS+ | Difference |
|---|---|---|---|
| **Cases Evaluated** | 9658 | 9,000 | --- |
| **Total Work Events** | 141567 | 46,152 | --- |
| **Mean Case Cycle Time** | 11.3 days | 13.5 days | 2.2 days |
| **Median Case Cycle Time** | 8.3 days | 3.6 days | --- |
| **Cycle Time Wasserstein Dist** | --- | --- | 6.6 days |

## Activity Execution Breakdown

| Activity | Simulated Count | Sim Mean Duration | Sim Mean Wait | Sim Mean Queue Wait |
|---|---|---|---|---|
| `W_Assess fraud` | 4,144 | 25.1s | 18.7 mins | 9.9 mins |
| `W_Call after incomplete files` | 7,793 | 4.6 mins | 1.8 days | 1.1 mins |
| `W_Call after offers` | 18,258 | 2.8 mins | 4.0 days | 2.2 mins |
| `W_Complete application` | 6,511 | 6.6 mins | 21.7 hours | 5.2 mins |
| `W_Handle leads` | 3,001 | 4.5 mins | 7.2 mins | 5.7 mins |
| `W_Validate application` | 6,445 | 38.6 mins | 4.3 days | 4.1 mins |
