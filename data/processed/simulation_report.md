# RIMS++ Simulation Benchmark Report

- **Simulation Mode:** `hybrid_residual`
- **Arrival Mode:** `prophet`
- **Simulated Cases:** 1,000
- **Total Work Events:** 5,690
- **Workers in Multi-Skilled Pool:** 58 human resources
- **Calendar Engine:** Working Hours 8:00 - 20:00 (Mon-Fri)
- **Simulated Log Path (CSV):** `/home/antonio/fbk_rims/data/generated/csv/simulated_log.csv`
- **Simulated Log Path (XES):** `/home/antonio/fbk_rims/data/generated/xes/simulated_log.xes`

## Multi-Perspective Quality Benchmark

| Dimension | Metric | Real Ground-Truth | Simulated RIMS++ | Benchmark Score |
|---|---|---|---|---|
| **Volume** | Cases Evaluated | 32,429 | 1,000 | --- |
| **Volume** | Completed Work Tasks | 215,334 | 5,690 | --- |
| **Volume** | Avg Tasks / Case | 6.64 | 5.69 | -0.95 |
| **Control-Flow** | Control-flow Log Distance (CFLD) | --- | --- | **1.0000** (Hungarian) |
| **Control-Flow** | Trace Length Error ($L^{traces}$) | --- | --- | **5.17** events |
| **Control-Flow** | 2-Gram Distance (NGD 2-Gram) | --- | --- | **1.0000** |
| **Control-Flow** | 3-Gram Distance (NGD 3-Gram) | --- | --- | **1.0000** |
| **Temporal** | Absolute Event Distribution (AED) | --- | --- | **12.69h** EMD |
| **Temporal** | Circadian Event Distribution (CED) | --- | --- | **11.81h** EMD |
| **Temporal** | Relative Event Distribution (RED) | --- | --- | **180.21h** EMD |
| **Congestion** | Cycle Time 1WD (CTD) | --- | --- | **12.0 days** |
| **Congestion** | Case Arrival Rate (CAR) | --- | --- | **12.50h** EMD |
| **Congestion** | Mean Case Cycle Time | 15.5 days | 3.5 days | 12.0 days MAE |
| **Congestion** | Median Case Cycle Time | 7.7 days | 2.2 days | --- |

## Activity Execution Breakdown

| Activity | Simulated Count | Sim Mean Duration | Sim Mean Wait | Sim Mean Queue Wait |
|---|---|---|---|---|
| `W_Assess fraud` | 451 | 12.4s | 10.1 hours | 31.4s |
| `W_Call after incomplete files` | 980 | 2.4 mins | 29.6 mins | 0.0s |
| `W_Call after offers` | 569 | 3.0 mins | 3.1 days | 24.9s |
| `W_Complete application` | 569 | 3.6 mins | 6.4 hours | 11.7s |
| `W_Handle leads` | 2,349 | 3.4 mins | 46.0 mins | 10.3s |
| `W_Validate application` | 772 | 25.9 mins | 2.1 days | 50.8s |
