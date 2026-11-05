# RIMS++ Simulation Benchmark Report

- **Simulation Mode:** `hybrid_residual`
- **Arrival Mode:** `replay`
- **Simulated Cases:** 1,000
- **Total Work Events:** 6,700
- **Workers in Multi-Skilled Pool:** 655 human resources
- **Calendar Engine:** Working Hours 8:00 - 20:00 (Mon-Fri)
- **Simulated Log Path (CSV):** `/home/antonio/fbk_rims/data/generated/csv/simulated_log.csv`
- **Simulated Log Path (XES):** `/home/antonio/fbk_rims/data/generated/xes/simulated_log.xes`

## Multi-Perspective Quality Benchmark (log-distance-measures / WISDOM)

| Dimension | Metric | Real Ground-Truth | Simulated RIMS++ | Benchmark Score |
|---|---|---|---|---|
| **Volume** | Cases Evaluated | 1,000 | 1,000 | --- |
| **Volume** | Completed Work Tasks | 6,585 | 6,700 | --- |
| **Volume** | Avg Tasks / Case | 6.58 | 6.70 | +0.12 |
| **Control-Flow** | Control-flow Log Distance (CFLD) | --- | --- | **0.0308** (log-distance) |
| **Control-Flow** | Trace Length Error ($L^{traces}$) | --- | --- | **0.15** events (Hungarian) |
| **Control-Flow** | 2-Gram Distance (NGD 2-Gram) | --- | --- | **0.0193** |
| **Control-Flow** | 3-Gram Distance (NGD 3-Gram) | --- | --- | **0.0245** |
| **Temporal** | Absolute Event Distribution (AED) | --- | --- | **4676.67h** EMD |
| **Temporal** | Circadian Event Distribution (CED) | --- | --- | **4.58h** EMD |
| **Temporal** | Relative Event Distribution (RED) | --- | --- | **75.01h** EMD |
| **Workforce** | Circadian Workforce Dist (CWD) | --- | --- | **3.82h** EMD |
| **Congestion** | Cycle Time Distribution (CTD) | --- | --- | **7.0 mins** |
| **Congestion** | Case Arrival Rate (CAR) | --- | --- | **4650.77h** EMD |
| **Congestion** | Mean Case Cycle Time | 13.8 days | 27.3 days | 13.6 days MAE |
| **Congestion** | Median Case Cycle Time | 7.6 days | 35.9 days | --- |

## Activity Execution Breakdown

| Activity | Simulated Count | Sim Mean Duration | Sim Mean Wait | Sim Mean Queue Wait |
|---|---|---|---|---|
| `Authorization Requested` | 379 | 1.0 hours | 1.0s | 0.0s |
| `Back-Office Adjustment Requested` | 104 | 55.2 mins | 0.0s | 0.0s |
| `Evaluating Request (NO registered letter)` | 368 | 38.9 mins | 0.0s | 0.0s |
| `Evaluating Request (WITH registered letter)` | 65 | 9.5 hours | 2.8 mins | 0.0s |
| `Network Adjustment Requested` | 61 | 5.1 days | 1.0s | 0.0s |
| `Pending Liquidation Request` | 906 | 2.7 days | 0.0s | 0.0s |
| `Pending Request for Network Information` | 19 | 38.2 days | 0.0s | 0.0s |
| `Pending Request for Reservation Closure` | 911 | 15.0 mins | 0.0s | 0.0s |
| `Pending Request for acquittance of heirs` | 52 | 6.7 hours | 0.0s | 0.0s |
| `Request completed with account closure` | 906 | 21.5 days | 0.0s | 0.0s |
| `Request completed with customer recovery` | 4 | 2.7 days | 1.0s | 0.0s |
| `Request created` | 1,000 | 34.0 mins | 0.0s | 0.0s |
| `Request deleted` | 90 | 8.5 hours | 0.0s | 0.0s |
| `Service closure Request with BO responsibility` | 948 | 4.5 days | 0.0s | 0.0s |
| `Service closure Request with network responsibility` | 887 | 4.0 mins | 1.0s | 0.0s |
