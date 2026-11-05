# RIMS++ Simulation Benchmark Report

- **Simulation Mode:** `hybrid_residual`
- **Arrival Mode:** `prophet`
- **Simulated Cases:** 1,000
- **Total Work Events:** 6,687
- **Workers in Multi-Skilled Pool:** 655 human resources
- **Calendar Engine:** Working Hours 8:00 - 20:00 (Mon-Fri)
- **Simulated Log Path (CSV):** `/home/antonio/fbk_rims/data/generated/csv/simulated_log.csv`
- **Simulated Log Path (XES):** `/home/antonio/fbk_rims/data/generated/xes/simulated_log.xes`

## Multi-Perspective Quality Benchmark (log-distance-measures / WISDOM)

| Dimension | Metric | Real Ground-Truth | Simulated RIMS++ | Benchmark Score |
|---|---|---|---|---|
| **Volume** | Cases Evaluated | 1,000 | 1,000 | --- |
| **Volume** | Completed Work Tasks | 6,585 | 6,687 | --- |
| **Volume** | Avg Tasks / Case | 6.58 | 6.69 | +0.10 |
| **Control-Flow** | Control-flow Log Distance (CFLD) | --- | --- | **0.0359** (log-distance) |
| **Control-Flow** | Trace Length Error ($L^{traces}$) | --- | --- | **0.13** events (Hungarian) |
| **Control-Flow** | 2-Gram Distance (NGD 2-Gram) | --- | --- | **0.0202** |
| **Control-Flow** | 3-Gram Distance (NGD 3-Gram) | --- | --- | **0.0285** |
| **Temporal** | Absolute Event Distribution (AED) | --- | --- | **12895.81h** EMD |
| **Temporal** | Circadian Event Distribution (CED) | --- | --- | **4.54h** EMD |
| **Temporal** | Relative Event Distribution (RED) | --- | --- | **102.26h** EMD |
| **Workforce** | Circadian Workforce Dist (CWD) | --- | --- | **4.04h** EMD |
| **Congestion** | Cycle Time Distribution (CTD) | --- | --- | **2.8 mins** |
| **Congestion** | Case Arrival Rate (CAR) | --- | --- | **12833.44h** EMD |
| **Congestion** | Mean Case Cycle Time | 13.8 days | 8.4 days | 5.4 days MAE |
| **Congestion** | Median Case Cycle Time | 7.6 days | 7.7 days | --- |

## Activity Execution Breakdown

| Activity | Simulated Count | Sim Mean Duration | Sim Mean Wait | Sim Mean Queue Wait |
|---|---|---|---|---|
| `Authorization Requested` | 412 | 45.4 mins | 1.0s | 0.0s |
| `Back-Office Adjustment Requested` | 101 | 51.4 mins | 0.0s | 0.0s |
| `Evaluating Request (NO registered letter)` | 342 | 37.6 mins | 0.0s | 0.0s |
| `Evaluating Request (WITH registered letter)` | 61 | 6.9 hours | 3.2 mins | 0.0s |
| `Network Adjustment Requested` | 58 | 3.4 days | 1.0s | 0.0s |
| `Pending Liquidation Request` | 905 | 2.7 days | 0.0s | 0.0s |
| `Pending Request for Network Information` | 31 | 39.2 days | 0.0s | 0.0s |
| `Pending Request for Reservation Closure` | 908 | 47.7 mins | 0.0s | 0.0s |
| `Pending Request for acquittance of heirs` | 63 | 5.9 hours | 0.0s | 0.0s |
| `Request completed with account closure` | 905 | 0.0s | 0.0s | 0.0s |
| `Request completed with customer recovery` | 2 | 0.0s | 1.0s | 0.0s |
| `Request created` | 1,000 | 1.0 hours | 9.8 hours | 0.0s |
| `Request deleted` | 93 | 0.0s | 0.0s | 0.0s |
| `Service closure Request with BO responsibility` | 932 | 4.8 days | 0.0s | 0.0s |
| `Service closure Request with network responsibility` | 874 | 3.0 mins | 1.0s | 0.0s |
