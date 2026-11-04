# RIMS++ Simulation Benchmark Report

- **Simulation Mode:** `hybrid_residual`
- **Arrival Mode:** `prophet`
- **Simulated Cases:** 1,000
- **Total Work Events:** 6,607
- **Workers in Multi-Skilled Pool:** 59 human resources
- **Simulated Log Path (CSV):** `/home/antonio/fbk_rims/data/generated/csv/simulated_log.csv`
- **Simulated Log Path (XES):** `/home/antonio/fbk_rims/data/generated/xes/simulated_log.xes`

## Multi-Perspective Quality Benchmark (Meneghello et al. 2025)

| Dimension | Metric | Real Ground-Truth | Simulated RIMS++ | Benchmark Score |
|---|---|---|---|---|
| **Volume** | Cases Evaluated | 32,429 | 1,000 | --- |
| **Volume** | Completed Work Tasks | 215,334 | 6,607 | --- |
| **Volume** | Avg Tasks / Case | 6.64 | 6.61 | -0.03 |
| **Control-Flow** | Control-flow Log Distance (CFLD) | --- | --- | **0.0253** (Hungarian) |
| **Control-Flow** | Trace Length Error ($L^{traces}$) | --- | --- | **0.09** events |
| **Control-Flow** | 2-Gram Distance (NGD 2-Gram) | --- | --- | **0.0117** |
| **Control-Flow** | 3-Gram Distance (NGD 3-Gram) | --- | --- | **0.0205** |
| **Temporal** | Absolute Event Distribution (AED) | --- | --- | **12.70h** EMD |
| **Temporal** | Circadian Event Distribution (CED) | --- | --- | **18.94h** EMD |
| **Temporal** | Relative Event Distribution (RED) | --- | --- | **158.50h** EMD |
| **Congestion** | Cycle Time 1WD (CTD) | --- | --- | **23.3 days** |
| **Congestion** | Case Arrival Rate (CAR) | --- | --- | **12.50h** EMD |
| **Congestion** | Mean Case Cycle Time | 15.5 days | 32.2 days | 16.8 days MAE |
| **Congestion** | Median Case Cycle Time | 7.7 days | 39.0 days | --- |

## Activity Execution Breakdown

| Activity | Simulated Count | Sim Mean Duration | Sim Mean Wait | Sim Mean Queue Wait |
|---|---|---|---|---|
| `Authorization Requested` | 371 | 2.9 hours | 9.1 mins | 0.0s |
| `Back-Office Adjustment Requested` | 110 | 19.0 mins | 2.4 days | 0.0s |
| `Evaluating Request (NO registered letter)` | 341 | 7.4 hours | 11.1 mins | 0.0s |
| `Evaluating Request (WITH registered letter)` | 71 | 1.6 hours | 6.2 days | 0.0s |
| `Network Adjustment Requested` | 58 | 1.1 days | 5.9 days | 0.0s |
| `Pending Liquidation Request` | 895 | 4.3 days | 3.2 hours | 0.0s |
| `Pending Request for Network Information` | 27 | 7.1 days | 36.8s | 0.0s |
| `Pending Request for Reservation Closure` | 897 | 52.0 mins | 13.7 hours | 0.0s |
| `Pending Request for acquittance of heirs` | 43 | 2.5 hours | 23.4 hours | 0.0s |
| `Request completed with account closure` | 895 | 22.8 days | 4.7 days | 0.0s |
| `Request completed with customer recovery` | 5 | 2.5 hours | 4.1 days | 0.0s |
| `Request created` | 1,000 | 1.2 hours | 9.8 hours | 0.0s |
| `Request deleted` | 100 | 19.1 hours | 32.1s | 0.0s |
| `Service closure Request with BO responsibility` | 926 | 1.3 days | 2.9 hours | 0.0s |
| `Service closure Request with network responsibility` | 868 | 35.4 mins | 2.0 mins | 0.0s |
