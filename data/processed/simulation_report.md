# RIMS++ Simulation Benchmark Report

- **Simulation Mode:** `hybrid_residual`
- **Arrival Mode:** `replay`
- **Simulated Cases:** 2
- **Total Work Events:** 4
- **Workers in Multi-Skilled Pool:** 59 human resources
- **Simulated Log Path (CSV):** `/home/antonio/fbk_rims/data/generated/csv/simulated_log.csv`

## Multi-Perspective Quality Benchmark (Meneghello et al. 2025)

| Dimension | Metric | Real Ground-Truth | Simulated RIMS++ | Benchmark Score |
|---|---|---|---|---|
| **Volume** | Cases Evaluated | 2 | 2 | --- |
| **Volume** | Completed Work Tasks | 4 | 4 | --- |
| **Volume** | Avg Tasks / Case | 2.00 | 2.00 | +0.00 |
| **Control-Flow** | Control-flow Log Distance (CFLD) | --- | --- | **0.0000** (Hungarian) |
| **Control-Flow** | Trace Length Error ($L^{traces}$) | --- | --- | **0.00** events |
| **Control-Flow** | 2-Gram Distance (NGD 2-Gram) | --- | --- | **0.0000** |
| **Control-Flow** | 3-Gram Distance (NGD 3-Gram) | --- | --- | **0.0000** |
| **Temporal** | Absolute Event Distribution (AED) | --- | --- | **0.00h** EMD |
| **Temporal** | Circadian Event Distribution (CED) | --- | --- | **0.00h** EMD |
| **Temporal** | Relative Event Distribution (RED) | --- | --- | **0.00h** EMD |
| **Congestion** | Cycle Time 1WD (CTD) | --- | --- | **1.0 hours** |
| **Congestion** | Case Arrival Rate (CAR) | --- | --- | **0.00h** EMD |
| **Congestion** | Mean Case Cycle Time | 2.5 hours | 3.5 hours | 1.0 hours MAE |
| **Congestion** | Median Case Cycle Time | 2.5 hours | 3.5 hours | --- |

## Activity Execution Breakdown

| Activity | Simulated Count | Sim Mean Duration | Sim Mean Wait | Sim Mean Queue Wait |
|---|---|---|---|---|
| `A` | 2 | 1.0 hours | 0.0s | 0.0s |
| `B` | 2 | 1.0 hours | 0.0s | 0.0s |
