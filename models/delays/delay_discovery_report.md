# RIMS++ Chapela-Campa Extraneous Delay & Human Latency Discovery Report

- **Source Log:** `/home/antonio/fbk_rims/data/processed/aligned_BPI_2012.xes`
- **Algorithm:** `ECLAW_PLUS` (Eclipse-Aware Extrapolation)
- **Placement:** `EX_ANTE`
- **Concurrency Threshold (ζ):** `0.2`
- **Outlier Threshold (δ):** `0.05`
- **Generated:** `2026-10-21 12:20:04`

## 1. Discovered External Uncoupled Delays (EUD)

These activity delays passed both the Chapela-Campa un-eclipsed availability test and RIMS++ WIP orthogonality ($|r| < 0.30$):

| Activity | Placement | Positive Ratio | Samples | Median Delay | WIP Corr ($r$) | Clock Type | Best Fit Distribution |
|---|---|---|---|---|---|---|---|
| `W_Call after offers` | `ex_ante` | 84.5% | 25,093 | **1.83 days** (43.92h) | 0.015 | `24_7` | `gamma` |

## 2. Human Inter-Ticket Latency (Worker Refractory Cooldowns)

- **Total Intra-Day Work Gaps Mined:** 51,731 consecutive tasks
- **Global Median Cooldown:** **34.1 seconds** (0.6 mins)
- **Global Mean Cooldown:** **325.4 seconds** (5.4 mins)
- **Champion Fit Distribution:** `lognormal`
- **Profiled Workers:** 49 human agents
