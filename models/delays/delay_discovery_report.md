# RIMS++ Chapela-Campa Extraneous Delay & Human Latency Discovery Report

- **Source Log:** `/home/antonio/fbk_rims/data/processed/aligned_BAC.xes`
- **Algorithm:** `ECLAW_PLUS` (Eclipse-Aware Extrapolation)
- **Placement:** `EX_ANTE`
- **Concurrency Threshold (ζ):** `0.2`
- **Outlier Threshold (δ):** `0.05`
- **Generated:** `2026-11-23 12:22:51`

## 1. Discovered External Uncoupled Delays (EUD)

These activity delays passed both the Chapela-Campa un-eclipsed availability test and RIMS++ WIP orthogonality ($|r| < 0.30$):

| Activity | Placement | Positive Ratio | Samples | Median Delay | WIP Corr ($r$) | Clock Type | Best Fit Distribution |
|---|---|---|---|---|---|---|---|

## 2. Human Inter-Ticket Latency (Worker Refractory Cooldowns)

- **Total Intra-Day Work Gaps Mined:** 42,273 consecutive tasks
- **Global Median Cooldown:** **75.0 seconds** (1.2 mins)
- **Global Mean Cooldown:** **640.2 seconds** (10.7 mins)
- **Champion Fit Distribution:** `lognormal`
- **Profiled Workers:** 41 human agents
