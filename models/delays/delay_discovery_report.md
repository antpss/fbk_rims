# RIMS+ Agnostic Causal Delay & Human Latency Discovery Report

- **Source Log:** `/home/antonio/fbk_rims/data/processed/aligned_BPI_2012.xes`
- **Engine:** RIMS+ Tier 4 Causal Discovery Engine
- **Generated:** 2026-10-02 15:44:22

## 1. Discovered External Uncoupled Delays (EUD)

These transitions passed the Triple Causal Test (No active human resource, WIP correlation $|r| < 0.30$, and 24/7 calendar clock progression):

| Trigger Activity | Resume Activity | Samples | Median Delay | WIP Corr ($r$) | 24/7 Off-Hour Ratio | Best Fit Distribution |
|---|---|---|---|---|---|---|
| `W_Complete application` | `W_Validate application` | 190 | **8.08 days** (193.99h) | -0.033 | 70.4% | `gamma` |
| `W_Complete application` | `O_SENT_BACK` | 561 | **5.93 days** (142.25h) | 0.027 | 64.6% | `exponential` |
| `W_Call after offers` | `A_REGISTERED` | 92 | **3.12 days** (74.92h) | 0.009 | 60.1% | `gamma` |
| `W_Call after offers` | `A_ACTIVATED` | 81 | **3.05 days** (73.17h) | -0.125 | 54.5% | `exponential` |
| `W_Call after offers` | `A_APPROVED` | 167 | **3.05 days** (73.18h) | -0.096 | 57.7% | `exponential` |
| `W_Call after offers` | `O_SENT_BACK` | 2,693 | **2.97 days** (71.21h) | -0.017 | 69.7% | `gamma` |
| `W_Call after offers` | `O_ACCEPTED` | 304 | **2.96 days** (71.11h) | 0.058 | 57.3% | `exponential` |
| `W_Call after offers` | `A_DECLINED` | 213 | **2.88 days** (69.18h) | -0.085 | 61.6% | `exponential` |
| `W_Call after offers` | `O_DECLINED` | 183 | **2.28 days** (54.68h) | -0.101 | 58.8% | `exponential` |
| `W_Call after offers` | `W_Validate application` | 2,474 | **2.21 days** (53.1h) | -0.027 | 57.4% | `exponential` |
| `W_Call after offers` | `W_Call after offers` | 13,162 | **1.84 days** (44.23h) | 0.034 | 63.0% | `gamma` |
| `W_Call after offers` | `O_CANCELLED` | 1,046 | **1.05 days** (25.11h) | -0.037 | 56.6% | `gamma` |
| `W_Call after incomplete files` | `O_SENT_BACK` | 190 | **1.0 days** (23.95h) | 0.049 | 56.0% | `gamma` |
| `A_PARTLYSUBMITTED` | `W_Assess fraud` | 66 | **0.89 days** (21.39h) | 0.110 | 63.1% | `gamma` |
| `W_Call after offers` | `A_CANCELLED` | 689 | **0.82 days** (19.75h) | -0.056 | 54.8% | `gamma` |
| `W_Call after offers` | `O_SELECTED` | 438 | **0.78 days** (18.6h) | -0.032 | 55.3% | `gamma` |
| `W_Complete application` | `W_Complete application` | 12,323 | **0.23 days** (5.49h) | 0.044 | 49.3% | `lognormal` |
| `A_PREACCEPTED` | `W_Complete application` | 2,791 | **0.15 days** (3.54h) | 0.122 | 52.6% | `gamma` |
| `W_Call after incomplete files` | `W_Call after incomplete files` | 8,921 | **0.14 days** (3.32h) | 0.052 | 40.1% | `lognormal` |
| `W_Complete application` | `A_CANCELLED` | 723 | **0.09 days** (2.11h) | -0.060 | 51.0% | `gamma` |
| `W_Complete application` | `A_DECLINED` | 685 | **0.09 days** (2.06h) | 0.033 | 44.3% | `lognormal` |

## 2. Human Inter-Ticket Latency (Worker Refractory Buffers)

- **Total Transitions Mined:** 51,731 intra-day consecutive tasks
- **Global Median Idle Gap:** **34.1 seconds** (0.6 mins)
- **Global Mean Idle Gap:** **325.4 seconds** (5.4 mins)
- **Champion Fit Distribution:** `lognormal`
- **Distinct Workers Profiled:** 49 human agents
