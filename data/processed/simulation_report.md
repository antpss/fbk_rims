# Simulation benchmark

Full-case metrics below are conditional on completion; censored and failed cases are excluded.
Replay uses paired completed IDs. Window metrics use observed intervals clipped at the cutoff; clipped endpoints are not recorded completions.

## Populations

```json
{
  "simulated_scheduled": 1000,
  "simulated_statuses": {
    "completed": 1000
  },
  "reference_cases": 1000,
  "completed_reference_evaluated": 1000,
  "completed_simulated_evaluated": 1000,
  "window_reference_intervals": 7563,
  "window_simulated_intervals": 6660
}
```

## Completed-case metrics

| Metric | Value | Status |
|---|---:|---|
| cfld | 0.1000 | {'status': 'ok'} |
| l_traces | 0.9030 | {'status': 'ok'} |
| ngram_2 | 0.1077 | {'status': 'ok'} |
| ngram_3 | 0.1359 | {'status': 'ok'} |
| aed | 2091.8646 | {'status': 'ok'} |
| ced | 1.5207 | {'status': 'ok'} |
| red_dist | 1998.9041 | {'status': 'ok'} |
| cwd | 1.9404 | {'status': 'ok'} |
| car | 0.2740 | {'status': 'ok'} |
| ctd | 126.1 days | {'status': 'ok'} |
| mean_cycle_difference | 126.1 days | {'status': 'ok'} |

## Observation-window metrics

```json
{
  "aed": 2091.8645636413885,
  "ced": 1.520748037900127,
  "cwd": 1.940359470237636
}
```

## Completed-case cycle times

Mean: reference 147.3 days; simulated 21.1 days.
Median: reference 108.8 days; simulated 16.5 days.

## Activity execution

| Activity | Completed tasks | Mean duration | Mean waiting | Mean worker queue |
|---|---:|---:|---:|---:|
| Authorization Requested | 457 | 6.7 days | 1.0s | 0.0s |
| Back-Office Adjustment Requested | 34 | 3.6 hours | 0.0s | 0.0s |
| Evaluating Request (NO registered letter) | 331 | 17.0 hours | 0.0s | 0.0s |
| Evaluating Request (WITH registered letter) | 69 | 13.0 hours | 2.5 mins | 0.0s |
| Network Adjustment Requested | 59 | 55.8 days | 8.5 hours | 0.0s |
| Pending Liquidation Request | 910 | 2.2 days | 0.0s | 0.0s |
| Pending Request for Network Information | 40 | 33.5 days | 1.7 mins | 0.0s |
| Pending Request for Reservation Closure | 914 | 3.2 mins | 2.5 mins | 0.0s |
| Pending Request for acquittance of heirs | 47 | 1.5 days | 0.0s | 0.0s |
| Request completed with account closure | 910 | 0.0s | 0.0s | 0.0s |
| Request completed with customer recovery | 2 | 0.0s | 1.0s | 0.0s |
| Request created | 1000 | 1.4 mins | 16.1 mins | 0.0s |
| Request deleted | 88 | 0.0s | 0.0s | 0.0s |
| Service closure Request with BO responsibility | 929 | 11.8 days | 58.8 mins | 0.0s |
| Service closure Request with network responsibility | 870 | 17.4 mins | 1.0s | 0.0s |

Activity statistics above describe completed tasks in the observation window, including tasks in unfinished cases.

## Arrival generation

```json
{
  "status": "ok"
}
```
