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
  "window_simulated_intervals": 6563
}
```

## Completed-case metrics

| Metric | Value | Status |
|---|---:|---|
| cfld | 0.1186 | {'status': 'ok'} |
| l_traces | 1.0000 | {'status': 'ok'} |
| ngram_2 | 0.1237 | {'status': 'ok'} |
| ngram_3 | 0.1429 | {'status': 'ok'} |
| aed | 2131.5853 | {'status': 'ok'} |
| ced | 1.3233 | {'status': 'ok'} |
| red_dist | 2056.6707 | {'status': 'ok'} |
| cwd | 1.7563 | {'status': 'ok'} |
| car | 0.2740 | {'status': 'ok'} |
| ctd | 130.5 days | {'status': 'ok'} |
| mean_cycle_difference | 130.5 days | {'status': 'ok'} |

## Observation-window metrics

```json
{
  "aed": 2131.585282982992,
  "ced": 1.3233019104886237,
  "cwd": 1.756303834464004
}
```

## Completed-case cycle times

Mean: reference 147.3 days; simulated 16.8 days.
Median: reference 108.8 days; simulated 11.9 days.

## Activity execution

| Activity | Completed tasks | Mean duration | Mean waiting | Mean worker queue |
|---|---:|---:|---:|---:|
| Authorization Requested | 368 | 1.7 days | 1.0s | 0.0s |
| Back-Office Adjustment Requested | 110 | 3.0 hours | 0.0s | 0.0s |
| Evaluating Request (NO registered letter) | 334 | 4.8 hours | 0.0s | 0.0s |
| Evaluating Request (WITH registered letter) | 60 | 1.9 hours | 28.6s | 0.0s |
| Network Adjustment Requested | 59 | 48.6 days | 9.5 hours | 0.0s |
| Pending Liquidation Request | 893 | 2.2 days | 0.0s | 0.0s |
| Pending Request for Network Information | 59 | 49.8 days | 1.3 mins | 0.0s |
| Pending Request for Reservation Closure | 894 | 4.2 mins | 3.8 mins | 0.0s |
| Pending Request for acquittance of heirs | 59 | 2.5 days | 0.0s | 0.0s |
| Request completed with account closure | 893 | 0.0s | 0.0s | 0.0s |
| Request completed with customer recovery | 1 | 0.0s | 1.0s | 0.0s |
| Request created | 1000 | 1.7 mins | 16.1 mins | 0.0s |
| Request deleted | 106 | 0.0s | 0.0s | 0.0s |
| Service closure Request with BO responsibility | 893 | 9.0 days | 54.7 mins | 0.0s |
| Service closure Request with network responsibility | 834 | 10.5 mins | 1.0s | 0.0s |

Activity statistics above describe completed tasks in the observation window, including tasks in unfinished cases.

## Arrival generation

```json
{
  "status": "ok"
}
```
