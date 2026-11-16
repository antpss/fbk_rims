# Simulation benchmark

Full-case metrics below are conditional on completion; censored and failed cases are excluded.
Replay uses paired completed IDs. Window metrics use observed intervals clipped at the cutoff; clipped endpoints are not recorded completions.

## Populations

```json
{
  "simulated_scheduled": 32429,
  "simulated_statuses": {
    "completed": 32429
  },
  "reference_cases": 32429,
  "completed_reference_evaluated": 1000,
  "completed_simulated_evaluated": 1000,
  "window_reference_intervals": 215314,
  "window_simulated_intervals": 216705
}
```

## Completed-case metrics

| Metric | Value | Status |
|---|---:|---|
| cfld | 0.0276 | {'status': 'ok'} |
| l_traces | 0.1010 | {'status': 'ok'} |
| ngram_2 | 0.0184 | {'status': 'ok'} |
| ngram_3 | 0.0269 | {'status': 'ok'} |
| aed | 69.3095 | {'status': 'ok'} |
| ced | 1.3555 | {'status': 'ok'} |
| red_dist | 115.4716 | {'status': 'ok'} |
| cwd | 1.5545 | {'status': 'ok'} |
| car | 0.6870 | {'status': 'ok'} |
| ctd | 7.4 days | {'status': 'ok'} |
| mean_cycle_difference | 6.8 days | {'status': 'ok'} |

## Observation-window metrics

```json
{
  "aed": 110.43794076541903,
  "ced": 1.5076062827829164,
  "cwd": 0.8526935661946767
}
```

## Completed-case cycle times

Mean: reference 13.8 days; simulated 6.9 days.
Median: reference 7.6 days; simulated 6.7 days.

## Activity execution

| Activity | Completed tasks | Mean duration | Mean waiting | Mean worker queue |
|---|---:|---:|---:|---:|
| Authorization Requested | 12812 | 7.8 hours | 1.0s | 0.0s |
| Back-Office Adjustment Requested | 3333 | 18.1 hours | 0.0s | 0.0s |
| Evaluating Request (NO registered letter) | 10817 | 1.7 hours | 0.0s | 0.0s |
| Evaluating Request (WITH registered letter) | 2203 | 1.1 hours | 4.0 mins | 0.0s |
| Network Adjustment Requested | 2184 | 3.3 days | 8.7 hours | 0.0s |
| Pending Liquidation Request | 29355 | 2.1 days | 0.0s | 0.0s |
| Pending Request for Network Information | 1368 | 15.1 days | 1.7 hours | 1.7 hours |
| Pending Request for Reservation Closure | 29482 | 3.4 mins | 3.6 mins | 0.0s |
| Pending Request for acquittance of heirs | 2135 | 1.2 days | 0.0s | 0.0s |
| Request completed with account closure | 29355 | 0.0s | 0.0s | 0.0s |
| Request completed with customer recovery | 89 | 0.0s | 1.0s | 0.0s |
| Request created | 32429 | 1.3 mins | 40.3 mins | 0.0s |
| Request deleted | 2985 | 0.0s | 0.0s | 0.0s |
| Service closure Request with BO responsibility | 30171 | 4.2 days | 48.6 mins | 0.0s |
| Service closure Request with network responsibility | 27987 | 4.1 mins | 1.0s | 0.0s |

Activity statistics above describe completed tasks in the observation window, including tasks in unfinished cases.

## Arrival generation

```json
{
  "status": "ok"
}
```
