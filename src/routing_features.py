"""Temporal contract shared only by routing training and inference."""

import math

import pandas as pd


TEMPORAL_FEATURE_VERSION = 1
TEMPORAL_COLUMNS = (
    "start_hour", "start_weekday", "last_event_hour",
    "last_event_weekday", "elapsed_since_start",
)


def routing_temporal_features(trace_start_dt, last_event_dt=None):
    """Use arrival until the first observed completion; age is elapsed seconds."""
    if trace_start_dt is None or pd.isna(trace_start_dt):
        raise ValueError("Routing requires a valid trace_start_dt")
    start = pd.Timestamp(trace_start_dt)
    last = start if last_event_dt is None else pd.Timestamp(last_event_dt)
    if pd.isna(last):
        raise ValueError("Routing requires a valid last_event_dt")
    try:
        elapsed = (last - start).total_seconds()
    except TypeError as exc:
        raise ValueError("Routing timestamps must use compatible timezones") from exc
    if not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("Routing last completion must not precede trace arrival")
    return dict(zip(TEMPORAL_COLUMNS, (
        start.hour, start.weekday(), last.hour, last.weekday(), float(elapsed),
    )))


def validate_routing_attribute_names(names):
    collisions = set(names).intersection((*TEMPORAL_COLUMNS, "hour", "weekday"))
    if collisions:
        raise ValueError(f"Case attributes collide with routing temporal features: {sorted(collisions)}")
