"""Feature snapshots from observed execution intervals and estimated eligibility."""
from collections import defaultdict, deque
import json

import pandas as pd

from features import FeatureContext, capacity_occupancy


def execution_instances(df):
    """Pair lifecycle events FIFO, retaining completion-only observations as context."""
    pending = defaultdict(deque)
    instances = []
    for order, row in enumerate(df.to_dict("records")):
        key = (str(row["case:concept:name"]), str(row["concept:name"]))
        transition = str(row.get("lifecycle:transition", "COMPLETE")).upper()
        if transition == "START":
            pending[key].append((order, row))
        elif transition == "COMPLETE":
            start_order, start_row = pending[key].popleft() if pending[key] else (order, None)
            instances.append({"id": str(order), "case": key[0], "act": key[1],
                              "start": start_row["time:timestamp"] if start_row else row["time:timestamp"],
                              "end": row["time:timestamp"], "start_row": start_row, "row": row,
                              "start_order": start_order, "end_order": order})
    return instances


def snapshot_rows(df, cfg, task, manager, calendar, role_data, activity_types,
                  activities, tracked, imputation, delay_records=None, manifest=None, diagnostics=None):
    df = df.copy()
    df["case:concept:name"] = df["case:concept:name"].astype(str)
    instances = execution_instances(df)
    bounds = df.groupby("case:concept:name")["time:timestamp"].agg(["min", "max"])
    if manifest:
        for cid in bounds.index:
            if cid in manifest["cases"]:
                bounds.loc[cid, "min"] = pd.Timestamp(manifest["cases"][cid]["arrival"])
                bounds.loc[cid, "max"] = pd.Timestamp(manifest["cases"][cid]["end"])
    case_first = df.drop_duplicates("case:concept:name", keep="first").set_index("case:concept:name").to_dict("index")
    delay_lookup = defaultdict(deque)
    if delay_records is not None:
        for rec in delay_records.to_dict("records"):
            key = (str(rec["case_id"]), str(rec["target_act"]), pd.Timestamp(rec["start_timestamp"]))
            delay_lookup[key].append(rec)
    last_completion = {}
    for inst in instances:
        previous = last_completion.get(inst["case"])
        if previous is None or (inst["end"], inst["end_order"]) > (previous["end"], previous["end_order"]):
            last_completion[inst["case"]] = inst
    timeline = []
    for cid, bound in bounds.iterrows():
        timeline.append((bound["min"], -1, 0, "arrival", cid))
        last = last_completion.get(cid)
        phase = 0 if last and last["end"] == bound["max"] and last["end"] > last["start"] else 2
        timeline.append((bound["max"], phase, float("inf"), "departure", cid))
    skipped = defaultdict(int)
    for inst in instances:
        act, cid = inst["act"], inst["case"]
        synthetic = str(inst["row"].get("is_synthetic", "false")).lower() == "true"
        typ = activity_types.get(act, "HUMAN_WORK_ITEM")
        inst["human"] = typ == "HUMAN_WORK_ITEM" and not synthetic
        inst["valid"] = not synthetic and (inst["start_row"] is not None or typ in ("AUTOMATED_TASK", "INSTANT_GATEWAY"))
        inst["worker"] = str((inst["start_row"] or inst["row"]).get("org:resource", "UNKNOWN"))
        inst["duration"] = calendar.calculate_net_working_seconds(inst["start"], inst["end"], role_or_act=act) if inst["start_row"] else 0.0
        if inst["start_row"] is not None and inst["human"]:
            timeline.append((inst["start"], 1, inst["start_order"] * 3, "start", inst))
        # Positive-duration completions release slots before same-time predictions.
        phase = 0 if inst["end"] > inst["start"] else 1
        timeline.append((inst["end"], phase, inst["end_order"] * 3 + 1, "complete", inst))
        if act not in activities:
            skipped["activity_outside_training_inventory"] += 1
            continue
        if not inst["valid"]:
            skipped["synthetic" if synthetic else "missing_start"] += 1
            continue
        if task == "duration":
            inst["eligible"] = inst["start"]
            timeline.append((inst["start"], 1, inst["start_order"] * 3 + .5, "snapshot", inst))
        else:
            key = (cid, act, pd.Timestamp(inst["start"]))
            if not delay_lookup[key]:
                skipped["missing_eligibility"] += 1
                continue
            record = delay_lookup[key].popleft()
            eligible = pd.Timestamp(record["predecessor_end_timestamp"])
            if eligible > inst["start"] or eligible < bounds.loc[cid, "min"]:
                skipped["invalid_eligibility"] += 1
                continue
            inst["eligible"] = eligible
            mode = cfg["tasks"][task].get("target_mode", "raw_wait")
            inst["wait"] = float(record["residual_delay_seconds"]) if mode == "residual_delay" else (inst["start"]-eligible).total_seconds()
            order = inst["start_order"] * 3 - .5 if eligible == inst["start"] else float("inf")
            timeline.append((eligible, 1, order, "snapshot", inst))
    active_cases, active_intervals = set(), set()
    workers, ac_wip = defaultdict(int), defaultdict(int)
    prefix, history, previous = defaultdict(list), defaultdict(lambda: defaultdict(int)), defaultdict(float)
    rows = []
    quotas = cfg.get("resources", {}).get("simulation", {}).get("enable_concurrency_quotas", True)
    for timestamp, _, _, kind, value in sorted(timeline, key=lambda item: item[:3]):
        if kind == "arrival":
            active_cases.add(value)
            continue
        if kind == "departure":
            active_cases.discard(value)
            continue
        inst = value
        cid, act = inst["case"], inst["act"]
        if kind == "start":
            active_intervals.add(inst["id"])
            workers[inst["worker"]] += 1
            ac_wip[act] += 1
        elif kind == "complete":
            if inst["id"] in active_intervals:
                active_intervals.remove(inst["id"])
                workers[inst["worker"]] -= 1
                ac_wip[act] -= 1
            prefix[cid].append(act)
            history[cid][act] += 1
            previous[cid] = inst["duration"]
        else:
            attrs = {}
            source = case_first[cid] if task == "waiting_time" else (inst["start_row"] or inst["row"])
            for attr in cfg.get("features", {}).get("case_attributes", []):
                col, name = attr["column"], attr["feature_name"]
                default = imputation.get(name, attr.get("fill_value", 0.0))
                val = source.get(col, source.get(col.replace("case:", ""), default))
                if pd.isna(val) or (attr.get("treat_zero_as_missing") and val == 0):
                    val = default
                attrs[name] = val
            events = {} if task == "waiting_time" else {c: source.get(c, "UNKNOWN") for c in cfg.get("features", {}).get("event_attributes", [])}
            ctx = FeatureContext(target_act=act, wip=len(active_cases), ac_wip=ac_wip,
                                 dt=timestamp, role_oc=capacity_occupancy(role_data, workers, quotas),
                                 prev_dur=previous[cid], trace_history=dict(history[cid]),
                                 tracked_activities=tracked, case_attrs=attrs, event_attrs=events,
                                 worker_id=inst["worker"] if task == "duration" else "UNKNOWN")
            features = manager.build_features(ctx)
            info = manifest["cases"].get(cid) if manifest else None
            row = {"Case_ID": cid, "Target_Activity": act,
                   "Prefix": json.dumps(prefix[cid][-cfg.get("features", {}).get("prefix_window_size", 10):]),
                   "Duration_Seconds": inst["duration"], "Wait_Time_Seconds": inst.get("wait", 0.0),
                   "Case_Arrival": info["arrival"] if info else bounds.loc[cid, "min"],
                   "Case_End": info["end"] if info else bounds.loc[cid, "max"],
                   "Partition": info["partition"] if info else "unassigned",
                   "Instance_ID": inst["id"], "Eligibility_Time": inst["eligible"], **features}
            rows.append(row)
    summary = {"observed_instances": len(instances), "emitted_rows": len(rows), "excluded": dict(skipped)}
    if diagnostics is not None:
        diagnostics.update(summary)
    print(f"Feature extraction summary: {summary}")
    return rows
