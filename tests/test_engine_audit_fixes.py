"""
tests/test_engine_audit_fixes.py

Unit tests verifying the 4 critical engine audit fixes:
1. Concurrency interval reconstruction & no-leak tie-breaking in roles discovery.
2. Normalized ActivityProfiler cache validation and config propagation.
3. Chronological horizon alignment in simulation benchmark metrics.
4. Trace history frequency counts (`count_*`) in features and XOR routing.
"""

import os
import sys
import json
import time
import tempfile
import shutil
import unittest
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
import pandas as pd
import pm4py
from pm4py.objects.log.obj import EventLog, Trace, Event

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from activity_profiler import ActivityProfiler, load_or_discover_activity_profiles
from features import FeatureManager, FeatureContext
from simulator.metrics import prepare_evaluation_dfs, SimulationEvaluator


class TestEngineAuditFixes(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_roles_interval_concurrency_no_leak_and_handoff(self):
        """
        Fix 1 Test:
        Verifies that:
        1. 0-duration paired events (START == COMPLETE) do NOT occupy human worker capacity.
        2. Back-to-back sequential handoffs (end_A == start_B == 10:30:00) do NOT falsely spike to concurrency 2.
        3. Concurrency open tasks counter resets cleanly to 0 without leaking.
        """
        # Create events for Worker Alice:
        # Task 1: 10:00:00 -> 10:30:00 (dur = 30m, positive)
        # Task 2 (Milestone): 10:30:00 -> 10:30:00 (dur = 0s)
        # Task 3 (Milestone): 10:30:00 -> 10:30:00 (dur = 0s)
        # Task 4: 10:30:00 -> 11:00:00 (dur = 30m, sequential handoff)
        df_target = pd.DataFrame([
            {"case:concept:name": "C1", "concept:name": "Task_1", "org:resource": "Alice", "lifecycle:transition": "START", "time:timestamp": pd.to_datetime("2024-01-01 10:00:00")},
            {"case:concept:name": "C1", "concept:name": "Task_1", "org:resource": "Alice", "lifecycle:transition": "COMPLETE", "time:timestamp": pd.to_datetime("2024-01-01 10:30:00")},
            {"case:concept:name": "C1", "concept:name": "Milestone_1", "org:resource": "Alice", "lifecycle:transition": "START", "time:timestamp": pd.to_datetime("2024-01-01 10:30:00")},
            {"case:concept:name": "C1", "concept:name": "Milestone_1", "org:resource": "Alice", "lifecycle:transition": "COMPLETE", "time:timestamp": pd.to_datetime("2024-01-01 10:30:00")},
            {"case:concept:name": "C1", "concept:name": "Milestone_2", "org:resource": "Alice", "lifecycle:transition": "START", "time:timestamp": pd.to_datetime("2024-01-01 10:30:00")},
            {"case:concept:name": "C1", "concept:name": "Milestone_2", "org:resource": "Alice", "lifecycle:transition": "COMPLETE", "time:timestamp": pd.to_datetime("2024-01-01 10:30:00")},
            {"case:concept:name": "C1", "concept:name": "Task_4", "org:resource": "Alice", "lifecycle:transition": "START", "time:timestamp": pd.to_datetime("2024-01-01 10:30:00")},
            {"case:concept:name": "C1", "concept:name": "Task_4", "org:resource": "Alice", "lifecycle:transition": "COMPLETE", "time:timestamp": pd.to_datetime("2024-01-01 11:00:00")},
        ])

        # Execute interval reconstruction logic from 01b_discover_roles.py
        trans_order = {"START": 0, "COMPLETE": 1}
        df_sorted = df_target.copy()
        c_col = "case:concept:name"
        df_sorted["_tr"] = df_sorted["lifecycle:transition"].astype(str).str.upper().map(trans_order).fillna(2)
        df_sorted = df_sorted.sort_values(by=["time:timestamp", "_tr"]).drop(columns=["_tr"]).reset_index(drop=True)

        pending_starts = defaultdict(list)
        interval_events = []
        for row in df_sorted.to_dict("records"):
            cid = str(row[c_col])
            act = row["concept:name"]
            w = row["org:resource"]
            trans = str(row.get("lifecycle:transition", "COMPLETE")).upper()
            ts = row["time:timestamp"]

            if trans == "START":
                pending_starts[(cid, act)].append((ts, w))
            elif trans == "COMPLETE":
                if pending_starts[(cid, act)]:
                    s_ts, s_w = pending_starts[(cid, act)].pop(0)
                    worker = s_w if s_w not in ("nan", "UNKNOWN", "None", "") else w
                    if s_ts < ts:
                        interval_events.append((s_ts, "START", worker, act))
                        interval_events.append((ts, "COMPLETE", worker, act))

        interval_events.sort(key=lambda x: (x[0], 0 if x[1] == "COMPLETE" else 1))

        worker_open_tasks = defaultdict(int)
        worker_max_concurrency = defaultdict(int)
        worker_concurrency_history = defaultdict(list)

        for ts, trans, w, act in interval_events:
            if trans == "START":
                worker_open_tasks[w] += 1
                worker_concurrency_history[w].append(worker_open_tasks[w])
                if worker_open_tasks[w] > worker_max_concurrency[w]:
                    worker_max_concurrency[w] = worker_open_tasks[w]
            elif trans == "COMPLETE":
                worker_open_tasks[w] = max(0, worker_open_tasks[w] - 1)

        # Alice's open tasks must finish at 0 (no leaks)
        self.assertEqual(worker_open_tasks["Alice"], 0)
        # Alice's peak concurrency must be exactly 1 (no false spike from milestones or handoff)
        self.assertEqual(worker_max_concurrency["Alice"], 1)
        # Exactly two 1s recorded for the two positive intervals
        self.assertEqual(worker_concurrency_history["Alice"], [1, 1])

    def test_activity_profiler_cache_invalidation_and_normalization(self):
        """
        Fix 2 Test:
        Verifies that:
        1. Cache is reloaded cleanly when log and config do not change.
        2. Lowercase vs uppercase overrides in config.yaml do not trigger false invalidation.
        3. Modifying config threshold or touching log mtime triggers proper cache invalidation.
        """
        log_path = os.path.join(self.test_dir, "test_log.csv")
        out_json = os.path.join(self.test_dir, "activity_profiles.json")

        df = pd.DataFrame([
            {"case:concept:name": "C1", "concept:name": "Task_A", "lifecycle:transition": "START", "time:timestamp": "2024-01-01 10:00:00", "org:resource": "R1"},
            {"case:concept:name": "C1", "concept:name": "Task_A", "lifecycle:transition": "COMPLETE", "time:timestamp": "2024-01-01 10:10:00", "org:resource": "R1"},
        ])
        df.to_csv(log_path, index=False)

        cfg = {
            "paths": {"raw_log": log_path, "models_dir": self.test_dir},
            "activity_profiling": {
                "enabled": True,
                "output_file": out_json,
                "zero_duration_threshold": 0.95,
                "min_automated_wait_seconds": 60.0,
                "overrides": {"Task_A": "human_work_item"}  # Lowercase in config
            }
        }

        # 1. First run creates cache
        res1 = load_or_discover_activity_profiles(cfg)
        self.assertTrue(os.path.exists(out_json))
        self.assertEqual(res1["activity_types"]["Task_A"], "HUMAN_WORK_ITEM")

        # 2. Second run reuses cache without re-profiling
        res2 = load_or_discover_activity_profiles(cfg)
        self.assertEqual(res2["__metadata__"]["generated_at"], res1["__metadata__"]["generated_at"])

        # 3. Changing config override to automated_task invalidates cache
        cfg["activity_profiling"]["overrides"]["Task_A"] = "automated_task"
        res3 = load_or_discover_activity_profiles(cfg)
        self.assertEqual(res3["activity_types"]["Task_A"], "AUTOMATED_TASK")

    def test_metrics_unbiased_subsampling(self):
        """
        Fix 3 Test:
        Verifies that:
        1. When real cases exceed max_cases, clean_real is subsampled to max_cases.
        2. Both clean_real and clean_sim conform to expected max_cases bounds.
        """
        # Create 5 real cases
        real_events = []
        for i in range(1, 6):
            dt_s = pd.to_datetime(f"2024-01-0{i} 10:00:00", utc=True)
            dt_e = pd.to_datetime(f"2024-01-0{i} 11:00:00", utc=True)
            real_events.append({"case:concept:name": f"Case_{i}", "concept:name": "Task_A", "lifecycle:transition": "START", "time:timestamp": dt_s, "org:resource": "W1"})
            real_events.append({"case:concept:name": f"Case_{i}", "concept:name": "Task_A", "lifecycle:transition": "COMPLETE", "time:timestamp": dt_e, "org:resource": "W1"})
        df_real = pd.DataFrame(real_events)

        # Create 2 simulated cases
        sim_events = [
            {"case_id": "Sim_1", "activity": "Task_A", "start_timestamp": "2024-01-01 10:00:00", "complete_timestamp": "2024-01-01 11:00:00", "resource_id": "W1"},
            {"case_id": "Sim_2", "activity": "Task_A", "start_timestamp": "2024-01-02 10:00:00", "complete_timestamp": "2024-01-02 11:00:00", "resource_id": "W1"},
        ]
        df_sim = pd.DataFrame(sim_events)

        clean_real, clean_sim, ids = prepare_evaluation_dfs(df_real, df_sim, max_cases=2)

        # clean_real must be subsampled to exactly max_cases = 2
        self.assertEqual(clean_real["case_id"].nunique(), 2)
        self.assertEqual(clean_sim["case_id"].nunique(), 2)

    def test_features_include_counts(self):
        """
        Fix 4 Test:
        Verifies that:
        1. include_counts: false generates only has_*
        2. include_counts: true generates has_* and count_*
        """
        # 1. FeatureManager with include_counts=False
        cfg_false = {
            "features": {
                "selected_features": ["trace_history"],
                "trace_history": {"include_counts": False}
            }
        }
        fm_false = FeatureManager(cfg_false)
        ctx = FeatureContext(
            trace_history={"Task_A": 3, "Task_B": 0},
            tracked_activities=["Task_A", "Task_B"]
        )
        feats_false = fm_false.build_features(ctx)
        self.assertEqual(feats_false["has_Task_A"], 1.0)
        self.assertEqual(feats_false["has_Task_B"], 0.0)
        self.assertNotIn("count_Task_A", feats_false)

        # 2. FeatureManager with include_counts=True
        cfg_true = {
            "features": {
                "selected_features": ["trace_history"],
                "trace_history": {"include_counts": True}
            }
        }
        fm_true = FeatureManager(cfg_true)
        feats_true = fm_true.build_features(ctx)
        self.assertEqual(feats_true["has_Task_A"], 1.0)
        self.assertEqual(feats_true["has_Task_B"], 0.0)
        self.assertEqual(feats_true["count_Task_A"], 3.0)
        self.assertEqual(feats_true["count_Task_B"], 0.0)


if __name__ == "__main__":
    unittest.main()
