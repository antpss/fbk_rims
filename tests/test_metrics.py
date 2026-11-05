#!/usr/bin/env python3
"""
tests/test_metrics.py

Comprehensive unit tests for the multi-perspective evaluation suite
powered by the official `log-distance-measures` library (Chapela-Campa et al. BPM 2023):
  - Control-Flow: CFLD, L_traces, NGD (2-Gram, 3-Gram)
  - Temporal: AED, CED, RED
  - Workforce: CWD (Circadian Workforce Distribution)
  - Congestion: CTD (Cycle Time Distribution / 1WD), CAR
"""

import unittest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from log_distance_measures.config import EventLogIDs
from log_distance_measures.control_flow_log_distance import control_flow_log_distance
from log_distance_measures.n_gram_distribution import n_gram_distribution_distance
from log_distance_measures.circadian_workforce_distribution import circadian_workforce_distribution_distance
from src.simulator.metrics import (
    compute_trace_length_error,
    prepare_evaluation_dfs,
    SimulationEvaluator,
    format_dur,
)


class TestMetrics(unittest.TestCase):
    def test_format_dur(self):
        self.assertEqual(format_dur(30), "30.0s")
        self.assertEqual(format_dur(120), "2.0 mins")
        self.assertEqual(format_dur(7200), "2.0 hours")
        self.assertEqual(format_dur(172800), "2.0 days")

    def test_compute_trace_length_error(self):
        # Clean logs with identical trace lengths
        df1 = pd.DataFrame({
            "case_id": ["c1", "c1", "c2", "c2"],
            "activity": ["A", "B", "A", "B"]
        })
        df2 = pd.DataFrame({
            "case_id": ["c1", "c1", "c2", "c2"],
            "activity": ["A", "B", "A", "B"]
        })
        self.assertAlmostEqual(compute_trace_length_error(df1, df2), 0.0)

        # Different trace lengths: df1 has lengths [2, 2], df3 has lengths [1, 3] -> difference |2-1| + |2-3| = 2 / 2 = 1.0
        df3 = pd.DataFrame({
            "case_id": ["c1", "c2", "c2", "c2"],
            "activity": ["A", "A", "B", "C"]
        })
        self.assertAlmostEqual(compute_trace_length_error(df1, df3), 1.0)

    def test_prepare_evaluation_dfs(self):
        base_time = datetime(2023, 5, 1, 8, 0, 0)
        # Real log with START and COMPLETE events
        real_events = [
            {"case:concept:name": "c1", "concept:name": "A", "time:timestamp": base_time, "lifecycle:transition": "START", "org:resource": "R1"},
            {"case:concept:name": "c1", "concept:name": "A", "time:timestamp": base_time + timedelta(minutes=30), "lifecycle:transition": "COMPLETE", "org:resource": "R1"},
            {"case:concept:name": "c1", "concept:name": "B", "time:timestamp": base_time + timedelta(hours=1), "lifecycle:transition": "START", "org:resource": "R2"},
            {"case:concept:name": "c1", "concept:name": "B", "time:timestamp": base_time + timedelta(hours=2), "lifecycle:transition": "COMPLETE", "org:resource": "R2"},
        ]
        df_real = pd.DataFrame(real_events)

        sim_events = [
            {"case_id": "c1", "activity": "A", "start_timestamp": base_time, "complete_timestamp": base_time + timedelta(minutes=30), "resource_id": "R1", "duration_seconds": 1800, "wait_time_seconds": 0, "queue_wait_seconds": 0},
            {"case_id": "c1", "activity": "B", "start_timestamp": base_time + timedelta(hours=1), "complete_timestamp": base_time + timedelta(hours=2), "resource_id": "R2", "duration_seconds": 3600, "wait_time_seconds": 0, "queue_wait_seconds": 0},
        ]
        df_sim = pd.DataFrame(sim_events)

        clean_real, clean_sim, ids = prepare_evaluation_dfs(df_real, df_sim)

        self.assertIsInstance(ids, EventLogIDs)
        self.assertEqual(len(clean_real), 2)
        self.assertEqual(len(clean_sim), 2)
        self.assertIn("case_id", clean_real.columns)
        self.assertIn("start_time", clean_real.columns)
        self.assertIn("end_time", clean_real.columns)
        self.assertIn("resource", clean_real.columns)

    def test_log_distance_measures_integration(self):
        base_time = datetime(2023, 5, 1, 8, 0, 0)
        df_real = pd.DataFrame([
            {"case_id": "c1", "activity": "A", "start_time": pd.to_datetime(base_time, utc=True), "end_time": pd.to_datetime(base_time + timedelta(hours=1), utc=True), "resource": "W1"},
            {"case_id": "c1", "activity": "B", "start_time": pd.to_datetime(base_time + timedelta(hours=1), utc=True), "end_time": pd.to_datetime(base_time + timedelta(hours=2), utc=True), "resource": "W1"},
            {"case_id": "c2", "activity": "A", "start_time": pd.to_datetime(base_time + timedelta(hours=2), utc=True), "end_time": pd.to_datetime(base_time + timedelta(hours=3), utc=True), "resource": "W2"},
            {"case_id": "c2", "activity": "B", "start_time": pd.to_datetime(base_time + timedelta(hours=3), utc=True), "end_time": pd.to_datetime(base_time + timedelta(hours=4), utc=True), "resource": "W2"},
        ])
        df_sim_identical = df_real.copy()

        ids = EventLogIDs(case="case_id", activity="activity", start_time="start_time", end_time="end_time", resource="resource")

        # Test CFLD on identical logs
        cfld = control_flow_log_distance(df_real, ids, df_sim_identical, ids, parallel=False)
        self.assertAlmostEqual(cfld, 0.0)

        # Test N-Gram distributions on identical logs
        ng2 = n_gram_distribution_distance(df_real, ids, df_sim_identical, ids, n=2)
        ng3 = n_gram_distribution_distance(df_real, ids, df_sim_identical, ids, n=3)
        self.assertAlmostEqual(ng2, 0.0)
        self.assertAlmostEqual(ng3, 0.0)

        # Test CWD (Circadian Workforce Distribution)
        cwd = circadian_workforce_distribution_distance(df_real, ids, df_sim_identical, ids)
        self.assertAlmostEqual(cwd, 0.0)

    def test_simulation_evaluator_full(self):
        base_time = datetime(2023, 5, 1, 8, 0, 0)
        real_events = [
            {"case:concept:name": "c1", "concept:name": "A", "time:timestamp": base_time, "lifecycle:transition": "COMPLETE", "org:resource": "W1"},
            {"case:concept:name": "c1", "concept:name": "B", "time:timestamp": base_time + timedelta(hours=2), "lifecycle:transition": "COMPLETE", "org:resource": "W1"},
            {"case:concept:name": "c2", "concept:name": "A", "time:timestamp": base_time + timedelta(hours=1), "lifecycle:transition": "COMPLETE", "org:resource": "W2"},
            {"case:concept:name": "c2", "concept:name": "B", "time:timestamp": base_time + timedelta(hours=4), "lifecycle:transition": "COMPLETE", "org:resource": "W2"},
        ]
        df_real = pd.DataFrame(real_events)

        sim_events = [
            {"case_id": "c1", "activity": "A", "start_timestamp": base_time, "complete_timestamp": base_time + timedelta(hours=1), "resource_id": "W1", "duration_seconds": 3600, "wait_time_seconds": 0, "queue_wait_seconds": 0},
            {"case_id": "c1", "activity": "B", "start_timestamp": base_time + timedelta(hours=2), "complete_timestamp": base_time + timedelta(hours=3), "resource_id": "W1", "duration_seconds": 3600, "wait_time_seconds": 0, "queue_wait_seconds": 0},
            {"case_id": "c2", "activity": "A", "start_timestamp": base_time + timedelta(hours=1), "complete_timestamp": base_time + timedelta(hours=2), "resource_id": "W2", "duration_seconds": 3600, "wait_time_seconds": 0, "queue_wait_seconds": 0},
            {"case_id": "c2", "activity": "B", "start_timestamp": base_time + timedelta(hours=4), "complete_timestamp": base_time + timedelta(hours=5), "resource_id": "W2", "duration_seconds": 3600, "wait_time_seconds": 0, "queue_wait_seconds": 0},
        ]
        df_sim = pd.DataFrame(sim_events)

        cfg = {"target_activities": ["A", "B"], "paths": {"output_base_dir": "data/processed"}}
        evaluator = SimulationEvaluator(cfg, export_xes=False)

        results = evaluator.evaluate(df_real, df_sim)

        # Check all perspectives are present
        self.assertIn("cfld", results)
        self.assertIn("l_traces", results)
        self.assertIn("ngram_2", results)
        self.assertIn("ngram_3", results)
        self.assertIn("aed", results)
        self.assertIn("ced", results)
        self.assertIn("cwd", results)
        self.assertIn("red_dist", results)
        self.assertIn("car", results)
        self.assertIn("ctd", results)
        self.assertIn("mae_cycle", results)

        # Control-flow should be identical (A -> B in both traces)
        self.assertAlmostEqual(results["cfld"], 0.0)
        self.assertAlmostEqual(results["l_traces"], 0.0)
        self.assertAlmostEqual(results["ngram_2"], 0.0)
        self.assertAlmostEqual(results["ngram_3"], 0.0)


if __name__ == "__main__":
    unittest.main()
