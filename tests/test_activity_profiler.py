"""
tests/test_activity_profiler.py

Unit tests for Universal Data-Driven Activity Profiling in RIMS++.
Verifies:
1. Automatic classification of HUMAN_WORK_ITEM (positive duration).
2. Automatic classification of AUTOMATED_TASK (p_zero >= 0.95, wait > 60s).
3. Automatic classification of INSTANT_GATEWAY (p_zero >= 0.95, wait <= 60s).
4. Manual configuration overrides in config.yaml.
"""

import os
import sys
import tempfile
import shutil
import unittest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from activity_profiler import ActivityProfiler


class TestActivityProfiler(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_automatic_classification(self):
        """Test rule-based classification on synthetic log."""
        events = [
            # Case 1
            # Human Task A: 10:00 -> 10:30 (dur = 1800s > 0)
            {"case:concept:name": "C1", "concept:name": "Human_Task_A", "lifecycle:transition": "START", "time:timestamp": "2024-01-01 10:00:00", "org:resource": "Alice"},
            {"case:concept:name": "C1", "concept:name": "Human_Task_A", "lifecycle:transition": "COMPLETE", "time:timestamp": "2024-01-01 10:30:00", "org:resource": "Alice"},
            # Instant Gateway G: single complete at 10:30:00 (wait = 0s, dur = 0s)
            {"case:concept:name": "C1", "concept:name": "Instant_Gateway_G", "lifecycle:transition": "COMPLETE", "time:timestamp": "2024-01-01 10:30:00", "org:resource": "SYSTEM"},
            # Automated Task with Delay: complete 2 days later at 2024-01-03 10:30:00 (wait = 172800s, dur = 0s)
            {"case:concept:name": "C1", "concept:name": "Auto_Closure", "lifecycle:transition": "COMPLETE", "time:timestamp": "2024-01-03 10:30:00", "org:resource": "SYSTEM"},

            # Case 2
            {"case:concept:name": "C2", "concept:name": "Human_Task_A", "lifecycle:transition": "START", "time:timestamp": "2024-01-01 11:00:00", "org:resource": "Bob"},
            {"case:concept:name": "C2", "concept:name": "Human_Task_A", "lifecycle:transition": "COMPLETE", "time:timestamp": "2024-01-01 11:45:00", "org:resource": "Bob"},
            {"case:concept:name": "C2", "concept:name": "Instant_Gateway_G", "lifecycle:transition": "COMPLETE", "time:timestamp": "2024-01-01 11:45:00", "org:resource": "SYSTEM"},
            {"case:concept:name": "C2", "concept:name": "Auto_Closure", "lifecycle:transition": "COMPLETE", "time:timestamp": "2024-01-03 11:45:00", "org:resource": "SYSTEM"},
        ]
        df = pd.DataFrame(events)
        profiler = ActivityProfiler(zero_duration_threshold=0.95, min_automated_wait_seconds=60.0)
        res = profiler.profile_dataframe(df)

        act_types = res["activity_types"]
        self.assertEqual(act_types["Human_Task_A"], "HUMAN_WORK_ITEM")
        self.assertEqual(act_types["Instant_Gateway_G"], "INSTANT_GATEWAY")
        self.assertEqual(act_types["Auto_Closure"], "AUTOMATED_TASK")

        # Check profiles metadata
        self.assertAlmostEqual(res["profiles"]["Human_Task_A"]["p_zero_duration"], 0.0)
        self.assertAlmostEqual(res["profiles"]["Instant_Gateway_G"]["p_zero_duration"], 1.0)
        self.assertAlmostEqual(res["profiles"]["Auto_Closure"]["p_zero_duration"], 1.0)
        self.assertGreater(res["profiles"]["Auto_Closure"]["median_wait_seconds"], 60.0)
        self.assertLessEqual(res["profiles"]["Instant_Gateway_G"]["median_wait_seconds"], 60.0)

    def test_overrides_precedence(self):
        """Test that explicit overrides in config take precedence over data-driven classification."""
        events = [
            # Human Task A with positive duration
            {"case:concept:name": "C1", "concept:name": "Human_Task_A", "lifecycle:transition": "START", "time:timestamp": "2024-01-01 10:00:00", "org:resource": "Alice"},
            {"case:concept:name": "C1", "concept:name": "Human_Task_A", "lifecycle:transition": "COMPLETE", "time:timestamp": "2024-01-01 10:30:00", "org:resource": "Alice"},
        ]
        df = pd.DataFrame(events)
        # Override Human_Task_A to INSTANT_GATEWAY
        profiler = ActivityProfiler(overrides={"Human_Task_A": "INSTANT_GATEWAY"})
        res = profiler.profile_dataframe(df)
        self.assertEqual(res["activity_types"]["Human_Task_A"], "INSTANT_GATEWAY")

    def test_paired_zero_duration(self):
        """Test that paired START/COMPLETE events with identical timestamp are detected as 0-duration."""
        events = [
            {"case:concept:name": "C1", "concept:name": "Fast_Task", "lifecycle:transition": "START", "time:timestamp": "2024-01-01 10:00:00", "org:resource": "SYSTEM"},
            {"case:concept:name": "C1", "concept:name": "Fast_Task", "lifecycle:transition": "COMPLETE", "time:timestamp": "2024-01-01 10:00:00", "org:resource": "SYSTEM"},
        ]
        df = pd.DataFrame(events)
        profiler = ActivityProfiler(zero_duration_threshold=0.95)
        res = profiler.profile_dataframe(df)
        self.assertEqual(res["profiles"]["Fast_Task"]["p_zero_duration"], 1.0)
        self.assertEqual(res["profiles"]["Fast_Task"]["median_duration_seconds"], 0.0)


if __name__ == "__main__":
    unittest.main()

