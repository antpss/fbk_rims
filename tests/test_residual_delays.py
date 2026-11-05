"""
tests/test_residual_delays.py

Unit and integration tests for Fix 2: Residual Waiting Time De-Coupling.
Verifies that:
1. CausalDelayDiscoveryEngine decomposes elapsed time into worker busy time,
   calendar off-hours, and residual delay (omega_ex <= raw_wait).
2. The instance-level parquet/csv table is serialized cleanly.
3. DatasetBuilder accurately maps instance-level residual delays into Wait_Time_Seconds
   when target_mode == "residual_delay".
"""

import os
import sys
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta
import pandas as pd
import pm4py

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from config_loader import PROJECT_ROOT


class TestResidualWaitingDelays(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_residual_delay_bounds_and_export(self):
        """Verify that omega_ex is bounded between 0 and raw_wait, and exports table."""
        # Create synthetic event log with 2 cases, 2 activities, with known worker busy interval
        # Worker 'Alice' works on A in Case 1 from 09:00 to 10:00.
        # Worker 'Alice' is busy on another task from 10:00 to 11:30.
        # Case 1 task B starts at 12:00.
        # Raw wait between A complete (10:00) and B start (12:00) is 2.0h (7200s).
        # Alice was busy 1.5h (5400s), leaving 0.5h (1800s) residual wait.
        
        events = [
            # Case 1: Task A by Alice
            {"case:concept:name": "C1", "concept:name": "Task_A", "lifecycle:transition": "START", "time:timestamp": pd.Timestamp("2024-01-08 09:00:00", tz="UTC"), "org:resource": "Alice"},
            {"case:concept:name": "C1", "concept:name": "Task_A", "lifecycle:transition": "COMPLETE", "time:timestamp": pd.Timestamp("2024-01-08 10:00:00", tz="UTC"), "org:resource": "Alice"},
            # Other task by Alice between 10:00 and 11:30
            {"case:concept:name": "C2", "concept:name": "Task_A", "lifecycle:transition": "START", "time:timestamp": pd.Timestamp("2024-01-08 10:00:00", tz="UTC"), "org:resource": "Alice"},
            {"case:concept:name": "C2", "concept:name": "Task_A", "lifecycle:transition": "COMPLETE", "time:timestamp": pd.Timestamp("2024-01-08 11:30:00", tz="UTC"), "org:resource": "Alice"},
            # Case 1: Task B by Alice starts at 12:00
            {"case:concept:name": "C1", "concept:name": "Task_B", "lifecycle:transition": "START", "time:timestamp": pd.Timestamp("2024-01-08 12:00:00", tz="UTC"), "org:resource": "Alice"},
            {"case:concept:name": "C1", "concept:name": "Task_B", "lifecycle:transition": "COMPLETE", "time:timestamp": pd.Timestamp("2024-01-08 13:00:00", tz="UTC"), "org:resource": "Alice"},
        ]
        df_log = pd.DataFrame(events)
        xes_path = os.path.join(self.test_dir, "test_log.xes")
        pm4py.write_xes(df_log, xes_path)

        cfg = {
            "paths": {
                "raw_log": xes_path,
                "aligned_log": xes_path,
                "models_dir": self.test_dir,
                "output_base_dir": self.test_dir
            },
            "delays": {
                "estimator": "eclaw_plus",
                "placement": "ex_ante",
                "concurrency_zeta": 0.2,
                "outlier_delta": 0.0,
                "min_samples": 1,
                "min_delay_seconds": 0.0,
                "min_break_seconds": 60.0,
                "output_dir": os.path.join(self.test_dir, "delays")
            },
            "simulation": {
                "calendar": {
                    "enabled": True,
                    "work_days": [0, 1, 2, 3, 4],
                    "hour_start": 8,
                    "hour_end": 20,
                    "use_discovered_role_calendars": False
                }
            }
        }

        from src.simulator.calendar import CalendarManager
        import importlib
        mod_02b = importlib.import_module("02b_discover_delays")
        ChapelaDelayDiscoveryEngine = mod_02b.ChapelaDelayDiscoveryEngine

        engine = ChapelaDelayDiscoveryEngine(cfg)
        engine.output_dir = os.path.join(self.test_dir, "delays")
        os.makedirs(engine.output_dir, exist_ok=True)
        manifest = engine.run()

        parquet_path = os.path.join(self.test_dir, "causal_delays_instance_level.parquet")
        csv_path = os.path.join(self.test_dir, "causal_delays_instance_level.csv.gz")

        self.assertTrue(os.path.exists(parquet_path), "causal_delays_instance_level.parquet must exist")
        self.assertTrue(os.path.exists(csv_path), "causal_delays_instance_level.csv.gz must exist")

        df_inst = pd.read_parquet(parquet_path)
        self.assertGreater(len(df_inst), 0)

        # Check bounds: 0 <= residual_delay <= raw_wait
        for _, row in df_inst.iterrows():
            raw_w = row["raw_wait_seconds"]
            res_w = row["residual_delay_seconds"]
            self.assertGreaterEqual(res_w, 0.0)
            self.assertLessEqual(res_w, raw_w + 1e-5)

    def test_dataset_builder_residual_mode(self):
        """Verify that DatasetBuilder ingests residual_delay_seconds when target_mode is residual_delay."""
        events = [
            {"case:concept:name": "C1", "concept:name": "Task_A", "lifecycle:transition": "START", "time:timestamp": pd.Timestamp("2024-01-08 09:00:00", tz="UTC"), "org:resource": "Alice", "is_synthetic": "false"},
            {"case:concept:name": "C1", "concept:name": "Task_A", "lifecycle:transition": "COMPLETE", "time:timestamp": pd.Timestamp("2024-01-08 10:00:00", tz="UTC"), "org:resource": "Alice", "is_synthetic": "false"},
            {"case:concept:name": "C1", "concept:name": "Task_B", "lifecycle:transition": "START", "time:timestamp": pd.Timestamp("2024-01-08 12:00:00", tz="UTC"), "org:resource": "Alice", "is_synthetic": "false"},
            {"case:concept:name": "C1", "concept:name": "Task_B", "lifecycle:transition": "COMPLETE", "time:timestamp": pd.Timestamp("2024-01-08 13:00:00", tz="UTC"), "org:resource": "Alice", "is_synthetic": "false"},
        ]
        df_log = pd.DataFrame(events)
        xes_path = os.path.join(self.test_dir, "test_log_builder.xes")
        pm4py.write_xes(df_log, xes_path)

        # Mock instance-level residual table with a custom residual delay
        t_epoch = int(pd.Timestamp("2024-01-08 12:00:00", tz="UTC").timestamp())
        inst_records = [
            {
                "case_id": "C1",
                "target_act": "Task_B",
                "trigger_act": "Task_A",
                "start_timestamp": "2024-01-08 12:00:00+00:00",
                "start_timestamp_epoch": t_epoch,
                "predecessor_end_timestamp": "2024-01-08 10:00:00+00:00",
                "raw_wait_seconds": 7200.0,
                "worker_busy_seconds": 5400.0,
                "calendar_nw_seconds": 0.0,
                "residual_delay_seconds": 1800.0
            }
        ]
        df_mock = pd.DataFrame(inst_records)
        df_mock.to_parquet(os.path.join(self.test_dir, "causal_delays_instance_level.parquet"))

        cfg = {
            "paths": {
                "raw_log": xes_path,
                "aligned_log": xes_path,
                "models_dir": self.test_dir,
                "output_base_dir": self.test_dir
            },
            "features": {
                "prefix_window_size": 5,
                "selected_features": ["wip"],
                "case_attributes": [],
                "event_attributes": []
            },
            "preprocessing": {
                "activity_mapping": {}
            },
            "target_activities": ["Task_A", "Task_B"],
            "tasks": {
                "waiting_time": {
                    "target_column": "Wait_Time_Seconds",
                    "target_mode": "residual_delay",
                    "output_file": os.path.join(self.test_dir, "dataset_waiting.csv"),
                    "filter_zero_duration": False,
                    "sample_weighting": {"enabled": False}
                }
            },
            "simulation": {
                "calendar": {
                    "enabled": True,
                    "work_days": [0, 1, 2, 3, 4],
                    "hour_start": 8,
                    "hour_end": 20,
                    "use_discovered_role_calendars": False
                }
            }
        }

        import importlib
        mod_03 = importlib.import_module("03_dataset_builder")
        mod_03.build_dataset(cfg, task_name="waiting_time")

        out_csv = os.path.join(self.test_dir, "dataset_waiting.csv")
        self.assertTrue(os.path.exists(out_csv))
        df_res = pd.read_csv(out_csv)
        row_b = df_res[df_res["Target_Activity"] == "Task_B"].iloc[0]
        # Must match residual_delay_seconds (1800.0), NOT raw elapsed (7200.0)
        self.assertEqual(row_b["Wait_Time_Seconds"], 1800.0)


if __name__ == "__main__":
    unittest.main()
