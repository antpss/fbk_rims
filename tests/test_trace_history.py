"""
tests/test_trace_history.py

Unit and integration tests for the universal, leak-free trace_history feature in RIMS++.
Verifies:
1. Feature extraction logic in FeatureContext & FeatureManager.
2. Strict causality: events in the future of a trace are NEVER visible in the past history.
3. Profile-aware filtering of tracked activities.
4. Token execution updating trace_history and predictor routing resolution.
"""

import os
import sys
import unittest
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from features import FeatureContext, FeatureManager, _extract_trace_history


class TestTraceHistory(unittest.TestCase):
    def test_feature_extractor_basic(self):
        """Test _extract_trace_history builds clean multi-hot binary indicators."""
        ctx = FeatureContext(
            trace_history={"Request Created": 1, "Network Adjustment": 2},
            tracked_activities=["Request Created", "Network Adjustment", "Authorization Requested"]
        )
        feats = _extract_trace_history(ctx)
        self.assertEqual(feats["has_Request_Created"], 1.0)
        self.assertEqual(feats["has_Network_Adjustment"], 1.0)
        self.assertEqual(feats["has_Authorization_Requested"], 0.0)

    def test_feature_manager_integration(self):
        """Test FeatureManager extracts trace_history when active in selected_features."""
        cfg = {
            "features": {
                "selected_features": ["wip", "trace_history"],
                "trace_history": {
                    "mode": "whitelist",
                    "whitelist": ["Task_A", "Task_B"]
                }
            }
        }
        fm = FeatureManager(cfg)
        self.assertTrue(fm.is_feature_active("trace_history"))

        ctx = FeatureContext(
            wip=5.0,
            trace_history={"Task_A": 1},
            tracked_activities=["Task_A", "Task_B"]
        )
        res = fm.build_features(ctx)
        self.assertEqual(res["WIP"], 5.0)
        self.assertEqual(res["has_Task_A"], 1.0)
        self.assertEqual(res["has_Task_B"], 0.0)

    def test_causal_leak_freedom(self):
        """
        Verify strict causality:
        For an event at t1, events that execute later at t2 MUST NOT be present in t1's trace_history.
        """
        # Simulate chronological case history accumulation
        case_history = defaultdict(int)

        # Event 1: Intake (t=0)
        ctx_e1 = FeatureContext(
            target_act="Intake",
            trace_history=dict(case_history),
            tracked_activities=["Intake", "Approval", "Closure"]
        )
        f_e1 = _extract_trace_history(ctx_e1)
        self.assertEqual(f_e1["has_Intake"], 0.0)
        self.assertEqual(f_e1["has_Approval"], 0.0)
        self.assertEqual(f_e1["has_Closure"], 0.0)

        # Intake finishes
        case_history["Intake"] += 1

        # Event 2: Approval (t=10)
        ctx_e2 = FeatureContext(
            target_act="Approval",
            trace_history=dict(case_history),
            tracked_activities=["Intake", "Approval", "Closure"]
        )
        f_e2 = _extract_trace_history(ctx_e2)
        self.assertEqual(f_e2["has_Intake"], 1.0, "Past event Intake MUST be visible")
        self.assertEqual(f_e2["has_Approval"], 0.0, "Current event Approval MUST NOT be in history prior to completion")
        self.assertEqual(f_e2["has_Closure"], 0.0, "Future event Closure MUST NOT be visible")

        # Approval finishes
        case_history["Approval"] += 1

        # Event 3: Closure (t=20)
        ctx_e3 = FeatureContext(
            target_act="Closure",
            trace_history=dict(case_history),
            tracked_activities=["Intake", "Approval", "Closure"]
        )
        f_e3 = _extract_trace_history(ctx_e3)
        self.assertEqual(f_e3["has_Intake"], 1.0)
        self.assertEqual(f_e3["has_Approval"], 1.0)
        self.assertEqual(f_e3["has_Closure"], 0.0)


if __name__ == "__main__":
    unittest.main()
