"""
tests/test_zero_duration_simulation.py

Integration tests for Zero-Duration and Automated Activity execution in RIMS++ Simulator.
Verifies:
1. INSTANT_GATEWAY fires immediately at t=env.now with duration=0.0s, wait=0.0s, resource='SYSTEM'.
2. AUTOMATED_TASK executes after its timer delay on 24/7 clock without acquiring human workers, with duration=0.0s.
3. HUMAN_WORK_ITEM requests human worker, respects calendar shift, runs for positive duration.
4. ChampionPredictor returns 0.0s duration for profiled automated tasks and instant gateways.
"""

import os
import sys
import unittest
import simpy
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils import petri_utils

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from datetime import datetime
from simulator.token import Token
from simulator.calendar import CalendarManager
from simulator.predictor import ChampionPredictor


class TestZeroDurationSimulation(unittest.TestCase):
    def test_predictor_zero_duration_dispatch(self):
        """Verify that ChampionPredictor explicitly returns 0.0s for automated/instant activities."""
        predictor = ChampionPredictor.__new__(ChampionPredictor)
        predictor.activity_profiles = {
            "Auto_Closure": "AUTOMATED_TASK",
            "Gateway_M": "INSTANT_GATEWAY",
            "Human_Task": "HUMAN_WORK_ITEM"
        }
        predictor.dur_max_caps = {}
        predictor.dur_global_cap = 3600.0
        predictor.wait_max_caps = {}
        predictor.wait_global_cap = 86400.0

        # Duration predictions
        self.assertEqual(predictor.predict_duration("Auto_Closure", [], 1, {}, None), 0.0)
        self.assertEqual(predictor.predict_duration("Gateway_M", [], 1, {}, None), 0.0)

        # Waiting predictions
        self.assertEqual(predictor.predict_waiting("Gateway_M", [], 1, {}, None), 0.0)

    def test_token_execution_by_profile(self):
        """
        Verify that Token executes:
          - INSTANT_GATEWAY immediately without delay or worker.
          - AUTOMATED_TASK with delay on 24/7 clock without worker.
          - HUMAN_WORK_ITEM with worker acquisition and calendar duration.
        """
        env = simpy.Environment()

        # Build Petri net: p_start -> t_gw -> p1 -> t_human -> p2 -> t_auto -> p_sink
        net = PetriNet("Test_Net")
        p_start = PetriNet.Place("start")
        p1 = PetriNet.Place("p1")
        p2 = PetriNet.Place("p2")
        p_sink = PetriNet.Place("sink")
        for p in [p_start, p1, p2, p_sink]:
            net.places.add(p)

        t_gw = PetriNet.Transition("t_gw", "Gateway_M")
        t_human = PetriNet.Transition("t_human", "Human_Task")
        t_auto = PetriNet.Transition("t_auto", "Auto_Closure")
        for t in [t_gw, t_human, t_auto]:
            net.transitions.add(t)

        petri_utils.add_arc_from_to(p_start, t_gw, net)
        petri_utils.add_arc_from_to(t_gw, p1, net)
        petri_utils.add_arc_from_to(p1, t_human, net)
        petri_utils.add_arc_from_to(t_human, p2, net)
        petri_utils.add_arc_from_to(p2, t_auto, net)
        petri_utils.add_arc_from_to(t_auto, p_sink, net)

        recorded_events = []
        worker_requests = []
        worker_releases = []

        class MockWorkers:
            def __init__(self, env):
                self.env = env
                self.all_workers = ["Alice"]

            def request_worker(self, act):
                worker_requests.append(act)
                ev = self.env.event()
                ev.succeed(value="Alice")
                return ev

            def release_worker(self, worker_id, act):
                worker_releases.append((worker_id, act))

            def get_role_occupancy(self, acts):
                return [0.5]

        class MockPredictor:
            def predict_waiting(self, act, *args, **kwargs):
                if act == "Auto_Closure":
                    return 3600.0  # 1 hour automated delay
                return 0.0

            def predict_duration(self, act, *args, **kwargs):
                if act == "Human_Task":
                    return 1800.0  # 30 mins
                return 0.0

        class MockProcess:
            def __init__(self, env):
                self.env = env
                self.net = net
                self.im = Marking({p_start: 1})
                self.fm = Marking({p_sink: 1})
                self.cfg = {}
                self.mode = "hybrid_residual"
                self.target_activities = ["Gateway_M", "Human_Task", "Auto_Closure"]
                self.proc_prefixes = ()
                self.ms_prefixes = ()
                self.branch_target_map = {}
                self.external_delays = {}
                self.active_cases_count = 1
                self.ac_wip = {"Gateway_M": 0, "Human_Task": 0, "Auto_Closure": 0}
                self.max_events = 20
                self.calendar = CalendarManager(ref_dt=datetime(2024, 1, 8, 9, 0, 0))
                self.workers = MockWorkers(env)
                self.predictor = MockPredictor()
                self.activity_profiles = {
                    "Gateway_M": "INSTANT_GATEWAY",
                    "Human_Task": "HUMAN_WORK_ITEM",
                    "Auto_Closure": "AUTOMATED_TASK"
                }

            def has_ex_ante_delay(self, act):
                return False

            def sample_external_delay(self, prev, act):
                return 0.0

            def record_event(self, event_dict):
                recorded_events.append(dict(event_dict))

        proc = MockProcess(env)
        token = Token("C1", arrival_offset=0, process=proc)
        env.process(token.run())
        env.run(until=10000)

        # 1. Verify that all 3 activities fired
        acts = [e["activity"] for e in recorded_events]
        self.assertEqual(acts, ["Gateway_M", "Human_Task", "Auto_Closure"])

        # 2. Verify INSTANT_GATEWAY
        gw_evt = recorded_events[0]
        self.assertEqual(gw_evt["resource_id"], "SYSTEM")
        self.assertEqual(gw_evt["duration_seconds"], 0.0)
        self.assertEqual(gw_evt["wait_time_seconds"], 0.0)

        # 3. Verify HUMAN_WORK_ITEM
        human_evt = recorded_events[1]
        self.assertEqual(human_evt["resource_id"], "Alice")
        self.assertGreater(human_evt["duration_seconds"], 0.0)
        self.assertIn("Human_Task", worker_requests)
        self.assertIn(("Alice", "Human_Task"), worker_releases)

        # 4. Verify AUTOMATED_TASK
        auto_evt = recorded_events[2]
        self.assertEqual(auto_evt["resource_id"], "SYSTEM")
        self.assertEqual(auto_evt["duration_seconds"], 0.0)
        self.assertEqual(auto_evt["wait_time_seconds"], 3600.0)  # Preserved 1h delay
        self.assertNotIn("Auto_Closure", worker_requests, "Automated task must NOT request human worker")


if __name__ == "__main__":
    unittest.main()
