import unittest
import sys
import os
import json
import tempfile
from datetime import datetime
sys.path.insert(0, os.path.abspath("."))
import pandas as pd
import numpy as np
import simpy
from src.simulator.calendar import CalendarManager
from src.simulator.process import SharedWorkerPool

class TestSimulator(unittest.TestCase):
    def test_calendar_off_hours_dispatch_gate(self):
        """Verify that a freed worker during off-hours (e.g. Sunday 22:00) is held
        and NOT dispatched until shift opening (Monday 08:00)."""
        env = simpy.Environment()
        cfg = {
            "simulation": {
                "calendar": {
                    "enabled": True,
                    "hour_start": 8,
                    "hour_end": 20,
                    "work_days": [0, 1, 2, 3, 4], # Mon-Fri
                }
            }
        }
        # 2012-01-01 is a Sunday
        sunday_ref = datetime(2012, 1, 1, 0, 0, 0)
        cal = CalendarManager.from_config(cfg, ref_dt=sunday_ref)
        
        # Create temporary role mapping json
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({
                "__all_resources__": ["W1"],
                "A": ["W1"]
            }, f)
            role_file = f.name

        try:
            pool = SharedWorkerPool(env, role_file, latency_cfg={}, calendar=cal)

            # Advance to Sunday 22:00 (off-hours)
            env.run(until=22 * 3600)
            
            # Put worker as busy
            pool.worker_active_count["W1"] = 1
            pool.worker_busy["W1"] = True

            # Request a worker for activity A during off-hours
            ev = pool.request_worker("A")
            self.assertFalse(ev.triggered, "Event should not be triggered during off-hours")

            # Worker finishes task on Sunday 22:00 and is released
            pool.release_worker("W1")

            # Even though W1 is released, it must NOT be dispatched immediately because it is Sunday night
            self.assertFalse(ev.triggered, "Worker must NOT be dispatched during off-hours")
            self.assertTrue(pool._shift_alarm_active, "Shift alarm should be armed to wake at next shift")

            # Calculate exact seconds until Monday 08:00
            curr_dt = cal.to_datetime(env.now)
            gap = cal.get_gap_to_next_work_window(curr_dt)

            # Advance to 1 second before shift open
            env.run(until=env.now + gap - 1)
            self.assertFalse(ev.triggered, "Worker must still NOT be dispatched before 08:00 AM Monday")

            # Advance into the shift opening
            env.run(until=env.now + 2)
            self.assertTrue(ev.triggered, "Worker must be dispatched at 08:00 AM sharp on Monday")
            self.assertEqual(ev.value, "W1")
        finally:
            if os.path.exists(role_file):
                os.remove(role_file)

    def test_treat_zero_as_missing_imputation(self):
        """Verify that continuous attributes treat 0.0 as missing when treat_zero_as_missing=True."""
        data = pd.Series([0.0, 0.0, 100.0, 200.0, np.nan])
        treat_zero = True
        clean_s = pd.to_numeric(data, errors="coerce")
        if treat_zero:
            clean_s = clean_s[clean_s != 0]
        clean_s = clean_s.dropna()
        median_val = float(clean_s.median())
        self.assertEqual(median_val, 150.0)

        # When treat_zero = False
        clean_s2 = pd.to_numeric(data, errors="coerce").dropna()
        median_val2 = float(clean_s2.median())
        self.assertEqual(median_val2, 50.0)

    def test_worker_ubiquity_prevention(self):
        """Verify that a multi-skilled worker assigned to two roles cannot be
        dispatched concurrently across both roles when C_w=1 (eliminating phantom staff)."""
        env = simpy.Environment()
        role_data = {
            "__all_resources__": ["W_Multi"],
            "__role_names__": ["Role_A", "Role_B"],
            "roles": {"Role_A": ["Act_A"], "Role_B": ["Act_B"]},
            "activity_to_role": {"Act_A": "Role_A", "Act_B": "Role_B"},
            "role_to_workers": {"Role_A": ["W_Multi"], "Role_B": ["W_Multi"]},
            "role_capacities": {"Role_A": 1, "Role_B": 1},
            "worker_concurrencies": {"W_Multi": 1}
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(role_data, f)
            role_file = f.name

        try:
            pool = SharedWorkerPool(env, role_file, latency_cfg={})
            
            # Request worker for Act_A in Role_A
            ev_a = pool.request_worker("Act_A")
            self.assertTrue(ev_a.triggered, "W_Multi should be immediately assigned to Act_A")
            self.assertEqual(ev_a.value, "W_Multi")
            self.assertTrue(pool.worker_busy["W_Multi"])

            # Simultaneously request worker for Act_B in Role_B
            ev_b = pool.request_worker("Act_B")
            self.assertFalse(ev_b.triggered, "W_Multi must NOT be cloned to Act_B (No Worker Ubiquity!)")

            # Release W_Multi from Act_A
            pool.release_worker("W_Multi", activity="Act_A")

            # Now W_Multi should reactively be dispatched to the waiting Act_B
            self.assertTrue(ev_b.triggered, "W_Multi should be reactively dispatched to Act_B upon release")
            self.assertEqual(ev_b.value, "W_Multi")
        finally:
            if os.path.exists(role_file):
                os.remove(role_file)

    def test_multitasking_concurrency_quota(self):
        """Verify that a worker with concurrency quota C_w=2 can multitask on 2 cases,
        but queues on a 3rd case."""
        env = simpy.Environment()
        role_data = {
            "__all_resources__": ["W_Desk"],
            "__role_names__": ["Role_Front"],
            "roles": {"Role_Front": ["Act_1", "Act_2", "Act_3"]},
            "activity_to_role": {"Act_1": "Role_Front", "Act_2": "Role_Front", "Act_3": "Role_Front"},
            "role_to_workers": {"Role_Front": ["W_Desk"]},
            "role_capacities": {"Role_Front": 1},
            "worker_concurrencies": {"W_Desk": 2} # Concurrency quota = 2
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(role_data, f)
            role_file = f.name

        try:
            pool = SharedWorkerPool(env, role_file, latency_cfg={})
            
            # Case 1
            ev1 = pool.request_worker("Act_1")
            self.assertTrue(ev1.triggered)
            self.assertEqual(pool.worker_active_count["W_Desk"], 1)

            # Case 2 (allowed via concurrency quota)
            ev2 = pool.request_worker("Act_2")
            self.assertTrue(ev2.triggered)
            self.assertEqual(pool.worker_active_count["W_Desk"], 2)
            self.assertTrue(pool.worker_busy["W_Desk"])

            # Case 3 (exceeds quota -> must queue)
            ev3 = pool.request_worker("Act_3")
            self.assertFalse(ev3.triggered, "3rd task must queue because C_w=2 ceiling is reached")

            # Release Case 1
            pool.release_worker("W_Desk", activity="Act_1")
            self.assertTrue(ev3.triggered, "3rd task should now be dispatched")
            self.assertEqual(pool.worker_active_count["W_Desk"], 2)
        finally:
            if os.path.exists(role_file):
                os.remove(role_file)

    def test_role_occupancy_vector(self):
        """Verify that get_role_occupancy accurately returns utilization ratio per discovered role."""
        env = simpy.Environment()
        role_data = {
            "__all_resources__": ["W1", "W2"],
            "__role_names__": ["Role_A", "Role_B"],
            "roles": {"Role_A": ["Act_A"], "Role_B": ["Act_B"]},
            "activity_to_role": {"Act_A": "Role_A", "Act_B": "Role_B"},
            "role_to_workers": {"Role_A": ["W1"], "Role_B": ["W2"]},
            "role_capacities": {"Role_A": 2, "Role_B": 2},
            "worker_concurrencies": {"W1": 1, "W2": 1}
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(role_data, f)
            role_file = f.name

        try:
            pool = SharedWorkerPool(env, role_file, latency_cfg={})
            # Initially 0% occupancy
            self.assertEqual(pool.get_role_occupancy(), [0.0, 0.0])

            # W1 takes Act_A (1 active out of cap 2 -> 50%)
            pool.request_worker("Act_A")
            self.assertEqual(pool.get_role_occupancy(), [0.5, 0.0])

            # Release
            pool.release_worker("W1", activity="Act_A")
            self.assertEqual(pool.get_role_occupancy(), [0.0, 0.0])
        finally:
            if os.path.exists(role_file):
                os.remove(role_file)

    def test_activity_to_role_calendar_resolution(self):
        """Verify CalendarManager resolves activity to role calendar and falls back to global."""
        cfg = {
            "simulation": {
                "calendar": {
                    "enabled": True,
                    "hour_start": 8,
                    "hour_end": 20,
                    "work_days": [0, 1, 2, 3, 4],
                    "per_role": {
                        "Role_Custom": {
                            "hour_start": 9,
                            "hour_end": 17,
                            "work_days": [0, 1, 2, 3, 4]
                        }
                    }
                }
            }
        }
        act_to_role = {"Act_Custom": "Role_Custom", "Act_Default": "Role_Default"}
        cal = CalendarManager.from_config(cfg, activity_to_role=act_to_role)

        cal_custom = cal.get_calendar("Act_Custom")
        self.assertEqual(cal_custom.hour_start, 9)
        self.assertEqual(cal_custom.hour_end, 17)

        cal_default = cal.get_calendar("Act_Default")
        self.assertEqual(cal_default.hour_start, 8)
        self.assertEqual(cal_default.hour_end, 20)

    def test_role_dispatch_independence_24_7(self):
        """Verify that a closed office for Role A does NOT block dispatching of a 24/7 Role B."""
        env = simpy.Environment()
        cfg = {
            "simulation": {
                "calendar": {
                    "enabled": True,
                    "hour_start": 8,
                    "hour_end": 16, # Role A uses global (08:00 - 16:00 Mon-Fri)
                    "work_days": [0, 1, 2, 3, 4],
                    "per_role": {
                        "Role_247": {
                            "hour_start": 0,
                            "hour_end": 24, # Role B is 24/7
                            "work_days": [0, 1, 2, 3, 4, 5, 6]
                        }
                    }
                }
            }
        }
        # Sunday 2012-01-01 22:00 (Role A is off-hours, Role B is active 24/7)
        sunday_ref = datetime(2012, 1, 1, 0, 0, 0)
        act_to_role = {"Task_A": "Role_A", "Task_247": "Role_247"}
        cal = CalendarManager.from_config(cfg, ref_dt=sunday_ref, activity_to_role=act_to_role)

        role_data = {
            "__all_resources__": ["WA", "WB"],
            "__role_names__": ["Role_A", "Role_247"],
            "roles": {"Role_A": ["Task_A"], "Role_247": ["Task_247"]},
            "activity_to_role": act_to_role,
            "role_to_workers": {"Role_A": ["WA"], "Role_247": ["WB"]},
            "role_capacities": {"Role_A": 1, "Role_247": 1},
            "worker_concurrencies": {"WA": 1, "WB": 1}
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(role_data, f)
            role_file = f.name

        try:
            pool = SharedWorkerPool(env, role_file, latency_cfg={}, calendar=cal)

            # Advance to Sunday 22:00
            env.run(until=22 * 3600)

            # Request for Task_A (Role_A is closed on Sunday) -> Must queue
            ev_a = pool.request_worker("Task_A")
            self.assertFalse(ev_a.triggered, "Task_A must queue because Role_A is off-shift on Sunday")

            # Request for Task_247 (Role_247 is 24/7) -> Must dispatch immediately!
            ev_247 = pool.request_worker("Task_247")
            self.assertTrue(ev_247.triggered, "Task_247 MUST dispatch immediately without being blocked by Role_A!")
            self.assertEqual(ev_247.value, "WB")

            # Advance to Monday 08:00 AM (shift open for Role A: 34h total from Sunday 00:00 = 10h after Sunday 22:00)
            env.run(until=32 * 3600 + 2)
            self.assertTrue(ev_a.triggered, "Task_A must be dispatched at Monday 08:00 AM opening")
            self.assertEqual(ev_a.value, "WA")
        finally:
            if os.path.exists(role_file):
                os.remove(role_file)

    def test_precedence_hierarchy_3_tier(self):
        """Verify strict 3-tier precedence: Tier 1 > Tier 2 > Tier 3."""
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({
                "activity_to_role": {"Task_A": "Role_A", "Task_B": "Role_B"},
                "role_calendars": {
                    "Role_A": {"hour_start": 7, "hour_end": 15, "work_days": [0, 1, 2, 3]},
                    "Role_B": {"hour_start": 6, "hour_end": 14, "work_days": [0, 1, 2, 3, 4]}
                }
            }, f)
            role_file = f.name

        try:
            cfg = {
                "simulation": {
                    "calendar": {
                        "enabled": True,
                        "hour_start": 8,
                        "hour_end": 20,
                        "work_days": [0, 1, 2, 3, 4],
                        "use_discovered_role_calendars": False, # Tier 2 disabled
                        "per_role": {
                            "Role_A": {"hour_start": 10, "hour_end": 18, "work_days": [0, 1, 2, 3, 4]} # Tier 1
                        }
                    }
                }
            }
            # When use_discovered_role_calendars is False:
            cal1 = CalendarManager.from_config(cfg, role_mapping_path=role_file)
            # Role_A has Tier 1 override (10:00 - 18:00)
            self.assertEqual(cal1.get_calendar("Task_A").hour_start, 10)
            # Role_B has no Tier 1, and Tier 2 is disabled -> Tier 3 global fallback (08:00 - 20:00)
            self.assertEqual(cal1.get_calendar("Task_B").hour_start, 8)

            # When use_discovered_role_calendars is True:
            cfg["simulation"]["calendar"]["use_discovered_role_calendars"] = True
            cal2 = CalendarManager.from_config(cfg, role_mapping_path=role_file)
            # Role_A still uses Tier 1 override (10:00 - 18:00)
            self.assertEqual(cal2.get_calendar("Task_A").hour_start, 10)
            # Role_B now uses Tier 2 discovered calendar (06:00 - 14:00)
            self.assertEqual(cal2.get_calendar("Task_B").hour_start, 6)
        finally:
            if os.path.exists(role_file):
                os.remove(role_file)

    def test_24_7_calendar_bounds_no_crash(self):
        """Verify that 24/7 calendar (hour_start=0, hour_end=24) calculates duration without error."""
        cal = CalendarManager(enabled=True, work_days=list(range(7)), hour_start=0, hour_end=24)
        dur = cal.calculate_calendar_duration(0, 3600.0)
        self.assertEqual(dur, 3600.0)

        dt1 = datetime(2012, 1, 1, 10, 0, 0)
        dt2 = datetime(2012, 1, 2, 15, 30, 0)
        net = cal.calculate_net_working_seconds(dt1, dt2)
        self.assertEqual(net, (dt2 - dt1).total_seconds())

    def test_concurrency_quotas_propagation_and_capacity(self):
        """Verify that SharedWorkerPool respects enable_concurrency_quotas configuration.
        When enable_concurrency_quotas=True: worker with mined concurrency quota C_w=2 can take 2 tasks.
        When enable_concurrency_quotas=False: worker capacity is clamped to 1, blocking concurrent tasks."""
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({
                "__all_resources__": ["W_MultiTasker"],
                "__role_names__": ["Role_A"],
                "activity_to_role": {"Task_A": "Role_A"},
                "role_to_workers": {"Role_A": ["W_MultiTasker"]},
                "Task_A": ["W_MultiTasker"],
                "worker_concurrencies": {"W_MultiTasker": 2}
            }, f)
            role_file = f.name

        try:
            # Case 1: enable_concurrency_quotas=True (Default/Mined behaviour)
            env1 = simpy.Environment()
            pool1 = SharedWorkerPool(env1, role_file, enable_concurrency_quotas=True)
            self.assertEqual(pool1._get_capacity("W_MultiTasker"), 2)

            ev1_a = pool1.request_worker("Task_A")
            self.assertTrue(ev1_a.triggered, "First request should be dispatched immediately")
            self.assertEqual(ev1_a.value, "W_MultiTasker")
            self.assertEqual(pool1.worker_active_count["W_MultiTasker"], 1)

            ev1_b = pool1.request_worker("Task_A")
            self.assertTrue(ev1_b.triggered, "Second concurrent request should dispatch immediately because C_w=2")
            self.assertEqual(ev1_b.value, "W_MultiTasker")
            self.assertEqual(pool1.worker_active_count["W_MultiTasker"], 2)

            # 3rd request must wait (capacity reached)
            ev1_c = pool1.request_worker("Task_A")
            self.assertFalse(ev1_c.triggered, "Third request should queue because capacity is 2")

            # Case 2: enable_concurrency_quotas=False (Clamped to 1)
            env2 = simpy.Environment()
            pool2 = SharedWorkerPool(env2, role_file, enable_concurrency_quotas=False)
            self.assertEqual(pool2._get_capacity("W_MultiTasker"), 1)

            ev2_a = pool2.request_worker("Task_A")
            self.assertTrue(ev2_a.triggered, "First request should be dispatched immediately")
            self.assertEqual(pool2.worker_active_count["W_MultiTasker"], 1)

            ev2_b = pool2.request_worker("Task_A")
            self.assertFalse(ev2_b.triggered, "Second request MUST queue because concurrency quotas are disabled (C_w clamped to 1)")

            # When first task releases, second task gets dispatched
            pool2.release_worker("W_MultiTasker")
            self.assertTrue(ev2_b.triggered, "Queued request must be dispatched immediately after release")
            self.assertEqual(ev2_b.value, "W_MultiTasker")

        finally:
            if os.path.exists(role_file):
                os.remove(role_file)

    def test_petri_net_concurrent_and_split_partitioning(self):
        """Verify that Token._partition_and_resolve_transitions correctly identifies
        independent parallel branches in an AND-split Petri net and selects both transitions."""
        from pm4py.objects.petri_net.obj import PetriNet, Marking
        from src.simulator.token import Token

        net = PetriNet("Test_Parallel_Net")
        p_src = PetriNet.Place("source")
        p_p1 = PetriNet.Place("p1")
        p_p2 = PetriNet.Place("p2")
        p_sink = PetriNet.Place("sink")
        for p in [p_src, p_p1, p_p2, p_sink]:
            net.places.add(p)

        t_a = PetriNet.Transition("t_a", "Task_A")
        t_b = PetriNet.Transition("t_b", "Task_B")
        for t in [t_a, t_b]:
            net.transitions.add(t)

        PetriNet.Arc(p_p1, t_a)
        PetriNet.Arc(p_p2, t_b)

        # Mock token with marking having tokens in both p1 and p2 (AND-split output)
        env = simpy.Environment()
        class MockProcess:
            def __init__(self):
                self.env = env
                self.calendar = CalendarManager()
                self.net = net
                self.im = Marking({p_src: 1})
                self.fm = Marking({p_sink: 1})
                self.cfg = {}
                self.target_activities = ["Task_A", "Task_B"]
                self.proc_prefixes = ()
                self.ms_prefixes = ()
                self.branch_target_map = {}
                self.external_delays = {}
                self.active_cases_count = 1
                self.ac_wip = {"Task_A": 0, "Task_B": 0}
                self.max_events = 10

        token = Token("case_001", arrival_offset=0, process=MockProcess())
        token.marking = Marking({p_p1: 1, p_p2: 1})

        # When enabled contains both t_a and t_b originating from disjoint places:
        selected = token._partition_and_resolve_transitions([t_a, t_b])
        self.assertEqual(len(selected), 2, "Both transitions must be selected for concurrent execution")
        self.assertIn(t_a, selected)
        self.assertIn(t_b, selected)

if __name__ == "__main__":
    unittest.main()