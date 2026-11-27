"""Case payloads survive simulation serialization as CSV columns/XES trace attributes."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd
import pm4py
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils.petri_utils import add_arc_from_to

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from simulator.process import SimulationProcess


class CaseAttributeExportTests(unittest.TestCase):
    def run_simulation(self, tmp, mode, horizon=None, attributes=True, silent=False):
        log_path, pnml = Path(tmp, "input.xes"), Path(tmp, "net.pnml")
        log_path.touch()
        pnml.touch()
        roles = Path(tmp, "roles.json")
        roles.write_text(json.dumps({"__all_resources__": ["w"], "A": ["w"]}))
        cfg = {
            "paths": {"aligned_log": str(log_path), "petri_net": str(pnml), "models_dir": tmp},
            "resources": {"discovery": {"output_file": str(roles)}},
            "features": {"case_attributes": [
                {"column": "case:Department", "feature_name": "Unit", "type": "categorical"},
                {"column": "Age", "feature_name": "PatientAge", "type": "continuous"},
            ] if attributes else []},
            "simulation": {
                "calendar": {"enabled": False},
                "arrival": {"mode": mode, "start_timestamp": "2024-01-01 00:00:00",
                            "distribution": "constant", "scale": 1},
                "output_simulated_log": str(Path(tmp, "out.csv")),
                "output_simulated_xes": str(Path(tmp, "out.xes")),
            },
        }
        frame = pd.DataFrame({
            "case:concept:name": ["one", "two"], "concept:name": ["A", "A"],
            "time:timestamp": pd.to_datetime(["2024-01-01T00:00:00Z", "2024-01-01T00:00:01Z"]),
            "case:Department": ["001", "002"], "Age": [30., None],
            "case:Undeclared": ["ignored", "ignored"],
        })
        net = PetriNet("single")
        p, q = PetriNet.Place("p"), PetriNet.Place("q")
        a = PetriNet.Transition("a", None if silent else "A")
        net.places.update([p, q])
        net.transitions.add(a)
        add_arc_from_to(p, a, net)
        add_arc_from_to(a, q, net)
        with patch("simulator.process.pm4py.read_pnml", return_value=(net, Marking({p: 1}), Marking({q: 1}))), \
             patch("simulator.arrivals.pm4py.read_xes", return_value=frame), \
             patch("activity_profiler.load_or_discover_activity_profiles", return_value={
                 "activity_types": {"A": "HUMAN_WORK_ITEM"}}), \
             patch("simulator.process.ChampionPredictor", return_value=SimpleNamespace(predict_duration=lambda *a, **kw: 10)), \
             patch("simulator.metrics.SimulationEvaluator.evaluate", return_value={}):
            sim = SimulationProcess(cfg, num_cases=2, mode="pure_physics", duration_days=horizon)
            sim.run()
        return sim

    def test_replay_and_generated_payloads_round_trip(self):
        for mode in ("replay", "generative"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                sim = self.run_simulation(tmp, mode)
                events = pd.read_csv(sim.evaluator.sim_log_path, dtype={"case:Department": str})
                cases = pd.read_csv(Path(sim.evaluator.sim_log_path).with_name("out_pure_physics_cases.csv"),
                                    dtype={"case:Department": str})
                self.assertEqual(len(events), 2)
                self.assertEqual(set(events["case:Age"]), {30.})  # imputed runtime payload
                self.assertTrue(set(events["case:Department"]) <= {"001", "002"})
                self.assertNotIn("case:Undeclared", events)
                expected = cases.set_index("case_id")
                for row in events.to_dict("records"):
                    self.assertEqual(row["case:Department"], expected.loc[row["case_id"], "case:Department"])
                log = pm4py.read_xes(sim.evaluator.output_simulated_xes, return_legacy_log_object=True)
                self.assertEqual(len(log), 2)
                for trace in log:
                    cid = trace.attributes["concept:name"]
                    self.assertEqual(trace.attributes["Department"], expected.loc[cid, "case:Department"])
                    self.assertEqual(trace.attributes["Age"], 30.)
                    self.assertEqual([e["lifecycle:transition"] for e in trace], ["start", "complete"])
                    self.assertTrue(all("Department" not in e and "Age" not in e for e in trace))

    def test_censored_cases_keep_payloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            sim = self.run_simulation(tmp, "replay", horizon=0.5 / 86400)
            events = pd.read_csv(sim.evaluator.sim_log_path, dtype={"case:Department": str})
            cases = pd.read_csv(Path(sim.evaluator.sim_log_path).with_name("out_pure_physics_cases.csv"),
                                dtype={"case:Department": str})
            self.assertEqual(events["case:Department"].tolist(), ["001"])
            self.assertTrue(events.complete_timestamp.isna().all())
            self.assertEqual(cases["case:Department"].tolist(), ["001"])
            self.assertEqual(cases.status.tolist(), ["censored"])

    def test_eventless_cases_keep_payloads_in_case_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            sim = self.run_simulation(tmp, "replay", silent=True)
            events = pd.read_csv(sim.evaluator.sim_log_path)
            cases = pd.read_csv(Path(sim.evaluator.sim_log_path).with_name("out_pure_physics_cases.csv"),
                                dtype={"case:Department": str})
            self.assertTrue(events.empty)
            self.assertIn("case:Department", events)
            self.assertEqual(cases["case:Department"].tolist(), ["001", "002"])
            self.assertEqual(cases["case:Age"].tolist(), [30., 30.])

    def test_no_configured_attributes_preserves_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            sim = self.run_simulation(tmp, "replay", attributes=False)
            events = pd.read_csv(sim.evaluator.sim_log_path)
            self.assertFalse(any(c.startswith("case:") for c in events))
