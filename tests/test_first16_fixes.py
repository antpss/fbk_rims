"""Regression examples for the first sixteen audit findings."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import simpy
import xgboost as xgb
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils.petri_utils import add_arc_from_to

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from features import FeatureManager, capacity_occupancy
from historical_features import snapshot_rows
from temporal_split import make_case_split
from model_artifacts import prediction_metrics
from simulator.calendar import CalendarManager
from simulator.token import Token
from simulator.predictor import ChampionPredictor
from simulator.metrics import SimulationEvaluator
from simulator.arrivals import ArrivalGenerator


def module(filename):
    spec = importlib.util.spec_from_file_location("audit_" + filename, Path(__file__).resolve().parents[1] / "src" / (filename + ".py"))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def event(cid, act, hour, lifecycle, worker="w"):
    return {"case:concept:name": cid, "concept:name": act, "time:timestamp": pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(hours=hour),
            "lifecycle:transition": lifecycle, "org:resource": worker}


class HistoricalTests(unittest.TestCase):
    def test_strict_split_uses_arrivals_and_excludes_crossing_cases(self):
        rows = [event(str(i), "A", i * 24, "START") for i in range(10)]
        rows += [event(str(i), "A", i * 24 + (200 if i == 0 else 1), "COMPLETE") for i in range(10)]
        manifest = make_case_split(pd.DataFrame(rows).sample(frac=1, random_state=2))
        self.assertEqual(manifest["cases"]["0"]["partition"], "excluded")
        self.assertEqual(manifest["cases"]["6"]["partition"], "train")
        self.assertEqual(manifest["cases"]["7"]["partition"], "validation")
        self.assertEqual(manifest["cases"]["8"]["partition"], "test")

    def test_snapshot_timing_prefix_and_unmatched_completion(self):
        rows = [event("x", "A", 8, "START"), event("y", "B", 8.5, "START"),
                event("x", "A", 9, "COMPLETE"), event("z", "B", 9.25, "COMPLETE"),
                event("x", "C", 10, "START"), event("y", "B", 11, "COMPLETE"), event("x", "C", 12, "COMPLETE")]
        frame = pd.DataFrame(rows)
        cfg = {"features": {"selected_features": ["wip", "ac_wip", "daytime", "weekday", "role_occupancy", "prev_duration", "trace_history"]},
               "tasks": {"waiting_time": {"target_mode": "residual_delay"}}}
        role = {"__role_names__": ["r"], "role_to_workers": {"r": ["w"]}, "worker_concurrencies": {"w": 2}}
        delays = pd.DataFrame([{"case_id": "x", "target_act": "C", "start_timestamp": rows[4]["time:timestamp"],
                               "predecessor_end_timestamp": rows[2]["time:timestamp"], "residual_delay_seconds": 123}])
        kwargs = dict(df=frame, cfg=cfg, manager=FeatureManager(cfg), calendar=CalendarManager(enabled=False), role_data=role,
                      activity_types={}, activities=["A", "B", "C"], tracked=["A", "B", "C"], imputation={})
        wait = snapshot_rows(task="waiting_time", delay_records=delays, **kwargs)[0]
        self.assertEqual(wait["Daytime"], 9/24)
        self.assertEqual(wait["WIP"], 2)
        self.assertEqual(wait["AC_WIP"], 0)
        self.assertEqual(wait["Role_0_OC"], .5)
        self.assertEqual(wait["Prev_Proc_Time"], 3600)
        self.assertEqual(wait["Prefix"], '["A"]')
        self.assertEqual(wait["Wait_Time_Seconds"], 123)
        duration = snapshot_rows(task="duration", **kwargs)
        c = next(row for row in duration if row["Target_Activity"] == "C")
        self.assertEqual(c["Role_0_OC"], 1)
        b = next(row for row in duration if row["Target_Activity"] == "B")
        self.assertEqual(b["Prefix"], '[]')
        self.assertEqual(len(duration), 3)  # unmatched human completion never becomes a sample

    def test_capacity_accounts_for_cross_role_work(self):
        role = {"__role_names__": ["a", "b"], "role_to_workers": {"a": ["w"], "b": ["w", "v"]},
                "worker_concurrencies": {"w": 3, "v": 1}}
        self.assertEqual(capacity_occupancy(role, {"w": 2}), [2/3, .5])
        self.assertEqual(capacity_occupancy(role, {"w": 2}, False), [1, .5])


class RuntimeTests(unittest.TestCase):
    def test_calendar_midnight_and_weekend(self):
        cal = CalendarManager(enabled=True, work_days=list(range(7)), hour_start=0, hour_end=24)
        cal.set_ref_dt(pd.Timestamp("2024-01-01"))
        self.assertEqual(cal.calculate_calendar_duration(0, 25*3600), 25*3600)
        cal = CalendarManager(enabled=True, work_days=list(range(5)), hour_start=0, hour_end=24)
        cal.set_ref_dt(pd.Timestamp("2024-01-05"))
        self.assertEqual(cal.calculate_calendar_duration(0, 25*3600), 73*3600)

    def test_overlapping_conflicts_reserve_tokens_without_serializing_independent_work(self):
        net = PetriNet("overlap")
        p, q = PetriNet.Place("p"), PetriNet.Place("q")
        net.places.update([p, q])
        a, b, c = [PetriNet.Transition(x, x) for x in "ABC"]
        net.transitions.update([a, b, c])
        for place, transition in [(p,a), (q,b), (p,c), (q,c)]:
            add_arc_from_to(place, transition, net)
        token = Token.__new__(Token)
        token.process = SimpleNamespace(net=net)
        token.marking = Marking({p:1, q:1})
        token._resolve_conflict_group = lambda group: c if c in group else group[0]
        self.assertEqual(token._partition_and_resolve_transitions([a,b,c]), [c])
        token._resolve_conflict_group = lambda group: group[0]
        self.assertEqual(token._partition_and_resolve_transitions([a,b,c]), [a,b])
        self.assertEqual(token.marking, Marking({p:1, q:1}))

    def test_silent_self_loop_raises_before_simpy_horizon(self):
        env = simpy.Environment()
        net = PetriNet("loop")
        p = PetriNet.Place("p")
        t = PetriNet.Transition("silent", None)
        net.places.add(p); net.transitions.add(t)
        add_arc_from_to(p,t,net); add_arc_from_to(t,p,net)
        process = SimpleNamespace(env=env, net=net, im=Marking({p:1}), fm=Marking(),
                                  calendar=CalendarManager(enabled=False), active_cases_count=0,
                                  max_events=80, max_silent_transitions=5)
        env.process(Token("c",0,process=process).run())
        with self.assertRaisesRegex(RuntimeError, "consecutive silent"):
            env.run(until=1)
        self.assertEqual(env.now, 0)

    def test_xgboost_runtime_limits_iterations(self):
        x = pd.DataFrame({"WIP": np.arange(30, dtype=float)})
        model = xgb.XGBRegressor(n_estimators=10, n_jobs=1, max_depth=2)
        model.fit(x, 3 + np.sin(x.WIP))
        model.get_booster().set_attr(best_iteration="1")
        predictor = ChampionPredictor.__new__(ChampionPredictor)
        predictor.max_seq_len = 1
        predictor.feature_cols = ["WIP"]
        predictor._build_feature_dict = lambda *a, **kw: {"WIP": 5.0}
        suite = {"models": {"A": {"type": "xgboost", "model": model}}}
        actual = predictor._predict_time(suite, {}, "A", [], 5, {}, None, 0, [], 600)
        self.assertAlmostEqual(actual, float(np.expm1(model.predict(pd.DataFrame({"WIP":[5.0]}))[0])), places=5)

    def test_wait_prediction_precedes_external_delay(self):
        env = simpy.Environment()
        observed = []
        process = SimpleNamespace(env=env, im=Marking(), calendar=CalendarManager(enabled=False), mode="hybrid_residual",
                                  target_activities=["A"], active_cases_count=1, ac_wip={}, external_delays={},
                                  workers=SimpleNamespace(get_role_occupancy=lambda _: []),
                                  predictor=SimpleNamespace(predict_waiting=lambda *a, **k: observed.append(env.now) or 10),
                                  has_ex_ante_delay=lambda act: True, sample_ex_ante_delay=lambda act: 5,
                                  activity_profiles={"A":"AUTOMATED_TASK"}, record_event=lambda r: None)
        token = Token("c", 0, process=process)
        env.process(token._execute_work_item(PetriNet.Transition("A", "A"), lambda: None))
        env.run()
        self.assertEqual(observed, [0])
        self.assertEqual(env.now, 10)  # preserve max(external, predicted) composition


class EvaluationTests(unittest.TestCase):
    def logs(self):
        real = pd.DataFrame([event("c","A",0,"START"), event("c","A",1,"COMPLETE")])
        sim = pd.DataFrame([{"case_id":"c", "activity":"A", "start_timestamp":real.iloc[0]["time:timestamp"],
                             "complete_timestamp":real.iloc[0]["time:timestamp"] + pd.Timedelta(hours=2),
                             "resource_id":"w", "duration_seconds":7200, "wait_time_seconds":0, "queue_wait_seconds":0}])
        return real, sim

    def test_units_and_failed_metric_are_not_zero(self):
        with tempfile.TemporaryDirectory() as tmp, patch("simulator.metrics.control_flow_log_distance", side_effect=ValueError("bad metric")):
            ev = SimulationEvaluator({"paths":{"output_base_dir":tmp}}, export_xes=False)
            result = ev.evaluate(*self.logs())
            self.assertEqual(result["ctd"], 3600)
            self.assertIsNone(result["cfld"])
            self.assertEqual(result["metric_status"]["cfld"]["status"], "error")
            self.assertIn("bad metric", Path(tmp,"simulation_report.md").read_text())

    def test_censored_case_excluded_from_full_case_metrics(self):
        real, sim = self.logs()
        with tempfile.TemporaryDirectory() as tmp:
            ev = SimulationEvaluator({"paths":{"output_base_dir":tmp}}, export_xes=False)
            ev.case_status = pd.DataFrame([{"case_id":"c", "arrival":sim.iloc[0].start_timestamp, "end":None, "status":"censored"}])
            ev.window_start = sim.iloc[0].start_timestamp
            ev.window_end = ev.window_start + pd.Timedelta(minutes=30)
            sim.loc[0,"complete_timestamp"] = None
            result = ev.evaluate(real,sim)
            self.assertIsNone(result["ctd"])
            self.assertEqual(result["populations"]["completed_simulated_evaluated"], 0)
            self.assertEqual(result["populations"]["window_simulated_intervals"], 1)
            self.assertEqual(result["populations"]["simulated_statuses"], {"censored":1})

    def test_empty_results_and_missing_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = SimulationEvaluator({"paths":{"output_base_dir":tmp}}, export_xes=False).evaluate(None,pd.DataFrame())
            self.assertIsNone(result["ctd"])
            self.assertIsNone(result["cfld"])
            self.assertEqual(result["events"],0)


class ArtifactTests(unittest.TestCase):
    def test_untrained_zero_predictions_not_scored(self):
        table = pd.DataFrame({"Actual":[0,10], "Predicted":[0,9], "Source_Model":["","trained"], "Target_Activity":["A","B"]})
        stats = prediction_metrics(table)
        self.assertNotIn("A", stats["activity_metrics"])
        self.assertEqual(stats["available_count"], 1)

    def test_validation_selection_and_fresh_self_contained_package(self):
        bench = module("04b_benchmark_models")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, val, test in [("global_xgboost",9,100), ("global_other",12,10)]:
                folder = root/name; folder.mkdir()
                model = xgb.XGBRegressor(n_estimators=2,n_jobs=1).fit(pd.DataFrame({"WIP":[0,1,2]}), [2,3,4])
                model.save_model(folder/"model.json")
                dispatch = {"experiment_id":"current", "dispatch":{"__default__":{"model_type":"xgboost", "file":"model.json"}}}
                (folder/"dispatch_config.json").write_text(json.dumps(dispatch))
                (folder/"training_metadata.json").write_text('{}')
                (folder/"categories.json").write_text('{"activities":["A"]}')
                table = pd.DataFrame({"Case_ID":["v","t"], "Instance_ID":["1","2"], "Partition":["validation","test"],
                                      "Target_Activity":["A","A"], "Actual":[10,10], "Predicted":[val,test], "Source_Model":[name,name]})
                table.to_csv(folder/"predictions.csv", index=False)
                metrics = {"task":"duration", "strategy":"global", "experiment_id":"current", "selection_partition":"validation",
                           **prediction_metrics(table.iloc[:1])}
                (folder/"metrics.json").write_text(json.dumps(metrics))
            stale = root/"hybrid_champion"; stale.mkdir()
            (stale/"categories.json").write_text('{"activities":["OLD"]}')
            result = bench.synthesize_champion(root, candidates=["global_xgboost","global_other"], identity="current")
            self.assertEqual(result["test_metrics"]["weighted_mae"], 90)
            manifest = json.loads((stale/"dispatch_config.json").read_text())
            self.assertEqual(manifest["dispatch"]["A"]["bundle"], "global_xgboost")
            self.assertFalse((stale/"categories.json").exists())
            self.assertTrue((stale/"global_xgboost/categories.json").exists())


class ArrivalTerminationTests(unittest.TestCase):
    def test_prophet_bounds_progress_even_when_forecast_empty(self):
        historical = pd.DataFrame([event(str(i), "A", i*24, "COMPLETE") for i in range(4)])
        class Forecast:
            calls = 0
            count = 0
            def __init__(self, **kwargs):
                pass
            def fit(self, frame):
                return self
            def predict(self, frame):
                Forecast.calls += 1
                if Forecast.calls > 3:
                    raise AssertionError("Forecast kept extending beyond its bound")
                return frame.assign(yhat=Forecast.count)
        for cases, duration, count in [(100, 1/24, 1), (None, 1/24, 0), (100, None, 0)]:
            with self.subTest(cases=cases, duration=duration), tempfile.TemporaryDirectory() as tmp:
                Forecast.calls, Forecast.count = 0, count
                cfg = {"paths":{"aligned_log":os.path.join(tmp,"not-needed.xes")}, "simulation":{"arrival":{
                    "start_timestamp":"2024-01-01 00:00:00", "calendar_bounded":False,
                    "prophet":{"cache_model":False, "model_cache_dir":tmp, "sampling_method":"round", "max_search_days":1/24}}}}
                generator = ArrivalGenerator(cfg,num_cases=cases,arrival_mode="prophet",duration_days=duration)
                with patch("prophet.Prophet", Forecast):
                    result = generator._prepare_prophet_arrivals(historical, [], None)
                self.assertEqual(len(result), count)
                self.assertEqual(Forecast.calls, 1)
                if cases:
                    self.assertEqual(generator.generation_status["status"], "shortfall")

    def test_replay_reference_contains_only_scheduled_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"log.xes"; path.touch()
            cfg = {"paths":{"aligned_log":str(path)}}
            log = pd.DataFrame([event(str(i),"A",i,"COMPLETE") for i in range(5)])
            with patch("simulator.arrivals.pm4py.read_xes", return_value=log):
                arrivals, reference = ArrivalGenerator(cfg,num_cases=2,arrival_mode="replay").prepare_arrivals()
            self.assertEqual(set(reference["case:concept:name"]), {a["case_id"] for a in arrivals})
            self.assertEqual(len(arrivals),2)

    def test_generated_arrival_beyond_horizon_is_not_scheduled(self):
        cfg = {"paths":{"aligned_log":"/nonexistent"}, "simulation":{"arrival":{
            "mode":"generative", "distribution":"constant", "scale":7200, "start_timestamp":"2024-01-01 00:00:00"}}}
        generator = ArrivalGenerator(cfg,num_cases=5,arrival_mode="generative",duration_days=1/24)
        arrivals,_ = generator.prepare_arrivals()
        self.assertEqual(arrivals, [])
        self.assertEqual(pd.Timestamp(generator.origin),pd.Timestamp("2024-01-01"))


class DeploymentTests(unittest.TestCase):
    def test_requested_strategy_wins_over_existing_hybrid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            routing = root / "routing"
            routing.mkdir()
            (routing / "routing_decisions.json").write_text('{"decision_points": {}}')
            model = xgb.XGBRegressor(n_estimators=2, n_jobs=1).fit(pd.DataFrame({"WIP": [0, 1, 2]}), [2, 3, 4])
            cfg = {"paths": {"models_dir": tmp, "routing_models_dir": str(routing)}, "features": {},
                   "activity_profiling": {"enabled": False}, "model_strategy": {"mode": "single_global", "single_global_engine": "xgboost"}}
            for task in ("duration", "waiting_time"):
                for folder in ("global_xgboost", "hybrid_champion", "global_champion"):
                    dest = root / task / folder
                    dest.mkdir(parents=True)
                    filename = folder + ".json"
                    model.save_model(dest / filename)
                    (dest / "dispatch_config.json").write_text(json.dumps({"schema_version": 2, "task": task,
                        "dispatch": {"__default__": {"file": filename, "model_type": "xgboost"}}}))
                    (dest / "training_metadata.json").write_text(json.dumps({"feature_config": {}, "capacity_quotas": True,
                        "feature_columns": ["WIP"], "feature_caps": {}, "activity_caps": {}, "global_cap": 100}))
            for mode, expected in [("single_global", "global_xgboost"), ("global_tournament", "global_champion"), ("hybrid", "hybrid_champion")]:
                cfg["model_strategy"]["mode"] = mode
                predictor = ChampionPredictor(cfg)
                self.assertEqual(predictor.dur_dispatch["__default__"]["file"], expected + ".json")
                self.assertEqual(predictor.wait_dispatch["__default__"]["file"], expected + ".json")
            cfg["model_strategy"].update(mode="single_global", single_global_engine="missing")
            with self.assertRaises(FileNotFoundError):
                ChampionPredictor(cfg)

    def test_neural_dataset_uses_its_input_partition(self):
        train = module("04_train")
        frame = pd.DataFrame({"Target_Duration": [10], "Target_Activity": ["A"], "Prefix": ['["B"]'], "WIP": [2]})
        data = train.EventLogDataset(frame, {"<PAD>":0, "<UNK>":1, "B":2}, {"A":0}, 2, ["WIP"], np.ones(1))
        self.assertEqual(data.prefixes.tolist(), [[0,2]])
        self.assertEqual(data.features.tolist(), [[2]])


if __name__ == "__main__":
    unittest.main()
