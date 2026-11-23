"""Task ordering, priority mapping, payload isolation and trace export contracts."""
import json
from collections import defaultdict
from types import SimpleNamespace
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import pm4py
import simpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from queue_policy import QueuePolicy
from features import FeatureContext, FeatureManager
from simulator.arrivals import ArrivalGenerator
from simulator.calendar import CalendarManager
from simulator.token import Token
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils.petri_utils import add_arc_from_to
from simulator.metrics import SimulationEvaluator
from simulator.process import SharedWorkerPool


def priority_config():
    return {"type": "priority", "priority": {
        "levels": 4, "source_attribute": "Amount", "mapping": {
            "kind": "numeric_ranges", "ranges": [
                {"max_exclusive": 1000, "priority": 4},
                {"min_inclusive": 1000, "max_exclusive": 5000, "priority": 3},
                {"min_inclusive": 5000, "max_exclusive": 10000, "priority": 2},
                {"min_inclusive": 10000, "priority": 1}]}}}


class QueuePolicyTests(unittest.TestCase):
    def test_numeric_boundaries_and_missing(self):
        cfg = priority_config()
        policy = QueuePolicy(cfg)
        self.assertEqual([policy.resolve(v) for v in (-1, 999, 1000, 4999, 5000, 9999, 10000)],
                         [4, 4, 3, 3, 2, 2, 1])
        for value in (None, float('nan'), 'bad', float('inf'), True):
            with self.assertRaisesRegex(ValueError, 'case.*example'):
                policy.resolve(value, 'example')
        cfg['priority'].update(on_missing='default', default_priority=4)
        policy = QueuePolicy(cfg)
        self.assertEqual(policy.resolve(None), 4)
        with self.assertRaises(ValueError):
            policy.resolve('bad')

    def test_categories_and_existing_priority(self):
        cfg = priority_config()
        cfg['priority']['mapping'] = {'kind': 'categorical', 'values': {'red': 1, 'white': 4}}
        policy = QueuePolicy(cfg)
        self.assertEqual(policy.resolve('red'), 1)
        with self.assertRaises(ValueError):
            policy.resolve('unknown')
        del cfg['priority']['mapping']
        policy = QueuePolicy(cfg)
        self.assertEqual(policy.resolve(2.0), 2)
        for value in (0, 5, 1.5, '1'):
            with self.assertRaises(ValueError):
                policy.resolve(value)

    def test_configuration_errors_and_uncovered_values(self):
        for update in ({'levels': 0}, {'levels': 2.5}, {'on_missing': 'ignore'},
                       {'on_missing': 'default'}, {'mapping': {'kind': 'unknown'}},
                       {'mapping': {'kind': 'categorical', 'values': {'red': 5}}},
                       {'mapping': {'kind': 'numeric_ranges', 'ranges': [
                           {'max_exclusive': 10, 'priority': 1},
                           {'min_inclusive': 5, 'priority': 2}]}}):
            cfg = priority_config()
            cfg['priority'].update(update)
            with self.subTest(update=update), self.assertRaises(ValueError):
                QueuePolicy(cfg)
        for policy_type in ('edd', 'spt', 'typo'):
            with self.assertRaises(ValueError):
                QueuePolicy({'type': policy_type})
        cfg = priority_config()
        cfg['priority']['mapping']['ranges'] = [{'min_inclusive': 0, 'max_exclusive': 10, 'priority': 1}]
        with self.assertRaisesRegex(ValueError, 'uncovered'):
            QueuePolicy(cfg).resolve(10)

    def pool(self, directory, queue, calendar=None):
        path = Path(directory, 'roles.json')
        path.write_text(json.dumps({'__all_resources__': ['w'], 'A': ['w'], 'B': ['w']}))
        return SharedWorkerPool(simpy.Environment(), str(path), queue_cfg=queue, calendar=calendar)

    def test_ordering_ties_and_no_preemption(self):
        with tempfile.TemporaryDirectory() as tmp:
            for queue, expected in (({}, [0, 1, 2]), ({'type': 'lifo'}, [2, 1, 0]),
                                    (priority_config(), [1, 2, 0])):
                with self.subTest(queue=queue):
                    pool = self.pool(tmp, queue)
                    running = pool.request_worker('A', priority=4)
                    requests = [pool.request_worker('A', priority=p) for p in (4, 1, 1)]
                    self.assertTrue(running.triggered)
                    self.assertTrue(all(not ev.triggered for ev in requests))
                    for index in expected:
                        pool.release_worker('w', 'A')
                        self.assertTrue(requests[index].triggered)
                        self.assertEqual(pool.worker_active_count['w'], 1)
                        requests[index] = pool.env.event()  # All other requests must still be waiting.
                        self.assertTrue(all(not ev.triggered for ev in requests))

    def test_priority_respects_shifts_and_eligible_workers(self):
        with tempfile.TemporaryDirectory() as tmp:
            pool = self.pool(tmp, priority_config())
            pool.request_worker('B', priority=4)
            with patch.object(pool, '_is_activity_working_now', side_effect=lambda a: a != 'A'), \
                 patch.object(pool, '_ensure_shift_alarm'):
                closed = pool.request_worker('A', priority=1)
                open_task = pool.request_worker('B', priority=4)
                pool.release_worker('w', 'B')
                self.assertFalse(closed.triggered)
                self.assertTrue(open_task.triggered)
            pool.release_worker('w', 'B')
            self.assertTrue(closed.triggered)

    def test_arrival_priority_is_separate_and_jointly_sampled(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, 'log.xes'); path.touch()
            frame = pd.DataFrame({'case:concept:name': ['one', 'two'], 'concept:name': ['A', 'A'],
                'time:timestamp': pd.to_datetime(['2024-01-08 09:00Z', '2024-01-08 10:00Z']),
                'case:amount': [500, 20000], 'case:TriageCode': ['white', 'red']})
            cfg = {'paths': {'aligned_log': str(path)}, 'features': {
                'selected_features': ['case_attributes'], 'case_attributes': [
                    {'column': 'case:amount', 'feature_name': 'Amount', 'type': 'continuous'}]},
                'simulation': {'queue': priority_config(), 'arrival': {'prophet': {
                    'cache_model': False, 'model_cache_dir': tmp, 'sampling_method': 'round'}}}}
            class Forecast:
                def __init__(self, **kwargs):
                    pass
                def fit(self, frame):
                    return self
                def predict(self, frame):
                    return frame.assign(yhat=1)

            with patch('simulator.arrivals.pm4py.read_xes', return_value=frame), patch('prophet.Prophet', Forecast):
                for mode in ('replay', 'generative', 'prophet'):
                    arrivals, _ = ArrivalGenerator(cfg, num_cases=8, arrival_mode=mode).prepare_arrivals()
                    for arrival in arrivals:
                        self.assertEqual(arrival['queue_priority'], 4 if arrival['case_attrs']['Amount'] == 500 else 1)
                        self.assertEqual(set(arrival['case_attrs']), {'Amount'})
                        self.assertEqual(set(FeatureManager(cfg).build_features(
                            FeatureContext(case_attrs=arrival['case_attrs']))), {'Amount'})
                # Scheduling-only source: no corresponding ML feature required.
                cfg['simulation']['queue']['priority'].update(source_attribute='TriageCode', mapping={
                    'kind': 'categorical', 'values': {'white': 4, 'red': 1}})
                arrivals, _ = ArrivalGenerator(cfg, num_cases=2).prepare_arrivals()
                self.assertEqual([a['queue_priority'] for a in arrivals], [4, 1])
                frame.loc[0, 'case:TriageCode'] = None
                with self.assertRaisesRegex(ValueError, 'missing source'):
                    ArrivalGenerator(cfg, num_cases=2).prepare_arrivals()
                cfg['simulation']['queue']['priority'].update(on_missing='default', default_priority=3)
                arrivals, _ = ArrivalGenerator(cfg, num_cases=2).prepare_arrivals()
                self.assertEqual(arrivals[0]['queue_priority'], 3)
            cfg['features']['case_attributes'].append({'column': 'case:Priority', 'feature_name': 'Hidden'})
            with self.assertRaisesRegex(ValueError, 'must not be configured as an ML feature'):
                FeatureManager(cfg)

    def test_tokens_pass_priority_to_each_task_without_ml_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            pool = self.pool(tmp, priority_config())
            net = PetriNet('sequence')
            p, q, r = [PetriNet.Place(n) for n in ('p', 'q', 'r')]
            a, b = PetriNet.Transition('a', 'A'), PetriNet.Transition('b', 'B')
            net.places.update([p, q, r]); net.transitions.update([a, b])
            for source, target in ((p, a), (a, q), (q, b), (b, r)):
                add_arc_from_to(source, target, net)
            records = []
            def duration(*args, **kwargs):
                self.assertEqual(kwargs['case_attrs'], {'Amount': 100})
                return 10
            process = SimpleNamespace(env=pool.env, net=net, im=Marking({p: 1}), fm=Marking({r: 1}),
                calendar=CalendarManager(enabled=False), mode='pure_physics', active_cases_count=0,
                max_events=10, target_activities=['A', 'B'], external_delays={},
                has_ex_ante_delay=lambda act: False, activity_profiles={}, workers=pool,
                ac_wip=defaultdict(int), predictor=SimpleNamespace(predict_duration=duration),
                record_event=records.append)
            with patch.object(pool, 'request_worker', wraps=pool.request_worker) as request:
                for name, arrival, priority in (('running', 0, 4), ('low', 1, 4), ('high', 2, 1)):
                    token = Token(name, arrival, process=process, case_attrs={'Amount': 100}, queue_priority=priority)
                    pool.env.process(token.run())
                pool.env.run()
                self.assertEqual(request.call_count, 6)
                self.assertTrue(all(call.kwargs['priority'] in (1, 4) for call in request.call_args_list))
            starts = sorted(records, key=lambda record: record['start_timestamp'])
            self.assertEqual([record['case_id'] for record in starts[:3]], ['running', 'high', 'low'])
            # Finishing A releases the worker before high's next task B is ready.
            # Priority is task-level and cannot interrupt low once it has started.

    def test_missing_raw_amount_is_not_replaced_by_ml_imputation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, 'log.xes'); path.touch()
            frame = pd.DataFrame({'case:concept:name': ['one', 'one'], 'concept:name': ['A', 'B'],
                'time:timestamp': pd.to_datetime(['2024-01-08 09:00Z', '2024-01-08 10:00Z']),
                'case:amount': [None, 20000]})
            cfg = {'paths': {'aligned_log': str(path)}, 'features': {'case_attributes': [
                {'column': 'case:amount', 'feature_name': 'Amount', 'type': 'continuous',
                 'fill_strategy': 'median'}]}, 'simulation': {'queue': priority_config()}}
            with patch('simulator.arrivals.pm4py.read_xes', return_value=frame):
                for mode in ('replay', 'generative'):
                    with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, 'missing source'):
                        ArrivalGenerator(cfg, num_cases=1, arrival_mode=mode).prepare_arrivals()
                cfg['simulation']['queue']['priority'].update(on_missing='default', default_priority=4)
                for mode in ('replay', 'generative'):
                    arrivals, _ = ArrivalGenerator(cfg, num_cases=1, arrival_mode=mode).prepare_arrivals()
                    self.assertEqual(arrivals[0]['queue_priority'], 4)
                    self.assertEqual(arrivals[0]['case_attrs']['Amount'], 20000)

    def test_priority_export_is_a_trace_attribute(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = {'simulation': {'output_simulated_log': str(Path(tmp, 'log.csv')),
                                  'output_simulated_xes': str(Path(tmp, 'log.xes'))}}
            evaluator = SimulationEvaluator(cfg)
            evaluator.save_xes(pd.DataFrame([{'case_id': 'one', 'activity': 'A', 'resource_id': 'w',
                'start_timestamp': pd.Timestamp('2024-01-08 09:00Z'),
                'complete_timestamp': pd.Timestamp('2024-01-08 10:00Z'), 'case:Priority': 2}]))
            log = pm4py.read_xes(str(Path(tmp, 'log.xes')), return_legacy_log_object=True)
            self.assertEqual(log[0].attributes['Priority'], 2)
            self.assertNotIn('Priority', log[0][0])


if __name__ == '__main__':
    unittest.main()
