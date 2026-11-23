"""Predicted-duration scheduling and explicit case-deadline contracts."""
from collections import defaultdict
import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import pm4py
import simpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from queue_policy import QueuePolicy
from features import FeatureManager
from simulator.arrivals import ArrivalGenerator
from simulator.calendar import CalendarManager
from simulator.predictor import ChampionPredictor
from simulator.process import SharedWorkerPool, SimulationProcess
from simulator.token import Token
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils.petri_utils import add_arc_from_to


class SchedulingTests(unittest.TestCase):
    def make_pool(self, tmp, config):
        path = Path(tmp, 'roles.json')
        path.write_text(json.dumps({'__all_resources__': ['w'], 'A': ['w']}))
        return SharedWorkerPool(simpy.Environment(), str(path), queue_cfg=config)

    def test_spt_edd_and_priority_tie_breakers(self):
        with tempfile.TemporaryDirectory() as tmp:
            for kind in ('spt', 'edd', 'priority'):
                for tie, order in (('fifo', [1, 2, 0]), ('lifo', [2, 1, 0])):
                    cfg = {'type': kind, 'tie_breaker': tie, 'edd': {'column': 'case:DueDate'},
                           'priority': {'levels': 4}}
                    pool = self.make_pool(tmp, cfg)
                    def request(rank):
                        return pool.request_worker('A', priority=rank, estimated_duration=rank,
                                                   due_date=f'2024-01-0{rank}T00:00:00Z')
                    running = request(4)
                    requests = [request(rank) for rank in (3, 1, 1)]
                    self.assertTrue(running.triggered)
                    self.assertFalse(any(e.triggered for e in requests))
                    for index in order:
                        pool.release_worker('w', 'A')
                        self.assertTrue(requests[index].triggered, (kind, tie, index))
                        requests[index] = pool.env.event()
                        self.assertFalse(any(e.triggered for e in requests))

    def test_configuration_and_request_validation(self):
        for cfg in ({'type': 'edd'}, {'type': 'edd', 'edd': None}, {'type': 'edd', 'edd': {'column': ''}},
                    {'type': 'edd', 'edd': {'column': 'time:timestamp'}},
                    {'type': 'spt', 'tie_breaker': 'random'}):
            with self.subTest(cfg=cfg), self.assertRaises(ValueError):
                QueuePolicy(cfg)
        with tempfile.TemporaryDirectory() as tmp:
            pool = self.make_pool(tmp, {'type': 'spt'})
            for value in (None, -1, float('inf'), float('nan'), True):
                with self.assertRaises(ValueError):
                    pool.request_worker('A', estimated_duration=value)
            self.assertEqual(pool.pending_requests, [])
            self.assertTrue(pool.request_worker('A', estimated_duration=0).triggered)
        for value in (None, '', 'bad', pd.NaT, 123):
            with self.assertRaises(ValueError):
                QueuePolicy.parse_deadline(value)
        self.assertEqual(QueuePolicy.parse_deadline('2024-01-01T02:00:00+02:00'),
                         QueuePolicy.parse_deadline('2024-01-01T00:00:00Z'))

    def test_spt_rejects_loaded_worker_features(self):
        predictor = ChampionPredictor.__new__(ChampionPredictor)
        predictor.feature_cols = ['WIP']
        predictor.dur_models = {'models': {}}
        predictor.validate_spt_models()
        for entry in ({'suite': {'metadata': {'feature_columns': ['Resource_ID']}}},
                      {'model': SimpleNamespace(feature_names_in_=np.array(['Resource_ID']))},
                      {'model': SimpleNamespace(feature_names=['Resource_ID'])},
                      {'scaler': SimpleNamespace(feature_names_in_=np.array(['Resource_ID']))}):
            predictor.dur_models = {'models': {'A': entry}}
            with self.assertRaisesRegex(ValueError, 'Resource_ID'):
                predictor.validate_spt_models()

    def test_spt_frozen_estimate_and_start_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            pool = self.make_pool(tmp, {'type': 'spt'})
            net = PetriNet('one_task'); p = PetriNet.Place('p'); q = PetriNet.Place('q')
            a = PetriNet.Transition('a', 'A')
            net.places.update([p, q]); net.transitions.add(a)
            add_arc_from_to(p, a, net); add_arc_from_to(a, q, net)
            records, predictions = [], []
            def duration(act, prefix, wip, ac_wip, dt, amount, occupancy, **kwargs):
                name = kwargs['case_attrs']['Name']
                predictions.append((name, kwargs['worker_id'], pool.env.now))
                if kwargs['worker_id'] is None:
                    before = dict(process.ac_wip)
                    ac_wip['A'] = 999
                    prefix.append('mutated_copy')
                    kwargs['case_attrs']['Name'] = 'mutated_copy'
                    self.assertEqual(dict(process.ac_wip), before)
                    return {'running': 10, 'long': 8, 'short': 1}[name]
                return {'running': 10, 'long': 2, 'short': 3}[name]
            process = SimpleNamespace(env=pool.env, net=net, im=Marking({p: 1}), fm=Marking({q: 1}),
                calendar=CalendarManager(enabled=False), mode='pure_physics', active_cases_count=0,
                max_events=10, target_activities=['A'], external_delays={},
                has_ex_ante_delay=lambda act: False, activity_profiles={}, workers=pool,
                ac_wip=defaultdict(int), predictor=SimpleNamespace(predict_duration=duration),
                record_event=records.append)
            tokens = []
            py_state, np_state = random.getstate(), np.random.get_state()
            for name, arrival in (('running', 0), ('long', 1), ('short', 2)):
                token = Token(name, arrival, process=process, case_attrs={'Name': name})
                tokens.append(token); pool.env.process(token.run())
            pool.env.run()
            self.assertEqual([r['case_id'] for r in records], ['running', 'short', 'long'])
            self.assertEqual([r['queue_estimated_duration_seconds'] for r in records], [10, 1, 8])
            self.assertEqual([r['duration_seconds'] for r in records], [10, 3, 2])
            self.assertEqual(predictions, [('running', None, 0), ('running', 'w', 0),
                ('long', None, 1), ('short', None, 2), ('short', 'w', 10), ('long', 'w', 13)])
            self.assertEqual([t.case_attrs['Name'] for t in tokens], ['running', 'long', 'short'])
            self.assertTrue(all(t.prefix == ['A'] for t in tokens))
            self.assertEqual(random.getstate(), py_state)
            self.assertTrue(np.array_equal(np.random.get_state()[1], np_state[1]))

    def fixture(self, tmp):
        path = Path(tmp, 'log.xes'); path.touch()
        frame = pd.DataFrame({'case:concept:name': ['one', 'two'], 'concept:name': ['A', 'A'],
            'time:timestamp': pd.to_datetime(['2024-01-01T00:00Z', '2024-01-02T00:00Z']),
            'case:DueDate': ['2024-01-03T00:00:00Z', '2024-01-07T00:00:00Z'],
            'case:Type': ['fast', 'slow']})
        cfg = {'paths': {'aligned_log': str(path)}, 'features': {'case_attributes': [
            {'column': 'case:Type', 'feature_name': 'Type', 'type': 'categorical'}]},
            'simulation': {'queue': {'type': 'edd', 'edd': {'column': 'case:DueDate'}},
                'arrival': {'start_timestamp': '2025-01-01 00:00:00', 'distribution': 'constant',
                    'scale': 30, 'prophet': {'cache_model': False, 'model_cache_dir': tmp,
                                           'sampling_method': 'round'}}}}
        return frame, cfg

    def test_edd_replay_and_joint_generation_including_prophet(self):
        class Forecast:
            def __init__(self, **kwargs): pass
            def fit(self, frame): return self
            def predict(self, frame): return frame.assign(yhat=1)
        with tempfile.TemporaryDirectory() as tmp:
            frame, cfg = self.fixture(tmp)
            with patch('simulator.arrivals.pm4py.read_xes', return_value=frame), patch('prophet.Prophet', Forecast):
                replay, _ = ArrivalGenerator(cfg, num_cases=2).prepare_arrivals()
                self.assertEqual([a['due_date'] for a in replay],
                                 [pd.Timestamp(x) for x in frame['case:DueDate']])
                for mode in ('generative', 'prophet'):
                    arrivals, _ = ArrivalGenerator(cfg, num_cases=8, arrival_mode=mode).prepare_arrivals()
                    self.assertEqual(len(arrivals), 8)
                    for arrival in arrivals:
                        days = 2 if arrival['case_attrs']['Type'] == 'fast' else 5
                        expected = pd.to_datetime(arrival['arrival_time'], utc=True) + pd.Timedelta(days=days)
                        self.assertEqual(arrival['due_date'], expected)
                        self.assertEqual(set(arrival['case_attrs']), {'Type'})
                        self.assertNotIn('deadline_allowance', arrival)
                # Generation only validates/samples training cases, never the held-out deadline.
                frame.loc[1, 'case:DueDate'] = 'invalid held-out date'
                with patch('temporal_split.training_rows', return_value=frame.iloc[:1]):
                    arrivals, _ = ArrivalGenerator(cfg, num_cases=3, arrival_mode='generative').prepare_arrivals()
                    self.assertTrue(all(a['case_attrs']['Type'] == 'fast' for a in arrivals))

    def test_edd_end_to_end_order_and_trace_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame, cfg = self.fixture(tmp)
            frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
            frame['case:concept:name'] = ['running', 'late', 'early']
            frame['time:timestamp'] = pd.to_datetime(['2024-01-01T00:00:00Z',
                '2024-01-01T00:00:01Z', '2024-01-01T00:00:02Z'])
            frame['case:DueDate'] = ['2024-01-05T00:00Z', '2024-01-07T00:00Z', '2024-01-03T00:00Z']
            pnml = Path(tmp, 'net.pnml'); pnml.touch()
            roles = Path(tmp, 'roles.json')
            roles.write_text(json.dumps({'__all_resources__': ['w'], 'A': ['w']}))
            cfg['paths'].update(petri_net=str(pnml), models_dir=tmp)
            cfg['resources'] = {'discovery': {'output_file': str(roles)}}
            cfg['simulation'].update(calendar={'enabled': False}, arrival={'mode': 'replay'},
                output_simulated_log=str(Path(tmp, 'out.csv')),
                output_simulated_xes=str(Path(tmp, 'out.xes')))
            net = PetriNet('single'); p, q = PetriNet.Place('p'), PetriNet.Place('q')
            a = PetriNet.Transition('a', 'A'); net.places.update([p, q]); net.transitions.add(a)
            add_arc_from_to(p, a, net); add_arc_from_to(a, q, net)
            def duration(*args, **kwargs):
                self.assertEqual(set(kwargs['case_attrs']), {'Type'})
                return 10
            with patch('simulator.process.pm4py.read_pnml', return_value=(net, Marking({p: 1}), Marking({q: 1}))), \
                 patch('simulator.arrivals.pm4py.read_xes', return_value=frame), \
                 patch('activity_profiler.load_or_discover_activity_profiles', return_value={'activity_types': {'A': 'HUMAN_WORK_ITEM'}}), \
                 patch('simulator.process.ChampionPredictor', return_value=SimpleNamespace(predict_duration=duration)), \
                 patch('simulator.metrics.SimulationEvaluator.evaluate', return_value={}):
                sim = SimulationProcess(cfg, num_cases=3, mode='pure_physics')
                sim.run()
            csv = pd.read_csv(sim.evaluator.sim_log_path)
            self.assertEqual(csv.case_id.tolist(), ['running', 'early', 'late'])
            self.assertEqual(pd.to_datetime(csv['case:DueDate'], utc=True).tolist(),
                             [pd.Timestamp(frame.loc[i, 'case:DueDate']) for i in (0, 2, 1)])
            log = pm4py.read_xes(sim.evaluator.output_simulated_xes, return_legacy_log_object=True)
            expected = dict(zip(frame['case:concept:name'], frame['case:DueDate']))
            for trace in log:
                self.assertEqual(pd.Timestamp(trace.attributes['DueDate']),
                                 pd.Timestamp(expected[trace.attributes['concept:name']]))
                self.assertNotIn('DueDate', trace[0])

    def test_edd_missing_invalid_and_changing_deadlines_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            original, cfg = self.fixture(tmp)
            variants = [original.drop(columns='case:DueDate')]
            for value in (None, 'invalid', 123):
                frame = original.copy(); frame.loc[0, 'case:DueDate'] = value; variants.append(frame)
            later = original.iloc[:1].copy(); later['case:DueDate'] = '2024-02-01T00:00:00Z'
            variants.append(pd.concat([original, later], ignore_index=True))
            for frame in variants:
                for mode in ('replay', 'generative', 'prophet'):
                    with self.subTest(mode=mode), patch('simulator.arrivals.pm4py.read_xes', return_value=frame):
                        with self.assertRaisesRegex(ValueError, 'EDD'):
                            ArrivalGenerator(cfg, num_cases=2, arrival_mode=mode).prepare_arrivals()
            cfg['features']['case_attributes'].append({'column': 'case:DueDate', 'feature_name': 'Deadline'})
            with self.assertRaisesRegex(ValueError, 'must not be configured as an ML feature'):
                FeatureManager(cfg)


if __name__ == '__main__':
    unittest.main()
