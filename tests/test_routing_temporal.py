"""Routing temporal contract, alignment causality and serving compatibility."""
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock

import pandas as pd
from pm4py.objects.petri_net.obj import PetriNet, Marking

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from routing_features import routing_temporal_features, TEMPORAL_COLUMNS
from simulator.predictor import ChampionPredictor
from simulator.token import Token
from test_remaining_six_fixes import module, event


class RoutingTemporalTests(unittest.TestCase):
    def test_elapsed_calendar_time_and_validation(self):
        start = pd.Timestamp('2024-01-05 23:00', tz='UTC')
        last = pd.Timestamp('2024-01-08 01:00', tz='UTC')
        self.assertEqual(routing_temporal_features(start, last), dict(zip(
            TEMPORAL_COLUMNS, [23, 4, 1, 0, 180000.0])))
        self.assertEqual(routing_temporal_features(start)['elapsed_since_start'], 0)
        for bad in (None, pd.NaT):
            with self.assertRaises(ValueError):
                routing_temporal_features(bad)
        with self.assertRaises(ValueError):
            routing_temporal_features(last, start)
        with self.assertRaises(ValueError):
            routing_temporal_features(start, pd.NaT)
        with self.assertRaises(ValueError):
            routing_temporal_features(start, last.tz_localize(None))

    def predictor(self, columns):
        p = ChampionPredictor.__new__(ChampionPredictor)
        p.cfg = {'tasks': {'routing': {'inference_mode': 'argmax'}}}
        model = Mock()
        model.predict.return_value = [0]
        p.routing_models = {'p': model}
        p.routing_manifest = {'p': {'strategy': 'machine_learning', 'features': list(columns),
                                   'label_encoding': {'B': 0}}}
        return p, model

    def test_serving_parity_legacy_and_required_start(self):
        start = pd.Timestamp('2024-01-05 23:00', tz='UTC')
        last = pd.Timestamp('2024-01-08 01:00', tz='UTC')
        p, model = self.predictor(TEMPORAL_COLUMNS)
        self.assertEqual(p.predict_routing('p', [], trace_start_dt=start, last_event_dt=last), 'B')
        self.assertEqual(model.predict.call_args.args[0][0].tolist(),
                         list(routing_temporal_features(start, last).values()))
        with self.assertRaises(ValueError):
            p.predict_routing('p', [], curr_dt=start)
        with self.assertRaises(ValueError):
            p.predict_routing('p', [], trace_start_dt=start, case_attrs={'start_hour': 3})
        p, model = self.predictor(['hour', 'weekday'])
        p.predict_routing('p', [], curr_dt=start, last_event_dt=last)
        self.assertEqual(model.predict.call_args.args[0][0].tolist(), [23, 4])
        p.routing_temporal_version = 1
        with self.assertRaises(ValueError):
            p.predict_routing('p', [], trace_start_dt=start)

    def test_alignment_consumes_only_observed_events_before_choice(self):
        routing = module('05_train_routing')
        net = PetriNet('routing')
        net.transitions.update([PetriNet.Transition('a', 'A'), PetriNet.Transition('tau'),
                                PetriNet.Transition('m', 'M'), PetriNet.Transition('b', 'B')])
        frame = pd.DataFrame([event('x', 'A', 9, 'START'), event('x', 'A', 11, 'COMPLETE'),
                              event('x', 'Extra', 12, 'COMPLETE'), event('x', 'B', 35, 'COMPLETE'),
                              event('x', 'B', 35, 'COMPLETE')]).iloc[::-1]
        moves = {'alignment': [(('a', 'a'), ('A', 'A')), (('extra', '>>'), ('Extra', '>>')),
                               (('>>', 'tau'), ('>>', None)), (('>>', 'm'), ('>>', 'M')),
                               (('b', 'b'), ('B', 'B')), (('b', 'b'), ('B', 'B'))]}
        with patch.object(routing.pm4py, 'read_xes', return_value=frame), \
             patch.object(routing.pm4py, 'read_pnml', return_value=(net, Marking(), Marking())), \
             patch.object(routing.alignments, 'apply', return_value=[moves]):
            samples = routing.extract_decision_samples_via_alignment('log', 'net',
                {'p': {'branches': {'a': {}, 'b': {}}}}, case_attrs_cfg=[])['p']
        self.assertEqual(samples.start_hour.tolist(), [9, 9, 9])
        self.assertEqual(samples.last_event_hour.tolist(), [9, 12, 11])
        self.assertEqual(samples.last_event_weekday.tolist(), [0, 0, 1])
        self.assertEqual(samples.elapsed_since_start.tolist(), [0, 10800, 93600])
        expected = routing_temporal_features(event('x', 'A', 9, 'START')['time:timestamp'],
                                             event('x', 'A', 12, 'COMPLETE')['time:timestamp'])
        self.assertEqual(samples.iloc[1][list(TEMPORAL_COLUMNS)].to_dict(), expected)

    def test_invalid_log_timestamp_identifies_case(self):
        routing = module('05_train_routing')
        frame = pd.DataFrame([event('broken', 'A', 9, 'COMPLETE')])
        frame['time:timestamp'] = pd.NaT
        with patch.object(routing.pm4py, 'read_xes', return_value=frame), \
             patch.object(routing.pm4py, 'read_pnml', return_value=(PetriNet('n'), Marking(), Marking())):
            with self.assertRaisesRegex(ValueError, 'broken'):
                routing.extract_decision_samples_via_alignment('log', 'net', {})

    def test_completion_clock_is_monotonic(self):
        start = pd.Timestamp('2024-01-01', tz='UTC')
        token = Token.__new__(Token)
        token.last_event_dt = start
        token.env = SimpleNamespace(now=3600)
        token.process = SimpleNamespace(calendar=SimpleNamespace(
            to_datetime=lambda t: start + pd.Timedelta(seconds=t)))
        for t in (3600, 3600, 1800, 7200):
            token.env.now = t
            token._record_completion_time()
        self.assertEqual(token.last_event_dt, start + pd.Timedelta(hours=2))


if __name__ == '__main__':
    unittest.main()
