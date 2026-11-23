"""Regressions for audit findings 17, 18, 19, 22, 31 and 34."""
from collections import defaultdict
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd
import simpy
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils.petri_utils import add_arc_from_to

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from simulator.calendar import CalendarManager
from simulator.metrics import prepare_evaluation_dfs
from simulator.process import SharedWorkerPool
from simulator.token import Token


def module(name):
    spec = importlib.util.spec_from_file_location('six_' + name, Path(__file__).resolve().parents[1] / 'src' / (name + '.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def event(case, act, hour, lifecycle, resource='w'):
    return {'case:concept:name': case, 'concept:name': act,
            'time:timestamp': pd.Timestamp('2024-01-08', tz='UTC') + pd.Timedelta(hours=hour),
            'lifecycle:transition': lifecycle, 'org:resource': resource}


class DelayAndRoutingTests(unittest.TestCase):
    def test_automated_availability_matches_runtime(self):
        discovery = module('02b_discover_delays')
        for estimator in ('eclaw_plus', 'eclaw', 'naive'):
            for start, end in ((20, 32), (9, 11)):
                with self.subTest(estimator=estimator, start=start), tempfile.TemporaryDirectory() as tmp:
                    log = Path(tmp, 'log.xes'); log.touch()
                    frame = pd.DataFrame([event('x', 'P', start, 'COMPLETE'), event('x', 'A', end, 'COMPLETE'),
                                          event('y', 'H', start, 'START'), event('y', 'H', end, 'COMPLETE')])
                    cfg = {'paths': {'aligned_log': str(log), 'models_dir': tmp, 'output_base_dir': tmp},
                           'simulation': {'calendar': {'enabled': True}},
                           'activity_profiling': {'overrides': {'A': 'AUTOMATED_TASK', 'H': 'HUMAN_WORK_ITEM'}}}
                    with patch.object(discovery.pm4py, 'read_xes', return_value=frame):
                        discovery.ChapelaDelayDiscoveryEngine(cfg, estimator=estimator, min_samples=1).run()
                    table = pd.read_csv(Path(tmp, 'causal_delays_instance_level.csv.gz'))
                    a = table[table.target_act == 'A'].iloc[0]
                    self.assertEqual(a.residual_delay_seconds, (end-start)*3600)
                    self.assertEqual(a.calendar_nw_seconds, 0)
                    self.assertEqual(a.worker_busy_seconds, 0)
                    # A human target keeps the physical unavailability correction.
                    cfg['activity_profiling']['overrides']['A'] = 'HUMAN_WORK_ITEM'
                    with patch.object(discovery.pm4py, 'read_xes', return_value=frame):
                        discovery.ChapelaDelayDiscoveryEngine(cfg, estimator=estimator, min_samples=1).run()
                    human = pd.read_csv(Path(tmp, 'causal_delays_instance_level.csv.gz'))
                    self.assertEqual(human[human.target_act == 'A'].iloc[0].residual_delay_seconds, 0)

    def test_routing_uses_arrival_and_observable_history(self):
        routing = module('05_train_routing')
        net = PetriNet('routing')
        net.transitions.update([PetriNet.Transition('a', 'A'), PetriNet.Transition('tau', None), PetriNet.Transition('b', 'B')])
        frame = pd.DataFrame([event('x', 'A', 9, 'START'), event('x', 'A', 11, 'COMPLETE'), event('x', 'B', 12, 'COMPLETE'),
                              event('y', 'A', 10, 'START'), event('y', 'A', 14, 'COMPLETE'), event('y', 'B', 15, 'COMPLETE')])
        moves = {'alignment': [(('a', 'a'), ('A', 'A')), (('>>', 'tau'), ('>>', None)), (('b', 'b'), ('B', 'B'))]}
        with patch.object(routing.pm4py, 'read_xes', return_value=frame), \
             patch.object(routing.pm4py, 'read_pnml', return_value=(net, Marking(), Marking())), \
             patch.object(routing.alignments, 'apply', return_value=[moves, moves]):
            samples = routing.extract_decision_samples_via_alignment('unused', 'unused', {'xor': {'branches': {'b': {}}}},
                                                                       case_attrs_cfg=[], routing_mode='both')['xor'].set_index('case_id')
        self.assertEqual(samples.loc['x', 'start_hour'], 9)
        self.assertEqual(samples.loc['y', 'start_hour'], 10)
        self.assertEqual(samples.loc['y', 'inter_arrival_seconds'], 3600)
        self.assertEqual(samples.loc['x', 'start_weekday'], 0)
        self.assertEqual(samples.prev_activity.tolist(), ['A', 'A'])
        self.assertEqual(samples.has_A.tolist(), [1, 1])


class EvaluationPairingTests(unittest.TestCase):
    def test_chronological_fifo_handles_missing_starts_and_overlap(self):
        real = pd.DataFrame([event('x', 'A', 9, 'COMPLETE'), event('x', 'A', 10, 'START'),
                             event('x', 'A', 10.5, 'START'), event('x', 'A', 11, 'COMPLETE'),
                             event('x', 'A', 12, 'COMPLETE'), event('y', 'A', 10, 'START'),
                             event('y', 'A', 10, 'COMPLETE'), event('z', 'A', 13, 'START')]).sample(frac=1, random_state=7)
        sim = pd.DataFrame([{'case_id': 'x', 'activity': 'A', 'start_timestamp': '2024-01-08 10:00',
                             'complete_timestamp': '2024-01-08 11:00', 'resource_id': 'w'}])
        real, _, _ = prepare_evaluation_dfs(real, sim)
        x = real[real.case_id == 'x'].sort_values('end_time')
        self.assertEqual((x.end_time-x.start_time).dt.total_seconds().tolist(), [0, 3600, 5400])
        y = real[real.case_id == 'y'].iloc[0]
        self.assertEqual(y.start_time, y.end_time)
        self.assertNotIn('z', set(real.case_id))  # no invented completion


class CooldownTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        path = Path(self.tmp.name, 'roles.json')
        path.write_text(json.dumps({'__all_resources__': ['w'], 'A': ['w'], 'worker_concurrencies': {'w': 2}}))
        self.env = simpy.Environment()
        self.pool = SharedWorkerPool(self.env, path, latency_cfg={'enabled': True, 'queue_aware': True})

    def test_discovery_distribution_schemas_and_bounds(self):
        for name, sampler, params in [('lognormal', 'lognorm', {'shape': 1, 'scale': 20}),
                                       ('gamma', 'gamma', {'shape': 2, 'scale': 10}),
                                       ('exponential', 'expon', {'loc': 0, 'scale': 15})]:
            with self.subTest(distribution=name):
                self.pool.latency_cfg = {'max_cooldown_seconds': 30, 'per_worker_profiles': {'w': {'distribution': name, 'params': params}}}
                with patch('simulator.process.st.' + sampler + '.rvs', return_value=12) as sample:
                    self.assertEqual(self.pool._sample_cooldown('w'), 12)
                    if name == 'exponential':
                        sample.assert_called_once_with(**params)
                    else:
                        sample.assert_called_once_with(params['shape'], loc=0.0, scale=params['scale'])
                with patch('simulator.process.st.' + sampler + '.rvs', return_value=100):
                    self.assertEqual(self.pool._sample_cooldown('w'), 30)
        self.pool.latency_cfg = {'global_distribution': 'exponential', 'global_params': {'scale': 20}, 'max_cooldown_seconds': 30}
        with patch('simulator.process.st.expon.rvs', return_value=12):
            self.assertEqual(self.pool._sample_cooldown('missing'), 12)
        self.pool.latency_cfg = {'max_cooldown_seconds': 1, 'per_worker_profiles': {'w': {'distribution': 'lognormal', 'params': {}}}}
        with patch('simulator.process.random.expovariate', return_value=42):
            self.assertEqual(self.pool._sample_cooldown('w'), 1)
        self.pool.latency_cfg['max_cooldown_seconds'] = 0
        self.assertEqual(self.pool._sample_cooldown('w'), 0)

    def test_break_only_after_final_release_and_old_cancel_cannot_clear_new_break(self):
        p = self.pool
        p.request_worker('A'); p.request_worker('A')
        with patch.object(p, '_sample_cooldown', side_effect=[10, 100]) as sample:
            p.release_worker('w', 'A')
            self.assertFalse(p.worker_in_cooldown['w'])
            sample.assert_not_called()
            p.release_worker('w', 'A')
            old = p.worker_cooldown_events['w']
            self.assertTrue(p.worker_in_cooldown['w'])
            # Interrupt and start a replacement break before the old process resumes.
            self.assertTrue(p.request_worker('A').triggered)
            self.assertTrue(old.triggered)
            p.release_worker('w', 'A')
            new = p.worker_cooldown_events['w']
            self.assertIsNot(old, new)
            self.env.run(until=11)
            self.assertTrue(p.worker_in_cooldown['w'])
            self.assertIs(p.worker_cooldown_events['w'], new)
            self.env.run(until=101)
            self.assertFalse(p.worker_in_cooldown['w'])

    def test_queue_aware_release_dispatches_without_break(self):
        p = self.pool
        p.request_worker('A'); p.request_worker('A')
        waiting = p.request_worker('A')
        with patch.object(p, '_sample_cooldown') as sample:
            p.release_worker('w', 'A')
            self.assertTrue(waiting.triggered)
            self.assertEqual(p.worker_active_count['w'], 2)
            sample.assert_not_called()

    def test_noninterruptible_break_dispatches_at_expiry(self):
        p = self.pool
        p.latency_cfg['queue_aware'] = False
        p.request_worker('A')
        with patch.object(p, '_sample_cooldown', return_value=10):
            p.release_worker('w', 'A')
        waiting = p.request_worker('A')
        self.env.run(until=9)
        self.assertFalse(waiting.triggered)
        self.env.run(until=11)
        self.assertTrue(waiting.triggered)
        self.assertEqual(p.worker_active_count['w'], 1)


class EventBudgetTests(unittest.TestCase):
    def make_process(self, budget, profiles=None):
        env = simpy.Environment(); net = PetriNet('parallel')
        p, q, f, g, sink = [PetriNet.Place(n) for n in ('p', 'q', 'f', 'g', 'sink')]
        net.places.update([p, q, f, g, sink])
        a, b, join = PetriNet.Transition('a', 'A'), PetriNet.Transition('b', 'B'), PetriNet.Transition('join', None)
        net.transitions.update([a, b, join])
        for source, target in [(p, a), (a, f), (q, b), (b, g), (f, join), (g, join), (join, sink)]:
            add_arc_from_to(source, target, net)
        workers = SimpleNamespace(get_role_occupancy=lambda *args: [], request_worker=lambda act: env.event().succeed('w'),
                                  release_worker=lambda *args: None)
        records = []
        process = SimpleNamespace(env=env, net=net, im=Marking({p: 1, q: 1}), fm=Marking({sink: 1}),
                                  calendar=CalendarManager(enabled=False), mode='pure_physics', active_cases_count=0,
                                  max_events=budget, target_activities=['A', 'B'], external_delays={},
                                  has_ex_ante_delay=lambda act: False, activity_profiles=profiles or {}, workers=workers,
                                  ac_wip=defaultdict(int), predictor=SimpleNamespace(predict_duration=lambda act, *args, **kwargs: 1 if act == 'A' else 10),
                                  record_event=records.append)
        token = Token('x', 0, process=process)
        env.process(token.run())
        return env, process, token, records

    def test_parallel_batch_cannot_exceed_budget(self):
        env, p, token, records = self.make_process(1)
        env.run(until=.5)
        self.assertEqual(token.events_done, 1)
        self.assertEqual(p.active_cases_count, 1)
        env.run()
        self.assertEqual(token.status, 'limit_failed')
        self.assertEqual([r['activity'] for r in records], ['A'])
        self.assertEqual(records[0]['wip'], 1)
        self.assertEqual(token.pending_work, [])
        self.assertEqual(p.active_cases_count, 0)

    def test_exact_budget_drains_parallel_work_and_silent_join(self):
        env, p, token, records = self.make_process(2)
        env.run(until=2)
        self.assertEqual(token.events_done, 2)
        self.assertEqual(p.active_cases_count, 1)
        self.assertEqual(token.status, 'running')
        self.assertEqual(len(token.pending_work), 1)
        env.run()
        self.assertEqual(token.status, 'completed')
        self.assertEqual(token.finished_at, 10)
        self.assertEqual([r['wip'] for r in records], [1, 1])
        self.assertEqual(p.active_cases_count, 0)

    def test_instant_and_automated_events_share_budget(self):
        for typ in ('INSTANT_GATEWAY', 'AUTOMATED_TASK'):
            with self.subTest(type=typ):
                env, p, token, records = self.make_process(1, {'A': typ, 'B': typ})
                env.run()
                self.assertEqual(len(records), 1)
                self.assertEqual(token.events_done, 1)
                self.assertEqual(token.status, 'limit_failed')
                self.assertEqual(p.active_cases_count, 0)


if __name__ == '__main__':
    unittest.main()
