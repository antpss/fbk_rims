"""Process-log contracts, optional fields, routing parity and experiment controls."""
from collections import defaultdict
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import pm4py
import simpy
import torch
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils.petri_utils import add_arc_from_to

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from config_loader import get_role_mapping_path, get_profile_paths, get_simulation_output_paths
from simulator.calendar import CalendarManager
from simulator.metrics import SimulationEvaluator
from simulator.predictor import ChampionPredictor
from simulator.process import SharedWorkerPool, SimulationProcess
from simulator.token import Token


def module(name):
    spec = importlib.util.spec_from_file_location('batch_three_' + name, ROOT / 'src' / (name + '.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def event(case, act, hour, lifecycle='COMPLETE'):
    return {'case:concept:name': case, 'concept:name': act,
            'time:timestamp': pd.Timestamp('2024-01-08', tz='UTC') + pd.Timedelta(hours=hour),
            'lifecycle:transition': lifecycle}


def sequence_net(labels):
    net = PetriNet('sequence')
    places = [PetriNet.Place(str(i)) for i in range(len(labels)+1)]
    net.places.update(places)
    for i, label in enumerate(labels):
        t = PetriNet.Transition('t'+str(i), label)
        net.transitions.add(t)
        add_arc_from_to(places[i], t, net)
        add_arc_from_to(t, places[i+1], net)
    return net, Marking({places[0]: 1}), Marking({places[-1]: 1})


class AlignmentTests(unittest.TestCase):
    def test_completion_projection_stays_aligned_with_auxiliary_states_and_tau_label(self):
        align = module('02_align_and_repair_log')
        net, im, fm = sequence_net(['A', None, 'tau_visible', 'B'])
        rows = [event('x', 'A', 8, 'SCHEDULE'), event('x', 'A', 8.5, 'ASSIGN'),
                event('x', 'A', 9, 'START'), event('x', 'A', 10),
                event('x', 'X', 10), event('x', 'tau_visible', 10), event('x', 'B', 10)]
        repaired, counts = align.repair_log(pd.DataFrame(rows), net, im, fm)
        completes = [e for e in repaired[0] if e['lifecycle:transition'] == 'COMPLETE']
        self.assertEqual([e['concept:name'] for e in completes], ['A', 'tau_visible', 'B'])
        self.assertEqual(sum(e['lifecycle:transition'] == 'START' for e in repaired[0]), 1)
        self.assertEqual(counts['other_lifecycle'], 2)
        self.assertEqual(counts['log'], 1)
        # Verify the reconstructed completion projection against the real net.
        from pm4py.objects.log.obj import EventLog, Trace
        result = pm4py.conformance_diagnostics_alignments(EventLog([Trace(completes)]), net, im, fm)
        self.assertTrue(all(a == b or b is None for a, b in result[0]['alignment']))

    def test_missing_visible_move_is_synthetic_and_bad_alignment_fails(self):
        align = module('02_align_and_repair_log')
        net, im, fm = sequence_net(['A', 'tau_visible', 'B'])
        df = pd.DataFrame([event('x', 'A', 9), event('x', 'B', 10)])
        repaired, counts = align.repair_log(df, net, im, fm)
        self.assertEqual([e['concept:name'] for e in repaired[0]], ['A', 'tau_visible', 'B'])
        self.assertEqual(repaired[0][1]['is_synthetic'], 'true')
        self.assertEqual(counts['model'], 1)
        with patch.object(align.pm4py, 'conformance_diagnostics_alignments', return_value=[{'alignment': [('B', 'B')]}]):
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                align.repair_log(df, net, im, fm)


class OptionalLogAndPathTests(unittest.TestCase):
    def test_missing_columns_pipeline_and_custom_role_paths(self):
        roles, delays, builder = [module(n) for n in ('01b_discover_roles', '02b_discover_delays', '03_dataset_builder')]
        for has_starts in (False, True):
            with self.subTest(has_starts=has_starts), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                log = root/'log.xes'; log.touch()
                rows = [event('x', 'P', 9), event('x', 'A', 10)]
                if has_starts:
                    rows.append(event('x', 'H', 10, 'START'))
                rows.append(event('x', 'H', 11))
                frame = pd.DataFrame(rows)
                if not has_starts:
                    frame = frame.drop(columns='lifecycle:transition')
                cfg = {'paths': {'raw_log': str(log), 'aligned_log': str(log), 'models_dir': str(root/'models'), 'output_base_dir': str(root/'data')},
                       'resources': {'discovery': {'output_file': str(root/'custom/roles.json'), 'report_file': str(root/'reports/roles.md')}},
                       'activity_profiling': {'output_file': str(root/'custom/profiles.json'), 'report_file': str(root/'reports/profiles.md'),
                                              'overrides': {'P': 'INSTANT_GATEWAY', 'A': 'AUTOMATED_TASK', 'H': 'HUMAN_WORK_ITEM'}},
                       'features': {'selected_features': ['wip', 'prev_duration']},
                       'simulation': {'calendar': {'use_discovered_role_calendars': True}},
                       'tasks': {'duration': {'target_column': 'Duration_Seconds', 'filter_zero_duration': True, 'output_file': 'duration.csv'},
                                 'waiting_time': {'target_column': 'Wait_Time_Seconds', 'target_mode': 'residual_delay', 'output_file': 'waiting.csv'}}}
                with patch.object(pm4py, 'read_xes', return_value=frame):
                    role_data = roles.discover_roles(cfg)
                    cal = CalendarManager.from_config(cfg)
                    self.assertEqual(cal.activity_to_role, role_data['activity_to_role'])
                    engine = delays.ChapelaDelayDiscoveryEngine(cfg)
                    self.assertEqual(engine.calendar_mgr.activity_to_role, role_data['activity_to_role'])
                    engine.run()
                    waiting = builder.build_dataset(cfg, 'waiting_time')
                    self.assertEqual(waiting[waiting.Target_Activity == 'A'].iloc[0].Wait_Time_Seconds, 3600)
                    if has_starts:
                        duration = builder.build_dataset(cfg, 'duration')
                        self.assertEqual(duration.Target_Activity.tolist(), ['H'])
                        self.assertEqual(duration.iloc[0].Duration_Seconds, 3600)
                    else:
                        self.assertNotIn('H', set(waiting.Target_Activity))
                        with self.assertRaisesRegex(ValueError, 'No usable duration'):
                            builder.build_dataset(cfg, 'duration')
                self.assertTrue(Path(get_role_mapping_path(cfg)).exists())
                self.assertTrue(all(Path(p).exists() for p in get_profile_paths(cfg)))
                self.assertFalse((root/'models/role_mapping.json').exists())
                # Runtime's calendar and worker pool must use the same custom manifest.
                pnml = root/'net.pnml'; pnml.touch()
                cfg['paths']['petri_net'] = str(pnml)
                with patch.object(pm4py, 'read_pnml', return_value=sequence_net(['H'])), \
                     patch('simulator.process.ChampionPredictor'):
                    process = SimulationProcess(cfg)
                self.assertEqual(process.workers.activity_to_role, role_data['activity_to_role'])
                self.assertEqual(process.calendar.activity_to_role, role_data['activity_to_role'])
                profiler = module('activity_profiler')
                with patch.object(sys, 'argv', ['profiler']), patch.object(profiler, 'load_config', return_value=cfg), \
                     patch.object(pm4py, 'read_xes', return_value=frame):
                    profiler.main()
                self.assertFalse((root/'models/activity_profiles.json').exists())


    def test_simulation_outputs_follow_config_in_both_modes(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = {'simulation': {'output_simulated_log': str(Path(tmp, 'nested/custom.csv')),
                                  'output_simulated_xes': str(Path(tmp, 'other/custom.xes'))}}
            for mode, suffix in [('hybrid_residual', ''), ('pure_physics', '_pure_physics')]:
                csv, xes = get_simulation_output_paths(cfg, mode)
                self.assertEqual(csv, str(Path(tmp, 'nested/custom'+suffix+'.csv')))
                self.assertEqual(xes, str(Path(tmp, 'other/custom'+suffix+'.xes')))
                evaluator = SimulationEvaluator(cfg, mode=mode)
                self.assertEqual((evaluator.sim_log_path, evaluator.output_simulated_xes), (csv, xes))


class RoutingAndTimingTests(unittest.TestCase):
    def test_categorical_xgboost_and_numeric_tree_runtime_parity(self):
        cats = ['A', 'B']
        X = pd.DataFrame({'prev_activity': pd.Categorical(['A', 'B']*30, categories=cats)})
        y = np.array([0, 1]*30)
        for model in (xgb.XGBClassifier(n_estimators=10, max_depth=2, enable_categorical=True, n_jobs=1), DecisionTreeClassifier(random_state=12)):
            is_xgb = isinstance(model, xgb.XGBClassifier)
            model.fit(X if is_xgb else X.assign(prev_activity=X.prev_activity.cat.codes), y)
            predictor = ChampionPredictor.__new__(ChampionPredictor)
            predictor.cfg = {'tasks': {'routing': {'inference_mode': 'argmax'}}}
            predictor.routing_models = {'p': model}
            predictor.routing_manifest = {'p': {'strategy': 'machine_learning', 'features': ['prev_activity'],
                                               'cat_mappings': {'prev_activity': cats}, 'label_encoding': {'left': 0, 'right': 1},
                                               'probabilities': {'fallback': 1}}}
            for value in ['A', 'B', 'unseen']:
                if is_xgb:
                    sample = pd.DataFrame({'prev_activity': pd.Categorical([value], categories=cats)})
                else:
                    sample = pd.DataFrame({'prev_activity': [cats.index(value) if value in cats else -1]})
                expected = ['left', 'right'][int(model.predict(sample)[0])]
                self.assertEqual(predictor.predict_routing('p', [value]), expected)

    def test_repeated_activity_keeps_predicted_wait(self):
        env = simpy.Environment()
        process = SimpleNamespace(env=env, im=Marking(), calendar=CalendarManager(enabled=False), mode='hybrid_residual',
                                  target_activities=['A'], active_cases_count=1, ac_wip={}, external_delays={},
                                  workers=SimpleNamespace(get_role_occupancy=lambda _: []),
                                  predictor=SimpleNamespace(predict_waiting=lambda *args, **kwargs: 100),
                                  has_ex_ante_delay=lambda act: True, sample_ex_ante_delay=lambda act: 40,
                                  activity_profiles={'A': 'AUTOMATED_TASK'}, record_event=lambda r: None)
        token = Token('x', 0, process=process)
        token.prefix = ['A', 'A']
        env.process(token._execute_work_item(PetriNet.Transition('a', 'A'), lambda: None))
        env.run()
        self.assertEqual(env.now, 100)  # existing max(40,100), without automatic quartering

    def test_cycle_difference_is_not_paired_mae(self):
        real = pd.DataFrame([event('x', 'A', 0, 'START'), event('x', 'A', 1),
                             event('y', 'A', 0, 'START'), event('y', 'A', 3)])
        sim = pd.DataFrame([{'case_id': c, 'activity': 'A', 'start_timestamp': '2024-01-08 00:00:00+00:00',
                             'complete_timestamp': f'2024-01-08 0{h}:00:00+00:00', 'resource_id': 'w'} for c,h in [('x',3),('y',1)]])
        with tempfile.TemporaryDirectory() as tmp:
            result = SimulationEvaluator({'paths': {'output_base_dir': tmp}}, export_xes=False).evaluate(real, sim)
            self.assertEqual(result['mean_cycle_difference'], 0)
            self.assertNotIn('mae_cycle', result)
            self.assertEqual(result['metric_units']['mean_cycle_difference'], 'seconds')


class ReproducibilityTests(unittest.TestCase):
    def test_neural_initialization_dropout_and_shuffle_repeat(self):
        train = module('04_train')
        def draw(seed):
            train.seed_training(seed)
            model = torch.nn.Sequential(torch.nn.Linear(3, 8), torch.nn.Dropout(.5), torch.nn.Linear(8, 1))
            loader = torch.utils.data.DataLoader(torch.arange(24).reshape(8, 3).float(), batch_size=2, shuffle=True)
            optimizer = torch.optim.Adam(model.parameters(), lr=.01)
            outputs = []
            for batch in loader:
                output = model(batch)
                outputs.extend(output.detach().flatten().tolist())
                optimizer.zero_grad(); output.square().mean().backward(); optimizer.step()
            return outputs, random.random(), np.random.random()
        self.assertEqual(draw(73), draw(73))
        self.assertNotEqual(draw(73), draw(74))

    def test_worker_dispatch_repeats_across_hash_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            role_file = Path(tmp, 'roles.json')
            role_file.write_text(json.dumps({'__all_resources__': ['alice','bob','carol'], 'A': ['alice','bob','carol']}))
            code = '''import sys, random, json, simpy
from simulator.process import SharedWorkerPool
random.seed(73)
p = SharedWorkerPool(simpy.Environment(), sys.argv[1])
p.pending_requests = [{'activity':'A', 'eligible':set(p.all_workers), 'event':p.env.event()} for _ in range(3)]
events = [r['event'] for r in p.pending_requests]
p._dispatch_all_pending()
print('DISPATCH=' + json.dumps([e.value for e in events]))
'''
            results = []
            for hash_seed in ['1', '77']:
                env = dict(os.environ, PYTHONHASHSEED=hash_seed, PYTHONPATH=str(ROOT/'src'))
                out = subprocess.check_output([sys.executable, '-c', code, str(role_file)], env=env, text=True)
                results.append(next(line for line in out.splitlines() if line.startswith('DISPATCH=')))
            self.assertEqual(*results)


if __name__ == '__main__':
    unittest.main()
