"""Regressions for the explicitly approved residual-audit repairs."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings

import pandas as pd
import simpy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from log_utils import read_delay_records
from model_artifacts import learning_settings, warn_learning_settings
from temporal_split import stamp_artifact, require_artifact
from simulator.calendar import CalendarManager
from simulator.process import SharedWorkerPool, SimulationProcess
from simulator.arrivals import ArrivalGenerator


def module(filename):
    spec = importlib.util.spec_from_file_location(filename, ROOT / 'src' / (filename + '.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class AuditFollowupTests(unittest.TestCase):
    def test_configuration_drift_warns_but_content_corruption_still_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, 'data.csv'); path.write_text('original')
            cfg = {'model_strategy': {}, '_split_manifest': {'split_id': 'x'},
                   'simulation': {'calendar': {'hour_end': 20}}}
            stamp_artifact(cfg, path)
            with warnings.catch_warnings(record=True) as messages:
                warnings.simplefilter('always')
                require_artifact(cfg, path)
            self.assertEqual(messages, [])
            cfg['simulation']['calendar']['hour_end'] = 16
            with self.assertWarnsRegex(UserWarning, 'calendar'):
                require_artifact(cfg, path)
            path.write_text('corrupt')
            with self.assertRaisesRegex(ValueError, 'provenance'):
                require_artifact(cfg, path)

    def test_training_settings_ignore_output_paths_and_warn_for_legacy(self):
        cfg = {'resources': {'discovery': {'output_file': 'a', 'threshold': .3}}}
        metadata = {'learning_settings': copy.deepcopy(learning_settings(cfg))}
        cfg['resources']['discovery']['output_file'] = 'b'
        with warnings.catch_warnings(record=True) as messages:
            warnings.simplefilter('always')
            warn_learning_settings(cfg, metadata, 'model')
        self.assertEqual(messages, [])
        cfg['resources']['discovery']['threshold'] = .5
        with self.assertWarnsRegex(UserWarning, 'resources'):
            warn_learning_settings(cfg, metadata, 'model')
        with self.assertWarnsRegex(UserWarning, 'no saved settings'):
            warn_learning_settings(cfg, {}, 'legacy')

    def test_role_calendar_queues_and_wakes_with_global_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, 'roles.json')
            path.write_text(json.dumps({'__all_resources__': ['w'], 'A': ['w'],
                                       'activity_to_role': {'A': 'r'}, 'role_to_workers': {'r': ['w']}}))
            env = simpy.Environment()
            calendar = CalendarManager(enabled=False, ref_dt=pd.Timestamp('2024-01-08 21:00'),
                                       activity_to_role={'A': 'r'},
                                       per_role={'r': {'enabled': True, 'hour_start': 8, 'hour_end': 20}})
            pool = SharedWorkerPool(env, path, calendar=calendar)
            request = pool.request_worker('A')
            self.assertFalse(request.triggered)
            env.run(until=11*3600-1)
            self.assertFalse(request.triggered)
            env.run(until=11*3600+1)
            self.assertTrue(request.triggered)
            self.assertEqual(request.value, 'w')

    def test_unsupported_placement_rejected_before_loading_artifacts(self):
        discovery = module('02b_discover_delays')
        for value in ('ex_post', 'causal_pair'):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, 'only.*ex_ante'):
                    discovery.ChapelaDelayDiscoveryEngine({}, placement=value)
                with self.assertRaisesRegex(ValueError, 'only.*ex_ante'):
                    SimulationProcess({'simulation': {'extraneous_delays': {'placement': value}}})

    def test_prefix_parser_accepts_lists_and_csv_strings(self):
        parse = module('04_train').parse_prefix
        for value, expected in [(['W_A', 'Triage'], ['W_A', 'Triage']),
                                ('["W_A", "Triage"]', ['W_A', 'Triage']),
                                ([], []), (None, []), (float('nan'), [])]:
            self.assertEqual(parse(value), expected)

    def test_csv_fallback_without_parquet_engine_does_not_hide_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = {'paths': {'output_base_dir': tmp}}
            parquet = Path(tmp, 'causal_delays_instance_level.parquet')
            parquet.touch()
            pd.DataFrame({'residual_delay_seconds': [12.5]}).to_csv(
                Path(tmp, 'causal_delays_instance_level.csv.gz'), index=False)
            with patch('pandas.read_parquet', side_effect=ImportError('no engine')):
                with self.assertWarnsRegex(UserWarning, 'CSV'):
                    table = read_delay_records(cfg)
            self.assertEqual(table.residual_delay_seconds.tolist(), [12.5])
            with patch('pandas.read_parquet', side_effect=ValueError('corrupt table')):
                with self.assertRaisesRegex(ValueError, 'corrupt table'):
                    read_delay_records(cfg)

    def test_prophet_cache_invalidates_activity_filter_changes(self):
        class Forecast:
            fits = 0
            def __init__(self, **kwargs): pass
            def fit(self, frame):
                Forecast.fits += 1
                return self
            def predict(self, frame): return frame.assign(yhat=1)
        historical = pd.DataFrame({'case:concept:name': ['1', '2', '3', '4'],
                                   'concept:name': ['A', 'B', 'A', 'B'],
                                   'time:timestamp': pd.date_range('2024-01-01', periods=4, freq='D')})
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp, 'log.xes'); log.touch()
            cfg = {'paths': {'aligned_log': str(log)}, 'simulation': {'arrival': {
                'start_timestamp': '2024-01-01 00:00:00', 'calendar_bounded': False,
                'prophet': {'cache_model': True, 'model_cache_dir': tmp, 'sampling_method': 'round'}}}}
            with patch('prophet.Prophet', Forecast), \
                 patch('prophet.serialize.model_to_json', return_value='fixture'), \
                 patch('prophet.serialize.model_from_json', side_effect=lambda _: Forecast()):
                for selected, expected_fits in [(['A', 'B'], 1), (['B', 'A'], 1), (['A'], 2)]:
                    cfg['target_activities'] = selected
                    ArrivalGenerator(cfg, num_cases=1)._prepare_prophet_arrivals(historical, [], None)
                    self.assertEqual(Forecast.fits, expected_fits)
